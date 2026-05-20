"""
Orchestrator — runs all three engines and assembles the OverallGrade.

Pipeline:
  1. Run FundamentalEngine, TechnicalEngine, QuantitativeEngine
  2. Weighted composite (weights from config or portfolio override)
  3. Apply circuit breakers (hard caps)
  4. Map composite → Grade
  5. Compute confidence (data completeness × signal agreement)
  6. Extract top-3 positive and negative drivers
  7. Return OverallGrade + EngineResults
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np

from stockgrader.config import get_config, get_engine_weights
from stockgrader.engines.fundamental  import FundamentalEngine
from stockgrader.engines.technical    import TechnicalEngine
from stockgrader.engines.quantitative import QuantitativeEngine
from stockgrader.grading.circuit_breakers import apply_circuit_breakers
from stockgrader.models import (
    EngineResults,
    FundamentalResult,
    OverallGrade,
    QuantitativeResult,
    TechnicalResult,
    composite_to_grade,
)

logger = logging.getLogger(__name__)


class Orchestrator:
    """
    Runs all three scoring engines and produces the overall grade.

    Designed to be stateless per-call: same inputs → same output.
    Can be re-used across many tickers in a session.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        self._cfg   = config or get_config()
        self._fund  = FundamentalEngine(self._cfg)
        self._tech  = TechnicalEngine(self._cfg)
        self._quant = QuantitativeEngine(self._cfg)

    # ── Public API ─────────────────────────────────────────────

    def analyze(
        self,
        data:      dict[str, Any],
        portfolio: str | None = None,
    ) -> tuple[OverallGrade, EngineResults]:
        """
        Score a stock and return (OverallGrade, EngineResults).

        Parameters
        ----------
        data:
            Normalized data dict from DataFetcher.normalize().
        portfolio:
            If given, use portfolio-specific engine weights (e.g. "conservative").
            If None, use the global overall weights (50/30/20 default).
        """
        weights = get_engine_weights(portfolio)

        # ── 1. Run engines ───────────────────────────────────
        fund_res  = self._fund.score(data)
        tech_res  = self._tech.score(data)
        quant_res = self._quant.score(data)

        # ── 2. Weighted composite ────────────────────────────
        raw = (
            weights.get("fundamental",  0.50) * fund_res.score  +
            weights.get("technical",    0.30) * tech_res.score  +
            weights.get("quantitative", 0.20) * quant_res.score
        )
        raw = float(np.clip(raw, 0.0, 100.0))

        # ── 3. Circuit breakers ──────────────────────────────
        composite, breakers = apply_circuit_breakers(
            raw, data, fund_res, self._cfg
        )

        # ── 4. Grade ─────────────────────────────────────────
        grade = composite_to_grade(composite)

        # ── 5. Confidence ────────────────────────────────────
        confidence = _confidence(data, fund_res, tech_res, quant_res)

        # ── 6. Drivers ───────────────────────────────────────
        drivers_pos, drivers_neg = _extract_drivers(
            fund_res, tech_res, quant_res, data
        )

        # ── 7. Assemble ──────────────────────────────────────
        overall = OverallGrade(
            grade              = grade,
            composite          = round(composite, 2),
            confidence         = round(confidence, 3),
            drivers_positive   = drivers_pos,
            drivers_negative   = drivers_neg,
            circuit_breakers   = breakers,
            fundamental_score  = round(fund_res.score, 2),
            technical_score    = round(tech_res.score, 2),
            quantitative_score = round(quant_res.score, 2),
            weights_used       = {k: round(v, 3) for k, v in weights.items()},
        )
        engines = EngineResults(
            fundamental  = fund_res,
            technical    = tech_res,
            quantitative = quant_res,
        )
        return overall, engines


# ──────────────────────────────────────────────────────────────
# Confidence computation
# ──────────────────────────────────────────────────────────────

def _confidence(
    data:       dict[str, Any],
    fund_res:   FundamentalResult,
    tech_res:   TechnicalResult,
    quant_res:  QuantitativeResult,
) -> float:
    """
    Composite confidence ∈ [0.05, 1.0].

    Three components:
      data_completeness  (50%) — fraction of required fields present
      signal_agreement   (30%) — how aligned are the three engine scores
      missing_penalty    (20%) — penalise each unique missing field
    """
    completeness = float(data.get("data_completeness", 0.80))

    scores    = [fund_res.score, tech_res.score, quant_res.score]
    divergence = float(np.std(scores))
    agreement  = max(0.0, 1.0 - divergence / 35.0)

    all_missing = set(
        fund_res.missing_fields +
        tech_res.missing_fields +
        quant_res.missing_fields
    )
    missing_penalty = max(0.0, 1.0 - len(all_missing) * 0.035)

    raw = 0.50 * completeness + 0.30 * agreement + 0.20 * missing_penalty
    return float(np.clip(raw, 0.05, 1.00))


# ──────────────────────────────────────────────────────────────
# Driver extraction
# ──────────────────────────────────────────────────────────────

def _extract_drivers(
    fund_res:  FundamentalResult,
    tech_res:  TechnicalResult,
    quant_res: QuantitativeResult,
    data:      dict[str, Any],
) -> tuple[list[str], list[str]]:
    """
    Return (top_3_positive_drivers, top_3_negative_drivers).

    Each driver is a concise human-readable string explaining the signal.
    Drivers are sorted by their representative score (0–100) so the spec
    requirement of "top 3 positive / top 3 negative" is met without black boxes.
    """
    items: list[tuple[float, str]] = []  # (score, label)

    # ── Fundamental pillars ──────────────────────────────────
    p = fund_res.pillars
    items += [
        (p.valuation,          f"Valuation {p.valuation:.0f}/100"),
        (p.profitability,      f"Profitability {p.profitability:.0f}/100"),
        (p.growth,             f"Growth {p.growth:.0f}/100"),
        (p.financial_health,   f"Financial health {p.financial_health:.0f}/100"),
        (p.capital_allocation, f"Capital allocation {p.capital_allocation:.0f}/100"),
    ]

    if fund_res.roic_vs_wacc is not None:
        pp  = fund_res.roic_vs_wacc * 100
        scr = min(100.0, max(0.0, 50.0 + pp * 3.0))
        dir_word = "above" if pp >= 0 else "below"
        items.append((scr, f"ROIC {pp:+.1f}ppt {dir_word} WACC"))

    if fund_res.altman_z is not None:
        z = fund_res.altman_z
        scr = min(100.0, z / 3.0 * 100.0)
        zone = "safe" if z >= 3.0 else ("grey zone" if z >= 1.8 else "distress")
        items.append((scr, f"Altman Z {z:.2f} ({zone})"))

    if fund_res.piotroski_f is not None:
        f_scr = fund_res.piotroski_f / 9.0 * 100.0
        items.append((f_scr, f"Piotroski F {fund_res.piotroski_f}/9"))

    # ── Technical pillars ────────────────────────────────────
    tp = tech_res.pillars
    items += [
        (tp.trend,            f"Trend {tp.trend:.0f}/100 (regime: {tech_res.regime})"),
        (tp.momentum,         f"Momentum {tp.momentum:.0f}/100"),
        (tp.volume_structure, f"Volume/structure {tp.volume_structure:.0f}/100"),
    ]

    # ── Quantitative factors ─────────────────────────────────
    qf = quant_res.factors
    for label, pct in [
        ("Value factor",      qf.value_pct),
        ("Quality factor",    qf.quality_pct),
        ("Momentum factor",   qf.momentum_pct),
        ("Low-vol factor",    qf.low_volatility_pct),
    ]:
        if pct is not None:
            items.append((pct, f"{label} {pct:.0f}th pct"))

    qr = quant_res.risk_metrics
    if qr.sharpe_1yr is not None:
        sh_scr = float(np.clip(50.0 + qr.sharpe_1yr * 30.0, 0.0, 100.0))
        items.append((sh_scr, f"Sharpe {qr.sharpe_1yr:.2f} (1yr)"))

    if qr.max_drawdown_3yr is not None:
        dd_pct = qr.max_drawdown_3yr * 100.0
        dd_scr = float(np.clip(100.0 + dd_pct * 2.0, 0.0, 100.0))
        items.append((dd_scr, f"Max drawdown {dd_pct:.1f}% (3yr)"))

    # ── Sort and select ──────────────────────────────────────
    items.sort(key=lambda x: x[0], reverse=True)
    drivers_pos = [label for _, label in items[:3]]
    drivers_neg = [label for _, label in reversed(items[-3:])]   # worst first

    return drivers_pos, drivers_neg


# ──────────────────────────────────────────────────────────────
# Convenience function for the CLI / reporters
# ──────────────────────────────────────────────────────────────

def grade_ticker(
    data:      dict[str, Any],
    portfolio: str | None = None,
    config:    dict[str, Any] | None = None,
) -> tuple[OverallGrade, EngineResults]:
    """
    Top-level convenience wrapper — create an Orchestrator and score data.

    Equivalent to Orchestrator().analyze(data, portfolio).
    """
    return Orchestrator(config).analyze(data, portfolio)
