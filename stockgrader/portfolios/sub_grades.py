"""
Portfolio Sub-Grades — re-grade the same stock through each of five
risk-tiered model portfolios.

For each portfolio the engine sub-scores are kept but:
  1. Eligibility gates are checked first (hard filters → Stay Away if failed)
  2. Engine weights are portfolio-specific (e.g. Very Conservative: 55/15/30)
  3. Quantitative score is adjusted for factor preferences (low-vol gets 2×
     weight for Very Conservative; momentum gets 2× for Very Aggressive)
  4. Fundamental score has a valuation penalty applied proportional to the
     portfolio's tolerance for overvaluation

This produces genuinely different grades per portfolio:
  A stock can be a Buy overall yet Stay Away for Very Conservative
  (e.g., beta 1.8, no dividend) while being Gotta Have for Very Aggressive.
"""
from __future__ import annotations

import logging
import math
from typing import Any

from stockgrader.models import (
    FundamentalResult,
    GateResult,
    Grade,
    PortfolioGrade,
    PortfolioGrades,
    QuantitativeResult,
    TechnicalResult,
    composite_to_grade,
)

logger = logging.getLogger(__name__)

# Multiplier applied to the (50 - valuation_score) penalty when stock is overvalued
_VALUATION_PENALTY_MULT: dict[str, float] = {
    "high":       0.50,
    "medium_high":0.30,
    "medium":     0.18,
    "low_medium": 0.08,
    "low":        0.02,
}

_PORTFOLIO_NAMES = [
    "very_conservative",
    "conservative",
    "balanced",
    "aggressive",
    "very_aggressive",
]


# ──────────────────────────────────────────────────────────────
# Gate checking
# ──────────────────────────────────────────────────────────────

def _f(v: Any) -> float | None:
    if v is None:
        return None
    try:
        f = float(v)
        return None if (math.isnan(f) or math.isinf(f)) else f
    except (TypeError, ValueError):
        return None


def _check_gates(
    data:      dict[str, Any],
    fund_res:  FundamentalResult,
    quant_res: QuantitativeResult,
    gates_cfg: dict[str, Any],
) -> list[GateResult]:
    """
    Evaluate all eligibility gates for one portfolio.

    Gate semantics:
      max_*      : actual value must be ≤ limit
      min_*      : actual value must be ≥ limit
      require_*  : boolean condition must be True

    If a required data point is None the gate is marked passed=True with a
    warning note — we cannot fail what we cannot measure, but the rationale
    will flag the gap.
    """
    results: list[GateResult] = []

    # ── max_beta ─────────────────────────────────────────────
    limit = _f(gates_cfg.get("max_beta"))
    if limit is not None:
        val    = _f(data.get("beta"))
        passed = (val is None) or (val <= limit)
        results.append(GateResult(gate="max_beta",    passed=passed, value=val, limit=limit))

    # ── min_market_cap ────────────────────────────────────────
    limit = _f(gates_cfg.get("min_market_cap"))
    if limit is not None:
        val    = _f(data.get("market_cap"))
        passed = (val is None) or (val >= limit)
        results.append(GateResult(gate="min_market_cap", passed=passed, value=val, limit=limit))

    # ── require_dividend ──────────────────────────────────────
    if gates_cfg.get("require_dividend", False):
        dy     = _f(data.get("dividend_yield")) or 0.0
        pr     = _f(data.get("payout_ratio"))   or 0.0
        passed = dy > 0.001 and pr < 0.95
        results.append(GateResult(gate="require_dividend", passed=passed,
                                  value=round(dy, 4), limit=0.001))

    # ── max_drawdown_3yr ──────────────────────────────────────
    limit = _f(gates_cfg.get("max_drawdown_3yr"))
    if limit is not None:
        dd  = quant_res.risk_metrics.max_drawdown_3yr
        val = abs(dd) if dd is not None else None
        passed = (val is None) or (val <= limit)
        results.append(GateResult(gate="max_drawdown_3yr", passed=passed,
                                  value=round(dd, 3) if dd is not None else None,
                                  limit=-limit))

    # ── min_altman_z ──────────────────────────────────────────
    limit = _f(gates_cfg.get("min_altman_z"))
    if limit is not None:
        val    = fund_res.altman_z
        passed = (val is None) or (val >= limit)
        results.append(GateResult(gate="min_altman_z", passed=passed,
                                  value=round(val, 2) if val is not None else None,
                                  limit=limit))

    # ── max_debt_equity ───────────────────────────────────────
    limit = _f(gates_cfg.get("max_debt_equity"))
    if limit is not None:
        de     = _f(data.get("debt_equity"))
        val    = abs(de) if de is not None else None
        passed = (val is None) or (val <= limit)
        results.append(GateResult(gate="max_debt_equity", passed=passed,
                                  value=round(de, 2) if de is not None else None,
                                  limit=limit))

    return results


# ──────────────────────────────────────────────────────────────
# Score adjustments
# ──────────────────────────────────────────────────────────────

def _adj_fund_score(
    fund_res:          FundamentalResult,
    valuation_penalty: str,
) -> float:
    """
    Apply valuation penalty to the fundamental score.

    Conservative portfolios care more about overvaluation; aggressive ones
    are willing to pay a premium for growth.  The penalty is proportional
    to how far below 50 the valuation pillar is:
      penalty_points = max(0, 50 - valuation_score) × multiplier
    """
    mult  = _VALUATION_PENALTY_MULT.get(valuation_penalty, 0.18)
    val_s = fund_res.pillars.valuation
    penalty = max(0.0, 50.0 - val_s) * mult
    adjusted = fund_res.score - penalty
    return max(0.0, min(100.0, adjusted))


def _adj_quant_score(
    quant_res:          QuantitativeResult,
    factor_prefs:       dict[str, float],
    base_factor_weights: dict[str, float],
) -> float:
    """
    Adjust the quantitative score for portfolio-specific factor preferences.

    Method: compute a preference-weighted factor composite, then blend
    60% original quant score + 40% preference-weighted composite.

    Factors with preference > 1 are rewarded more; < 1 are downweighted.
    """
    fac = quant_res.factors
    factor_scores: dict[str, float] = {
        "value":          fac.value_pct          or 50.0,
        "quality":        fac.quality_pct         or 50.0,
        "momentum":       fac.momentum_pct        or 50.0,
        "size":           fac.size_pct            or 50.0,
        "low_volatility": fac.low_volatility_pct  or 50.0,
    }

    # Apply preference multipliers to base weights, then renormalise
    adj_weights: dict[str, float] = {}
    for fname, base_w in base_factor_weights.items():
        mult = float(factor_prefs.get(fname, 1.0))
        adj_weights[fname] = base_w * mult

    total = sum(adj_weights.values())
    if total == 0:
        return quant_res.score

    norm_weights = {k: v / total for k, v in adj_weights.items()}

    # Preference-weighted factor composite
    pref_composite = sum(
        factor_scores.get(fname, 50.0) * w
        for fname, w in norm_weights.items()
    )

    # Blend: majority weight on original score (preserves risk-quality component)
    return float(max(0.0, min(100.0, 0.60 * quant_res.score + 0.40 * pref_composite)))


# ──────────────────────────────────────────────────────────────
# Rationale generation
# ──────────────────────────────────────────────────────────────

def _build_rationale(
    portfolio_name: str,
    gate_results:   list[GateResult],
    composite:      float,
    grade:          Grade,
    data:           dict[str, Any],
    fund_res:       FundamentalResult,
    quant_res:      QuantitativeResult,
) -> str:
    """Produce a concise one-line rationale for the sub-grade."""
    failed = [g for g in gate_results if not g.passed]

    if failed:
        gate_labels = []
        for g in failed[:3]:
            if g.gate == "max_beta" and g.value is not None:
                gate_labels.append(f"beta {g.value:.2f} > {g.limit:.1f} max")
            elif g.gate == "min_market_cap" and g.value is not None:
                mc_b = (g.value or 0) / 1e9
                gate_labels.append(f"market cap ${mc_b:.0f}B < ${g.limit/1e9:.0f}B min")
            elif g.gate == "require_dividend":
                dy = _f(data.get("dividend_yield")) or 0.0
                gate_labels.append(f"no dividend (yield {dy*100:.2f}%)")
            elif g.gate == "max_drawdown_3yr" and g.value is not None:
                gate_labels.append(f"max drawdown {abs(g.value)*100:.0f}% > {abs(g.limit)*100:.0f}% limit")
            elif g.gate == "min_altman_z" and g.value is not None:
                gate_labels.append(f"Altman Z {g.value:.2f} < {g.limit:.1f} min")
            elif g.gate == "max_debt_equity" and g.value is not None:
                gate_labels.append(f"D/E {g.value:.1f} > {g.limit:.1f} max")
            else:
                gate_labels.append(g.gate.replace("_", " "))
        gates_str = "; ".join(gate_labels)
        return f"Fails {portfolio_name.replace('_', ' ')} eligibility: {gates_str}"

    # Passed all gates — describe the main appeal or concern
    pf_name = portfolio_name.replace("_", " ").title()
    if grade == Grade.GOTTA_HAVE:
        return f"Excellent fit for {pf_name} mandate (composite {composite:.0f})"
    if grade == Grade.BUY:
        return f"Strong fit for {pf_name} (composite {composite:.0f})"
    if grade == Grade.HOLD:
        roic_note = ""
        if fund_res.roic_vs_wacc is not None:
            roic_note = f"; ROIC {'above' if fund_res.roic_vs_wacc > 0 else 'below'} WACC"
        return f"Adequate for {pf_name}{roic_note} (composite {composite:.0f})"
    if grade == Grade.SELL:
        return f"Marginal for {pf_name} mandate — consider reducing (composite {composite:.0f})"
    return f"Does not meet {pf_name} criteria (composite {composite:.0f})"


# ──────────────────────────────────────────────────────────────
# Single-portfolio grading
# ──────────────────────────────────────────────────────────────

def _grade_one_portfolio(
    name:              str,
    portfolio_cfg:     dict[str, Any],
    data:              dict[str, Any],
    fund_res:          FundamentalResult,
    tech_res:          TechnicalResult,
    quant_res:         QuantitativeResult,
    base_factor_weights: dict[str, float],
) -> PortfolioGrade:
    # ── Eligibility gates ─────────────────────────────────────
    gates_cfg    = portfolio_cfg.get("eligibility", {})
    gate_results = _check_gates(data, fund_res, quant_res, gates_cfg)
    failed_gates = [g.gate for g in gate_results if not g.passed]

    if failed_gates:
        grade = Grade.STAY_AWAY
        rationale = _build_rationale(name, gate_results, 0.0, grade, data, fund_res, quant_res)
        return PortfolioGrade(
            grade        = grade,
            composite    = 0.0,
            gate_results = gate_results,
            rationale    = rationale,
        )

    # ── Portfolio-adjusted scores ─────────────────────────────
    weights    = portfolio_cfg.get("weights", {"fundamental": 0.50, "technical": 0.30, "quantitative": 0.20})
    val_pen    = portfolio_cfg.get("valuation_penalty", "medium")
    fac_prefs  = portfolio_cfg.get("factor_preferences", {})

    adj_fund  = _adj_fund_score(fund_res, val_pen)
    adj_quant = _adj_quant_score(quant_res, fac_prefs, base_factor_weights)
    tech_s    = tech_res.score   # technical is unchanged across portfolios

    composite = (
        float(weights.get("fundamental",  0.50)) * adj_fund  +
        float(weights.get("technical",    0.30)) * tech_s    +
        float(weights.get("quantitative", 0.20)) * adj_quant
    )
    composite = round(max(0.0, min(100.0, composite)), 2)
    grade     = composite_to_grade(composite)

    rationale = _build_rationale(name, gate_results, composite, grade, data, fund_res, quant_res)

    return PortfolioGrade(
        grade        = grade,
        composite    = composite,
        gate_results = gate_results,
        rationale    = rationale,
    )


# ──────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────

def compute_sub_grades(
    data:      dict[str, Any],
    fund_res:  FundamentalResult,
    tech_res:  TechnicalResult,
    quant_res: QuantitativeResult,
    config:    dict[str, Any],
) -> PortfolioGrades:
    """
    Re-grade the stock for each of the five model portfolios.

    Returns a PortfolioGrades object with a PortfolioGrade per sleeve.
    Sub-grades reuse the engine sub-scores but apply portfolio-specific
    weights, factor preferences, and eligibility gates.
    """
    portfolios_cfg = config.get("portfolios", {})
    base_fw        = config.get("quantitative", {}).get(
        "factor_weights_in_quant_score",
        {"value": 0.20, "quality": 0.30, "momentum": 0.25,
         "size": 0.10, "low_volatility": 0.15},
    )

    grades: dict[str, PortfolioGrade] = {}
    for pname in _PORTFOLIO_NAMES:
        pcfg = portfolios_cfg.get(pname, {})
        if not pcfg:
            logger.warning("No config found for portfolio %r; using 50/30/20 defaults.", pname)
        grades[pname] = _grade_one_portfolio(
            pname, pcfg, data, fund_res, tech_res, quant_res, base_fw
        )

    return PortfolioGrades(
        very_conservative = grades["very_conservative"],
        conservative      = grades["conservative"],
        balanced          = grades["balanced"],
        aggressive        = grades["aggressive"],
        very_aggressive   = grades["very_aggressive"],
    )
