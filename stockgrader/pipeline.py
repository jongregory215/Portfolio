"""
Pipeline — assembles the full analysis from raw data to AnalysisResult.

Used by analyze.py (CLI) and can be called directly from Python:

    from stockgrader.pipeline import run_analysis, assemble_result
    result = run_analysis("AAPL")
    result = run_analysis("MSFT", portfolio="aggressive", no_cache=True)

The two-pass design:
  Pass 1 — runs all three engines to get the fundamental score, which is
            needed by the price-ladder module to quality-adjust the margin
            of safety bands.
  Pass 2 — re-runs after injecting _ladder_stay_away_above so the
            extreme-overvaluation circuit breaker can fire correctly.

Both passes use cached data, so there is no extra I/O.
"""
from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Any

from stockgrader.config import get_config, get_config_hash
from stockgrader.data.fetcher import DataFetcher
from stockgrader.grading.composite import Orchestrator
from stockgrader.grading.price_ladder import build_price_ladder
from stockgrader.models import (
    AnalysisResult, DataQuality,
    FairValueSensitivity, PriceLadder, SensitivityGrid,
)
from stockgrader.portfolios.sub_grades import compute_sub_grades

logger = logging.getLogger(__name__)


def run_analysis(
    ticker:    str,
    portfolio: str  = "all",
    no_cache:  bool = False,
    config:    dict[str, Any] | None = None,
    fetcher:   DataFetcher  | None = None,
) -> AnalysisResult:
    """
    Full pipeline: fetch → price ladder → grade → sub-grades → AnalysisResult.

    Parameters
    ----------
    ticker:    Equity ticker symbol (case-insensitive).
    portfolio: "all" (use global 50/30/20 weights) or a portfolio name
               (uses that portfolio's weights for the overall grade).
    no_cache:  If True, bypass disk cache and fetch live data.
    config:    Optional config dict override (defaults to config.yaml).
    fetcher:   Optional DataFetcher override (useful for testing).

    Returns
    -------
    AnalysisResult — fully assembled, ready for reporters.

    Raises
    ------
    ValueError  if price data is unavailable for the ticker.
    """
    cfg    = config or get_config()
    ticker = ticker.upper().strip()
    run_id = uuid.uuid4().hex[:8]

    # ── 1. Fetch data ─────────────────────────────────────────
    _fetcher = fetcher or DataFetcher()
    data     = _fetcher.fetch(ticker, use_cache=not no_cache)

    price = data.get("price")
    if price is None or float(price) <= 0:
        raise ValueError(
            f"No valid price data for {ticker}. "
            "Check the ticker symbol and API connectivity."
        )

    return assemble_result(
        ticker    = ticker,
        data      = data,
        portfolio = portfolio,
        config    = cfg,
        run_id    = run_id,
    )


def assemble_result(
    ticker:    str,
    data:      dict[str, Any],
    portfolio: str = "all",
    config:    dict[str, Any] | None = None,
    run_id:    str | None = None,
) -> AnalysisResult:
    """
    Run engines, price ladder, and sub-grades on a pre-fetched data dict.

    Use this function directly when you already have the normalized data
    (e.g., in the backtester or unit tests).
    """
    cfg      = config or get_config()
    run_id   = run_id or uuid.uuid4().hex[:8]
    port_arg = portfolio if portfolio not in ("all", "") else None
    price    = float(data.get("price", 0))

    orch = Orchestrator(cfg)

    # ── Pass 1: score engines ─────────────────────────────────
    overall_1, engines_1 = orch.analyze(data, port_arg)

    # ── Price ladder (uses fundamental score) ─────────────────
    ladder = build_price_ladder(data, engines_1.fundamental.score, cfg)
    if ladder is None:
        logger.warning("Price ladder unavailable for %s; using price-anchored fallback.", ticker)
        ladder = _fallback_ladder(price)

    # Inject stay_away threshold for the overvaluation circuit breaker
    data["_ladder_stay_away_above"] = ladder.stay_away_above

    # ── Pass 2: final grade with all circuit breakers ─────────
    overall, engines = orch.analyze(data, port_arg)

    # ── Portfolio sub-grades ──────────────────────────────────
    portfolios = compute_sub_grades(
        data,
        engines.fundamental,
        engines.technical,
        engines.quantitative,
        cfg,
    )

    # ── Assemble ──────────────────────────────────────────────
    return AnalysisResult(
        ticker       = ticker,
        as_of        = datetime.now(tz=timezone.utc),
        price        = price,
        overall      = overall,
        price_ladder = ladder,
        engines      = engines,
        portfolios   = portfolios,
        data_quality = DataQuality(
            missing_fields    = data.get("missing_fields",    []),
            imputed_fields    = data.get("imputed_fields",    []),
            sources           = data.get("sources",           {}),
            data_completeness = float(data.get("data_completeness", 0.0)),
            warnings          = data.get("warnings",          []),
        ),
        config_hash  = get_config_hash(),
        run_id       = run_id,
    )


def _fallback_ladder(price: float) -> PriceLadder:
    """
    Price-anchored fallback when DCF computation fails.
    Applies default 30/15/15/40 MoS bands around current price as a
    placeholder.  The report flags this as estimated/unreliable.
    """
    return PriceLadder(
        fair_value      = round(price, 2),
        gotta_have_at   = round(price * 0.70, 2),
        buy_at          = round(price * 0.85, 2),
        hold_low        = round(price * 0.85, 2),
        hold_high       = round(price * 1.15, 2),
        sell_above      = round(price * 1.15, 2),
        stay_away_above = round(price * 1.40, 2),
        upside_to_fv_pct = 0.0,
        implied_growth_rate = None,
        fair_value_sensitivity = FairValueSensitivity(
            low  = round(price * 0.85, 2),
            base = round(price, 2),
            high = round(price * 1.15, 2),
        ),
        sensitivity_grid = SensitivityGrid(),
    )
