"""
Price Ladder — fair-value estimation and grade-boundary prices.

Two valuation methods (blended):
  DCF (50% weight):
    Two-stage: 5-yr explicit forecast + 5-yr linear fade to terminal growth rate.
    Discount at WACC.  All assumptions in config.yaml / fair_value.dcf.
  Multiple-based (50% weight):
    Quality-adjusted peer median PE applied to forward EPS.
    Falls back to absolute sector PE when peers are sparse.

Grade boundaries are derived by applying margin-of-safety bands around fair
value, tightened for high-quality businesses (spec §8.2):
  Gotta Have  ≤  FV × (1 - mos_gotta_have)    default -30%
  Buy         ≤  FV × (1 - mos_buy)            default -15%
  Hold range  :  [Buy, Sell]
  Sell        ≥  FV × (1 + mos_sell)           default +15%
  Stay Away   ≥  FV × (1 + mos_stay_away)      default +40%

Quality adjustment: higher-quality companies earn tighter bands
(a great business with a durable moat deserves less margin of safety).
"""
from __future__ import annotations

import logging
import math
from typing import Any

from stockgrader.models import FairValueSensitivity, PriceLadder, SensitivityGrid

logger = logging.getLogger(__name__)

# Growth rate hard caps (prevents unrealistic DCF inflation)
_MAX_GROWTH_STAGE1  = 0.35   # 35% max explicit-period growth
_MIN_WACC           = 0.05   # 5% minimum discount rate
_MIN_TERMINAL_SPREAD = 0.01  # WACC must exceed terminal growth by at least 1%


# ──────────────────────────────────────────────────────────────
# Pure computation helpers
# ──────────────────────────────────────────────────────────────

def _f(v: Any) -> float | None:
    if v is None:
        return None
    try:
        f = float(v)
        return None if (math.isnan(f) or math.isinf(f)) else f
    except (TypeError, ValueError):
        return None


def _safe_rate(r: float | None, default: float, lo: float = -0.20, hi: float = 0.40) -> float:
    if r is None or math.isnan(r) or math.isinf(r):
        return default
    return max(lo, min(hi, float(r)))


def _dcf_intrinsic_value(
    fcf_per_share:   float,
    growth_stage1:   float,    # explicit-period annual growth
    terminal_growth: float,    # perpetual growth
    wacc:            float,
    stage1_years:    int = 5,
    stage2_years:    int = 5,
) -> float:
    """
    Two-stage DCF returning intrinsic value per share.

    Stage 1 (years 1–stage1_years): constant growth at growth_stage1.
    Stage 2 (years stage1_years+1 – stage1_years+stage2_years):
        linear fade from growth_stage1 to terminal_growth.
    Terminal value: Gordon Growth Model at end of stage 2.
    """
    # Guard against math failure
    wacc = max(wacc, _MIN_WACC)
    if terminal_growth >= wacc - _MIN_TERMINAL_SPREAD:
        terminal_growth = wacc - _MIN_TERMINAL_SPREAD

    pv  = 0.0
    fcf = fcf_per_share

    # Stage 1 — explicit forecast
    for t in range(1, stage1_years + 1):
        fcf  = fcf * (1.0 + growth_stage1)
        pv  += fcf / (1.0 + wacc) ** t

    # Stage 2 — transition
    for t in range(1, stage2_years + 1):
        alpha    = t / stage2_years
        growth_t = growth_stage1 * (1.0 - alpha) + terminal_growth * alpha
        fcf      = fcf * (1.0 + growth_t)
        pv      += fcf / (1.0 + wacc) ** (stage1_years + t)

    # Terminal value
    total_years = stage1_years + stage2_years
    fcf_tv      = fcf * (1.0 + terminal_growth)
    tv          = fcf_tv / (wacc - terminal_growth)
    pv         += tv / (1.0 + wacc) ** total_years

    return pv


def _reverse_dcf(
    target_price:    float,
    fcf_per_share:   float,
    wacc:            float,
    terminal_growth: float,
    stage1_years:    int = 5,
    stage2_years:    int = 5,
    lo:              float = -0.05,
    hi:              float = 0.50,
    iterations:      int = 50,
) -> float | None:
    """
    Reverse DCF: binary-search for the growth rate that makes
    intrinsic value equal to target_price.

    Returns the implied annual growth rate, or None if outside [lo, hi].
    """
    if fcf_per_share <= 0 or target_price <= 0:
        return None

    def _dcf(g: float) -> float:
        return _dcf_intrinsic_value(fcf_per_share, g, terminal_growth, wacc,
                                    stage1_years, stage2_years)

    # Check if target is within the reachable range
    if _dcf(lo) < target_price:
        return None   # target requires growth below lo → not meaningful
    if _dcf(hi) > target_price:
        return None   # even at max growth, DCF < target → extreme overvaluation

    mid = (lo + hi) / 2.0
    for _ in range(iterations):
        mid = (lo + hi) / 2.0
        val = _dcf(mid)
        if abs(val - target_price) < 0.01:
            break
        if val > target_price:
            lo = mid
        else:
            hi = mid
    return mid


def _peer_median_pe(peer_metrics: dict[str, dict]) -> float | None:
    """Median positive P/E across peers."""
    pes = []
    for m in peer_metrics.values():
        v = _f(m.get("priceEarningsRatioTTM"))
        if v and 3.0 < v < 200.0:   # exclude nonsensical values
            pes.append(v)
    if not pes:
        return None
    pes.sort()
    n = len(pes)
    return float((pes[n // 2] + pes[(n - 1) // 2]) / 2) if n >= 2 else pes[0]


def _quality_adjusted_mos(
    base_mos:          float,
    fundamental_score: float,
    quality_band_scale: float = 0.50,
) -> float:
    """
    Reduce margin-of-safety for high-quality companies.

    quality_fraction ∈ [0, 1]:  0 at score 50, 1 at score 100.
    A score of 80 → fraction 0.6 → band reduced by 30% (if scale=0.5).
    """
    quality_fraction = max(0.0, (fundamental_score - 50.0) / 50.0)
    reduction        = quality_fraction * quality_band_scale
    return max(0.02, base_mos * (1.0 - reduction))   # floor at 2%


def _grade_boundaries(
    fair_value:        float,
    fundamental_score: float,
    fv_cfg:            dict,
) -> dict[str, float]:
    """
    Compute the five grade-boundary price levels.
    Quality-adjusts each MoS band, then derives prices around fair_value.
    """
    mos_cfg   = fv_cfg.get("margin_of_safety", {})
    qa        = bool(mos_cfg.get("quality_adjustment", True))
    band_scale = float(mos_cfg.get("quality_band_scale", 0.50))

    base_gh  = float(mos_cfg.get("gotta_have_discount", 0.30))
    base_buy = float(mos_cfg.get("buy_discount",        0.15))
    base_sell = float(mos_cfg.get("sell_premium",       0.15))
    base_sa  = float(mos_cfg.get("stay_away_premium",   0.40))

    if qa:
        mos_gh  = _quality_adjusted_mos(base_gh,  fundamental_score, band_scale)
        mos_buy = _quality_adjusted_mos(base_buy, fundamental_score, band_scale)
        # Sell / Stay Away premiums also tighten for quality (market prices it in)
        mos_sell = _quality_adjusted_mos(base_sell, fundamental_score, band_scale)
        mos_sa   = _quality_adjusted_mos(base_sa,  fundamental_score, band_scale)
    else:
        mos_gh, mos_buy, mos_sell, mos_sa = base_gh, base_buy, base_sell, base_sa

    return {
        "gotta_have_at":  round(fair_value * (1.0 - mos_gh),   2),
        "buy_at":         round(fair_value * (1.0 - mos_buy),  2),
        "sell_above":     round(fair_value * (1.0 + mos_sell), 2),
        "stay_away_above":round(fair_value * (1.0 + mos_sa),   2),
    }


def _sensitivity_grid(
    fcf_per_share:   float,
    growth_base:     float,
    wacc_base:       float,
    terminal_growth: float,
    stage1_years:    int,
    stage2_years:    int,
    growth_deltas:   list[float],
    dr_deltas:       list[float],
) -> SensitivityGrid:
    """Compute a (len(growth_deltas) × len(dr_deltas)) fair-value grid."""
    cells:  dict[str, float] = {}
    for dg in growth_deltas:
        for dr in dr_deltas:
            g    = _safe_rate(growth_base + dg,  growth_base)
            wacc = max(_MIN_WACC, wacc_base + dr)
            fv   = _dcf_intrinsic_value(fcf_per_share, g, terminal_growth, wacc,
                                         stage1_years, stage2_years)
            key  = f"{dg:+.2f}/{dr:+.2f}"
            cells[key] = round(fv, 2)

    return SensitivityGrid(
        cells                = cells,
        growth_deltas        = growth_deltas,
        discount_rate_deltas = dr_deltas,
    )


# ──────────────────────────────────────────────────────────────
# Input extraction helper
# ──────────────────────────────────────────────────────────────

def _extract_inputs(data: dict, config: dict) -> dict[str, Any]:
    """
    Pull all raw values needed for the valuation from the normalized data dict.
    Returns a flat dict with defaults where data is missing.
    """
    dcf_cfg = config.get("fair_value", {}).get("dcf", {})
    rf      = _f(data.get("risk_free_rate_3mo")) or 0.05
    erp     = float(dcf_cfg.get("equity_risk_premium", 0.055))
    beta    = _f(data.get("beta")) or 1.0

    # WACC: prefer normalizer-computed; fallback to CAPM
    wacc = _f(data.get("wacc"))
    if wacc is None:
        wacc = rf + beta * erp

    price       = _f(data.get("price"))
    market_cap  = _f(data.get("market_cap"))
    forward_eps = _f(data.get("forward_eps"))
    eps_ttm     = _f(data.get("eps_ttm")) or (
        _f(_first(data.get("eps")))
    )

    # Growth rate: forward estimate > 3yr CAGR > eps_ttm-based
    fwd_growth  = _f(data.get("forward_eps_growth"))
    hist_cagr   = _f(data.get("eps_cagr_3yr")) or _f(data.get("revenue_cagr_3yr"))
    growth_base = _safe_rate(
        fwd_growth if fwd_growth is not None else hist_cagr,
        default = 0.05,
        lo = -0.10,
        hi = _MAX_GROWTH_STAGE1,
    )

    # FCF per share
    fcf_series  = data.get("free_cf", [])
    sh_series   = data.get("shares_diluted", [])
    fcf_ps: float | None = None

    fcf_0 = _f(_first(fcf_series))
    sh_0  = _f(_first(sh_series))
    if fcf_0 and sh_0 and sh_0 > 0:
        fcf_ps = fcf_0 / sh_0

    if fcf_ps is None:
        fy = _f(data.get("fcf_yield"))
        if fy and price:
            fcf_ps = fy * price

    if fcf_ps is None:
        fcf_conv = _f(data.get("fcf_conversion")) or 0.85
        if eps_ttm:
            fcf_ps = eps_ttm * fcf_conv

    # Peer median PE
    peer_pe = _peer_median_pe(data.get("peer_metrics", {}))

    # Total debt and cash for net-debt adjustment (not used in per-share DCF)
    net_debt_ps: float | None = None
    nd = _f(data.get("net_debt"))
    if nd is not None and sh_0 and sh_0 > 0:
        net_debt_ps = nd / sh_0

    return {
        "price":            price,
        "forward_eps":      forward_eps,
        "eps_ttm":          eps_ttm,
        "fcf_per_share":    fcf_ps,
        "growth_base":      growth_base,
        "wacc":             wacc,
        "terminal_growth":  float(dcf_cfg.get("terminal_growth_rate", 0.03)),
        "stage1_years":     int(dcf_cfg.get("stage1_years", 5)),
        "stage2_years":     int(dcf_cfg.get("stage2_years", 5)),
        "peer_median_pe":   peer_pe,
        "market_cap":       market_cap,
        "net_debt_ps":      net_debt_ps,
    }


def _first(lst: Any) -> Any:
    if isinstance(lst, list) and lst:
        return lst[0]
    return None


# ──────────────────────────────────────────────────────────────
# Main builder
# ──────────────────────────────────────────────────────────────

def build_price_ladder(
    data:              dict[str, Any],
    fundamental_score: float,
    config:            dict[str, Any],
) -> PriceLadder | None:
    """
    Compute the fair-value estimate and grade-boundary price ladder.

    Returns None if insufficient data is available for any valuation.
    On partial data, logs a warning and uses the available method only.

    Side-effect: injects ``data["_ladder_stay_away_above"]`` so the
    extreme-overvaluation circuit breaker can use it in Step 6.
    """
    fv_cfg = config.get("fair_value", {})
    inp    = _extract_inputs(data, config)
    price  = inp["price"]

    if price is None or price <= 0:
        logger.warning("No valid price for price-ladder computation.")
        return None

    # ── 1. DCF fair value ─────────────────────────────────────
    fv_dcf: float | None = None
    data_completeness = _f(data.get("data_completeness")) or 1.0

    if inp["fcf_per_share"] and inp["fcf_per_share"] > 0:
        fv_dcf = _dcf_intrinsic_value(
            fcf_per_share   = inp["fcf_per_share"],
            growth_stage1   = inp["growth_base"],
            terminal_growth = inp["terminal_growth"],
            wacc            = inp["wacc"],
            stage1_years    = inp["stage1_years"],
            stage2_years    = inp["stage2_years"],
        )
        if fv_dcf is not None and (math.isnan(fv_dcf) or math.isinf(fv_dcf)):
            fv_dcf = None

        # Sanity check: if DCF is more than 5× current price AND data
        # completeness is below 85%, the inputs are likely distorted
        # (post-restructuring, missing fields, etc.) — discard DCF and
        # rely on multiples only.
        if (fv_dcf is not None and price is not None and price > 0
                and fv_dcf > price * 5.0 and data_completeness < 0.85):
            logger.warning(
                "DCF fair value (%.2f) is >5x price (%.2f) with data completeness "
                "%.0f%% — discarding DCF, using multiples only.",
                fv_dcf, price, data_completeness * 100,
            )
            fv_dcf = None
    else:
        logger.info("No positive FCF per share — skipping DCF; using multiples only.")

    # ── 2. Multiple-based fair value ──────────────────────────
    fv_multiple: float | None = None
    fwd_eps = inp["forward_eps"] or (inp["eps_ttm"] and inp["eps_ttm"] * (1 + inp["growth_base"]))
    if fwd_eps and fwd_eps > 0:
        peer_pe = inp["peer_median_pe"]
        if peer_pe is None:
            # Absolute fallback: rough sector-neutral PE based on growth
            g = inp["growth_base"]
            peer_pe = max(10.0, min(35.0, 15.0 + g * 100))   # crude PEG-based floor

        qa = bool(fv_cfg.get("margin_of_safety", {}).get("quality_adjustment", True))
        qs = float(fv_cfg.get("margin_of_safety", {}).get("quality_band_scale", 0.50))
        if qa:
            quality_fraction = max(0.0, (fundamental_score - 50.0) / 50.0)
            premium          = quality_fraction * 0.20
            justified_pe     = peer_pe * (1.0 + premium)
        else:
            justified_pe = peer_pe

        fv_multiple = round(justified_pe * fwd_eps, 2)
        if math.isnan(fv_multiple) or math.isinf(fv_multiple):
            fv_multiple = None

    # ── 3. Blended fair value ─────────────────────────────────
    if fv_dcf is None and fv_multiple is None:
        logger.warning("Neither DCF nor multiple FV could be computed for price ladder.")
        return None

    if fv_dcf is not None and fv_multiple is not None:
        fair_value = round(0.50 * fv_dcf + 0.50 * fv_multiple, 2)
    else:
        fair_value = round((fv_dcf or fv_multiple), 2)  # type: ignore[arg-type]

    if fair_value <= 0:
        logger.warning("Computed fair value ≤ 0 (%.2f); aborting price ladder.", fair_value)
        return None

    # ── 4. Grade boundaries ───────────────────────────────────
    bounds = _grade_boundaries(fair_value, fundamental_score, fv_cfg)

    # ── 5. Sensitivity grid ───────────────────────────────────
    sg_cfg         = fv_cfg.get("sensitivity_grid", {})
    growth_deltas  = list(sg_cfg.get("growth_delta",        [-0.02, 0.00, 0.02]))
    dr_deltas      = list(sg_cfg.get("discount_rate_delta", [-0.01, 0.00, 0.01]))

    if inp["fcf_per_share"] and inp["fcf_per_share"] > 0:
        sens_grid = _sensitivity_grid(
            fcf_per_share   = inp["fcf_per_share"],
            growth_base     = inp["growth_base"],
            wacc_base       = inp["wacc"],
            terminal_growth = inp["terminal_growth"],
            stage1_years    = inp["stage1_years"],
            stage2_years    = inp["stage2_years"],
            growth_deltas   = growth_deltas,
            dr_deltas       = dr_deltas,
        )
        fv_low   = min(sens_grid.cells.values())
        fv_high  = max(sens_grid.cells.values())
    else:
        # No FCF → approximate sensitivity from multiple method only
        sens_grid = SensitivityGrid()
        fv_low  = fair_value * 0.85
        fv_high = fair_value * 1.15

    sensitivity = FairValueSensitivity(
        low  = round(fv_low,  2),
        base = round(fair_value, 2),
        high = round(fv_high, 2),
    )

    # ── 6. Reverse DCF (implied growth at current price) ──────
    implied_growth: float | None = None
    if inp["fcf_per_share"] and inp["fcf_per_share"] > 0:
        implied_growth = _reverse_dcf(
            target_price    = price,
            fcf_per_share   = inp["fcf_per_share"],
            wacc            = inp["wacc"],
            terminal_growth = inp["terminal_growth"],
            stage1_years    = inp["stage1_years"],
            stage2_years    = inp["stage2_years"],
        )

    # ── 7. Current price metrics ──────────────────────────────
    upside_pct = (fair_value - price) / price

    # ── 8. Inject stay_away threshold for circuit breaker ─────
    data["_ladder_stay_away_above"] = bounds["stay_away_above"]

    return PriceLadder(
        fair_value             = fair_value,
        fair_value_sensitivity = sensitivity,
        sensitivity_grid       = sens_grid,
        gotta_have_at          = bounds["gotta_have_at"],
        buy_at                 = bounds["buy_at"],
        hold_low               = bounds["buy_at"],
        hold_high              = bounds["sell_above"],
        sell_above             = bounds["sell_above"],
        stay_away_above        = bounds["stay_away_above"],
        upside_to_fv_pct       = round(upside_pct, 4),
        implied_growth_rate    = round(implied_growth, 4) if implied_growth is not None else None,
    )
