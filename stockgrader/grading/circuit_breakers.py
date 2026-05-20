"""
Circuit breakers — hard caps applied to the composite score before grade banding.

Each breaker can only reduce the composite (never increase it).
Breakers are applied in order; most restrictive wins.

Configured in config.yaml under circuit_breakers:
  altman_z_distress        → cap at Sell when Z < 1.8
  negative_fcf_high_leverage → cap at Sell when FCF negative N yrs + high D/E
  going_concern            → cap at Stay Away on any going-concern flag
  extreme_overvaluation    → cap at Sell when price > stay_away ladder level
                             (populated by the price-ladder module in Step 7)
"""
from __future__ import annotations

import math
from typing import Any

from stockgrader.models import FundamentalResult

# Maximum composite score allowed for each cap level
_CAP_COMPOSITE = {
    "stay_away": 19.0,
    "sell":      39.0,
    "hold":      59.0,
}


def _f(val: Any) -> float | None:
    if val is None:
        return None
    try:
        f = float(val)
        return None if (math.isnan(f) or math.isinf(f)) else f
    except (TypeError, ValueError):
        return None


def apply_circuit_breakers(
    composite:   float,
    data:        dict[str, Any],
    fund_result: FundamentalResult,
    config:      dict[str, Any],
) -> tuple[float, list[str]]:
    """
    Apply all enabled circuit breakers.

    Returns
    -------
    (capped_composite, list_of_human_readable_messages)
    Composite can only be decreased; messages describe what fired and why.
    """
    triggered: list[str] = []
    capped     = float(composite)
    cb         = config.get("circuit_breakers", {})

    # ── 1. Altman Z distress ──────────────────────────────────
    az_cfg = cb.get("altman_z_distress", {})
    if az_cfg.get("enabled", True):
        z         = fund_result.altman_z
        threshold = float(az_cfg.get("threshold", 1.8))
        cap_at    = az_cfg.get("cap_at", "sell")
        if z is not None and z < threshold:
            max_c = _CAP_COMPOSITE.get(cap_at, 39.0)
            if capped > max_c:
                capped = max_c
            triggered.append(
                f"Altman Z {z:.2f} < {threshold:.1f} (distress zone) "
                f"→ capped at {cap_at.replace('_', ' ').title()}"
            )

    # ── 2. Negative FCF + high leverage ──────────────────────
    fcf_cfg = cb.get("negative_fcf_high_leverage", {})
    if fcf_cfg.get("enabled", True):
        n_yrs  = int(fcf_cfg.get("fcf_negative_years", 2))
        max_de = float(fcf_cfg.get("min_debt_equity", 1.5))
        cap_at = fcf_cfg.get("cap_at", "sell")

        fcf_series = data.get("free_cf", [])
        neg_years  = 0
        if isinstance(fcf_series, list):
            for v in fcf_series[:n_yrs]:
                fv = _f(v)
                if fv is not None and fv < 0:
                    neg_years += 1

        de = _f(data.get("debt_equity"))
        if neg_years >= n_yrs and de is not None and abs(de) > max_de:
            max_c = _CAP_COMPOSITE.get(cap_at, 39.0)
            if capped > max_c:
                capped = max_c
            triggered.append(
                f"Negative FCF for {neg_years} consecutive years "
                f"with D/E {de:.2f} (>{max_de}) "
                f"→ capped at {cap_at.replace('_', ' ').title()}"
            )

    # ── 3. Going concern / data integrity failure ─────────────
    gc_cfg = cb.get("going_concern", {})
    if gc_cfg.get("enabled", True):
        going_concern = bool(data.get("going_concern_flag", False))
        cap_at        = gc_cfg.get("cap_at", "stay_away")
        if going_concern:
            max_c = _CAP_COMPOSITE.get(cap_at, 19.0)
            if capped > max_c:
                capped = max_c
            triggered.append(
                f"Going-concern flag active → {cap_at.replace('_', ' ').title()}"
            )

    # ── 4. Extreme overvaluation (vs price ladder) ───────────
    # Populated by the price-ladder module (Step 7); no-op here if not set.
    ev_cfg = cb.get("extreme_overvaluation", {})
    if ev_cfg.get("enabled", True):
        price          = _f(data.get("price"))
        stay_away_above = _f(data.get("_ladder_stay_away_above"))  # set by price ladder
        cap_at          = ev_cfg.get("cap_at", "sell")
        if price and stay_away_above and price > stay_away_above:
            max_c = _CAP_COMPOSITE.get(cap_at, 39.0)
            if capped > max_c:
                capped = max_c
            triggered.append(
                f"Price ${price:.2f} > Stay Away threshold ${stay_away_above:.2f} "
                f"(extreme overvaluation) → capped at {cap_at.replace('_', ' ').title()}"
            )

    return float(capped), triggered


def breaker_summary(triggered: list[str]) -> str:
    """One-line summary for the report header."""
    if not triggered:
        return "No circuit breakers triggered."
    return f"{len(triggered)} circuit breaker(s): " + "; ".join(triggered)
