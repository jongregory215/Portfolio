"""
Weight Calibrator — Layer 3.

Systematic search over engine weight combinations (fundamental / technical /
quantitative) to find the split that maximises out-of-sample IC on a held-out
test period.

Protocol
--------
  1. Train period  : fit nothing; define the candidate weight grid.
  2. Validation    : rank every grid point by mean IC-12m on the val set.
  3. Test (OOS)    : evaluate the single best-val weight set on the test set only.
     The test result is reported as the true OOS performance.

The walk-forward constraint still applies within each period: forward returns
are only used as labels, never as inputs to scoring.

Output:
  CalibrationResult with:
    - best_weights (fund/tech/quant fractions, sum=1)
    - val_ic, test_ic (mean IC-12m under best weights)
    - full grid results (GridPoint list)
    - contamination label
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date
from itertools import product
from typing import Any

import numpy as np
import pandas as pd

from .grade_validator import compute_ic_series, compute_ic_stats, ICStats

logger = logging.getLogger(__name__)

# Weight candidates — must sum to 1.0 within each combination
_WEIGHT_STEP  = 0.10          # 10-point grid
_MIN_WEIGHT   = 0.10          # at least 10% for each engine
_HORIZONS     = [12]          # calibrate against 12-month forward returns


# ──────────────────────────────────────────────────────────────
# Data classes
# ──────────────────────────────────────────────────────────────

@dataclass
class GridPoint:
    fund_w:  float
    tech_w:  float
    quant_w: float
    val_ic:  float      # mean IC on validation set
    val_ir:  float      # IC information ratio on validation set
    test_ic: float | None = None   # filled only for best point


@dataclass
class CalibrationResult:
    contamination_label: str
    train_start:    date
    train_end:      date
    val_start:      date
    val_end:        date
    test_start:     date
    test_end:       date

    best_weights:   dict[str, float]   # {"fundamental": w, "technical": w, "quantitative": w}
    val_ic:         float
    val_ir:         float
    test_ic:        float | None
    test_ir:        float | None

    grid_results:   list[GridPoint] = field(default_factory=list)
    n_grid_points:  int = 0

    caveat: str = (
        "OOS weight calibration. Best weights selected on validation period only. "
        "Test-period IC is the true OOS estimate. Past performance does not "
        "guarantee future results."
    )


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────

def _weight_grid(step: float = _WEIGHT_STEP, min_w: float = _MIN_WEIGHT) -> list[tuple[float, float, float]]:
    """All (fund, tech, quant) combinations on the step grid that sum to 1."""
    candidates = np.arange(min_w, 1.0 - 2 * min_w + step / 2, step)
    result = []
    for f, t in product(candidates, candidates):
        q = round(1.0 - f - t, 10)
        if min_w - 1e-9 <= q <= 1.0 - 2 * min_w + 1e-9:
            result.append((round(f, 2), round(t, 2), round(q, 2)))
    return result


def _blend_composite(
    records: pd.DataFrame,
    fund_w:  float,
    tech_w:  float,
    quant_w: float,
) -> pd.DataFrame:
    """
    Re-compute composite score from per-engine scores under trial weights.

    Requires columns: fund_score, tech_score, quant_score.
    Returns a copy with composite replaced.
    """
    df = records.copy()
    needed = {"fund_score", "tech_score", "quant_score"}
    if not needed.issubset(df.columns):
        raise ValueError(f"records must contain {needed}")
    df["composite"] = (
        fund_w  * df["fund_score"] +
        tech_w  * df["tech_score"] +
        quant_w * df["quant_score"]
    ).clip(0, 100)
    return df


def _eval_weights(
    records:   pd.DataFrame,
    fund_w:    float,
    tech_w:    float,
    quant_w:   float,
    return_col: str = "fwd_12m",
    min_obs:   int = 5,
) -> tuple[float, float]:
    """
    Compute mean IC and IC-IR for a given weight triple on records.
    Returns (mean_ic, ic_ir).
    """
    blended = _blend_composite(records, fund_w, tech_w, quant_w)
    ic_series = compute_ic_series(blended, score_col="composite",
                                  return_col=return_col, min_obs_per_date=min_obs)
    if len(ic_series) < 2:
        return 0.0, 0.0
    stats = compute_ic_stats(ic_series, horizon_months=12)
    return stats.mean, stats.information_ratio


def _split_periods(
    records:     pd.DataFrame,
    train_end:   date,
    val_end:     date,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split records into train / validation / test subsets."""
    dt = pd.to_datetime(records["date"]).dt.date
    train = records[dt <  train_end].copy()
    val   = records[(dt >= train_end) & (dt < val_end)].copy()
    test  = records[dt >= val_end].copy()
    return train, val, test


# ──────────────────────────────────────────────────────────────
# Main calibrator
# ──────────────────────────────────────────────────────────────

def calibrate_weights(
    records:     pd.DataFrame,
    train_end:   date,
    val_end:     date,
    contamination_label: str = "OUT-OF-SAMPLE (PIT)",
    return_col:  str = "fwd_12m",
    weight_step: float = _WEIGHT_STEP,
    min_weight:  float = _MIN_WEIGHT,
) -> CalibrationResult:
    """
    Grid-search engine weights using validation IC; report OOS test IC.

    Parameters
    ----------
    records:     Full scored panel (train + val + test).
                 Must contain: date, fund_score, tech_score, quant_score, fwd_12m
    train_end:   End of training period (exclusive).  Val starts here.
    val_end:     End of validation period (exclusive). Test starts here.
    contamination_label: PIT label from require_pit().
    return_col:  Forward-return column for IC computation.
    weight_step: Grid step size (default 0.10).
    min_weight:  Minimum weight per engine (default 0.10).
    """
    df = records.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.date

    train_df, val_df, test_df = _split_periods(df, train_end, val_end)

    start_date = min(df["date"])
    end_date   = max(df["date"])

    if val_df.empty:
        raise ValueError("Validation period is empty — check train_end / val_end dates.")

    grid = _weight_grid(weight_step, min_weight)
    logger.info("Calibrating over %d weight combinations …", len(grid))

    grid_results: list[GridPoint] = []
    for f, t, q in grid:
        mic, mir = _eval_weights(val_df, f, t, q, return_col)
        grid_results.append(GridPoint(fund_w=f, tech_w=t, quant_w=q,
                                       val_ic=mic, val_ir=mir))

    # Select best by val IC
    best = max(grid_results, key=lambda p: p.val_ic)

    # Evaluate best on test set
    test_ic: float | None = None
    test_ir: float | None = None
    if not test_df.empty:
        test_ic, test_ir = _eval_weights(test_df, best.fund_w, best.tech_w,
                                         best.quant_w, return_col)
        best.test_ic = test_ic

    train_dates = sorted(train_df["date"].unique()) if not train_df.empty else [start_date, train_end]
    val_dates   = sorted(val_df["date"].unique())
    test_dates  = sorted(test_df["date"].unique()) if not test_df.empty else [val_end, end_date]

    logger.info(
        "Best weights: fund=%.2f tech=%.2f quant=%.2f | val_IC=%.4f | test_IC=%s",
        best.fund_w, best.tech_w, best.quant_w, best.val_ic,
        f"{test_ic:.4f}" if test_ic is not None else "N/A",
    )

    return CalibrationResult(
        contamination_label = contamination_label,
        train_start  = train_dates[0],
        train_end    = train_end,
        val_start    = val_dates[0],
        val_end      = val_end,
        test_start   = test_dates[0],
        test_end     = test_dates[-1],
        best_weights = {
            "fundamental":  best.fund_w,
            "technical":    best.tech_w,
            "quantitative": best.quant_w,
        },
        val_ic       = best.val_ic,
        val_ir       = best.val_ir,
        test_ic      = test_ic,
        test_ir      = test_ir,
        grid_results = sorted(grid_results, key=lambda p: -p.val_ic),
        n_grid_points = len(grid_results),
    )
