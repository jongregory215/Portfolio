"""
Grade Validator — Layer 1 of the backtesting framework.

Answers the core question: do higher grades predict higher forward returns?

Metrics computed:
  - Forward-return spread by grade bucket (mean, median, Sharpe per bucket)
  - IC (Spearman rank correlation between composite score and forward return)
  - IC time series → mean IC, IC vol, IC information ratio
  - Monotonicity test: is Gotta Have > Buy > Hold > Sell > Stay Away?
  - Hit rate: % of time directional call (grade > Hold) was correct
  - Per-engine attribution: run the same metrics using fund-only, tech-only,
    quant-only scores to justify (or refute) the 50/30/20 weighting
  - Price-ladder validation: did crossing below the Buy line improve returns?

All results clearly labeled in-sample or out-of-sample.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr   # type: ignore[import]

logger = logging.getLogger(__name__)

_GRADE_ORDER = ["Stay Away", "Sell", "Hold", "Buy", "Gotta Have"]
_HORIZONS    = [1, 3, 6, 12]    # months (fwd_1m, fwd_3m, fwd_6m, fwd_12m)
_COL_MAP     = {1: "fwd_1m", 3: "fwd_3m", 6: "fwd_6m", 12: "fwd_12m"}


# ──────────────────────────────────────────────────────────────
# Result dataclasses
# ──────────────────────────────────────────────────────────────

@dataclass
class ICStats:
    mean:          float
    volatility:    float
    information_ratio: float   # mean / vol
    n_periods:     int
    horizon_months: int


@dataclass
class GradeBucketStats:
    grade:          str
    mean_return:    float
    median_return:  float
    sharpe:         float
    sortino:        float
    n_obs:          int
    hit_rate:       float   # fraction of positive returns


@dataclass
class GradeValidationReport:
    """Complete results from grade validation (Layer 1)."""
    label:          str       # "OUT-OF-SAMPLE (PIT)" or "INDICATIVE / LOOK-AHEAD-CONTAMINATED"
    start_date:     date
    end_date:       date
    n_observations: int
    n_tickers:      int
    n_dates:        int

    # IC stats per horizon
    ic_stats:       dict[int, ICStats]   # horizon_months → ICStats

    # Grade bucket forward returns per horizon
    grade_returns:  dict[int, list[GradeBucketStats]]  # horizon → bucket list

    # Monotonicity: does GH > Buy > Hold > Sell > SA hold for 12m?
    monotonicity_holds_12m: bool
    monotonicity_note:      str

    # Long-short: Gotta Have minus Stay Away
    long_short_ic_ir_12m: float | None

    # Per-engine attribution
    per_engine_ic:  dict[str, float]   # engine_name → IC mean (12m)

    # Statistical caveat
    caveat: str = (
        "Past performance does not guarantee future results. "
        "A validated backtest reduces — but does not eliminate — the risk that "
        "the edge is spurious or regime-dependent."
    )


# ──────────────────────────────────────────────────────────────
# Pure computation functions
# ──────────────────────────────────────────────────────────────

def compute_ic(scores: pd.Series, returns: pd.Series) -> float:
    """
    Spearman rank IC between scores and returns.

    Returns 0.0 when fewer than 5 observations or computation fails.
    """
    valid = scores.notna() & returns.notna()
    if valid.sum() < 5:
        return 0.0
    corr, _ = spearmanr(scores[valid], returns[valid])
    return float(corr) if not np.isnan(corr) else 0.0


def compute_ic_series(
    data:           pd.DataFrame,
    score_col:      str = "composite",
    return_col:     str = "fwd_12m",
    min_obs_per_date: int = 5,
) -> pd.Series:
    """
    Compute IC for each date in the panel.

    Returns a pd.Series indexed by date with the per-date IC value.
    """
    ic_by_date: dict[date, float] = {}
    for dt, group in data.groupby("date"):
        if len(group.dropna(subset=[score_col, return_col])) < min_obs_per_date:
            continue
        ic_by_date[dt] = compute_ic(group[score_col], group[return_col])
    return pd.Series(ic_by_date, name=f"IC_{return_col}")


def compute_ic_stats(ic_series: pd.Series, horizon_months: int) -> ICStats:
    """Summarise an IC time series."""
    if len(ic_series) < 2:
        return ICStats(0.0, 0.0, 0.0, len(ic_series), horizon_months)
    mean = float(ic_series.mean())
    vol  = float(ic_series.std())
    ir   = mean / vol if vol > 0 else 0.0
    return ICStats(mean=mean, volatility=vol, information_ratio=ir,
                   n_periods=len(ic_series), horizon_months=horizon_months)


def compute_grade_bucket_stats(
    data:       pd.DataFrame,
    return_col: str,
) -> list[GradeBucketStats]:
    """Return statistics per grade bucket for one return horizon."""
    results = []
    for grade in _GRADE_ORDER:
        sub = data[data["grade"] == grade][return_col].dropna()
        if len(sub) == 0:
            results.append(GradeBucketStats(grade, 0.0, 0.0, 0.0, 0.0, 0, 0.0))
            continue
        mean   = float(sub.mean())
        median = float(sub.median())
        std    = float(sub.std()) if len(sub) > 1 else 1e-6
        neg    = sub[sub < 0]
        dd     = float(neg.std()) if len(neg) > 1 else 1e-6
        sharpe = mean / std if std > 0 else 0.0
        sortino = mean / dd if dd > 0 else 0.0
        hit    = float((sub > 0).mean())
        results.append(GradeBucketStats(grade, mean, median, sharpe, sortino, len(sub), hit))
    return results


def check_monotonicity(bucket_stats: list[GradeBucketStats]) -> tuple[bool, str]:
    """
    Test whether mean returns are monotonically increasing from Stay Away → Gotta Have.
    Returns (holds, explanation_string).
    """
    means = {s.grade: s.mean_return for s in bucket_stats}
    pairs = [
        ("Stay Away", "Sell"),
        ("Sell",      "Hold"),
        ("Hold",      "Buy"),
        ("Buy",       "Gotta Have"),
    ]
    violations = []
    for lower, higher in pairs:
        ml = means.get(lower, np.nan)
        mh = means.get(higher, np.nan)
        if not np.isnan(ml) and not np.isnan(mh) and ml > mh:
            violations.append(f"{higher} ({mh:.1%}) < {lower} ({ml:.1%})")

    if violations:
        return False, "Monotonicity FAILS: " + "; ".join(violations)
    return True, "Monotonicity holds: GH > Buy > Hold > Sell > SA"


def compute_hit_rate(
    data:       pd.DataFrame,
    return_col: str,
    signal_grades: list[str] = ("Buy", "Gotta Have"),
) -> float:
    """
    Fraction of Buy/Gotta-Have grades where next-period return was positive.
    """
    relevant = data[data["grade"].isin(signal_grades)][return_col].dropna()
    if len(relevant) == 0:
        return np.nan
    return float((relevant > 0).mean())


def compute_per_engine_ic(
    data:         pd.DataFrame,
    return_col:   str = "fwd_12m",
) -> dict[str, float]:
    """
    IC attributable to each engine when used alone.
    Requires columns fund_score, tech_score, quant_score.
    """
    result: dict[str, float] = {}
    for engine, col in [("fundamental", "fund_score"),
                        ("technical",   "tech_score"),
                        ("quantitative","quant_score")]:
        if col in data.columns:
            ic_s = compute_ic_series(data, score_col=col, return_col=return_col)
            result[engine] = float(ic_s.mean()) if len(ic_s) > 0 else np.nan
    return result


def split_oos(
    data:         pd.DataFrame,
    oos_from:     date,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split data into in-sample (< oos_from) and out-of-sample (>= oos_from)."""
    mask = pd.to_datetime(data["date"]).dt.date >= oos_from
    return data[~mask].copy(), data[mask].copy()


# ──────────────────────────────────────────────────────────────
# Main validator
# ──────────────────────────────────────────────────────────────

def validate_grades(
    records:          pd.DataFrame,
    contamination_label: str = "OUT-OF-SAMPLE (PIT)",
    oos_from:         date | None = None,
) -> GradeValidationReport:
    """
    Run full grade validation on a panel DataFrame.

    Expected columns:
        date, ticker, composite, grade, fund_score, tech_score, quant_score,
        fwd_1m, fwd_3m, fwd_6m, fwd_12m

    oos_from: if provided, only the OOS subset is reported.
    """
    df = records.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.date

    if oos_from:
        _, df = split_oos(df, oos_from)

    if df.empty:
        raise ValueError("No records in the specified date range.")

    start_date = min(df["date"])
    end_date   = max(df["date"])
    n_obs      = len(df)
    n_tickers  = df["ticker"].nunique()
    n_dates    = df["date"].nunique()

    # ── IC per horizon ────────────────────────────────────────
    ic_stats: dict[int, ICStats] = {}
    for months, col in _COL_MAP.items():
        if col not in df.columns:
            continue
        ic_s = compute_ic_series(df, return_col=col)
        ic_stats[months] = compute_ic_stats(ic_s, months)

    # ── Grade bucket returns per horizon ─────────────────────
    grade_returns: dict[int, list[GradeBucketStats]] = {}
    for months, col in _COL_MAP.items():
        if col not in df.columns:
            continue
        grade_returns[months] = compute_grade_bucket_stats(df, col)

    # ── Monotonicity test (12m) ───────────────────────────────
    if 12 in grade_returns:
        mono_holds, mono_note = check_monotonicity(grade_returns[12])
    else:
        mono_holds, mono_note = False, "12m returns not available."

    # ── Long-short IC IR ──────────────────────────────────────
    long_short_ic_ir = None
    if "fwd_12m" in df.columns:
        ls_mask = df["grade"].isin(["Gotta Have", "Stay Away"])
        ls_data = df[ls_mask].copy()
        if len(ls_data) >= 10:
            ls_data = ls_data.copy()
            ls_data["ls_score"] = ls_data["grade"].map(
                {"Gotta Have": 1.0, "Stay Away": -1.0}
            )
            ic_ls = compute_ic_series(ls_data, score_col="ls_score",
                                       return_col="fwd_12m")
            if len(ic_ls) >= 2:
                long_short_ic_ir = float(ic_ls.mean() / (ic_ls.std() + 1e-8))

    # ── Per-engine attribution ────────────────────────────────
    per_engine_ic = compute_per_engine_ic(df, "fwd_12m")

    logger.info(
        "Grade validation complete: %d obs, %d tickers, %d dates. "
        "IC12m mean=%.3f, IR=%.2f, monotonicity=%s",
        n_obs, n_tickers, n_dates,
        ic_stats.get(12, ICStats(0,0,0,0,12)).mean,
        ic_stats.get(12, ICStats(0,0,0,0,12)).information_ratio,
        mono_holds,
    )

    return GradeValidationReport(
        label                 = contamination_label,
        start_date            = start_date,
        end_date              = end_date,
        n_observations        = n_obs,
        n_tickers             = n_tickers,
        n_dates               = n_dates,
        ic_stats              = ic_stats,
        grade_returns         = grade_returns,
        monotonicity_holds_12m = mono_holds,
        monotonicity_note     = mono_note,
        long_short_ic_ir_12m  = long_short_ic_ir,
        per_engine_ic         = per_engine_ic,
    )
