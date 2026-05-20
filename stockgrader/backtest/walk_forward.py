"""
Walk-forward Portfolio Backtester — Layer 2.

Walk-forward only: on each historical rebalance date, construct the portfolio
using ONLY data available then (PIT universe → screen → grade → optimize).
Never look ahead.

Key outputs per fund:
  - Equity curve (daily values) vs. benchmark
  - Realized return, volatility, Sharpe, Sortino, max drawdown (net of costs)
  - Annual turnover
  - Benchmark comparison (did we beat the passive analog?)
  - Regime breakdown (bull / bear / high-vol)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
# Data classes
# ──────────────────────────────────────────────────────────────

@dataclass
class CostModel:
    commission_per_trade: float = 0.00    # per-leg commission
    slippage_bps:         float = 5.0     # round-trip basis points

    def round_trip_cost(self, turnover_fraction: float) -> float:
        """Annual cost given annual turnover fraction."""
        return turnover_fraction * self.slippage_bps / 10_000


@dataclass
class PeriodResult:
    start:      date
    end:        date
    weights:    dict[str, float]   # ticker → weight
    gross_ret:  float
    cost:       float
    net_ret:    float
    turnover:   float              # fraction of portfolio repositioned


@dataclass
class BacktestResult:
    portfolio_name:     str
    start_date:         date
    end_date:           date
    rebalance_freq:     str
    contamination_label: str

    # Equity curve: date → cumulative portfolio value (starting at 1.0)
    equity_curve:       pd.Series
    benchmark_curve:    pd.Series
    benchmark_name:     str

    # Summary statistics (net of costs)
    ann_return:         float
    ann_vol:            float
    sharpe:             float
    sortino:            float
    max_drawdown:       float
    ann_turnover:       float
    cost_drag:          float      # annual cost from transaction costs
    outperformance:     float      # vs benchmark, annualized

    # Per-period records
    periods:            list[PeriodResult] = field(default_factory=list)

    # Regime breakdown
    regime_performance: dict[str, float] = field(default_factory=dict)

    caveat: str = (
        "Walk-forward net-of-cost backtest. Past performance does not guarantee "
        "future results. Short-term/long-term capital gains mix not modeled (v1)."
    )


# ──────────────────────────────────────────────────────────────
# Rebalance calendar
# ──────────────────────────────────────────────────────────────

def get_rebalance_dates(
    start: date,
    end:   date,
    freq:  str,
) -> list[date]:
    """
    Generate rebalance dates between start and end (inclusive).

    freq: "monthly" | "quarterly" | "semi_annual"
    Returns business-day-adjusted month-start dates.
    """
    freq_map = {"monthly": "MS", "quarterly": "QS", "semi_annual": "2QS"}
    pd_freq  = freq_map.get(freq, "QS")
    dates    = pd.date_range(start=start, end=end, freq=pd_freq)
    return [d.date() for d in dates]


# ──────────────────────────────────────────────────────────────
# Return computation helpers
# ──────────────────────────────────────────────────────────────

def compute_period_return(
    weights:     dict[str, float],
    price_rows:  pd.DataFrame,       # columns: ticker, start_price, end_price
) -> float:
    """
    Compute weighted portfolio return for a holding period.

    price_rows: DataFrame with one row per ticker, start_price and end_price columns.
    Missing tickers are treated as zero return.
    """
    total = 0.0
    for ticker, w in weights.items():
        row = price_rows[price_rows["ticker"] == ticker]
        if row.empty:
            continue
        sp = float(row["start_price"].iloc[0])
        ep = float(row["end_price"].iloc[0])
        if sp > 0:
            total += w * (ep / sp - 1.0)
    return total


def compute_turnover(
    prev_weights: dict[str, float],
    new_weights:  dict[str, float],
) -> float:
    """
    Portfolio turnover = sum of absolute weight changes / 2.
    A full rebalance from one portfolio to another = 1.0 (100%).
    """
    all_tickers = set(prev_weights) | set(new_weights)
    return sum(
        abs(new_weights.get(t, 0.0) - prev_weights.get(t, 0.0))
        for t in all_tickers
    ) / 2.0


def compute_equity_curve(returns: pd.Series) -> pd.Series:
    """Cumulative product from a series of period returns."""
    return (1.0 + returns).cumprod()


def compute_max_drawdown(equity: pd.Series) -> float:
    """Maximum peak-to-trough drawdown (negative fraction)."""
    roll_max  = equity.cummax()
    drawdowns = (equity - roll_max) / roll_max
    return float(drawdowns.min())


def annualize_stats(
    returns:      pd.Series,
    risk_free:    float = 0.05,
    periods_per_yr: float | None = None,
) -> dict[str, float]:
    """Compute annualized return, vol, Sharpe, Sortino, max drawdown."""
    n = len(returns)
    if n < 2:
        return {"ann_return": 0.0, "ann_vol": 0.0, "sharpe": 0.0,
                "sortino": 0.0, "max_drawdown": 0.0}

    if periods_per_yr is None:
        # Infer from the index if possible, else default to 12 (monthly)
        periods_per_yr = 12.0

    ann_r = float((1 + returns.mean()) ** periods_per_yr - 1)
    ann_v = float(returns.std() * np.sqrt(periods_per_yr))
    sharpe = (ann_r - risk_free) / ann_v if ann_v > 0 else 0.0

    neg_r  = returns[returns < risk_free / periods_per_yr]
    dd_dev = float(neg_r.std() * np.sqrt(periods_per_yr)) if len(neg_r) > 1 else 1e-6
    sortino = (ann_r - risk_free) / dd_dev if dd_dev > 0 else 0.0

    equity = compute_equity_curve(returns)
    mdd    = compute_max_drawdown(equity)

    return {"ann_return": ann_r, "ann_vol": ann_v, "sharpe": sharpe,
            "sortino": sortino, "max_drawdown": mdd}


# ──────────────────────────────────────────────────────────────
# Walk-forward engine (pre-scored records variant)
# ──────────────────────────────────────────────────────────────

def walk_forward_from_scores(
    scored_records:  pd.DataFrame,
    portfolio_name:  str,
    rebalance_freq:  str,
    cost_model:      CostModel,
    config:          dict,
    contamination_label: str = "OUT-OF-SAMPLE (PIT)",
    risk_free:       float = 0.05,
    benchmark_name:  str = "equal_weight",
) -> BacktestResult:
    """
    Walk-forward backtest operating on pre-scored records.

    Expected columns in scored_records:
        date, ticker, composite, grade, weight_equity, period_return
        (period_return = gross return for the holding period after this date)

    This variant is used when grading has been done offline (e.g., Sharadar
    pipeline).  The live `run_full_funnel` variant replaces this for real runs.

    Parameters
    ----------
    scored_records:  Pre-scored panel.  Must include period_return column.
    portfolio_name:  Which portfolio to evaluate.
    rebalance_freq:  "monthly" | "quarterly" | "semi_annual"
    cost_model:      Transaction cost model.
    config:          Full config dict.
    contamination_label: PIT label from require_pit().
    """
    df = scored_records.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.date

    dates = sorted(df["date"].unique())
    if len(dates) < 2:
        raise ValueError("Need at least 2 rebalance dates for walk-forward.")

    port_returns:    list[tuple[date, float]] = []
    bench_returns:   list[tuple[date, float]] = []
    periods:         list[PeriodResult] = []
    prev_weights:    dict[str, float] = {}
    total_turnover   = 0.0

    for i, rebal_date in enumerate(dates[:-1]):
        end_date = dates[i + 1]

        # Candidates for this rebalance date
        day_df = df[df["date"] == rebal_date].copy()
        eligible = day_df[day_df["grade"].isin(["Buy", "Gotta Have"])]

        if eligible.empty:
            # No candidates: hold cash (0 return)
            net_ret = 0.0
            new_weights: dict[str, float] = {}
        else:
            # Use provided weights if available, else equal-weight eligible
            if "weight_equity" in eligible.columns:
                total_w = eligible["weight_equity"].sum()
                new_weights = {
                    row["ticker"]: row["weight_equity"] / total_w
                    for _, row in eligible.iterrows()
                    if total_w > 0
                }
            else:
                n  = len(eligible)
                new_weights = {row["ticker"]: 1.0 / n for _, row in eligible.iterrows()}

            # Gross return
            gross_ret = 0.0
            for _, row in eligible.iterrows():
                w = new_weights.get(row["ticker"], 0.0)
                if "period_return" in row:
                    gross_ret += w * float(row["period_return"])

            # Cost
            turnover = compute_turnover(prev_weights, new_weights)
            cost     = cost_model.round_trip_cost(turnover)
            net_ret  = gross_ret - cost
            total_turnover += turnover

            periods.append(PeriodResult(
                start     = rebal_date,
                end       = end_date,
                weights   = new_weights,
                gross_ret = gross_ret,
                cost      = cost,
                net_ret   = net_ret,
                turnover  = turnover,
            ))

        # Benchmark: equal-weight of all tickers in the universe for this date
        bench_ret = 0.0
        all_day = df[df["date"] == rebal_date]
        if "period_return" in all_day.columns and not all_day.empty:
            bench_ret = float(all_day["period_return"].mean())

        port_returns.append((end_date, net_ret))
        bench_returns.append((end_date, bench_ret))
        prev_weights = new_weights

    # Build equity curves
    port_df  = pd.Series(dict(port_returns))
    bench_df = pd.Series(dict(bench_returns))

    n_periods = len(port_df)
    period_map = {"monthly": 12, "quarterly": 4, "semi_annual": 2}
    ppy        = period_map.get(rebalance_freq, 4)

    stats      = annualize_stats(port_df,  risk_free, ppy)
    bench_stats = annualize_stats(bench_df, risk_free, ppy)

    ann_turnover = total_turnover * ppy / max(n_periods, 1)
    cost_drag    = ann_turnover * cost_model.slippage_bps / 10_000

    return BacktestResult(
        portfolio_name      = portfolio_name,
        start_date          = dates[0],
        end_date            = dates[-1],
        rebalance_freq      = rebalance_freq,
        contamination_label = contamination_label,
        equity_curve        = compute_equity_curve(port_df),
        benchmark_curve     = compute_equity_curve(bench_df),
        benchmark_name      = benchmark_name,
        ann_return          = stats["ann_return"],
        ann_vol             = stats["ann_vol"],
        sharpe              = stats["sharpe"],
        sortino             = stats["sortino"],
        max_drawdown        = stats["max_drawdown"],
        ann_turnover        = ann_turnover,
        cost_drag           = cost_drag,
        outperformance      = stats["ann_return"] - bench_stats["ann_return"],
        periods             = periods,
    )
