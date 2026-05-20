"""
Backtest report generation — Markdown and JSON for all three layers.

  render_validation_report(GradeValidationReport)  → str (markdown)
  render_backtest_report(BacktestResult)            → str (markdown)
  render_calibration_report(CalibrationResult)      → str (markdown)

  save_report(content, path)  → writes UTF-8 file, creates parent dirs
  to_json(obj)                → dict serialisable by json.dumps
"""
from __future__ import annotations

import json
import logging
from datetime import date, datetime
from pathlib import Path
from typing import Any

from .grade_validator import GradeValidationReport, ICStats, GradeBucketStats
from .walk_forward import BacktestResult
from .calibrator import CalibrationResult

logger = logging.getLogger(__name__)

_GRADE_ORDER = ["Stay Away", "Sell", "Hold", "Buy", "Gotta Have"]
_EMOJI = {
    "Stay Away":  "🚫",
    "Sell":       "⚠️",
    "Hold":       "⚖️",
    "Buy":        "✅",
    "Gotta Have": "⭐",
}


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────

def _pct(v: float | None, decimals: int = 1) -> str:
    if v is None:
        return "N/A"
    return f"{v * 100:.{decimals}f}%"


def _f(v: float | None, decimals: int = 4) -> str:
    if v is None:
        return "N/A"
    return f"{v:.{decimals}f}"


def _date(d: date | None) -> str:
    return d.isoformat() if d else "N/A"


def _sep(width: int = 70) -> str:
    return "─" * width


# ──────────────────────────────────────────────────────────────
# Layer 1: Grade validation report
# ──────────────────────────────────────────────────────────────

def render_validation_report(report: GradeValidationReport) -> str:
    lines: list[str] = []

    lines += [
        f"# Grade Validation Report",
        f"",
        f"**Label:** `{report.label}`  ",
        f"**Period:** {_date(report.start_date)} → {_date(report.end_date)}  ",
        f"**Observations:** {report.n_observations:,}  ({report.n_tickers} tickers, {report.n_dates} dates)",
        f"",
        f"> {report.caveat}",
        f"",
        _sep(),
        f"",
    ]

    # IC table
    lines += ["## Information Coefficient (Spearman IC)", ""]
    lines += ["| Horizon | Mean IC | IC Vol | IC IR | N Periods |",
              "|---------|---------|--------|-------|-----------|"]
    for months in sorted(report.ic_stats):
        s = report.ic_stats[months]
        lines.append(
            f"| {months}m | {_f(s.mean)} | {_f(s.volatility)} "
            f"| {_f(s.information_ratio, 2)} | {s.n_periods} |"
        )
    lines.append("")

    # Grade return spreads
    lines += ["## Forward-Return Spread by Grade", ""]
    for months in sorted(report.grade_returns):
        lines += [f"### {months}-Month Forward Returns", ""]
        lines += ["| Grade | Mean | Median | Sharpe | Hit Rate | N |",
                  "|-------|------|--------|--------|----------|---|"]
        for b in report.grade_returns[months]:
            emoji = _EMOJI.get(b.grade, "")
            lines.append(
                f"| {emoji} {b.grade} | {_pct(b.mean_return)} | {_pct(b.median_return)} "
                f"| {_f(b.sharpe, 2)} | {_pct(b.hit_rate)} | {b.n_obs:,} |"
            )
        lines.append("")

    # Monotonicity
    lines += ["## Monotonicity Test (12-Month)", ""]
    status = "✅ HOLDS" if report.monotonicity_holds_12m else "❌ FAILS"
    lines += [f"**Result:** {status}", f"", f"{report.monotonicity_note}", ""]

    # Long-short
    lines += ["## Long-Short Portfolio (Gotta Have − Stay Away)", ""]
    if report.long_short_ic_ir_12m is not None:
        lines.append(f"**IC-IR (12m):** {_f(report.long_short_ic_ir_12m, 2)}")
    else:
        lines.append("*Insufficient data for long-short computation.*")
    lines.append("")

    # Per-engine IC
    lines += ["## Per-Engine Attribution (IC-12m mean)", ""]
    lines += ["| Engine | IC Mean |", "|--------|---------|"]
    for engine, ic in report.per_engine_ic.items():
        lines.append(f"| {engine} | {_f(ic)} |")
    lines.append("")

    lines.append(f"*Generated {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}*")
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────
# Layer 2: Walk-forward portfolio report
# ──────────────────────────────────────────────────────────────

def render_backtest_report(result: BacktestResult) -> str:
    lines: list[str] = []

    lines += [
        f"# Walk-Forward Backtest: {result.portfolio_name}",
        f"",
        f"**Label:** `{result.contamination_label}`  ",
        f"**Period:** {_date(result.start_date)} → {_date(result.end_date)}  ",
        f"**Rebalance:** {result.rebalance_freq}  ",
        f"**Benchmark:** {result.benchmark_name}",
        f"",
        f"> {result.caveat}",
        f"",
        _sep(),
        f"",
    ]

    # Summary stats
    lines += ["## Performance Summary (Net of Costs)", ""]
    lines += ["| Metric | Portfolio | Benchmark | Active |",
              "|--------|-----------|-----------|--------|"]
    ann_bench = result.ann_return - result.outperformance
    lines += [
        f"| Ann. Return | {_pct(result.ann_return)} | {_pct(ann_bench)} | {_pct(result.outperformance)} |",
        f"| Ann. Vol    | {_pct(result.ann_vol)} | — | — |",
        f"| Sharpe      | {_f(result.sharpe, 2)} | — | — |",
        f"| Sortino     | {_f(result.sortino, 2)} | — | — |",
        f"| Max Drawdown| {_pct(result.max_drawdown)} | — | — |",
        f"| Ann. Turnover| {_pct(result.ann_turnover)} | — | — |",
        f"| Cost Drag   | {_pct(result.cost_drag, 2)} | — | — |",
    ]
    lines.append("")

    # Equity curve (last 5 values as sample)
    lines += ["## Equity Curve (sample — last 5 dates)", ""]
    lines += ["| Date | Portfolio | Benchmark |",
              "|------|-----------|-----------|"]
    curve_tail = result.equity_curve.tail(5)
    bench_tail = result.benchmark_curve.tail(5)
    for dt in curve_tail.index:
        pv = curve_tail.get(dt, float("nan"))
        bv = bench_tail.get(dt, float("nan"))
        lines.append(f"| {dt} | {_f(pv, 4)} | {_f(bv, 4)} |")
    lines.append("")

    # Period results
    if result.periods:
        lines += ["## Period-by-Period Results", ""]
        lines += ["| Start | End | Net Return | Turnover | # Holdings |",
                  "|-------|-----|-----------|---------|------------|"]
        for p in result.periods[-12:]:   # last 12 periods
            n_hold = len(p.weights)
            lines.append(
                f"| {_date(p.start)} | {_date(p.end)} | {_pct(p.net_ret)} "
                f"| {_pct(p.turnover)} | {n_hold} |"
            )
        if len(result.periods) > 12:
            lines.append(f"*… {len(result.periods) - 12} earlier periods omitted …*")
        lines.append("")

    lines.append(f"*Generated {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}*")
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────
# Layer 3: Weight calibration report
# ──────────────────────────────────────────────────────────────

def render_calibration_report(result: CalibrationResult) -> str:
    lines: list[str] = []

    lines += [
        f"# Weight Calibration Report",
        f"",
        f"**Label:** `{result.contamination_label}`  ",
        f"**Train:**      {_date(result.train_start)} → {_date(result.train_end)}  ",
        f"**Validation:** {_date(result.val_start)} → {_date(result.val_end)}  ",
        f"**Test (OOS):** {_date(result.test_start)} → {_date(result.test_end)}",
        f"",
        f"> {result.caveat}",
        f"",
        _sep(),
        f"",
    ]

    # Best weights
    lines += ["## Optimal Engine Weights (selected on validation IC)", ""]
    lines += ["| Engine | Weight |", "|--------|--------|"]
    for engine, w in result.best_weights.items():
        lines.append(f"| {engine} | {w:.0%} |")
    lines.append("")

    # IC results
    lines += ["## IC Results", ""]
    lines += ["| Period | Mean IC | IC IR |", "|--------|---------|-------|"]
    lines += [
        f"| Validation | {_f(result.val_ic)} | {_f(result.val_ir, 2)} |",
        f"| Test (OOS) | {_f(result.test_ic)} | {_f(result.test_ir, 2)} |",
    ]
    lines.append("")

    # Top-10 grid points by val IC
    lines += ["## Top 10 Grid Points (by Validation IC)", ""]
    lines += ["| Rank | Fund% | Tech% | Quant% | Val IC | Val IR |",
              "|------|-------|-------|--------|--------|--------|"]
    for i, pt in enumerate(result.grid_results[:10], 1):
        lines.append(
            f"| {i} | {pt.fund_w:.0%} | {pt.tech_w:.0%} | {pt.quant_w:.0%} "
            f"| {_f(pt.val_ic)} | {_f(pt.val_ir, 2)} |"
        )
    lines.append(f"")
    lines.append(f"*Grid size: {result.n_grid_points} combinations evaluated.*")
    lines.append("")
    lines.append(f"*Generated {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}*")
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────
# JSON serialiser
# ──────────────────────────────────────────────────────────────

def _default(obj: Any) -> Any:
    if isinstance(obj, date):
        return obj.isoformat()
    if hasattr(obj, "__dataclass_fields__"):
        return obj.__dict__
    raise TypeError(f"Not JSON serialisable: {type(obj)}")


def to_json(obj: Any) -> str:
    """Serialise a backtest result object to a JSON string."""
    if hasattr(obj, "__dataclass_fields__"):
        d = obj.__dict__.copy()
        # Convert pandas Series to dicts
        for k, v in d.items():
            import pandas as pd
            if isinstance(v, pd.Series):
                d[k] = {str(idx): float(val) for idx, val in v.items()
                        if not (isinstance(val, float) and val != val)}
        return json.dumps(d, default=_default, indent=2)
    return json.dumps(obj, default=_default, indent=2)


# ──────────────────────────────────────────────────────────────
# File I/O
# ──────────────────────────────────────────────────────────────

def save_report(content: str, path: Path) -> None:
    """Write content to path, creating parent directories as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    logger.info("Saved report → %s", path)
