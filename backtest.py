"""
Backtest CLI — entry point for all three validation layers.

Usage
-----
  python backtest.py grades   --start 2010-01-01 --end 2024-12-31 [--oos-from 2021-01-01]
  python backtest.py portfolio --fund conservative --rebalance quarterly [--costs default]
  python backtest.py calibrate --train-end 2018-12-31 --val-end 2020-12-31 [--start 2010-01-01]

All results are written to runs/backtest/ and printed to stdout.
Non-PIT data sources are rejected in strict mode (default).
"""
from __future__ import annotations

import json
import logging
import os
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Optional

import typer

app = typer.Typer(
    name="backtest",
    help="Stock Grader backtesting & validation suite.",
    add_completion=False,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

_RUNS_DIR = Path("runs") / "backtest"


# ──────────────────────────────────────────────────────────────
# Shared helpers
# ──────────────────────────────────────────────────────────────

def _load_config() -> dict:
    from stockgrader.config import get_config
    return get_config()


def _make_adapter(config: dict):
    """
    Build the best available PIT adapter.

    Priority:
      1. Sharadar (if NASDAQ_DATA_LINK_API_KEY set)
      2. SyntheticPITAdapter (for --dry-run / testing)
         → labelled CONTAMINATED so reports are clearly marked.
    """
    from stockgrader.backtest import (
        SharadarAdapter, SyntheticPITAdapter, require_pit
    )
    api_key = os.environ.get("NASDAQ_DATA_LINK_API_KEY", "")
    if api_key:
        adapter = SharadarAdapter(api_key=api_key)
    else:
        typer.echo(
            "⚠️  NASDAQ_DATA_LINK_API_KEY not set. "
            "Falling back to SyntheticPITAdapter (test data only).",
            err=True,
        )
        adapter = SyntheticPITAdapter()

    label = require_pit(adapter, config)
    return adapter, label


def _datestamp() -> str:
    return datetime.utcnow().strftime("%Y-%m-%d_%H%M")


def _save(content: str, filename: str) -> Path:
    from stockgrader.backtest import save_report
    path = _RUNS_DIR / filename
    save_report(content, path)
    return path


# ──────────────────────────────────────────────────────────────
# Subcommand: grades
# ──────────────────────────────────────────────────────────────

@app.command("grades")
def cmd_grades(
    start: str = typer.Option("2015-01-01", help="Start date (YYYY-MM-DD)"),
    end:   str = typer.Option(...,           help="End date (YYYY-MM-DD)"),
    oos_from: Optional[str] = typer.Option(None, "--oos-from", help="OOS start (YYYY-MM-DD)"),
    save:  bool = typer.Option(True, help="Write report to runs/backtest/"),
):
    """
    Layer 1: Validate that higher grades predict higher forward returns.

    Computes Spearman IC, grade-bucket return spreads, monotonicity test,
    and per-engine attribution.  Uses the SyntheticPITAdapter when no
    Nasdaq Data Link key is present.
    """
    from stockgrader.backtest import (
        validate_grades, render_validation_report, to_json
    )

    config  = _load_config()
    adapter, label = _make_adapter(config)

    # Retrieve pre-scored records from adapter
    if hasattr(adapter, "get_all_scores"):
        records = adapter.get_all_scores()
    else:
        typer.echo("❌ This adapter does not expose pre-scored records.", err=True)
        typer.echo("   Use the Sharadar adapter with a scored panel CSV.", err=True)
        raise typer.Exit(1)

    oos_date: date | None = None
    if oos_from:
        oos_date = date.fromisoformat(oos_from)

    report = validate_grades(records, contamination_label=label, oos_from=oos_date)
    md     = render_validation_report(report)
    typer.echo(md)

    if save:
        ts   = _datestamp()
        path = _save(md, f"grades_{ts}.md")
        _save(to_json(report), f"grades_{ts}.json")
        typer.echo(f"\n✅ Reports saved to {path.parent}/")


# ──────────────────────────────────────────────────────────────
# Subcommand: portfolio
# ──────────────────────────────────────────────────────────────

@app.command("portfolio")
def cmd_portfolio(
    fund:      str = typer.Option("balanced",  help="Fund name (conservative/balanced/aggressive/…)"),
    rebalance: str = typer.Option("quarterly", help="monthly | quarterly | semi_annual"),
    costs:     str = typer.Option("default",   help="'default' or 'none' to disable costs"),
    save:      bool = typer.Option(True, help="Write report to runs/backtest/"),
):
    """
    Layer 2: Walk-forward portfolio backtest (net of costs).

    Rebalances on the specified schedule using only PIT-available data.
    Reports equity curve, Sharpe, Sortino, max drawdown, turnover, cost drag.
    """
    from stockgrader.backtest import (
        walk_forward_from_scores, CostModel,
        render_backtest_report, to_json
    )

    config  = _load_config()
    adapter, label = _make_adapter(config)

    if hasattr(adapter, "get_all_scores"):
        records = adapter.get_all_scores()
        # Rename fwd_12m → period_return for walk_forward compatibility
        if "fwd_12m" in records.columns and "period_return" not in records.columns:
            records = records.copy()
            records["period_return"] = records["fwd_12m"]
    else:
        typer.echo("❌ Adapter does not expose pre-scored records.", err=True)
        raise typer.Exit(1)

    cost_model = CostModel() if costs == "default" else CostModel(slippage_bps=0.0)

    result = walk_forward_from_scores(
        scored_records      = records,
        portfolio_name      = fund,
        rebalance_freq      = rebalance,
        cost_model          = cost_model,
        config              = config,
        contamination_label = label,
    )
    md = render_backtest_report(result)
    typer.echo(md)

    if save:
        ts   = _datestamp()
        path = _save(md, f"portfolio_{fund}_{rebalance}_{ts}.md")
        _save(to_json(result), f"portfolio_{fund}_{rebalance}_{ts}.json")
        typer.echo(f"\n✅ Reports saved to {path.parent}/")


# ──────────────────────────────────────────────────────────────
# Subcommand: calibrate
# ──────────────────────────────────────────────────────────────

@app.command("calibrate")
def cmd_calibrate(
    train_end: str = typer.Option(..., "--train-end", help="End of training period (YYYY-MM-DD)"),
    val_end:   str = typer.Option(..., "--val-end",   help="End of validation period (YYYY-MM-DD)"),
    save:      bool = typer.Option(True, help="Write report to runs/backtest/"),
):
    """
    Layer 3: Grid-search engine weights; report OOS test-period IC.

    Selects the best fund/tech/quant split on the validation IC, then
    evaluates it once on the held-out test set.  Never touches the test
    set during selection.
    """
    from stockgrader.backtest import (
        calibrate_weights, render_calibration_report, to_json
    )

    config  = _load_config()
    adapter, label = _make_adapter(config)

    if hasattr(adapter, "get_all_scores"):
        records = adapter.get_all_scores()
    else:
        typer.echo("❌ Adapter does not expose pre-scored records.", err=True)
        raise typer.Exit(1)

    t_end = date.fromisoformat(train_end)
    v_end = date.fromisoformat(val_end)

    result = calibrate_weights(
        records             = records,
        train_end           = t_end,
        val_end             = v_end,
        contamination_label = label,
    )
    md = render_calibration_report(result)
    typer.echo(md)

    typer.echo(
        f"\n🏆 Best weights: "
        f"Fundamental={result.best_weights['fundamental']:.0%}  "
        f"Technical={result.best_weights['technical']:.0%}  "
        f"Quantitative={result.best_weights['quantitative']:.0%}"
    )
    if result.test_ic is not None:
        typer.echo(f"   OOS test IC = {result.test_ic:.4f}  |  IR = {result.test_ir:.2f}")

    if save:
        ts   = _datestamp()
        path = _save(md, f"calibrate_{ts}.md")
        _save(to_json(result), f"calibrate_{ts}.json")
        typer.echo(f"\n✅ Reports saved to {path.parent}/")


# ──────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app()
