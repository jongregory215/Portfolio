#!/usr/bin/env python3
"""
Stock Analysis Engine — CLI entry point.

Usage:
    python analyze.py AAPL
    python analyze.py MSFT --format json
    python analyze.py JPM  --format md --save
    python analyze.py TSLA --portfolio aggressive --no-cache

The pipeline (per build spec §16):
  1. DataFetcher.fetch()          → normalized data dict
  2. First-pass Orchestrator      → engine scores + fundamental score
  3. build_price_ladder()         → DCF + multiple fair value + grade boundaries
  4. Inject stay_away threshold   → arms the extreme-overvaluation circuit breaker
  5. Second-pass Orchestrator     → final grade with all circuit breakers active
  6. compute_sub_grades()         → per-portfolio eligibility + adjusted composite
  7. Assemble AnalysisResult      → typed canonical model
  8. Reporter(s)                  → JSON / Markdown / terminal (rich)
"""
from __future__ import annotations

import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import typer
from rich.console import Console

app = typer.Typer(
    name="stockgrader",
    help="Stock Analysis Engine — grade an equity ticker on demand.",
    no_args_is_help=True,
)
_err = Console(stderr=True)   # progress / warnings always to stderr


@app.command()
def analyze(
    ticker: str = typer.Argument(..., help="Equity ticker symbol, e.g. AAPL"),
    portfolio: str = typer.Option(
        "all",
        "--portfolio", "-p",
        help="Apply a portfolio-specific weight profile: all | very_conservative | "
             "conservative | balanced | aggressive | very_aggressive",
        metavar="PORTFOLIO",
    ),
    format: str = typer.Option(
        "term",
        "--format", "-f",
        help="Output format: term | json | md | all",
        metavar="FORMAT",
    ),
    save: bool = typer.Option(
        False, "--save", "-s",
        help="Save JSON + Markdown reports to --runs-dir",
    ),
    runs_dir: str = typer.Option(
        "runs", "--runs-dir",
        help="Directory for saved output files",
    ),
    no_cache: bool = typer.Option(
        False, "--no-cache",
        help="Bypass disk cache and fetch live data",
    ),
):
    """
    Analyze TICKER and emit an overall grade, price ladder, and portfolio sub-grades.

    Exit codes:
      0  Normal (any grade)
      1  Fatal error (bad ticker, no data, etc.)
      2  Stay Away grade (useful for scripts / alerting)
    """
    from stockgrader.pipeline import run_analysis

    ticker = ticker.upper().strip()
    fmt    = format.lower()

    _err.print(f"[dim]Analyzing {ticker}…[/dim]")

    try:
        result = run_analysis(ticker, portfolio=portfolio, no_cache=no_cache)
    except SystemExit:
        raise
    except Exception as exc:
        _err.print(f"[bold red]Error:[/bold red] {exc}")
        raise typer.Exit(1)

    # ── Output ────────────────────────────────────────────────
    from stockgrader.reporting import JSONReporter, MarkdownReporter, TerminalReporter

    if fmt == "json":
        print(JSONReporter().render(result))
    elif fmt in ("md", "markdown"):
        print(MarkdownReporter().render(result))
    elif fmt == "all":
        TerminalReporter().print_full(result)
        print()
        print(MarkdownReporter().render(result))
    else:                                       # "term" (default)
        TerminalReporter().print_full(result)

    # ── Save ──────────────────────────────────────────────────
    if save or fmt == "all":
        rd     = Path(runs_dir)
        j_path = JSONReporter().save(result, rd)
        m_path = MarkdownReporter().save(result, rd)
        _err.print(f"[dim]Saved: {j_path}[/dim]")
        _err.print(f"[dim]Saved: {m_path}[/dim]")

    # ── Exit code ─────────────────────────────────────────────
    from stockgrader.models import Grade
    if result.overall.grade == Grade.STAY_AWAY:
        raise typer.Exit(2)


if __name__ == "__main__":
    app()
