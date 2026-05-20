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

# Load .env from project root automatically (if it exists).
# This lets you store FMP_API_KEY / FRED_API_KEY in .env without
# setting them in every shell session.
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env", override=False)
except ImportError:
    pass

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
    deep: bool = typer.Option(
        False, "--deep",
        help="Use FMP for richer fundamentals + peer-relative scoring "
             "(requires FMP_API_KEY env var; free tier: 250 calls/day)",
    ),
):
    """
    Analyze TICKER and emit an overall grade, price ladder, and portfolio sub-grades.

    Default:  yfinance only — no API key required.
    --deep:   adds FMP fundamentals, TTM ratios, and GICS peer percentile scoring.
              Set FMP_API_KEY in your environment or .env file first.

    Exit codes:
      0  Normal (any grade)
      1  Fatal error (bad ticker, no data, etc.)
      2  Stay Away grade (useful for scripts / alerting)
    """
    import os
    from stockgrader.pipeline import run_analysis
    from stockgrader.data.fetcher import DataFetcher

    ticker = ticker.upper().strip()
    fmt    = format.lower()

    if deep and not os.environ.get("FMP_API_KEY"):
        _err.print("[yellow]Warning:[/yellow] --deep requested but FMP_API_KEY is not set. "
                   "Running in standard yfinance mode.")
        deep = False

    mode_label = "deep/FMP" if deep else "standard"
    _err.print(f"[dim]Analyzing {ticker} ({mode_label})…[/dim]")

    fetcher = DataFetcher(deep=deep)

    try:
        result = run_analysis(ticker, portfolio=portfolio, no_cache=no_cache,
                              fetcher=fetcher)
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
