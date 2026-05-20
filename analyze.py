#!/usr/bin/env python3
"""
CLI entry point — analyze a single equity ticker.

Usage:
    python analyze.py TICKER [--portfolio all] [--format json|md|term]

Or after `pip install -e .`:
    stockgrader analyze TICKER
"""
import typer
from typing import Optional

app = typer.Typer(
    name="stockgrader",
    help="Stock Analysis Engine — grade a ticker on demand.",
    no_args_is_help=True,
)


@app.command()
def analyze(
    ticker: str = typer.Argument(..., help="Equity ticker, e.g. AAPL"),
    portfolio: str = typer.Option("all", help="Portfolio filter: all | very_conservative | conservative | balanced | aggressive | very_aggressive"),
    format: str = typer.Option("term", help="Output format: json | md | term"),
    no_cache: bool = typer.Option(False, help="Bypass disk cache and fetch live data"),
):
    """Analyze TICKER and emit an overall grade, price ladder, and portfolio sub-grades."""
    # Implementation wired in Step 2 (data layer) through Step 10 (reporters)
    typer.echo(f"[stockgrader] Analyzing {ticker.upper()} …  (data layer not yet implemented)")
    raise SystemExit(1)


if __name__ == "__main__":
    app()
