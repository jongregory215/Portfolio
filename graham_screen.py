"""
Graham Intelligent Investor Stock Screener

Evaluates stocks against Benjamin Graham's criteria from
The Intelligent Investor (Chapters 14 & 15).

Usage:
  python graham_screen.py AAPL
  python graham_screen.py AAPL MSFT KO
  python graham_screen.py --file tickers.txt
  python graham_screen.py AAPL --output json
  python graham_screen.py AAPL --output markdown
  python graham_screen.py AAPL --mode defensive
  python graham_screen.py AAPL --delay 2
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

# Force UTF-8 output on Windows so Unicode box-drawing chars and ✓/✗ render
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from stockgrader.graham.criteria import evaluate_defensive, evaluate_enterprising
from stockgrader.graham.fetcher import GrahamFetcher
from stockgrader.graham.models import GrahamResult
from stockgrader.graham.reporter import print_results

# Load .env so FMP_API_KEY / FRED_API_KEY are available without manual export
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

logging.basicConfig(
    level=logging.WARNING,
    format="%(levelname)s %(name)s: %(message)s",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate stocks against Benjamin Graham's Intelligent Investor criteria.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "tickers",
        nargs="*",
        metavar="TICKER",
        help="One or more stock tickers (e.g. AAPL KO JNJ)",
    )
    p.add_argument(
        "--file", "-f",
        metavar="FILE",
        help="Path to a text file with one ticker per line",
    )
    p.add_argument(
        "--output", "-o",
        choices=["terminal", "json", "markdown"],
        default="terminal",
        help="Output format (default: terminal)",
    )
    p.add_argument(
        "--mode", "-m",
        choices=["both", "defensive", "enterprising"],
        default="both",
        help="Which investor type to evaluate (default: both)",
    )
    p.add_argument(
        "--delay", "-d",
        type=float,
        default=0.5,
        metavar="SECONDS",
        help="Delay between tickers in batch mode (default: 0.5s)",
    )
    p.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show debug logging",
    )
    return p.parse_args()


def load_tickers(args: argparse.Namespace) -> list[str]:
    tickers: list[str] = []

    if args.file:
        path = Path(args.file)
        if not path.exists():
            print(f"Error: file not found: {path}", file=sys.stderr)
            sys.exit(1)
        lines = path.read_text().splitlines()
        tickers.extend(
            line.strip().upper()
            for line in lines
            if line.strip() and not line.startswith("#")
        )

    if args.tickers:
        tickers.extend(t.upper() for t in args.tickers)

    seen: set[str] = set()
    deduped: list[str] = []
    for t in tickers:
        if t not in seen:
            seen.add(t)
            deduped.append(t)

    return deduped


def evaluate_ticker(ticker: str, fetcher: GrahamFetcher, mode: str) -> GrahamResult | None:
    print(f"Fetching {ticker}...", end="\r", flush=True)
    try:
        data = fetcher.fetch(ticker)
    except Exception as exc:
        print(f"Error fetching {ticker}: {exc}", file=sys.stderr)
        return None

    price = data.get("price")
    if price is None:
        print(f"Warning: no price available for {ticker} — skipping.", file=sys.stderr)
        return None

    defensive   = evaluate_defensive(data)   if mode in ("both", "defensive")    else None
    enterprising = evaluate_enterprising(data) if mode in ("both", "enterprising") else None

    # If only one mode, provide a placeholder for the other
    from stockgrader.graham.models import DefensiveResult, EnterprisingResult
    if defensive is None:
        defensive = DefensiveResult(
            criteria=[], criteria_met=0, total_criteria=7,
            graham_number=None, price_vs_graham_pct=None,
            verdict="Not evaluated",
        )
    if enterprising is None:
        enterprising = EnterprisingResult(
            criteria=[], criteria_met=0, total_criteria=5,
            ncav_per_share=None, price_vs_ncav_pct=None,
            verdict="Not evaluated",
        )

    return GrahamResult(
        ticker=data["ticker"],
        as_of=data["as_of"],
        price=price,
        company_name=data.get("company_name", ticker),
        eps_source=data.get("eps_source", "yfinance"),
        defensive=defensive,
        enterprising=enterprising,
    )


def main() -> None:
    args = parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    tickers = load_tickers(args)
    if not tickers:
        print("No tickers provided. Use --help for usage.", file=sys.stderr)
        sys.exit(1)

    fetcher = GrahamFetcher()
    results: list[GrahamResult] = []

    for i, ticker in enumerate(tickers):
        if i > 0 and args.delay > 0:
            time.sleep(args.delay)

        result = evaluate_ticker(ticker, fetcher, args.mode)
        if result is not None:
            results.append(result)

    if not results:
        print("No results to display.", file=sys.stderr)
        sys.exit(1)

    print_results(results, output=args.output)


if __name__ == "__main__":
    main()
