"""
Howard Marks Investing Guide Screener

Evaluates stocks through the lens of Howard Marks' "The Most Important
Thing" and "Mastering the Market Cycle" — avoiding permanent capital loss,
durability through cycles, the pendulum of sentiment, margin of safety vs.
Earnings Power Value, balance-sheet resilience, and risk/reward asymmetry.
Also prints a one-time "Where We Stand in the Cycle" banner based on the
yield curve, high-yield credit spreads, and the VIX.

Usage:
  python marks_screen.py AAPL
  python marks_screen.py AAPL MSFT KO
  python marks_screen.py --file tickers.txt
  python marks_screen.py AAPL --output json
  python marks_screen.py AAPL --output markdown
  python marks_screen.py --universe marks --output summary
  python marks_screen.py AAPL --delay 2
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

from stockgrader.howard_marks.criteria import evaluate_marks
from stockgrader.howard_marks.cycle import get_cycle_reading
from stockgrader.howard_marks.fetcher import MarksFetcher
from stockgrader.howard_marks.models import CycleReading, MarksResult
from stockgrader.howard_marks.reporter import (
    format_cycle_banner,
    format_cycle_banner_markdown,
    format_summary,
    print_results,
)

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
        description="Evaluate stocks through the lens of Howard Marks' investing philosophy.",
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
        choices=["terminal", "json", "markdown", "summary"],
        default="terminal",
        help="Output format (default: terminal; 'summary' = one-line-per-stock table)",
    )
    p.add_argument(
        "--universe", "-u",
        choices=["marks", "sp500", "djia", "yaml"],
        default=None,
        help="Screen a preset universe: marks (curated cyclical/value list), sp500 (Wikipedia S&P 500), djia (Wikipedia DJIA 30), yaml (universe.yaml)",
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
    p.add_argument(
        "--qualifiers-only", "-q",
        action="store_true",
        help="Only show stocks rated 'Worth a Closer Look' or 'Compelling Opportunity'",
    )
    return p.parse_args()


_MARKS_UNIVERSE = [
    # Industrials — capital-cycle classics
    "CAT", "DE", "BA", "GE", "HON", "MMM", "EMR",
    # Energy — boom/bust commodity cycle
    "XOM", "CVX", "OXY", "SLB", "MPC", "VLO", "COP",
    # Financials — credit cycle sensitive
    "JPM", "BAC", "GS", "C", "WFC", "MS",
    # Materials — commodity cycle
    "FCX", "NUE", "X", "DOW", "LIN",
    # Semiconductors — sharp boom/bust cycles
    "INTC", "MU", "AMD", "TXN", "AVGO",
    # Autos — deeply cyclical
    "F", "GM",
    # Homebuilders — rate/credit cycle sensitive
    "DHI", "LEN", "PHM",
    # Airlines / transports — classic cyclicals
    "DAL", "UAL", "UPS", "FDX",
    # REITs — rate-sensitive, sentiment swings
    "SPG", "VNO",
]


def _load_universe_tickers(name: str) -> list[str]:
    if name == "marks":
        return _MARKS_UNIVERSE
    if name == "yaml":
        import yaml
        path = Path("universe.yaml")
        if not path.exists():
            print("Error: universe.yaml not found.", file=sys.stderr)
            sys.exit(1)
        with open(path) as f:
            data = yaml.safe_load(f)
        return [str(t).upper().strip() for t in data.get("universe", [])]
    if name == "sp500":
        import pandas as pd
        try:
            url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
            headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
            tables = pd.read_html(url, attrs={"id": "constituents"}, storage_options={"User-Agent": headers["User-Agent"]})
            return tables[0]["Symbol"].str.replace(".", "-", regex=False).tolist()
        except Exception as exc:
            print(f"Error fetching S&P 500: {exc}", file=sys.stderr)
            sys.exit(1)
    if name == "djia":
        import pandas as pd
        try:
            url = "https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average"
            headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
            tables = pd.read_html(url, storage_options={"User-Agent": headers["User-Agent"]})
            for table in tables:
                # Flatten multi-level columns to strings
                table.columns = [
                    " ".join(str(c) for c in col).strip() if isinstance(col, tuple) else str(col)
                    for col in table.columns
                ]
                cols_lower = [c.lower() for c in table.columns]
                match = next((c for c, cl in zip(table.columns, cols_lower)
                              if "symbol" in cl or "ticker" in cl), None)
                if match:
                    return table[match].dropna().astype(str).str.replace(".", "-", regex=False).tolist()
            print("Error: could not find DJIA components table on Wikipedia.", file=sys.stderr)
            sys.exit(1)
        except Exception as exc:
            print(f"Error fetching DJIA: {exc}", file=sys.stderr)
            sys.exit(1)
    return []


def load_tickers(args: argparse.Namespace) -> list[str]:
    tickers: list[str] = []

    if args.universe:
        tickers.extend(_load_universe_tickers(args.universe))

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


def evaluate_ticker(ticker: str, fetcher: MarksFetcher, cycle: CycleReading) -> MarksResult | None:
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

    return evaluate_marks(data, cycle)


def main() -> None:
    args = parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    tickers = load_tickers(args)
    if not tickers:
        print("No tickers provided. Use --help for usage.", file=sys.stderr)
        sys.exit(1)

    print("Reading the market cycle...", end="\r", flush=True)
    cycle = get_cycle_reading()
    print(" " * 40, end="\r")

    if args.output == "markdown":
        print(format_cycle_banner_markdown(cycle))
    elif args.output != "json":
        print(format_cycle_banner(cycle))

    fetcher = MarksFetcher()
    results: list[MarksResult] = []

    batch = len(tickers) > 1
    for i, ticker in enumerate(tickers):
        if i > 0 and args.delay > 0:
            time.sleep(args.delay)

        if batch:
            print(f"[{i+1}/{len(tickers)}] {ticker}...", end="\r", flush=True)

        result = evaluate_ticker(ticker, fetcher, cycle)
        if result is not None:
            results.append(result)

    if batch:
        print(" " * 40, end="\r")  # clear progress line

    if not results:
        print("No results to display.", file=sys.stderr)
        sys.exit(1)

    if args.qualifiers_only:
        results = [r for r in results if r.verdict != "Pass"]
        if not results:
            print("No stocks rated 'Worth a Closer Look' or better.")
            return

    if args.output == "summary":
        print(format_summary(results))
    else:
        print_results(results, output=args.output)


if __name__ == "__main__":
    main()
