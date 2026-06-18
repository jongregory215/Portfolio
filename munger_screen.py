"""
Charlie Munger Screener

Evaluates stocks through Munger's "Poor Charlie's Almanack" framework:
exceptional ROIC (> 20%), asset-light model (capex < 5% of revenue), fortress
pricing power (gross margin ≥ 45%), earnings consistency and strong growth
(EPS CAGR > 8%), owner earnings quality, extreme debt aversion (LT debt < 3×
FCF), and a 20% margin of safety to intrinsic value.

Usage:
  python munger_screen.py COST
  python munger_screen.py COST V MA
  python munger_screen.py --file tickers.txt
  python munger_screen.py COST --output json
  python munger_screen.py COST --output markdown
  python munger_screen.py --universe munger --output summary
  python munger_screen.py --universe sp500 --output summary
  python munger_screen.py COST --delay 2
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from stockgrader.charlie_munger.criteria import evaluate_munger
from stockgrader.charlie_munger.fetcher import MungerFetcher
from stockgrader.charlie_munger.models import MungerResult
from stockgrader.charlie_munger.reporter import format_summary, print_results

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
        description="Evaluate stocks through Charlie Munger's investing framework.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("tickers", nargs="*", metavar="TICKER")
    p.add_argument("--file", "-f", metavar="FILE")
    p.add_argument(
        "--output", "-o",
        choices=["terminal", "json", "markdown", "summary"],
        default="terminal",
    )
    p.add_argument(
        "--universe", "-u",
        choices=["munger", "sp500", "djia", "yaml"],
        default=None,
        help=(
            "Screen a preset universe: munger (curated Munger-style holdings), "
            "sp500, djia, yaml"
        ),
    )
    p.add_argument("--delay", "-d", type=float, default=0.5, metavar="SECONDS")
    p.add_argument("--verbose", "-v", action="store_true")
    p.add_argument(
        "--qualifiers-only", "-q",
        action="store_true",
        help="Only show 'Strong Candidate' or 'Munger-Grade Business'",
    )
    return p.parse_args()


_MUNGER_UNIVERSE = [
    # Munger's most celebrated holding — the Costco flywheel
    "COST",
    # Payment networks — asset-light, near-zero capex, monopoly economics
    "V", "MA",
    # Rating agencies / data monopolies — Munger's "toll bridges"
    "MCO", "SPGI",
    # Platform / software — asset-light compounders
    "MSFT", "GOOGL", "AAPL", "ADBE", "INTU",
    # Consumer brand moats with pricing power
    "KO", "PEP", "MO", "PM",
    # Financial services (high ROE, asset-light)
    "AXP", "JPM",
    # Healthcare franchises
    "LLY", "UNH", "ABT",
    # Industrial asset-light businesses
    "UNP", "WM",
    # Specialty / niche dominance
    "NKE", "SBUX", "EL",
    # Insurance
    "CB", "TRV",
    # Other wide-moat businesses
    "JNJ", "PG", "ISRG", "TMO",
]


def _load_universe_tickers(name: str) -> list[str]:
    if name == "munger":
        return _MUNGER_UNIVERSE
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
                table.columns = [
                    " ".join(str(c) for c in col).strip() if isinstance(col, tuple) else str(col)
                    for col in table.columns
                ]
                cols_lower = [c.lower() for c in table.columns]
                match = next(
                    (c for c, cl in zip(table.columns, cols_lower) if "symbol" in cl or "ticker" in cl),
                    None,
                )
                if match:
                    return table[match].dropna().astype(str).str.replace(".", "-", regex=False).tolist()
            print("Error: could not find DJIA components table.", file=sys.stderr)
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
        tickers.extend(line.strip().upper() for line in lines if line.strip() and not line.startswith("#"))
    if args.tickers:
        tickers.extend(t.upper() for t in args.tickers)
    seen: set[str] = set()
    deduped: list[str] = []
    for t in tickers:
        if t not in seen:
            seen.add(t)
            deduped.append(t)
    return deduped


def evaluate_ticker(ticker: str, fetcher: MungerFetcher) -> MungerResult | None:
    try:
        data = fetcher.fetch(ticker)
    except Exception as exc:
        print(f"Error fetching {ticker}: {exc}", file=sys.stderr)
        return None
    if data.get("price") is None:
        print(f"Warning: no price for {ticker} — skipping.", file=sys.stderr)
        return None
    return evaluate_munger(data)


def main() -> None:
    args = parse_args()
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    tickers = load_tickers(args)
    if not tickers:
        print("No tickers provided. Use --help for usage.", file=sys.stderr)
        sys.exit(1)

    fetcher = MungerFetcher()
    results: list[MungerResult] = []

    batch = len(tickers) > 1
    for i, ticker in enumerate(tickers):
        if i > 0 and args.delay > 0:
            time.sleep(args.delay)
        if batch:
            print(f"[{i+1}/{len(tickers)}] {ticker}...", end="\r", flush=True)
        result = evaluate_ticker(ticker, fetcher)
        if result is not None:
            results.append(result)

    if batch:
        print(" " * 40, end="\r")

    if not results:
        print("No results to display.", file=sys.stderr)
        sys.exit(1)

    if args.qualifiers_only:
        results = [r for r in results if r.verdict != "Pass"]
        if not results:
            print("No stocks rated 'Strong Candidate' or better.")
            return

    if args.output == "summary":
        print(format_summary(results))
    else:
        print_results(results, output=args.output)


if __name__ == "__main__":
    main()
