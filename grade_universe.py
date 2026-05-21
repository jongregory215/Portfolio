"""
Universe Grader — weekly full analysis of every ticker in the universe.

Sources
-------
  yaml    (default) universe.yaml — your hand-curated list
  sp500             S&P 500 constituents from Wikipedia  (~503 tickers)
  nasdaq            All US-listed equities from Nasdaq FTP (~7 000 tickers)
                    auto-filtered to investable names by market cap + volume

Two-phase for large universes
------------------------------
  Phase 1  Fast parallel screen — yfinance .info only (price, market cap,
           volume, sector). Drops non-investable names in seconds.
  Phase 2  Full parallel analysis — run_analysis() on Phase 1 survivors.

Output
------
  runs/weekly/YYYY-MM-DD/universe_grades.json
  runs/weekly/YYYY-MM-DD/.grade_checkpoint.json  (resume marker)

Usage
-----
  python grade_universe.py                          # yaml source, 8 workers
  python grade_universe.py --source sp500           # S&P 500
  python grade_universe.py --source nasdaq          # full US market
  python grade_universe.py --source nasdaq --workers 16
  python grade_universe.py --tickers AAPL MSFT NVDA
  python grade_universe.py --dry-run
"""
from __future__ import annotations

import json
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path
from typing import Any

import typer
import yaml

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env", override=False)
except ImportError:
    pass

app = typer.Typer(name="grade_universe", add_completion=False,
                  help="Grade every ticker in the investable universe.")

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

_RUNS_DIR      = Path("runs") / "weekly"
_UNIVERSE_FILE = Path("universe.yaml")
_ALL_FUNDS     = ["very_conservative", "conservative", "balanced",
                  "aggressive", "very_aggressive"]
_FUND_LABELS   = {
    "very_conservative": "Very Conservative",
    "conservative":      "Conservative",
    "balanced":          "Balanced",
    "aggressive":        "Aggressive",
    "very_aggressive":   "Very Aggressive",
}


# ──────────────────────────────────────────────────────────────
# Universe sources
# ──────────────────────────────────────────────────────────────

def _load_yaml_universe(overrides: list[str] | None = None) -> list[str]:
    if overrides:
        return [t.upper() for t in overrides]
    if _UNIVERSE_FILE.exists():
        with open(_UNIVERSE_FILE, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return [str(t).upper().strip() for t in data.get("universe", [])]
    return []


def _fetch_sp500() -> list[str]:
    """Fetch S&P 500 constituents from Wikipedia (free, ~503 tickers)."""
    import requests, pandas as pd
    typer.echo("Fetching S&P 500 from Wikipedia …")
    try:
        url  = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        html = requests.get(url, headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/124.0"
        }, timeout=20).text
        df = pd.read_html(StringIO(html), attrs={"id": "constituents"})[0]
        tickers = df["Symbol"].str.replace(".", "-", regex=False).tolist()
        typer.echo(f"  S&P 500: {len(tickers)} tickers")
        return tickers
    except Exception as exc:
        typer.echo(f"  Wikipedia fetch failed ({exc}) — falling back to yaml.")
        return _load_yaml_universe()


def _fetch_nasdaq_universe(min_market_cap: float, min_avg_volume: float) -> list[str]:
    """
    Fetch all US-listed equities from the SEC EDGAR company tickers list.

    https://www.sec.gov/files/company_tickers.json — free, no auth, ~10 000 entries.
    Returns raw ticker list BEFORE Stage 1 filtering.
    """
    import requests

    typer.echo("Fetching full US equity universe from SEC EDGAR …")
    try:
        r = requests.get(
            "https://www.sec.gov/files/company_tickers.json",
            headers={"User-Agent": "stockgrader research@example.com"},
            timeout=30,
        )
        r.raise_for_status()
        data    = r.json()
        raw_all = [entry["ticker"] for entry in data.values()
                   if isinstance(entry, dict) and entry.get("ticker")]
        typer.echo(f"  SEC EDGAR: {len(raw_all)} entries")
    except Exception as exc:
        typer.echo(f"  SEC EDGAR fetch failed ({exc}) — falling back to S&P 1500.")
        return _fetch_sp1500()

    # Clean: keep simple US-style tickers only
    clean = []
    seen  = set()
    for t in raw_all:
        t = str(t).upper().strip()
        if not t or t in seen:
            continue
        if any(c in t for c in [".", "$", "^", "+", " ", "/", "\\"]):
            continue
        if len(t) > 5:
            continue
        seen.add(t)
        clean.append(t)

    typer.echo(f"  Total after cleaning: {len(clean)}")
    return clean


def _fetch_sp1500() -> list[str]:
    """
    S&P 1500 = S&P 500 + S&P 400 (mid-cap) + S&P 600 (small-cap) from Wikipedia.
    Free, no API key, ~1 500 tickers covering large / mid / small cap.
    """
    import requests, pandas as pd
    typer.echo("Fetching S&P 1500 from Wikipedia …")

    _HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0 Safari/537.36"
        )
    }
    pages = [
        ("S&P 500", "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
         "constituents", "Symbol"),
        ("S&P 400", "https://en.wikipedia.org/wiki/List_of_S%26P_400_companies",
         None, "Ticker"),
        ("S&P 600", "https://en.wikipedia.org/wiki/List_of_S%26P_600_companies",
         None, "Ticker"),
    ]
    tickers: list[str] = []
    for name, url, table_id, col in pages:
        try:
            html = requests.get(url, headers=_HEADERS, timeout=20).text
            kwargs = {"attrs": {"id": table_id}} if table_id else {}
            tables = pd.read_html(StringIO(html), **kwargs)
            for df in tables:
                # column name varies slightly across Wikipedia pages
                match = next((c for c in df.columns
                              if str(c).strip().lower() in ("symbol", "ticker")), None)
                if match:
                    t = df[match].astype(str).str.replace(".", "-", regex=False).dropna().tolist()
                    tickers.extend(t)
                    typer.echo(f"  {name}: {len(t)} tickers")
                    break
            else:
                typer.echo(f"  {name}: no matching column found")
        except Exception as exc:
            typer.echo(f"  {name} failed: {exc}")

    return list(dict.fromkeys(t.upper() for t in tickers if t and t != "NAN"))


# ──────────────────────────────────────────────────────────────
# Stage 1 fast screen (parallel)
# ──────────────────────────────────────────────────────────────

def _screen_one(ticker: str, yf_provider, min_market_cap: float,
                min_avg_volume: float, delay: float = 0.0) -> tuple[str, dict | None]:
    """Return (ticker, basic_info) if it passes Stage 1, else (ticker, None)."""
    if delay > 0:
        time.sleep(delay)
    try:
        info = yf_provider.get_fundamentals(ticker)
        mc  = info.get("marketCap")  or 0
        vol = info.get("averageVolume") or 0
        if mc >= min_market_cap and vol >= min_avg_volume:
            return ticker, {
                "price":      info.get("currentPrice") or info.get("regularMarketPrice"),
                "sector":     info.get("sector", ""),
                "market_cap": mc,
            }
    except Exception:
        pass
    return ticker, None


def _quick_screen_parallel(
    tickers:        list[str],
    yf_provider,
    workers:        int,
    min_market_cap: float,
    min_avg_volume: float,
    delay:          float = 0.0,
) -> list[str]:
    """Parallel Stage 1 screen — returns tickers that pass liquidity floors."""
    survivors: list[str] = []
    done = 0
    lock = threading.Lock()
    n    = len(tickers)

    def _cb(future):
        nonlocal done
        ticker, info = future.result()
        with lock:
            done += 1
            if info is not None:
                survivors.append(ticker)
            if done % 200 == 0 or done == n:
                typer.echo(f"  Stage 1: {done}/{n} screened, "
                           f"{len(survivors)} survivors …", err=True)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {
            ex.submit(_screen_one, t, yf_provider, min_market_cap, min_avg_volume, delay): t
            for t in tickers
        }
        for f in as_completed(futures):
            _cb(f)

    return survivors


# ──────────────────────────────────────────────────────────────
# Full analysis (parallel)
# ──────────────────────────────────────────────────────────────

def _grade_one(
    ticker:  str,
    config:  dict,
    fetcher,
    delay:   float = 0.0,
) -> tuple[str, dict | None, str | None]:
    """Grade a single ticker. Returns (ticker, record, error_msg)."""
    if delay > 0:
        time.sleep(delay)
    try:
        raw_data = fetcher.fetch(ticker)
        from stockgrader.pipeline import run_analysis
        result   = run_analysis(ticker, config=config, fetcher=fetcher)
        return ticker, _extract_ticker_record(result, raw_data), None
    except Exception as exc:
        return ticker, None, str(exc)


def _grade_parallel(
    tickers:      list[str],
    config:       dict,
    fetcher,
    workers:      int,
    existing:     dict[str, Any],
    out_dir:      Path,
    delay:        float = 0.0,
) -> dict[str, Any]:
    """Run full analysis in parallel, resuming from existing checkpoint."""
    results: dict[str, Any] = dict(existing)
    lock    = threading.Lock()
    skipped: list[str] = []
    done    = len(existing)
    total   = len(tickers)
    start   = time.time()

    pending = [t for t in tickers if t not in results]
    if not pending:
        return results

    def _cb(future):
        nonlocal done
        ticker, record, err = future.result()
        with lock:
            done += 1
            if record:
                results[ticker] = record
            else:
                skipped.append(ticker)
                logger.warning("Failed %s: %s", ticker, err)

            # Progress + checkpoint every 20 tickers
            if done % 20 == 0 or done == total:
                elapsed = time.time() - start
                rate    = (done - len(existing)) / max(elapsed, 1)
                eta     = (total - done) / rate if rate > 0 else 0
                typer.echo(
                    f"  {done}/{total} ({done/total:.0%})  "
                    f"elapsed {elapsed:.0f}s  "
                    f"ETA {eta:.0f}s",
                    err=True,
                )
                _save_checkpoint(out_dir, results)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(_grade_one, t, config, fetcher, delay): t for t in pending}
        for f in as_completed(futures):
            _cb(f)

    return results


# ──────────────────────────────────────────────────────────────
# Checkpointing
# ──────────────────────────────────────────────────────────────

def _checkpoint_path(out_dir: Path) -> Path:
    return out_dir / ".grade_checkpoint.json"


def _load_checkpoint(out_dir: Path) -> dict[str, Any]:
    cp = _checkpoint_path(out_dir)
    if cp.exists():
        try:
            return json.loads(cp.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def _save_checkpoint(out_dir: Path, results: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    _checkpoint_path(out_dir).write_text(
        json.dumps(results, indent=2, default=str), encoding="utf-8"
    )


# ──────────────────────────────────────────────────────────────
# Prior grades (for change detection)
# ──────────────────────────────────────────────────────────────

def _load_prior_grades(runs_dir: Path, today: str) -> dict[str, str]:
    dated_dirs = sorted(
        (d for d in runs_dir.glob("????-??-??") if d.is_dir() and d.name != today),
        reverse=True,
    )
    for ddir in dated_dirs:
        p = ddir / "universe_grades.json"
        if p.exists():
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                return {t: v["grade"] for t, v in data.get("tickers", {}).items()}
            except Exception:
                pass
    return {}


# ──────────────────────────────────────────────────────────────
# Record extraction
# ──────────────────────────────────────────────────────────────

def _extract_ticker_record(result, raw_data: dict) -> dict[str, Any]:
    """Flatten AnalysisResult + raw data dict into a JSON-friendly dict."""
    overall = result.overall
    ladder  = result.price_ladder

    price      = getattr(result, "price", None)
    fair_value = getattr(ladder, "fair_value", None) if ladder else None
    upside     = (fair_value / price - 1.0) if price and fair_value and price > 0 else None

    portfolio_grades:     dict[str, str]   = {}
    portfolio_composites: dict[str, float] = {}
    pg = getattr(result, "portfolios", None)
    if pg:
        for fund in _ALL_FUNDS:
            sleeve = getattr(pg, fund, None)
            if sleeve:
                portfolio_grades[fund]     = sleeve.grade.value
                portfolio_composites[fund] = (
                    sleeve.composite if sleeve.composite > 0
                    else (overall.composite if overall else 0.0)
                )

    fund_eng  = result.engines.fundamental  if result.engines else None
    tech_eng  = result.engines.technical    if result.engines else None
    quant_eng = result.engines.quantitative if result.engines else None
    risk      = quant_eng.risk_metrics      if quant_eng else None

    cbs = getattr(overall, "circuit_breakers", None) if overall else None
    if isinstance(cbs, dict):
        cb_list = [k for k, v in cbs.items() if v]
    elif isinstance(cbs, list):
        cb_list = list(cbs)
    else:
        cb_list = []

    return {
        "grade":                overall.grade.value if overall else "Unknown",
        "composite":            round(overall.composite, 1)  if overall else 0.0,
        "confidence":           round(overall.confidence, 2) if overall else 0.0,
        "price":                round(price, 2)      if price      else None,
        "fair_value":           round(fair_value, 2) if fair_value else None,
        "upside_pct":           round(upside * 100, 1) if upside is not None else None,
        "sector":               raw_data.get("sector")   or "",
        "industry":             raw_data.get("industry") or "",
        "fund_score":           round(fund_eng.score, 1)  if fund_eng  else None,
        "tech_score":           round(tech_eng.score, 1)  if tech_eng  else None,
        "quant_score":          round(quant_eng.score, 1) if quant_eng else None,
        "portfolio_grades":     portfolio_grades,
        "portfolio_composites": portfolio_composites,
        "drivers_positive":     list(overall.drivers_positive) if overall else [],
        "drivers_negative":     list(overall.drivers_negative) if overall else [],
        "circuit_breakers":     cb_list,
        "price_ladder": {
            "gotta_have_at":   getattr(ladder, "gotta_have_at", None),
            "buy_at":          getattr(ladder, "buy_at", None),
            "hold_low":        (getattr(ladder, "hold_range", None) or [None, None])[0],
            "hold_high":       (getattr(ladder, "hold_range", None) or [None, None])[1],
            "sell_above":      getattr(ladder, "sell_above", None),
            "stay_away_above": getattr(ladder, "stay_away_above", None),
            "fair_value":      fair_value,
        } if ladder else {},
        "altman_z":     getattr(fund_eng,  "altman_z",     None),
        "piotroski_f":  getattr(fund_eng,  "piotroski_f",  None),
        "roic_vs_wacc": getattr(fund_eng,  "roic_vs_wacc", None),
        "beta":         getattr(risk, "beta_1yr",         None),
        "max_drawdown": getattr(risk, "max_drawdown_3yr", None),
        "sharpe":       getattr(risk, "sharpe_1yr",       None),
        "regime":       getattr(tech_eng,  "regime",       None),
    }


# ──────────────────────────────────────────────────────────────
# Portfolio construction
# ──────────────────────────────────────────────────────────────

def _build_portfolio_summaries(config: dict, fetcher) -> dict[str, Any]:
    from stockgrader.portfolios.construction import run_full_funnel
    import weekly_run as wr

    typer.echo("\nBuilding portfolio holdings …")
    tickers         = _load_yaml_universe()
    universe_basics = wr._build_universe_basics(tickers, fetcher._yf, delay_secs=0.0)
    summaries: dict[str, Any] = {}

    for fund in _ALL_FUNDS:
        try:
            result = run_full_funnel(
                portfolio_name  = fund,
                universe_basics = universe_basics,
                fetch_full_data = fetcher.fetch,
                config          = config,
            )
            holdings = []
            if result and result.holdings:
                for h in sorted(result.holdings, key=lambda x: -x.weight):
                    holdings.append({
                        "ticker":    h.ticker,
                        "weight":    round(h.weight, 4),
                        "grade":     h.grade,
                        "composite": round(h.composite, 1),
                        "sector":    h.sector,
                        "sleeve":    h.sleeve,
                    })
            analytics: dict = {}
            if result and result.analytics:
                a = result.analytics
                analytics = {
                    "projected_return":     round(a.projected_return, 4),
                    "projected_volatility": round(a.projected_volatility, 4),
                    "weighted_beta":        round(a.weighted_beta, 2),
                    "projected_yield":      round(a.projected_yield, 4),
                    "n_holdings":           len(result.holdings) if result.holdings else 0,
                }
            summaries[fund] = {"holdings": holdings, "analytics": analytics}
            typer.echo(f"  {_FUND_LABELS[fund]}: {len(holdings)} holdings")
        except Exception as exc:
            logger.warning("Portfolio construction failed for %s: %s", fund, exc)
            summaries[fund] = {"holdings": [], "analytics": {}}

    return summaries


def _grade_rank(grade: str) -> int:
    return {"Stay Away": 0, "Sell": 1, "Hold": 2, "Buy": 3, "Gotta Have": 4}.get(grade, 2)


# ──────────────────────────────────────────────────────────────
# Main command
# ──────────────────────────────────────────────────────────────

@app.command()
def main(
    tickers: list[str] = typer.Argument(None,
        help="Optional subset of tickers (overrides --source)"),
    source:  str  = typer.Option("yaml", "--source",
        help="Universe source: yaml | sp500 | nasdaq"),
    workers: int  = typer.Option(8, "--workers",
        help="Parallel workers for analysis (8 is a safe default)"),
    min_market_cap: float = typer.Option(500_000_000, "--min-market-cap",
        help="Stage 1 market cap floor in USD (nasdaq source only)"),
    min_volume: float = typer.Option(500_000, "--min-volume",
        help="Stage 1 average daily volume floor (nasdaq source only)"),
    deep:       bool  = typer.Option(False, "--deep",
        help="Use FMP for richer data (requires FMP_API_KEY)"),
    dry_run:    bool  = typer.Option(False, "--dry-run",
        help="Validate config and universe only, no analysis"),
    resume:     bool  = typer.Option(True, "--resume/--no-resume",
        help="Resume from today's checkpoint if present"),
    portfolios: bool  = typer.Option(True, "--portfolios/--no-portfolios",
        help="Also run portfolio construction for all 5 funds"),
    delay:      float = typer.Option(0.0, "--delay",
        help="Seconds to sleep between requests per worker. "
             "Use 1.0–2.0 for overnight full-market runs to avoid throttling."),
):
    """
    Grade every ticker in the investable universe and save results for the dashboard.

    Examples
    --------
      python grade_universe.py                          # 136-ticker yaml, 8 workers
      python grade_universe.py --source sp500           # S&P 500 (~503)
      python grade_universe.py --source nasdaq          # full US market, filtered
      python grade_universe.py --source nasdaq --workers 16 --min-market-cap 1e9
    """
    import os
    from stockgrader.config import get_config
    from stockgrader.data.fetcher import DataFetcher

    if dry_run:
        cfg = get_config()
        typer.echo(f"Config OK | Source: {source} | Workers: {workers}")
        raise typer.Exit(0)

    if deep and not os.environ.get("FMP_API_KEY"):
        typer.echo("Warning: --deep set but FMP_API_KEY absent. Using standard mode.")
        deep = False

    config  = get_config()
    fetcher = DataFetcher(deep=deep)

    # ── Build universe ────────────────────────────────────────
    if tickers:
        universe = [t.upper() for t in tickers]
        typer.echo(f"Manual subset: {len(universe)} tickers")
    elif source == "nasdaq":
        raw      = _fetch_nasdaq_universe(min_market_cap, min_volume)
        typer.echo(f"\nStage 1 screening {len(raw)} tickers "
                   f"(cap>${min_market_cap/1e6:.0f}M, vol>{min_volume/1e3:.0f}K) "
                   f"with {workers} workers …")
        universe = _quick_screen_parallel(
            raw, fetcher._yf, workers, min_market_cap, min_volume, delay
        )
        typer.echo(f"Stage 1 survivors: {len(universe)} tickers")
    elif source == "sp500":
        universe = _fetch_sp500()
    elif source == "sp1500":
        universe = _fetch_sp1500()
    else:
        universe = _load_yaml_universe()

    if not universe:
        typer.echo("No tickers found. Check --source or universe.yaml.", err=True)
        raise typer.Exit(1)

    run_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    out_dir  = _RUNS_DIR / run_date
    out_dir.mkdir(parents=True, exist_ok=True)

    existing = _load_checkpoint(out_dir) if resume else {}
    prior_grades = _load_prior_grades(_RUNS_DIR, run_date)

    typer.echo(f"\nGrading {len(universe)} tickers "
               f"({'deep/FMP' if deep else 'standard'}) "
               f"with {workers} workers …")
    if existing:
        typer.echo(f"  Resuming: {len(existing)} already done, "
                   f"{len(universe) - len(existing)} remaining.")

    # ── Full parallel analysis ────────────────────────────────
    ticker_results = _grade_parallel(
        universe, config, fetcher, workers, existing, out_dir, delay
    )

    # ── Grade changes ─────────────────────────────────────────
    grade_changes: dict[str, dict] = {}
    for t, rec in ticker_results.items():
        prior = prior_grades.get(t)
        curr  = rec["grade"]
        if prior and prior != curr:
            grade_changes[t] = {"from": prior, "to": curr}

    # ── Portfolio construction ────────────────────────────────
    portfolio_summaries: dict[str, Any] = {}
    if portfolios and not tickers:
        try:
            portfolio_summaries = _build_portfolio_summaries(config, fetcher)
        except Exception as exc:
            logger.error("Portfolio construction failed: %s", exc)

    # ── Save JSON ─────────────────────────────────────────────
    output = {
        "run_date":      run_date,
        "source":        source,
        "mode":          "deep" if deep else "standard",
        "n_tickers":     len(ticker_results),
        "n_skipped":     len(universe) - len(ticker_results),
        "grade_changes": grade_changes,
        "tickers":       ticker_results,
        "portfolios":    portfolio_summaries,
    }
    out_path = out_dir / "universe_grades.json"
    out_path.write_text(json.dumps(output, indent=2, default=str), encoding="utf-8")

    # ── Summary ───────────────────────────────────────────────
    grade_dist: dict[str, int] = {}
    for rec in ticker_results.values():
        g = rec["grade"]
        grade_dist[g] = grade_dist.get(g, 0) + 1

    typer.echo(f"\n{'='*52}")
    typer.echo(f"Graded {len(ticker_results)}  |  "
               f"Skipped {len(universe)-len(ticker_results)}  |  "
               f"Grade changes {len(grade_changes)}")
    typer.echo("Grade distribution:")
    for grade in ["Gotta Have", "Buy", "Hold", "Sell", "Stay Away"]:
        n   = grade_dist.get(grade, 0)
        pct = n / max(len(ticker_results), 1) * 100
        bar = "#" * min(n // 2, 40)
        typer.echo(f"  {grade:12s} {n:4d} ({pct:4.1f}%)  {bar}")

    if grade_changes:
        upgrades   = [(t, c) for t, c in grade_changes.items()
                      if _grade_rank(c["to"]) > _grade_rank(c["from"])]
        downgrades = [(t, c) for t, c in grade_changes.items()
                      if _grade_rank(c["to"]) < _grade_rank(c["from"])]
        if upgrades:
            typer.echo(f"\nUpgrades ({len(upgrades)}):")
            for t, c in sorted(upgrades)[:20]:
                typer.echo(f"  {t:8s}  {c['from']} -> {c['to']}")
        if downgrades:
            typer.echo(f"\nDowngrades ({len(downgrades)}):")
            for t, c in sorted(downgrades)[:20]:
                typer.echo(f"  {t:8s}  {c['from']} -> {c['to']}")

    n_skipped = len(universe) - len(ticker_results)
    if n_skipped:
        typer.echo(f"\nSkipped {n_skipped} tickers — data unavailable or error.")

    typer.echo(f"\nSaved: {out_path}")
    typer.echo("Dashboard: streamlit run dashboard.py")


if __name__ == "__main__":
    app()
