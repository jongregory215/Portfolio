"""
Universe Grader — weekly full analysis of every ticker in universe.yaml.

For each ticker, runs the complete analysis pipeline (same as analyze.py)
and saves a comprehensive JSON used by the dashboard.

Output: runs/weekly/YYYY-MM-DD/universe_grades.json

Also runs portfolio construction for all 5 funds, so the dashboard can
show the optimal portfolio alongside the full ticker ranking.

Usage
-----
  python grade_universe.py                    # grade all tickers
  python grade_universe.py --deep             # use FMP for richer data
  python grade_universe.py --dry-run          # validate config only
  python grade_universe.py --tickers AAPL MSFT NVDA  # subset
"""
from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
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
                  help="Grade every ticker in universe.yaml and save results for the dashboard.")

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
# Helpers
# ──────────────────────────────────────────────────────────────

def _load_universe(tickers_override: list[str] | None = None) -> list[str]:
    if tickers_override:
        return [t.upper() for t in tickers_override]
    if _UNIVERSE_FILE.exists():
        with open(_UNIVERSE_FILE, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return [str(t).upper().strip() for t in data.get("universe", [])]
    return []


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


def _load_prior_grades(runs_dir: Path, today: str) -> dict[str, str]:
    """Find the most recent prior universe_grades.json and extract ticker→grade."""
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


def _extract_ticker_record(result, raw_data: dict) -> dict[str, Any]:
    """Pull the fields we need from an AnalysisResult + raw data dict."""
    overall = result.overall
    ladder  = result.price_ladder

    price      = getattr(result, "price", None)
    fair_value = getattr(ladder, "fair_value", None) if ladder else None
    upside     = ((fair_value / price - 1.0) if price and fair_value and price > 0 else None)

    # Portfolio sub-grades (attribute is 'portfolios', not 'portfolio_grades')
    portfolio_grades     : dict[str, str]   = {}
    portfolio_composites : dict[str, float] = {}
    pg = getattr(result, "portfolios", None)
    if pg:
        for fund in _ALL_FUNDS:
            sleeve = getattr(pg, fund, None)
            if sleeve:
                portfolio_grades[fund]     = sleeve.grade.value
                # Use overall composite as proxy for portfolio ranking score
                # since sleeve.composite is 0 when gates fail
                portfolio_composites[fund] = (
                    sleeve.composite if sleeve.composite > 0
                    else overall.composite if overall else 0.0
                )

    # Sector comes from the raw data dict, not the AnalysisResult
    sector   = raw_data.get("sector")   or ""
    industry = raw_data.get("industry") or ""

    # Convenience refs
    fund_eng  = result.engines.fundamental  if result.engines else None
    tech_eng  = result.engines.technical    if result.engines else None
    quant_eng = result.engines.quantitative if result.engines else None
    risk      = quant_eng.risk_metrics      if quant_eng else None

    return {
        "grade":                overall.grade.value if overall else "Unknown",
        "composite":            round(overall.composite, 1) if overall else 0.0,
        "confidence":           round(overall.confidence, 2) if overall else 0.0,
        "price":                round(price, 2) if price else None,
        "fair_value":           round(fair_value, 2) if fair_value else None,
        "upside_pct":           round(upside * 100, 1) if upside is not None else None,
        "sector":               sector,
        "industry":             industry,
        "fund_score":           round(fund_eng.score, 1)  if fund_eng  else None,
        "tech_score":           round(tech_eng.score, 1)  if tech_eng  else None,
        "quant_score":          round(quant_eng.score, 1) if quant_eng else None,
        "portfolio_grades":     portfolio_grades,
        "portfolio_composites": portfolio_composites,
        "drivers_positive":     list(overall.drivers_positive) if overall else [],
        "drivers_negative":     list(overall.drivers_negative) if overall else [],
        "circuit_breakers":     list(overall.circuit_breakers.keys()) if overall and overall.circuit_breakers else [],
        "price_ladder": {
            "gotta_have_at":   getattr(ladder, "gotta_have_at", None),
            "buy_at":          getattr(ladder, "buy_at", None),
            "hold_low":        (getattr(ladder, "hold_range", None) or [None, None])[0],
            "hold_high":       (getattr(ladder, "hold_range", None) or [None, None])[1],
            "sell_above":      getattr(ladder, "sell_above", None),
            "stay_away_above": getattr(ladder, "stay_away_above", None),
            "fair_value":      fair_value,
        } if ladder else {},
        "altman_z":     getattr(fund_eng,  "altman_z",    None),
        "piotroski_f":  getattr(fund_eng,  "piotroski_f", None),
        "roic_vs_wacc": getattr(fund_eng,  "roic_vs_wacc",None),
        "beta":         getattr(risk, "beta_1yr",        None),
        "max_drawdown": getattr(risk, "max_drawdown_3yr", None),
        "sharpe":       getattr(risk, "sharpe_1yr",      None),
        "regime":       getattr(tech_eng, "regime",      None),
    }


# ──────────────────────────────────────────────────────────────
# Portfolio construction summary
# ──────────────────────────────────────────────────────────────

def _build_portfolio_summaries(config: dict, fetcher) -> dict[str, Any]:
    """Run the 4-stage funnel for each fund using cached data."""
    from stockgrader.data.yfinance_provider import YFinanceProvider
    from stockgrader.portfolios.construction import run_full_funnel
    import weekly_run as wr

    yf = fetcher._yf
    typer.echo("\nBuilding portfolio holdings …")

    tickers = _load_universe()
    universe_basics = wr._build_universe_basics(tickers, yf, delay_secs=0.0)

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
            analytics = {}
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


# ──────────────────────────────────────────────────────────────
# Main command
# ──────────────────────────────────────────────────────────────

@app.command()
def main(
    tickers:  list[str] = typer.Argument(None, help="Optional ticker subset (default: all from universe.yaml)"),
    deep:     bool = typer.Option(False, "--deep", help="Use FMP for richer fundamentals (requires FMP_API_KEY)"),
    dry_run:  bool = typer.Option(False, "--dry-run", help="Validate config only, no data fetched"),
    resume:   bool = typer.Option(True,  "--resume/--no-resume", help="Resume from checkpoint if interrupted"),
    portfolios: bool = typer.Option(True, "--portfolios/--no-portfolios", help="Also run portfolio construction"),
):
    """
    Grade every ticker in universe.yaml and save results for the dashboard.

    Run this once a week (Sunday evening) before reviewing the dashboard.
    Subsequent runs the same day are fast due to 24-hour caching.
    """
    if dry_run:
        from stockgrader.config import get_config
        cfg = get_config()
        typer.echo(f"Config OK. Universe: {_UNIVERSE_FILE}. Funds: {_ALL_FUNDS}")
        raise typer.Exit(0)

    import os
    from stockgrader.config import get_config
    from stockgrader.pipeline import run_analysis
    from stockgrader.data.fetcher import DataFetcher

    if deep and not os.environ.get("FMP_API_KEY"):
        typer.echo("Warning: --deep requested but FMP_API_KEY not set. Using standard mode.")
        deep = False

    config  = get_config()
    fetcher = DataFetcher(deep=deep)

    universe = _load_universe(list(tickers) if tickers else None)
    if not universe:
        typer.echo("No tickers found. Check universe.yaml.", err=True)
        raise typer.Exit(1)

    run_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    out_dir  = _RUNS_DIR / run_date
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load checkpoint and prior grades
    ticker_results: dict[str, Any] = _load_checkpoint(out_dir) if resume else {}
    prior_grades = _load_prior_grades(_RUNS_DIR, run_date)
    skipped:  list[str] = []
    n_done   = len(ticker_results)

    typer.echo(f"\nGrading {len(universe)} tickers"
               f" ({'deep/FMP' if deep else 'standard'}) …")
    if n_done:
        typer.echo(f"  Resuming from checkpoint: {n_done} already done.")

    start_time = time.time()

    for i, ticker in enumerate(universe, 1):
        if ticker in ticker_results:
            continue

        try:
            raw_data = fetcher.fetch(ticker)          # cached after first call
            result   = run_analysis(ticker, config=config, fetcher=fetcher)
            ticker_results[ticker] = _extract_ticker_record(result, raw_data)
        except Exception as exc:
            logger.warning("Failed %s: %s", ticker, exc)
            skipped.append(ticker)
            continue

        # Progress update every 10 tickers
        if i % 10 == 0 or i == len(universe):
            elapsed = time.time() - start_time
            pct     = (len(ticker_results) / len(universe)) * 100
            typer.echo(f"  {len(ticker_results)}/{len(universe)} ({pct:.0f}%)  "
                       f"elapsed {elapsed:.0f}s", err=True)
            _save_checkpoint(out_dir, ticker_results)

        # Polite delay every 10 tickers to avoid yfinance throttling
        if i % 10 == 0:
            time.sleep(0.5)

    # Grade change detection
    grade_changes: dict[str, dict] = {}
    for ticker, rec in ticker_results.items():
        prior = prior_grades.get(ticker)
        curr  = rec["grade"]
        if prior and prior != curr:
            grade_changes[ticker] = {"from": prior, "to": curr}

    # Portfolio construction
    portfolio_summaries: dict[str, Any] = {}
    if portfolios and not tickers:   # skip if running a subset
        try:
            portfolio_summaries = _build_portfolio_summaries(config, fetcher)
        except Exception as exc:
            logger.error("Portfolio construction failed: %s", exc)

    # Save final output
    output = {
        "run_date":   run_date,
        "mode":       "deep" if deep else "standard",
        "n_tickers":  len(ticker_results),
        "n_skipped":  len(skipped),
        "skipped":    skipped,
        "grade_changes": grade_changes,
        "tickers":    ticker_results,
        "portfolios": portfolio_summaries,
    }
    out_path = out_dir / "universe_grades.json"
    out_path.write_text(json.dumps(output, indent=2, default=str), encoding="utf-8")

    # Summary
    grade_dist: dict[str, int] = {}
    for rec in ticker_results.values():
        g = rec["grade"]
        grade_dist[g] = grade_dist.get(g, 0) + 1

    typer.echo(f"\n{'='*50}")
    typer.echo(f"Graded {len(ticker_results)} tickers  |  {len(skipped)} skipped")
    typer.echo(f"Grade distribution:")
    for grade in ["Gotta Have", "Buy", "Hold", "Sell", "Stay Away"]:
        n = grade_dist.get(grade, 0)
        bar = "█" * n
        typer.echo(f"  {grade:12s} {n:3d}  {bar}")

    if grade_changes:
        typer.echo(f"\nGrade changes vs prior week ({len(grade_changes)}):")
        upgrades   = {t: c for t, c in grade_changes.items()
                      if _grade_rank(c["to"]) > _grade_rank(c["from"])}
        downgrades = {t: c for t, c in grade_changes.items()
                      if _grade_rank(c["to"]) < _grade_rank(c["from"])}
        if upgrades:
            typer.echo("  Upgrades:")
            for t, c in sorted(upgrades.items()):
                typer.echo(f"    {t:8s} {c['from']} → {c['to']}")
        if downgrades:
            typer.echo("  Downgrades:")
            for t, c in sorted(downgrades.items()):
                typer.echo(f"    {t:8s} {c['from']} → {c['to']}")
    else:
        typer.echo("\nNo grade changes vs prior week.")

    typer.echo(f"\nSaved → {out_path}")
    typer.echo("Run 'streamlit run dashboard.py' to view the dashboard.")


def _grade_rank(grade: str) -> int:
    return {"Stay Away": 0, "Sell": 1, "Hold": 2, "Buy": 3, "Gotta Have": 4}.get(grade, 2)


if __name__ == "__main__":
    app()
