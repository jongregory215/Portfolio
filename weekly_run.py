"""
Weekly portfolio rebuild runner.

Runs the full Section 11 portfolio-construction funnel for all five funds:
  universe screen → sub-grade / rank → constrained optimization → assemble

Universe source (resolution order):
  1. universe.yaml  — user-editable list in project root (~150 tickers default)
  2. Wikipedia S&P 500 — fetched live if universe.yaml is deleted
  3. Hardcoded fallback of 50 large-caps

Output per run (written to runs/weekly/YYYY-MM-DD/):
  {fund}_holdings.md       — holdings table: ticker, weight, grade, composite
  {fund}_analytics.json    — expected return, vol, Sharpe, weights dict
  {fund}_mandate.md        — mandate check: PASS/FAIL + violations
  {fund}_funnel.md         — funnel transparency: names at each of 4 stages
  drift_report.md          — weight drift vs. prior week

Checkpointing
-------------
Completed funds are saved to runs/weekly/YYYY-MM-DD/.checkpoint.json.
An interrupted run resumes from the last checkpoint. Use --no-resume to
force a full rebuild.

API requirements
----------------
No paid API key required. All data comes from yfinance (free).
FRED_API_KEY is optional (improves risk-free rate accuracy; falls back to 5%).

Exit codes
----------
  0 — success
  1 — runtime error (some funds may have failed; check logs)
  2 — --dry-run

Usage
-----
  python weekly_run.py
  python weekly_run.py --funds balanced,aggressive
  python weekly_run.py --dry-run
  python weekly_run.py --no-save
  python weekly_run.py --no-resume   # ignore checkpoint, full rebuild
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import typer

app = typer.Typer(
    name="weekly_run",
    help="Weekly full-universe portfolio rebuild (yfinance, no API key required).",
    add_completion=False,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

_RUNS_DIR  = Path("runs") / "weekly"
_ALL_FUNDS = ["very_conservative", "conservative", "balanced",
              "aggressive", "very_aggressive"]


# ──────────────────────────────────────────────────────────────
# Checkpointing
# ──────────────────────────────────────────────────────────────

def _checkpoint_path(out_dir: Path) -> Path:
    return out_dir / ".checkpoint.json"


def _load_checkpoint(out_dir: Path) -> dict[str, bool]:
    cp = _checkpoint_path(out_dir)
    if cp.exists():
        try:
            return json.loads(cp.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def _save_checkpoint(out_dir: Path, completed: dict[str, bool]) -> None:
    cp = _checkpoint_path(out_dir)
    cp.parent.mkdir(parents=True, exist_ok=True)
    cp.write_text(json.dumps(completed, indent=2), encoding="utf-8")


# ──────────────────────────────────────────────────────────────
# Universe loading
# ──────────────────────────────────────────────────────────────

def _build_universe_basics(
    tickers: list[str],
    yf_provider,
) -> list[dict[str, Any]]:
    """
    Fetch minimal per-ticker info from yfinance .info for Stage 1 screening.

    Returns a list of dicts with the fields run_full_funnel expects:
    ticker, price, avg_volume, market_cap, beta, dividend_yield,
    debt_equity, sector.
    """
    basics: list[dict[str, Any]] = []
    for ticker in tickers:
        try:
            info = yf_provider.get_fundamentals(ticker)
            price = (info.get("currentPrice")
                     or info.get("regularMarketPrice")
                     or info.get("previousClose"))
            basics.append({
                "ticker":         ticker,
                "price":          float(price) if price else None,
                "avg_volume":     info.get("averageVolume"),
                "market_cap":     info.get("marketCap"),
                "beta":           info.get("beta"),
                "dividend_yield": info.get("dividendYield"),
                "debt_equity":    info.get("debtToEquity"),
                "sector":         info.get("sector", ""),
            })
        except Exception as exc:
            logger.warning("Skipping %s — yfinance .info failed: %s", ticker, exc)
    return basics


# ──────────────────────────────────────────────────────────────
# Prior holdings for drift report
# ──────────────────────────────────────────────────────────────

def _load_prior_holdings(fund: str, runs_dir: Path) -> dict[str, float]:
    dated_dirs = sorted(
        (d for d in runs_dir.glob("????-??-??") if d.is_dir()),
        reverse=True,
    )
    for ddir in dated_dirs:
        p = ddir / f"{fund}_analytics.json"
        if p.exists():
            try:
                return json.loads(p.read_text(encoding="utf-8")).get("weights", {})
            except Exception:
                pass
    return {}


# ──────────────────────────────────────────────────────────────
# Report builders
# ──────────────────────────────────────────────────────────────

def _drift_report(fund: str, prior: dict[str, float], current: dict[str, float]) -> str:
    all_tickers = set(prior) | set(current)
    lines = [f"## {fund} — Weight Drift vs. Prior Week", ""]
    entered = sorted(t for t in current if t not in prior)
    exited  = sorted(t for t in prior   if t not in current)
    if entered:
        lines.append(f"**Entered portfolio:** {', '.join(entered)}")
    if exited:
        lines.append(f"**Exited portfolio:** {', '.join(exited)}")
    if entered or exited:
        lines.append("")
    lines += ["| Ticker | Prior | Current | Δ |", "|--------|-------|---------|---|"]
    changed = False
    for t in sorted(all_tickers):
        pw, cw = prior.get(t, 0.0), current.get(t, 0.0)
        delta  = cw - pw
        if abs(delta) > 0.001:
            lines.append(f"| {t} | {pw:.1%} | {cw:.1%} | {'↑' if delta>0 else '↓'} {abs(delta):.1%} |")
            changed = True
    if not changed:
        lines.append("*No meaningful weight changes.*")
    lines.append("")
    return "\n".join(lines)


def _holdings_md(fund: str, result, run_date: str) -> str:
    label = fund.replace("_", " ").title()
    lines = [f"# {label} — Holdings Report", f"*{run_date}*", ""]
    if not result or not result.holdings:
        lines.append("*No holdings (insufficient candidates passed the funnel).*")
        return "\n".join(lines)
    lines += ["| # | Ticker | Weight | Grade | Composite |",
              "|---|--------|--------|-------|-----------|"]
    for i, h in enumerate(sorted(result.holdings, key=lambda x: -x.weight), 1):
        lines.append(f"| {i} | {h.ticker} | {h.weight:.1%} | {h.grade} | {h.composite:.1f} |")
    lines.append("")
    if result.analytics:
        a = result.analytics
        lines += [
            "## Analytics",
            f"- Projected annual return: {a.projected_return:.1%}",
            f"- Projected volatility:    {a.projected_volatility:.1%}",
            f"- Weighted beta:           {a.weighted_beta:.2f}",
            f"- Holdings count:          {len(result.holdings)}",
            "",
        ]
    return "\n".join(lines)


def _mandate_md(fund: str, result) -> str:
    label = fund.replace("_", " ").title()
    lines = [f"# {label} — Mandate Check", ""]
    if result is None or result.mandate is None:
        lines.append("*Mandate check unavailable.*")
        return "\n".join(lines)
    mc = result.mandate
    status = "✅ PASS" if mc.passed else "❌ FAIL"
    lines += [f"**Overall: {status}**", ""]
    if mc.violations:
        lines.append("**Violations:**")
        for v in mc.violations:
            lines.append(f"- {v}")
    else:
        lines.append("No violations.")
    return "\n".join(lines)


def _funnel_md(fund: str, result, run_date: str, n_universe: int) -> str:
    label = fund.replace("_", " ").title()
    lines = [f"# {label} — Funnel Transparency", f"*{run_date}*", ""]
    funnel = getattr(result, "funnel", None) if result else None
    lines += ["| Stage | In | Out | Description |",
              "|-------|----|-----|-------------|"]
    if funnel:
        n_hold = len(result.holdings) if result and result.holdings else 0
        lines += [
            f"| 1 — Universe screen | {n_universe} | {funnel.universe_entered} | Liquidity / mandate pre-filter |",
            f"| 2 — Grade & rank    | {funnel.universe_entered} | {funnel.after_grading} | Buy/Gotta Have, top 40 by composite |",
            f"| 3 — Optimize        | {funnel.after_grading} | {funnel.after_optimize} | Mean-variance (Ledoit-Wolf) |",
            f"| 4 — Assemble        | {funnel.after_optimize} | {n_hold} | Equity + bond sleeve, mandate check |",
        ]
    else:
        lines.append("*Funnel statistics not available.*")
    lines.append("")
    return "\n".join(lines)


def _analytics_json(fund: str, result) -> dict:
    weights = {h.ticker: h.weight for h in result.holdings} if result and result.holdings else {}
    data: dict = {"fund": fund, "weights": weights}
    if result and result.analytics:
        a = result.analytics
        data.update({
            "projected_return":     a.projected_return,
            "projected_volatility": a.projected_volatility,
            "weighted_beta":        a.weighted_beta,
            "n_holdings":           len(result.holdings) if result.holdings else 0,
        })
    return data


# ──────────────────────────────────────────────────────────────
# Main command
# ──────────────────────────────────────────────────────────────

@app.command()
def main(
    dry_run: bool = typer.Option(False, "--dry-run",
                                 help="Validate config only; no data or writes. Exit 2."),
    funds:   str  = typer.Option("all", "--funds",
                                 help="Comma-separated fund names, or 'all'"),
    save:    bool = typer.Option(True, "--save/--no-save",
                                 help="Write output to runs/weekly/ (default on)"),
    resume:  bool = typer.Option(True, "--resume/--no-resume",
                                 help="Skip funds already checkpointed for today"),
):
    """
    Full-universe portfolio rebuild for all five risk-tiered funds.

    No API key required — all data sourced from yfinance (free).
    Optional: set FRED_API_KEY for live risk-free rates.
    """
    if dry_run:
        typer.echo("--dry-run: validating configuration only, no live data fetched.")
        from stockgrader.config import get_config
        try:
            cfg = get_config()
            typer.echo(f"✅ Config loaded. Portfolios: {list(cfg.get('portfolios', {}).keys())}")
        except Exception as exc:
            typer.echo(f"❌ Config error: {exc}", err=True)
            raise typer.Exit(1)
        raise typer.Exit(2)

    fund_list = _ALL_FUNDS if funds == "all" else [f.strip() for f in funds.split(",")]

    from stockgrader.config import get_config
    from stockgrader.data.fetcher import DataFetcher
    from stockgrader.data.yfinance_provider import YFinanceProvider
    from stockgrader.portfolios.construction import run_full_funnel

    config   = get_config()
    fetcher  = DataFetcher()
    yf       = fetcher._yf
    run_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    out_dir  = _RUNS_DIR / run_date

    if save:
        out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load universe ─────────────────────────────────────────
    typer.echo("Loading universe …")
    universe_tickers = yf.get_universe()
    typer.echo(f"  {len(universe_tickers)} tickers in universe.")

    typer.echo("Fetching basic info for universe screening …")
    universe_basics = _build_universe_basics(universe_tickers, yf)
    typer.echo(f"  {len(universe_basics)} tickers with valid data for Stage 1.")

    # ── Checkpointing ─────────────────────────────────────────
    checkpoint = _load_checkpoint(out_dir) if (resume and save) else {}
    if checkpoint:
        done = [f for f in fund_list if checkpoint.get(f)]
        if done:
            typer.echo(f"  Resuming — skipping already-done: {done}")

    drift_sections: list[str] = [f"# Drift Report — {run_date}", ""]
    had_error = False

    # ── Per-fund loop ─────────────────────────────────────────
    for fund in fund_list:
        if checkpoint.get(fund):
            typer.echo(f"\n  ⏭  {fund} (checkpointed — skipped)")
            prior = _load_prior_holdings(fund, _RUNS_DIR)
            curr_w: dict[str, float] = {}
            anl = out_dir / f"{fund}_analytics.json"
            if anl.exists():
                try:
                    curr_w = json.loads(anl.read_text(encoding="utf-8")).get("weights", {})
                except Exception:
                    pass
            drift_sections.append(_drift_report(fund, prior, curr_w))
            continue

        typer.echo(f"\n{'─'*60}")
        typer.echo(f"  Building portfolio: {fund} …")

        try:
            result = run_full_funnel(
                portfolio_name  = fund,
                universe_basics = universe_basics,
                fetch_full_data = fetcher.fetch,
                config          = config,
            )
        except Exception as exc:
            logger.error("Failed to build %s: %s", fund, exc)
            typer.echo(f"  ❌ {fund}: {exc}", err=True)
            had_error = True
            continue

        n_hold = len(result.holdings) if result and result.holdings else 0
        if n_hold:
            typer.echo(f"  ✅ {fund}: {n_hold} holdings")
        else:
            typer.echo(f"  ⚠️  {fund}: funnel returned empty")

        curr_weights = {h.ticker: h.weight for h in result.holdings} if result and result.holdings else {}
        prior_weights = _load_prior_holdings(fund, _RUNS_DIR)
        drift_sections.append(_drift_report(fund, prior_weights, curr_weights))

        if save:
            (out_dir / f"{fund}_holdings.md").write_text(
                _holdings_md(fund, result, run_date), encoding="utf-8")
            (out_dir / f"{fund}_analytics.json").write_text(
                json.dumps(_analytics_json(fund, result), indent=2), encoding="utf-8")
            (out_dir / f"{fund}_mandate.md").write_text(
                _mandate_md(fund, result), encoding="utf-8")
            (out_dir / f"{fund}_funnel.md").write_text(
                _funnel_md(fund, result, run_date, len(universe_basics)), encoding="utf-8")

            checkpoint[fund] = True
            _save_checkpoint(out_dir, checkpoint)

    # ── Drift report ──────────────────────────────────────────
    drift_md = "\n".join(drift_sections)
    if save:
        (out_dir / "drift_report.md").write_text(drift_md, encoding="utf-8")
        typer.echo(f"\n✅ Reports saved → {out_dir}/")
        if had_error:
            typer.echo("   ⚠️  Some funds failed — see logs above.")
    else:
        typer.echo(drift_md)

    if had_error:
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
