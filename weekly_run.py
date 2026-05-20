"""
Weekly portfolio rebuild runner.

Runs the full Section 11 portfolio-construction funnel for all five funds:
  universe screen → sub-grade and rank → constrained optimization → assemble

Output per run (written to runs/weekly/YYYY-MM-DD/):
  - {fund}_holdings.md        — holdings table with weights and grades
  - {fund}_analytics.json     — Sharpe, vol, sector concentrations, etc.
  - {fund}_mandate.md         — mandate check results
  - drift_report.md           — weight drift vs. last week

Exit codes:
  0 — success
  1 — runtime error
  2 — --dry-run (validation only, no writes)

Usage
-----
  python weekly_run.py [--dry-run] [--funds conservative,balanced,aggressive]
  python weekly_run.py --no-save    # run but don't write files
"""
from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import typer
import yaml

app = typer.Typer(
    name="weekly_run",
    help="Weekly full-universe portfolio rebuild.",
    add_completion=False,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

_RUNS_DIR   = Path("runs") / "weekly"
_ALL_FUNDS  = ["very_conservative", "conservative", "balanced",
               "aggressive", "very_aggressive"]


# ──────────────────────────────────────────────────────────────
# Drift report
# ──────────────────────────────────────────────────────────────

def _load_prior_holdings(fund: str, runs_dir: Path) -> dict[str, float]:
    """Load last week's holdings weights from the most recent dated subfolder."""
    dated_dirs = sorted(runs_dir.glob("????-??-??"), reverse=True)
    for ddir in dated_dirs:
        p = ddir / f"{fund}_analytics.json"
        if p.exists():
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                return data.get("weights", {})
            except Exception:
                pass
    return {}


def _drift_report(fund: str, prior: dict[str, float], current: dict[str, float]) -> str:
    all_tickers = set(prior) | set(current)
    lines = [f"## {fund} — Weight Drift vs. Prior Week", ""]
    lines += ["| Ticker | Prior | Current | Δ |", "|--------|-------|---------|---|"]
    for t in sorted(all_tickers):
        pw = prior.get(t, 0.0)
        cw = current.get(t, 0.0)
        delta = cw - pw
        if abs(delta) > 0.001:  # only show meaningful changes
            sign = "↑" if delta > 0 else "↓"
            lines.append(f"| {t} | {pw:.1%} | {cw:.1%} | {sign} {abs(delta):.1%} |")
    lines.append("")
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────
# Holdings report
# ──────────────────────────────────────────────────────────────

def _holdings_md(fund: str, result, run_date: str) -> str:
    lines = [
        f"# {fund.replace('_', ' ').title()} — Holdings Report",
        f"*{run_date}*",
        "",
    ]
    if not result or not result.holdings:
        lines.append("*No holdings (insufficient candidates passed the funnel).*")
        return "\n".join(lines)

    lines += ["| Ticker | Weight | Grade | Composite |", "|--------|--------|-------|-----------|"]
    for h in sorted(result.holdings, key=lambda x: -x.weight):
        lines.append(
            f"| {h.ticker} | {h.weight:.1%} | {h.grade} | {h.composite:.1f} |"
        )
    lines.append("")

    if result.analytics:
        a = result.analytics
        lines += [
            "## Analytics",
            f"- Expected return: {a.expected_return:.1%}",
            f"- Volatility: {a.volatility:.1%}",
            f"- Sharpe: {a.sharpe:.2f}",
            f"- N holdings: {a.n_holdings}",
            "",
        ]
    return "\n".join(lines)


def _mandate_md(fund: str, result) -> str:
    lines = [f"## {fund} — Mandate Check", ""]
    if result is None or result.mandate_check is None:
        lines.append("*Mandate check unavailable.*")
        return "\n".join(lines)

    mc = result.mandate_check
    status = "✅ PASS" if mc.passed else "❌ FAIL"
    lines.append(f"**Overall: {status}**")
    lines.append("")
    if mc.violations:
        lines.append("**Violations:**")
        for v in mc.violations:
            lines.append(f"- {v}")
    else:
        lines.append("No violations.")
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────
# Main command
# ──────────────────────────────────────────────────────────────

@app.command()
def main(
    dry_run: bool = typer.Option(False, "--dry-run", help="Validate config, no live data or writes"),
    funds:   str  = typer.Option("all", "--funds", help="Comma-separated fund names, or 'all'"),
    save:    bool = typer.Option(True, "--save/--no-save", help="Write output to runs/weekly/"),
):
    """
    Full-universe portfolio rebuild for all five risk-tiered funds.
    """
    if dry_run:
        typer.echo("--dry-run: validating configuration only, no live data fetched.")
        from stockgrader.config import get_config
        try:
            cfg = get_config()
            typer.echo(f"✅ Config loaded. Funds: {list(cfg.get('portfolios', {}).keys())}")
        except Exception as exc:
            typer.echo(f"❌ Config error: {exc}", err=True)
            raise typer.Exit(1)
        raise typer.Exit(2)

    if funds == "all":
        fund_list = _ALL_FUNDS
    else:
        fund_list = [f.strip() for f in funds.split(",")]

    from stockgrader.config import get_config
    from stockgrader.portfolios.construction import run_full_funnel

    config   = get_config()
    run_date = datetime.utcnow().strftime("%Y-%m-%d")
    out_dir  = _RUNS_DIR / run_date

    if save:
        out_dir.mkdir(parents=True, exist_ok=True)

    drift_sections: list[str] = [f"# Drift Report — {run_date}", ""]

    for fund in fund_list:
        typer.echo(f"\n{'─'*60}")
        typer.echo(f"  Building portfolio: {fund} …")

        try:
            result = run_full_funnel(fund, config)
        except Exception as exc:
            logger.error("Failed to build %s: %s", fund, exc)
            typer.echo(f"  ❌ {fund}: {exc}", err=True)
            continue

        # Weights dict for drift report
        curr_weights: dict[str, float] = {}
        if result and result.holdings:
            curr_weights = {h.ticker: h.weight for h in result.holdings}
            typer.echo(f"  ✅ {fund}: {len(result.holdings)} holdings")
        else:
            typer.echo(f"  ⚠️  {fund}: no holdings (funnel returned empty)")

        # Drift
        prior_weights = _load_prior_holdings(fund, _RUNS_DIR)
        drift_sections.append(_drift_report(fund, prior_weights, curr_weights))

        if not save:
            continue

        # Holdings report
        holdings_md = _holdings_md(fund, result, run_date)
        (out_dir / f"{fund}_holdings.md").write_text(holdings_md, encoding="utf-8")

        # Analytics JSON
        analytics_data: dict = {"weights": curr_weights}
        if result and result.analytics:
            a = result.analytics
            analytics_data.update({
                "expected_return": a.expected_return,
                "volatility":      a.volatility,
                "sharpe":          a.sharpe,
                "n_holdings":      a.n_holdings,
            })
        (out_dir / f"{fund}_analytics.json").write_text(
            json.dumps(analytics_data, indent=2), encoding="utf-8"
        )

        # Mandate check
        mandate_md = _mandate_md(fund, result)
        (out_dir / f"{fund}_mandate.md").write_text(mandate_md, encoding="utf-8")

    # Drift report
    drift_md = "\n".join(drift_sections)
    if save:
        (out_dir / "drift_report.md").write_text(drift_md, encoding="utf-8")
        typer.echo(f"\n✅ All reports saved to {out_dir}/")
    else:
        typer.echo(drift_md)


if __name__ == "__main__":
    app()
