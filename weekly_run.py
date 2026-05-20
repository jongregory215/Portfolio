"""
Weekly portfolio rebuild runner.

Runs the full Section 11 portfolio-construction funnel for all five funds:
  universe screen → sub-grade / rank → constrained optimization → assemble

Output per run (written to runs/weekly/YYYY-MM-DD/):
  {fund}_holdings.md       — holdings table with weights, grades, composites
  {fund}_analytics.json    — Sharpe, vol, sector concentrations, weights
  {fund}_mandate.md        — mandate check results
  {fund}_funnel.md         — funnel transparency: how many names passed each stage
  drift_report.md          — weight drift vs. prior week

Checkpointing
-------------
Completed fund results are cached to runs/weekly/YYYY-MM-DD/.checkpoint.json
so an interrupted run resumes without re-fetching data.  Delete the checkpoint
to force a full rebuild.

Rate-limit handling
-------------------
- Aggressive caching via stockgrader's disk cache (Stage-1 screening data has
  a 7-day TTL; intraday prices 15 min).  The weekly run should hit the cache
  for most Stage-1 calls if daily_run has run recently.
- Configurable concurrency cap: `weekly.concurrency` in config.yaml (default 1,
  i.e. serial).  Increase with care — FMP free tier allows ~250 calls/day.
- Expected runtime: ~20–60 min for 3 000-ticker universe on FMP Starter.
  The free tier (250 calls/day) can't complete a full universe run in one day;
  FMP Starter ($14/mo) is the recommended minimum for weekly rebuilds.

Exit codes
----------
  0 — success
  1 — runtime error
  2 — --dry-run

Usage
-----
  python weekly_run.py [--dry-run] [--funds conservative,balanced] [--no-save]
  python weekly_run.py --funds all     # default; all five funds
"""
from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timezone
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
# Checkpointing
# ──────────────────────────────────────────────────────────────

def _checkpoint_path(out_dir: Path) -> Path:
    return out_dir / ".checkpoint.json"


def _load_checkpoint(out_dir: Path) -> dict[str, bool]:
    """Return {fund: True} for already-completed funds."""
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
# Prior holdings for drift report
# ──────────────────────────────────────────────────────────────

def _load_prior_holdings(fund: str, runs_dir: Path) -> dict[str, float]:
    """Load last week's weights from the most recent completed dated subfolder."""
    dated_dirs = sorted(
        (d for d in runs_dir.glob("????-??-??") if d.is_dir()),
        reverse=True,
    )
    for ddir in dated_dirs:
        p = ddir / f"{fund}_analytics.json"
        if p.exists():
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                return data.get("weights", {})
            except Exception:
                pass
    return {}


# ──────────────────────────────────────────────────────────────
# Report builders
# ──────────────────────────────────────────────────────────────

def _drift_report(fund: str, prior: dict[str, float], current: dict[str, float]) -> str:
    all_tickers = set(prior) | set(current)
    lines = [f"## {fund} — Weight Drift vs. Prior Week", ""]

    entered  = sorted(t for t in current if t not in prior)
    exited   = sorted(t for t in prior   if t not in current)
    if entered:
        lines.append(f"**Entered portfolio:** {', '.join(entered)}")
    if exited:
        lines.append(f"**Exited portfolio:** {', '.join(exited)}")
    if entered or exited:
        lines.append("")

    lines += ["| Ticker | Prior | Current | Δ |", "|--------|-------|---------|---|"]
    changed = False
    for t in sorted(all_tickers):
        pw = prior.get(t, 0.0)
        cw = current.get(t, 0.0)
        delta = cw - pw
        if abs(delta) > 0.001:
            sign = "↑" if delta > 0 else "↓"
            lines.append(f"| {t} | {pw:.1%} | {cw:.1%} | {sign} {abs(delta):.1%} |")
            changed = True
    if not changed:
        lines.append("*No meaningful weight changes.*")
    lines.append("")
    return "\n".join(lines)


def _holdings_md(fund: str, result, run_date: str) -> str:
    fund_label = fund.replace("_", " ").title()
    lines = [f"# {fund_label} — Holdings Report", f"*{run_date}*", ""]
    if not result or not result.holdings:
        lines.append("*No holdings (insufficient candidates passed the funnel).*")
        return "\n".join(lines)

    lines += ["| # | Ticker | Weight | Grade | Composite |",
              "|---|--------|--------|-------|-----------|"]
    for i, h in enumerate(sorted(result.holdings, key=lambda x: -x.weight), 1):
        lines.append(
            f"| {i} | {h.ticker} | {h.weight:.1%} | {h.grade} | {h.composite:.1f} |"
        )
    lines.append("")

    if result.analytics:
        a = result.analytics
        lines += [
            "## Analytics",
            f"- Expected annual return: {a.expected_return:.1%}",
            f"- Estimated volatility:   {a.volatility:.1%}",
            f"- Estimated Sharpe:       {a.sharpe:.2f}",
            f"- Holdings count:         {a.n_holdings}",
            "",
        ]
    return "\n".join(lines)


def _mandate_md(fund: str, result) -> str:
    fund_label = fund.replace("_", " ").title()
    lines = [f"# {fund_label} — Mandate Check", ""]
    if result is None or result.mandate_check is None:
        lines.append("*Mandate check unavailable.*")
        return "\n".join(lines)

    mc = result.mandate_check
    status = "✅ PASS" if mc.passed else "❌ FAIL"
    lines += [f"**Overall: {status}**", ""]
    if mc.violations:
        lines.append("**Violations:**")
        for v in mc.violations:
            lines.append(f"- {v}")
    else:
        lines.append("No violations.")
    return "\n".join(lines)


def _funnel_md(fund: str, result, run_date: str) -> str:
    """Funnel transparency report: how many names passed each construction stage."""
    fund_label = fund.replace("_", " ").title()
    lines = [f"# {fund_label} — Funnel Transparency", f"*{run_date}*", ""]

    stats = getattr(result, "funnel_stats", None) if result else None
    if stats is None:
        lines.append("*Funnel statistics not available.*")
        return "\n".join(lines)

    lines += ["| Stage | Names In | Names Out | Reason |",
              "|-------|----------|-----------|--------|"]

    stage_data = [
        ("Stage 1 — Universe screen",
         getattr(stats, "n_universe",      None),
         getattr(stats, "n_after_screen",  None),
         "Liquidity / exchange / circuit-breaker pre-filter"),
        ("Stage 2 — Grade & rank",
         getattr(stats, "n_after_screen",  None),
         getattr(stats, "n_after_rank",    None),
         "Only Buy / Gotta Have eligible; top 40 by composite"),
        ("Stage 3 — Optimize",
         getattr(stats, "n_after_rank",    None),
         getattr(stats, "n_after_optimize",None),
         "Mean-variance optimiser (Ledoit-Wolf cov, shrunk ER)"),
        ("Stage 4 — Assemble",
         getattr(stats, "n_after_optimize",None),
         getattr(result.holdings, "__len__", lambda: None)() if result and result.holdings else None,
         "Equity sleeve + bond sleeve, mandate check"),
    ]
    for stage, n_in, n_out, reason in stage_data:
        n_in_s  = str(n_in)  if n_in  is not None else "—"
        n_out_s = str(n_out) if n_out is not None else "—"
        lines.append(f"| {stage} | {n_in_s} | {n_out_s} | {reason} |")

    lines.append("")
    return "\n".join(lines)


def _analytics_json(fund: str, result) -> dict:
    """Build the analytics dict saved to {fund}_analytics.json."""
    curr_weights: dict[str, float] = {}
    if result and result.holdings:
        curr_weights = {h.ticker: h.weight for h in result.holdings}

    data: dict = {"fund": fund, "weights": curr_weights}
    if result and result.analytics:
        a = result.analytics
        data.update({
            "expected_return": a.expected_return,
            "volatility":      a.volatility,
            "sharpe":          a.sharpe,
            "n_holdings":      a.n_holdings,
        })
    return data


# ──────────────────────────────────────────────────────────────
# Main command
# ──────────────────────────────────────────────────────────────

@app.command()
def main(
    dry_run: bool = typer.Option(False, "--dry-run",
                                 help="Validate config only; no live data or writes. Exit 2."),
    funds:   str  = typer.Option("all", "--funds",
                                 help="Comma-separated fund names, or 'all'"),
    save:    bool = typer.Option(True, "--save/--no-save",
                                 help="Write output to runs/weekly/ (default on)"),
    resume:  bool = typer.Option(True, "--resume/--no-resume",
                                 help="Skip funds already checkpointed for today"),
):
    """
    Full-universe portfolio rebuild for all five risk-tiered funds.

    Runs the Stage 1-4 funnel per fund, writes holdings + analytics +
    mandate check + funnel transparency, and produces a drift report vs.
    the prior week.  Supports checkpointing so interrupted runs resume
    where they left off.

    Note: FMP Starter ($14/mo) or higher is required for the full-universe
    screen.  The free tier (250 calls/day) is insufficient for a complete
    rebuild.  See README §API Tiers.
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

    if funds == "all":
        fund_list = _ALL_FUNDS
    else:
        fund_list = [f.strip() for f in funds.split(",")]

    from stockgrader.config import get_config
    from stockgrader.portfolios.construction import run_full_funnel

    config   = get_config()
    run_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    out_dir  = _RUNS_DIR / run_date

    if save:
        out_dir.mkdir(parents=True, exist_ok=True)

    # ── Checkpointing ─────────────────────────────────────────
    checkpoint: dict[str, bool] = _load_checkpoint(out_dir) if resume and save else {}
    if checkpoint:
        already_done = [f for f in fund_list if checkpoint.get(f)]
        if already_done:
            typer.echo(f"  Resuming — skipping already-done funds: {already_done}")

    drift_sections: list[str] = [f"# Drift Report — {run_date}", ""]
    had_error = False

    # ── Per-fund loop ─────────────────────────────────────────
    for fund in fund_list:
        if checkpoint.get(fund):
            typer.echo(f"\n  ⏭  {fund} (checkpointed — skipped)")
            # Still need prior holdings for drift
            prior = _load_prior_holdings(fund, _RUNS_DIR)
            anl_path = out_dir / f"{fund}_analytics.json"
            curr_w: dict[str, float] = {}
            if anl_path.exists():
                try:
                    curr_w = json.loads(anl_path.read_text(encoding="utf-8")).get("weights", {})
                except Exception:
                    pass
            drift_sections.append(_drift_report(fund, prior, curr_w))
            continue

        typer.echo(f"\n{'─'*60}")
        typer.echo(f"  Building portfolio: {fund} …")

        try:
            result = run_full_funnel(fund, config)
        except Exception as exc:
            logger.error("Failed to build %s: %s", fund, exc)
            typer.echo(f"  ❌ {fund}: {exc}", err=True)
            had_error = True
            continue

        # Summary
        n_hold = len(result.holdings) if result and result.holdings else 0
        if n_hold:
            typer.echo(f"  ✅ {fund}: {n_hold} holdings")
        else:
            typer.echo(f"  ⚠️  {fund}: funnel returned empty (no holdings)")

        # Drift
        curr_weights = {h.ticker: h.weight for h in result.holdings} if result and result.holdings else {}
        prior_weights = _load_prior_holdings(fund, _RUNS_DIR)
        drift_sections.append(_drift_report(fund, prior_weights, curr_weights))

        if not save:
            continue

        # ── Write per-fund output files ────────────────────────
        (out_dir / f"{fund}_holdings.md").write_text(
            _holdings_md(fund, result, run_date), encoding="utf-8"
        )
        (out_dir / f"{fund}_analytics.json").write_text(
            json.dumps(_analytics_json(fund, result), indent=2), encoding="utf-8"
        )
        (out_dir / f"{fund}_mandate.md").write_text(
            _mandate_md(fund, result), encoding="utf-8"
        )
        (out_dir / f"{fund}_funnel.md").write_text(
            _funnel_md(fund, result, run_date), encoding="utf-8"
        )

        # Mark fund as done in checkpoint
        checkpoint[fund] = True
        _save_checkpoint(out_dir, checkpoint)

    # ── Drift report ──────────────────────────────────────────
    drift_md = "\n".join(drift_sections)
    if save:
        (out_dir / "drift_report.md").write_text(drift_md, encoding="utf-8")
        typer.echo(f"\n✅ All reports saved to {out_dir}/")
        if had_error:
            typer.echo("   ⚠️  Some funds failed — check logs. Checkpoint saved for completed funds.")
    else:
        typer.echo(drift_md)

    if had_error:
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
