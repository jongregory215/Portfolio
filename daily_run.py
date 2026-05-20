"""
Daily watchlist alert runner.

Grades each ticker in watchlist.yaml each morning and prints ONLY actionable
alerts:
  - Overall grade changed vs. the prior run
  - Any fund sub-grade changed (e.g., AAPL went Hold → Buy in the Aggressive sleeve)
  - Price crossed a grade-boundary on the price ladder
  - Circuit breaker newly triggered
  - alert_price hit (price fell to or below the configured level)

State and change detection
--------------------------
Each run's grades and ladder positions are persisted to:
  runs/daily/YYYY-MM-DD.json   (dated history, append-only)
  runs/.daily_state.json       (rolling latest — used as the diff baseline)

First run: no prior state → state is recorded, no change-alerts fired.  A
plain message is printed so the owner knows this was a baseline run.

Output
------
  Terminal: tight alert list.  Silence ("No alerts. N names checked …") is the
            expected outcome when the market is quiet — that IS the signal.
  Markdown: dated report written to runs/daily/YYYY-MM-DD.md

Notification hook
-----------------
Set `notifications.enabled: true` in config.yaml and implement
`_send_notification()` below to wire alerts to email / SMS / Slack.
The hook is off by default and is intentionally left as a stub.

Exit codes
----------
  0 — success (alerts may or may not be present)
  1 — runtime error (data outage, bad config)
  2 — --dry-run (no live data fetched, no files written)

Usage
-----
  python daily_run.py [--dry-run] [--watchlist watchlist.yaml] [--format md]
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

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env", override=False)
except ImportError:
    pass

app = typer.Typer(
    name="daily_run",
    help="Morning watchlist alert runner.",
    add_completion=False,
)

logging.basicConfig(
    level=logging.WARNING,
    format="%(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

_RUNS_DIR    = Path("runs") / "daily"
_STATE_FILE  = Path("runs") / ".daily_state.json"
_DEFAULT_WL  = Path("watchlist.yaml")

_GRADE_EMOJI = {
    "Stay Away":  "🚫",
    "Sell":       "⚠️",
    "Hold":       "⚖️",
    "Buy":        "✅",
    "Gotta Have": "⭐",
}

_GRADE_ORDER = ["Stay Away", "Sell", "Hold", "Buy", "Gotta Have"]


# ──────────────────────────────────────────────────────────────
# State persistence
# ──────────────────────────────────────────────────────────────

def _load_state() -> dict:
    """Load rolling prior-run state.  Returns {} on first run."""
    if _STATE_FILE.exists():
        try:
            return json.loads(_STATE_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def _save_state(state: dict, run_date: str, runs_dir: Path) -> None:
    """Persist rolling state and a dated snapshot for the historical record."""
    _STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    _STATE_FILE.write_text(json.dumps(state, indent=2), encoding="utf-8")
    # Dated snapshot (append-only history)
    snapshot = runs_dir / f"{run_date}.json"
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    snapshot.write_text(json.dumps(state, indent=2), encoding="utf-8")


# ──────────────────────────────────────────────────────────────
# Notification hook (stub — wire to email/SMS/Slack here)
# ──────────────────────────────────────────────────────────────

def _send_notification(alerts: list[tuple[str, list[str]]], config: dict) -> None:
    """
    Optional outbound notification hook.

    Called after all alerts are collected when `notifications.enabled` is True
    in config.yaml.  Currently a no-op stub — implement email/SMS/Slack below.

    Example config.yaml addition:
        notifications:
          enabled: false
          method: email       # or sms, slack
          recipient: you@example.com
    """
    if not config.get("notifications", {}).get("enabled", False):
        return
    # TODO: implement notification delivery
    # method = config["notifications"].get("method", "log")
    # For now, just log so the seam is visible
    logger.info(
        "NOTIFICATION HOOK: %d alert(s) — %s",
        len(alerts),
        ", ".join(t for t, _ in alerts),
    )


# ──────────────────────────────────────────────────────────────
# Alert detection
# ──────────────────────────────────────────────────────────────

def _grade_changed(prev: dict | None, curr_grade: str) -> bool:
    if prev is None:
        return False
    return prev.get("grade") != curr_grade


def _sub_grade_changes(prev: dict | None, result) -> list[str]:
    """Return messages for any fund-level sub-grade changes."""
    msgs: list[str] = []
    if prev is None:
        return msgs
    prev_sub: dict[str, str] = prev.get("sub_grades", {})

    curr_sub: dict[str, str] = {}
    if hasattr(result, "portfolio_grades") and result.portfolio_grades:
        pg = result.portfolio_grades
        for fund_attr in ["very_conservative", "conservative", "balanced",
                          "aggressive", "very_aggressive"]:
            sleeve = getattr(pg, fund_attr, None)
            if sleeve is not None:
                curr_sub[fund_attr] = sleeve.grade.value

    for fund, curr_g in curr_sub.items():
        prev_g = prev_sub.get(fund)
        if prev_g is not None and prev_g != curr_g:
            fund_label = fund.replace("_", " ").title()
            msgs.append(
                f"Sub-grade changed ({fund_label}): "
                f"{prev_g} → {curr_g} {_GRADE_EMOJI.get(curr_g, '')}"
            )
    return msgs


def _price_crossed_boundary(prev: dict | None, result) -> list[str]:
    """Return boundary-crossing messages when price moved across a grade line."""
    msgs: list[str] = []
    if prev is None:
        return msgs
    ladder = getattr(result, "price_ladder", None)
    if ladder is None:
        return msgs
    prev_price = prev.get("price")
    curr_price = getattr(result, "price", None)
    if prev_price is None or curr_price is None:
        return msgs

    boundaries = {
        "Gotta Have upper": getattr(ladder, "gotta_have_max", None),
        "Buy upper":        getattr(ladder, "buy_max", None),
        "Hold upper":       getattr(ladder, "hold_max", None),
        "Sell upper":       getattr(ladder, "sell_max", None),
    }
    for label, boundary in boundaries.items():
        if boundary is None:
            continue
        if prev_price > boundary >= curr_price:
            msgs.append(f"Price dropped below {label} boundary (${boundary:.2f})")
        elif prev_price < boundary <= curr_price:
            msgs.append(f"Price rose above {label} boundary (${boundary:.2f})")
    return msgs


def _circuit_breaker_fired(prev: dict | None, result) -> list[str]:
    """Return names of circuit breakers that are newly triggered vs. last run."""
    curr_cbs: set[str] = set()
    overall = getattr(result, "overall", None)
    if overall is not None:
        cbs = getattr(overall, "circuit_breakers", None)
        if cbs:
            curr_cbs = {cb for cb, v in cbs.items() if v}

    prev_cbs: set[str] = set(prev.get("circuit_breakers", [])) if prev else set()
    return sorted(curr_cbs - prev_cbs)


def _alert_price_hit(entry: dict, result) -> bool:
    """Return True if price has dropped to or below the watchlist alert_price."""
    alert = entry.get("alert_price")
    curr_price = getattr(result, "price", None)
    if alert is None or curr_price is None:
        return False
    return float(curr_price) <= float(alert)


# ──────────────────────────────────────────────────────────────
# State extraction from a result
# ──────────────────────────────────────────────────────────────

def _extract_state(result) -> dict:
    """Build the per-ticker state record to persist for the next diff."""
    overall   = getattr(result, "overall", None)
    grade_str = overall.grade.value if overall else "Unknown"
    price     = getattr(result, "price", None)

    cbs: list[str] = []
    if overall and hasattr(overall, "circuit_breakers"):
        raw = overall.circuit_breakers or {}
        cbs = [k for k, v in raw.items() if v]

    sub_grades: dict[str, str] = {}
    if hasattr(result, "portfolio_grades") and result.portfolio_grades:
        pg = result.portfolio_grades
        for fund_attr in ["very_conservative", "conservative", "balanced",
                          "aggressive", "very_aggressive"]:
            sleeve = getattr(pg, fund_attr, None)
            if sleeve is not None:
                sub_grades[fund_attr] = sleeve.grade.value

    ladder_prices: dict[str, float | None] = {}
    ladder = getattr(result, "price_ladder", None)
    if ladder is not None:
        for attr in ["gotta_have_max", "buy_max", "hold_max", "sell_max"]:
            ladder_prices[attr] = getattr(ladder, attr, None)

    return {
        "grade":          grade_str,
        "price":          price,
        "circuit_breakers": cbs,
        "sub_grades":     sub_grades,
        "ladder_prices":  ladder_prices,
    }


# ──────────────────────────────────────────────────────────────
# Markdown report builder
# ──────────────────────────────────────────────────────────────

def _grade_emoji(grade_str: str) -> str:
    return _GRADE_EMOJI.get(grade_str, "")


def _make_report(
    alerts:     list[tuple[str, list[str]]],
    no_alerts:  list[str],
    skipped:    list[str],
    run_date:   str,
    first_run:  bool,
) -> str:
    n_checked = len(alerts) + len(no_alerts) + len(skipped)
    lines = [f"# Daily Watchlist Alert Report — {run_date}", ""]

    if first_run:
        lines += [
            "> **First run** — no prior baseline existed.  State recorded.  "
            "No change-alerts generated.  Run again tomorrow to start diffing.",
            "",
        ]

    if alerts and not first_run:
        lines += [f"## Actionable Alerts ({len(alerts)} tickers)", ""]
        for ticker, msgs in alerts:
            lines.append(f"### {ticker}")
            for m in msgs:
                lines.append(f"- {m}")
            lines.append("")
    else:
        lines += ["## No Actionable Alerts", ""]
        lines.append(
            f"No alerts.  {n_checked} name(s) checked, "
            "all grades unchanged, no ladder breaches."
        )
        lines.append("")

    if no_alerts:
        lines += ["## Tickers — No Change", ""]
        lines.append(", ".join(no_alerts))
        lines.append("")

    if skipped:
        lines += ["## Skipped (data unavailable)", ""]
        lines.append(", ".join(skipped))
        lines.append("")

    lines.append(f"*Generated {run_date} UTC*")
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────
# Main command
# ──────────────────────────────────────────────────────────────

@app.command()
def main(
    dry_run:   bool = typer.Option(False, "--dry-run",
                                   help="No live data fetched, no files written. Exit 2."),
    watchlist: str  = typer.Option(str(_DEFAULT_WL), "--watchlist",
                                   help="Path to watchlist YAML"),
    fmt:       str  = typer.Option("term", "--format",
                                   help="Output format: term | md"),
    save:      bool = typer.Option(True,
                                   help="Write dated report + state to runs/daily/"),
):
    """
    Grade watchlist tickers and emit actionable alerts only.
    Silence ('No alerts. N names checked …') is the expected outcome when
    the market is quiet — it IS the signal.
    """
    if dry_run:
        typer.echo("--dry-run: no live data fetched, no files written.")
        raise typer.Exit(2)

    # ── Load watchlist ────────────────────────────────────────
    wl_path = Path(watchlist)
    if not wl_path.exists():
        typer.echo(f"❌ Watchlist not found: {wl_path}", err=True)
        raise typer.Exit(1)
    with open(wl_path, encoding="utf-8") as f:
        wl_data = yaml.safe_load(f)
    entries: list[dict] = wl_data.get("watchlist", [])
    if not entries:
        typer.echo("Watchlist is empty. Nothing to do.")
        raise typer.Exit(0)

    from stockgrader.pipeline import run_analysis
    from stockgrader.config import get_config

    config   = get_config()
    state    = _load_state()
    first_run = len(state) == 0

    run_ts   = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M")
    run_date = run_ts[:10]

    alerts:    list[tuple[str, list[str]]] = []
    no_alerts: list[str] = []
    skipped:   list[str] = []
    new_state: dict = {}

    if first_run:
        typer.echo(
            "\n📋 First run — no prior baseline.  Recording state; "
            "no change-alerts will fire today.",
            err=True,
        )

    # ── Grade each ticker ─────────────────────────────────────
    for entry in entries:
        ticker = entry["ticker"]
        typer.echo(f"  Grading {ticker} …", err=True)

        try:
            result = run_analysis(ticker, config=config)
        except Exception as exc:
            logger.warning("Skipping %s — data unavailable: %s", ticker, exc)
            skipped.append(ticker)
            continue

        new_state[ticker] = _extract_state(result)
        prev = state.get(ticker)

        if first_run:
            # No diff on first run — just record state
            no_alerts.append(ticker)
            continue

        ticker_alerts: list[str] = []
        grade_str = new_state[ticker]["grade"]

        # Overall grade change
        if _grade_changed(prev, grade_str):
            prev_grade = prev.get("grade", "unknown") if prev else "unknown"
            ticker_alerts.append(
                f"Overall grade: {prev_grade} → {grade_str} {_grade_emoji(grade_str)}"
            )

        # Sub-grade changes
        ticker_alerts += _sub_grade_changes(prev, result)

        # Price ladder crossings
        ticker_alerts += _price_crossed_boundary(prev, result)

        # New circuit breakers
        for cb in _circuit_breaker_fired(prev, result):
            ticker_alerts.append(f"Circuit breaker triggered: {cb}")

        # Alert price
        if _alert_price_hit(entry, result):
            curr_price = new_state[ticker].get("price")
            ticker_alerts.append(
                f"Alert price hit: "
                f"${curr_price:.2f} ≤ ${entry['alert_price']:.2f}"
            )

        if ticker_alerts:
            alerts.append((ticker, ticker_alerts))
            if fmt == "term":
                typer.echo(f"\n{'─'*50}")
                typer.echo(f"  {_grade_emoji(grade_str)} {ticker}  ({grade_str})")
                for msg in ticker_alerts:
                    typer.echo(f"    ⚡ {msg}")
        else:
            no_alerts.append(ticker)

    # ── Persist state ─────────────────────────────────────────
    if save:
        _save_state(new_state, run_date, _RUNS_DIR)

    # ── Notification hook ─────────────────────────────────────
    if alerts and not first_run:
        _send_notification(alerts, config)

    # ── Terminal summary ──────────────────────────────────────
    n_checked = len(alerts) + len(no_alerts) + len(skipped)
    report_md = _make_report(alerts, no_alerts, skipped, run_date, first_run)

    if fmt == "md":
        typer.echo(report_md)
    elif first_run:
        typer.echo(f"\n📋 Baseline recorded. {n_checked} name(s) checked.")
    elif not alerts:
        typer.echo(
            f"\n✅ No alerts.  {n_checked} name(s) checked, "
            "all grades unchanged, no ladder breaches."
        )
    else:
        typer.echo(f"\n⚡ {len(alerts)} alert(s).  {n_checked} name(s) checked.")

    if skipped:
        typer.echo(f"   ⚠️  Skipped {len(skipped)} ticker(s): {', '.join(skipped)}")

    # ── Write dated Markdown report ───────────────────────────
    if save:
        out = _RUNS_DIR / f"{run_date}.md"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(report_md, encoding="utf-8")
        typer.echo(f"   Report saved → {out}", err=True)


if __name__ == "__main__":
    app()
