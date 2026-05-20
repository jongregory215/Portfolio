"""
Daily watchlist alert runner.

Grades each ticker in watchlist.yaml each morning and prints ONLY actionable
alerts:
  - Overall or portfolio sub-grade changed vs. prior run
  - Price crossed a grade-boundary on the price ladder
  - Circuit breaker newly triggered
  - alert_price hit (price fell to or below the configured alert level)

Output:
  - Color-coded terminal summary
  - Dated Markdown file in runs/daily/

Exit codes:
  0 — success (alerts may or may not be present)
  1 — runtime error
  2 — called with --dry-run (no live data fetched, no files written)

Usage
-----
  python daily_run.py [--dry-run] [--watchlist /path/to/watchlist.yaml]
  python daily_run.py --format md     # Markdown to stdout instead of terminal
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
    name="daily_run",
    help="Morning watchlist alert runner.",
    add_completion=False,
)

logging.basicConfig(
    level=logging.WARNING,
    format="%(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

_RUNS_DIR       = Path("runs") / "daily"
_STATE_FILE     = Path("runs") / ".daily_state.json"
_DEFAULT_WL     = Path("watchlist.yaml")

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
    if _STATE_FILE.exists():
        try:
            return json.loads(_STATE_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def _save_state(state: dict) -> None:
    _STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    _STATE_FILE.write_text(json.dumps(state, indent=2), encoding="utf-8")


# ──────────────────────────────────────────────────────────────
# Alert detection
# ──────────────────────────────────────────────────────────────

def _grade_changed(prev: dict | None, curr_grade: str) -> bool:
    if prev is None:
        return False
    return prev.get("grade") != curr_grade


def _price_crossed_boundary(prev: dict | None, result) -> list[str]:
    """Return list of boundary-crossing messages."""
    msgs: list[str] = []
    if prev is None or result.price_ladder is None:
        return msgs
    prev_price = prev.get("price")
    curr_price = getattr(result, "price", None)
    if prev_price is None or curr_price is None:
        return msgs

    ladder = result.price_ladder
    boundaries = {
        "Gotta Have upper": getattr(ladder, "gotta_have_max", None),
        "Buy upper":        getattr(ladder, "buy_max", None),
        "Hold upper":       getattr(ladder, "hold_max", None),
        "Sell upper":       getattr(ladder, "sell_max", None),
    }
    for label, boundary in boundaries.items():
        if boundary is None:
            continue
        crossed_down = prev_price > boundary >= curr_price
        crossed_up   = prev_price < boundary <= curr_price
        if crossed_down:
            msgs.append(f"Price dropped below {label} boundary (${boundary:.2f})")
        elif crossed_up:
            msgs.append(f"Price rose above {label} boundary (${boundary:.2f})")
    return msgs


def _circuit_breaker_fired(prev: dict | None, result) -> list[str]:
    """Return list of newly triggered circuit breaker names."""
    curr_cbs: set[str] = set()
    if hasattr(result, "overall") and result.overall is not None:
        cbs = getattr(result.overall, "circuit_breakers", None)
        if cbs:
            curr_cbs = {cb for cb in cbs if cbs[cb]}

    prev_cbs: set[str] = set(prev.get("circuit_breakers", [])) if prev else set()
    return sorted(curr_cbs - prev_cbs)


def _alert_price_hit(entry: dict, result) -> bool:
    """Return True if price has dropped to or below alert_price."""
    alert = entry.get("alert_price")
    curr_price = getattr(result, "price", None)
    if alert is None or curr_price is None:
        return False
    return curr_price <= alert


# ──────────────────────────────────────────────────────────────
# Formatting
# ──────────────────────────────────────────────────────────────

def _grade_emoji(grade_str: str) -> str:
    return _GRADE_EMOJI.get(grade_str, "")


def _alert_lines_md(ticker: str, alerts: list[str]) -> str:
    lines = [f"### {ticker}"]
    for a in alerts:
        lines.append(f"- {a}")
    return "\n".join(lines)


def _make_report(
    alerts: list[tuple[str, list[str]]],
    no_alerts: list[str],
    run_date: str,
) -> str:
    lines = [
        f"# Daily Watchlist Alert Report — {run_date}",
        "",
    ]
    if alerts:
        lines += [f"## Actionable Alerts ({len(alerts)} tickers)", ""]
        for ticker, msgs in alerts:
            lines.append(_alert_lines_md(ticker, msgs))
            lines.append("")
    else:
        lines += ["## No Actionable Alerts", ""]
        lines.append("All watchlist tickers are within expected ranges.")
        lines.append("")

    if no_alerts:
        lines += ["## Tickers with No Change", ""]
        lines.append(", ".join(no_alerts))
        lines.append("")

    lines.append(f"*Generated {run_date} UTC*")
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────
# Main command
# ──────────────────────────────────────────────────────────────

@app.command()
def main(
    dry_run:   bool = typer.Option(False, "--dry-run", help="No live data, no file writes"),
    watchlist: str  = typer.Option(str(_DEFAULT_WL), "--watchlist", help="Path to watchlist YAML"),
    fmt:       str  = typer.Option("term", "--format", help="term | md"),
    save:      bool = typer.Option(True, help="Write dated report to runs/daily/"),
):
    """
    Grade watchlist tickers and emit actionable alerts only.
    """
    if dry_run:
        typer.echo("--dry-run: no live data fetched, no files written.")
        raise typer.Exit(2)

    # Load watchlist
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
    from stockgrader.reporting.terminal_reporter import TerminalReporter

    config  = get_config()
    state   = _load_state()
    run_ts  = datetime.utcnow().strftime("%Y-%m-%d_%H%M")
    run_date = run_ts[:10]

    alerts:    list[tuple[str, list[str]]] = []
    no_alerts: list[str] = []
    new_state: dict = {}

    reporter = TerminalReporter()

    for entry in entries:
        ticker = entry["ticker"]
        typer.echo(f"  Grading {ticker} …", err=True)

        try:
            result = run_analysis(ticker, config=config)
        except Exception as exc:
            logger.warning("Failed to grade %s: %s", ticker, exc)
            continue

        grade_str  = result.overall.grade.value if result.overall else "Unknown"
        curr_price = getattr(result, "price", None)
        cbs        = {}
        if result.overall and hasattr(result.overall, "circuit_breakers"):
            cbs = {k: bool(v) for k, v in (result.overall.circuit_breakers or {}).items()}

        new_state[ticker] = {
            "grade":           grade_str,
            "price":           curr_price,
            "circuit_breakers": [k for k, v in cbs.items() if v],
        }

        prev = state.get(ticker)
        ticker_alerts: list[str] = []

        # Grade change
        if _grade_changed(prev, grade_str):
            prev_grade = prev.get("grade", "unknown") if prev else "unknown"
            ticker_alerts.append(
                f"Grade changed: {prev_grade} → {grade_str} {_grade_emoji(grade_str)}"
            )

        # Price ladder crossings
        ticker_alerts += _price_crossed_boundary(prev, result)

        # New circuit breakers
        for cb in _circuit_breaker_fired(prev, result):
            ticker_alerts.append(f"Circuit breaker triggered: {cb}")

        # Alert price
        if _alert_price_hit(entry, result):
            ticker_alerts.append(
                f"Alert price hit: ${curr_price:.2f} ≤ ${entry['alert_price']:.2f}"
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

    # Update persisted state
    _save_state(new_state)

    # Summary
    report_md = _make_report(alerts, no_alerts, run_date)
    if fmt == "md":
        typer.echo(report_md)
    elif not alerts:
        typer.echo(f"\n✅ No actionable alerts for {run_date}.")
    else:
        typer.echo(f"\n⚡ {len(alerts)} alert(s) for {run_date}.")

    if save:
        out = _RUNS_DIR / f"{run_date}.md"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(report_md, encoding="utf-8")
        typer.echo(f"   Report saved → {out}", err=True)


if __name__ == "__main__":
    app()
