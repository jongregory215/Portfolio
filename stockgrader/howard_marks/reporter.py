"""
Terminal, JSON, and Markdown reporters for Howard Marks evaluation results.
"""
from __future__ import annotations

import json
from typing import Sequence

from stockgrader.howard_marks.models import CriterionResult, CycleReading, MarksResult

# ── ANSI color codes (no dependency on rich) ─────────────────────────────────
_GREEN  = "\033[92m"
_RED    = "\033[91m"
_YELLOW = "\033[93m"
_BOLD   = "\033[1m"
_RESET  = "\033[0m"
_DIM    = "\033[2m"

_PASS = f"{_GREEN}✓{_RESET}"
_FAIL = f"{_RED}✗{_RESET}"


def _check(passed: bool) -> str:
    return _PASS if passed else _FAIL


def _verdict_color(verdict: str) -> str:
    if verdict == "Compelling Opportunity":
        return _GREEN
    if verdict == "Worth a Closer Look":
        return _YELLOW
    return _DIM


def _verdict_str(verdict: str) -> str:
    color = _verdict_color(verdict)
    return f"{color}{_BOLD}{verdict}{_RESET}"


def _bar(met: int, total: int) -> str:
    ratio_str = f"{met} / {total} criteria"
    if met >= total - 1:
        color = _GREEN
    elif met >= total // 2:
        color = _YELLOW
    else:
        color = _RED
    return f"{color}[{ratio_str}]{_RESET}"


def _fmt_criterion(c: CriterionResult, indent: int = 2) -> str:
    pad = " " * indent
    check = _check(c.passed)
    name_col = (c.name + " ").ljust(38, "·")
    line = f"{pad}{check} {name_col} {c.label}"
    if c.note:
        line += f"\n{pad}  {_DIM}↳ {c.note}{_RESET}"
    return line


def _wide_sep(char: str = "═", width: int = 70) -> str:
    return char * width


# ── Cycle banner ──────────────────────────────────────────────────────────────

def format_cycle_banner(cycle: CycleReading) -> str:
    lines: list[str] = []
    lines.append("")
    lines.append(f"{_BOLD}Where We Stand in the Cycle{_RESET}  ({cycle.as_of})")
    lines.append(_wide_sep("─"))

    yc = f"{cycle.yield_curve_spread:+.2f} pts" if cycle.yield_curve_spread is not None else "N/A"
    hy = f"{cycle.high_yield_spread:.2f}%" if cycle.high_yield_spread is not None else "N/A"
    vix = f"{cycle.vix:.1f}" if cycle.vix is not None else "N/A"

    lines.append(f"  10yr-2yr yield curve: {_BOLD}{yc}{_RESET}    "
                  f"High-yield spread: {_BOLD}{hy}{_RESET}    "
                  f"VIX: {_BOLD}{vix}{_RESET}")

    zone_color = {
        "Fear / Capitulation": _GREEN,
        "Greed / Late-Cycle":  _RED,
        "Neutral":             _YELLOW,
    }.get(cycle.zone, _RESET)

    lines.append(f"  Zone: {zone_color}{_BOLD}{cycle.zone}{_RESET}   "
                  f"(EPV margin-of-safety multiplier: {_BOLD}{cycle.mos_multiplier:.2f}×{_RESET})")
    lines.append(f"  {_DIM}{cycle.commentary}{_RESET}")
    lines.append("")
    return "\n".join(lines)


def format_cycle_banner_markdown(cycle: CycleReading) -> str:
    yc = f"{cycle.yield_curve_spread:+.2f} pts" if cycle.yield_curve_spread is not None else "N/A"
    hy = f"{cycle.high_yield_spread:.2f}%" if cycle.high_yield_spread is not None else "N/A"
    vix = f"{cycle.vix:.1f}" if cycle.vix is not None else "N/A"

    lines = [
        f"## Where We Stand in the Cycle ({cycle.as_of})",
        "",
        f"- **10yr-2yr yield curve:** {yc}",
        f"- **High-yield spread:** {hy}",
        f"- **VIX:** {vix}",
        f"- **Zone:** {cycle.zone} (EPV margin-of-safety multiplier: {cycle.mos_multiplier:.2f}×)",
        "",
        cycle.commentary,
        "",
    ]
    return "\n".join(lines)


# ── Per-ticker formats ───────────────────────────────────────────────────────

def format_terminal(result: MarksResult) -> str:
    lines: list[str] = []
    price = result.price

    lines.append("")
    header = f"{_BOLD}{result.ticker}{_RESET} — Howard Marks Evaluation ({result.as_of})  Price: {_BOLD}${price:.2f}{_RESET}"
    if result.company_name and result.company_name != result.ticker:
        header += f"  {_DIM}{result.company_name}{_RESET}"
    lines.append(header)
    lines.append(_wide_sep("═"))

    lines.append(f"{_BOLD}HOWARD MARKS LENS{_RESET}  {_wide_sep('─', 30)}  {_bar(result.criteria_met, result.total_criteria)}")
    lines.append(_wide_sep("─"))

    for c in result.criteria:
        lines.append(_fmt_criterion(c))

    lines.append("")

    if result.epv is not None:
        pct = result.price_vs_epv_pct or 0.0
        direction = "above" if pct > 0 else "below"
        color = _RED if pct > 0 else _GREEN
        lines.append(
            f"  Earnings Power Value: {_BOLD}${result.epv:.2f}{_RESET}  |  "
            f"Price (${price:.2f}) is {color}{abs(pct):.0f}% {direction} EPV{_RESET}"
        )
    else:
        lines.append(f"  {_DIM}Earnings Power Value: N/A (insufficient data){_RESET}")

    if result.range_position_pct is not None:
        lines.append(f"  52-week range position: {_BOLD}{result.range_position_pct:.0f}%{_RESET} (0% = 52wk low, 100% = 52wk high)")

    if result.reward_risk_ratio is not None:
        lines.append(f"  Reward/Risk ratio: {_BOLD}{result.reward_risk_ratio:.2f}{_RESET}")

    lines.append(f"  Verdict: {_verdict_str(result.verdict)}")
    lines.append("")

    return "\n".join(lines)


def format_json(result: MarksResult) -> str:
    return json.dumps(result.to_dict(), indent=2, default=str)


def format_markdown(result: MarksResult) -> str:
    price = result.price
    lines: list[str] = []

    lines.append(f"# {result.ticker} — Howard Marks Evaluation ({result.as_of})")
    lines.append(f"**Price:** ${price:.2f}  |  **Company:** {result.company_name}")
    lines.append("")

    lines.append(f"## Howard Marks Lens  [{result.criteria_met} / {result.total_criteria} criteria]")
    lines.append("")
    lines.append("| # | Criterion | Value | Pass? |")
    lines.append("|---|-----------|-------|-------|")
    for i, c in enumerate(result.criteria, 1):
        mark = "✓" if c.passed else "✗"
        lines.append(f"| {i} | {c.name} | {c.label} | {mark} |")
    lines.append("")

    if result.epv is not None:
        pct = result.price_vs_epv_pct or 0.0
        direction = "above" if pct > 0 else "below"
        lines.append(f"**Earnings Power Value:** ${result.epv:.2f}  |  Price is {abs(pct):.0f}% {direction} EPV")
    if result.range_position_pct is not None:
        lines.append(f"**52-week range position:** {result.range_position_pct:.0f}%")
    if result.reward_risk_ratio is not None:
        lines.append(f"**Reward/Risk ratio:** {result.reward_risk_ratio:.2f}")

    lines.append(f"**Verdict: {result.verdict}**")
    return "\n".join(lines)


def format_summary(results: Sequence[MarksResult]) -> str:
    """Condensed table: one row per stock, sorted by criteria_met desc."""
    lines: list[str] = []
    lines.append("")
    header = (
        f"{'Ticker':<7} {'Company':<28} {'Price':>8}  "
        f"{'Score':>6}  {'Range%':>7}  {'Verdict'}"
    )
    lines.append(f"{_BOLD}{header}{_RESET}")
    lines.append("─" * 78)

    sorted_results = sorted(results, key=lambda r: r.criteria_met, reverse=True)

    for r in sorted_results:
        score_str = f"{r.criteria_met}/{r.total_criteria}"
        if r.criteria_met >= r.total_criteria - 1:
            score_color = _GREEN
        elif r.criteria_met >= r.total_criteria // 2:
            score_color = _YELLOW
        else:
            score_color = _RED

        range_str = f"{r.range_position_pct:.0f}%" if r.range_position_pct is not None else "N/A"
        name = (r.company_name[:26] + "..") if len(r.company_name) > 28 else r.company_name

        row = (
            f"{r.ticker:<7} {name:<28} ${r.price:>7.2f}  "
            f"{score_color}{score_str:>6}{_RESET}  "
            f"{range_str:>7}  "
            f"{_verdict_str(r.verdict)}"
        )
        lines.append(row)

    lines.append("")
    compelling = [r for r in results if r.verdict == "Compelling Opportunity"]
    closer_look = [r for r in results if r.verdict == "Worth a Closer Look"]
    lines.append(
        f"  {_BOLD}{len(compelling)} compelling opportunity(ies){_RESET}, "
        f"{_BOLD}{len(closer_look)} worth a closer look{_RESET} "
        f"out of {len(results)} stocks screened."
    )
    lines.append("")
    return "\n".join(lines)


def print_results(results: Sequence[MarksResult], output: str = "terminal") -> None:
    for result in results:
        if output == "json":
            print(format_json(result))
        elif output == "markdown":
            print(format_markdown(result))
        elif output == "summary":
            pass  # handled separately — caller uses format_summary()
        else:
            print(format_terminal(result))
