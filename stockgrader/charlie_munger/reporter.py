"""
Terminal, JSON, and Markdown reporters for Charlie Munger evaluation results.
"""
from __future__ import annotations

import json
from typing import Sequence

from stockgrader.charlie_munger.models import CriterionResult, MungerResult

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
    if verdict == "Munger-Grade Business":
        return _GREEN
    if verdict == "Strong Candidate":
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
    name_col = (c.name + " ").ljust(48, "·")
    line = f"{pad}{check} {name_col} {c.label}"
    if c.note:
        line += f"\n{pad}  {_DIM}↳ {c.note}{_RESET}"
    return line


def _wide_sep(char: str = "═", width: int = 70) -> str:
    return char * width


def format_terminal(result: MungerResult) -> str:
    lines: list[str] = []
    price = result.price

    lines.append("")
    header = (
        f"{_BOLD}{result.ticker}{_RESET} — Charlie Munger Evaluation ({result.as_of})"
        f"  Price: {_BOLD}${price:.2f}{_RESET}"
    )
    if result.company_name and result.company_name != result.ticker:
        header += f"  {_DIM}{result.company_name}{_RESET}"
    lines.append(header)
    lines.append(_wide_sep("═"))

    lines.append(
        f"{_BOLD}MUNGER LENS{_RESET}  {_wide_sep('─', 35)}  "
        f"{_bar(result.criteria_met, result.total_criteria)}"
    )
    lines.append(_wide_sep("─"))

    for c in result.criteria:
        lines.append(_fmt_criterion(c))

    lines.append("")

    if result.intrinsic_value is not None:
        pct = result.price_vs_intrinsic_pct or 0.0
        direction = "above" if pct > 0 else "below"
        color = _RED if pct > 0 else _GREEN
        mos_pct = -pct
        lines.append(
            f"  Intrinsic Value (owner earnings DCF): {_BOLD}${result.intrinsic_value:.2f}{_RESET}  |  "
            f"Price (${price:.2f}) is {color}{abs(pct):.0f}% {direction} IV{_RESET}  "
            f"({'MoS ' + _GREEN + f'{mos_pct:.0f}%' + _RESET if mos_pct > 0 else _RED + f'{mos_pct:.0f}% (overvalued)' + _RESET})"
        )
    else:
        lines.append(f"  {_DIM}Intrinsic Value: N/A (insufficient data){_RESET}")

    if result.roic is not None:
        roic_color = _GREEN if result.roic > 0.20 else _YELLOW
        lines.append(f"  ROIC: {roic_color}{_BOLD}{result.roic*100:.1f}%{_RESET}")

    if result.capex_intensity is not None:
        ci_color = _GREEN if result.capex_intensity < 0.05 else _RED
        lines.append(f"  Capex intensity: {ci_color}{_BOLD}{result.capex_intensity*100:.1f}% of revenue{_RESET}")

    lines.append(f"  Verdict: {_verdict_str(result.verdict)}")
    lines.append("")

    return "\n".join(lines)


def format_json(result: MungerResult) -> str:
    return json.dumps(result.to_dict(), indent=2, default=str)


def format_markdown(result: MungerResult) -> str:
    price = result.price
    lines: list[str] = []

    lines.append(f"# {result.ticker} — Charlie Munger Evaluation ({result.as_of})")
    lines.append(f"**Price:** ${price:.2f}  |  **Company:** {result.company_name}")
    lines.append("")

    lines.append(f"## Munger Lens  [{result.criteria_met} / {result.total_criteria} criteria]")
    lines.append("")
    lines.append("| # | Criterion | Value | Pass? |")
    lines.append("|---|-----------|-------|-------|")
    for i, c in enumerate(result.criteria, 1):
        mark = "✓" if c.passed else "✗"
        lines.append(f"| {i} | {c.name} | {c.label} | {mark} |")
    lines.append("")

    if result.intrinsic_value is not None:
        pct = result.price_vs_intrinsic_pct or 0.0
        direction = "above" if pct > 0 else "below"
        lines.append(
            f"**Intrinsic Value (owner earnings DCF):** ${result.intrinsic_value:.2f}  |  "
            f"Price is {abs(pct):.0f}% {direction} IV  (MoS: {-pct:.0f}%)"
        )
    if result.roic is not None:
        lines.append(f"**ROIC:** {result.roic*100:.1f}%")
    if result.capex_intensity is not None:
        lines.append(f"**Capex intensity:** {result.capex_intensity*100:.1f}% of revenue")

    lines.append(f"**Verdict: {result.verdict}**")
    return "\n".join(lines)


def format_summary(results: Sequence[MungerResult]) -> str:
    lines: list[str] = []
    lines.append("")
    header = (
        f"{'Ticker':<7} {'Company':<28} {'Price':>8}  "
        f"{'Score':>6}  {'MoS%':>6}  {'Verdict'}"
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

        if r.price_vs_intrinsic_pct is not None:
            mos_pct = -r.price_vs_intrinsic_pct
            mos_str = f"{mos_pct:+.0f}%"
            mos_color = _GREEN if mos_pct > 0 else _RED
        else:
            mos_str = "N/A"
            mos_color = _DIM

        name = (r.company_name[:26] + "..") if len(r.company_name) > 28 else r.company_name

        row = (
            f"{r.ticker:<7} {name:<28} ${r.price:>7.2f}  "
            f"{score_color}{score_str:>6}{_RESET}  "
            f"{mos_color}{mos_str:>6}{_RESET}  "
            f"{_verdict_str(r.verdict)}"
        )
        lines.append(row)

    lines.append("")
    munger_grade = [r for r in results if r.verdict == "Munger-Grade Business"]
    strong       = [r for r in results if r.verdict == "Strong Candidate"]
    lines.append(
        f"  {_BOLD}{len(munger_grade)} Munger-grade business(es){_RESET}, "
        f"{_BOLD}{len(strong)} strong candidate(s){_RESET} "
        f"out of {len(results)} stocks screened."
    )
    lines.append("")
    return "\n".join(lines)


def print_results(results: Sequence[MungerResult], output: str = "terminal") -> None:
    for result in results:
        if output == "json":
            print(format_json(result))
        elif output == "markdown":
            print(format_markdown(result))
        elif output == "summary":
            pass
        else:
            print(format_terminal(result))
