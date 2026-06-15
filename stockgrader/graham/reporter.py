"""
Terminal, JSON, and Markdown reporters for Graham evaluation results.
"""
from __future__ import annotations

import json
from typing import Sequence

from stockgrader.graham.models import CriterionResult, DefensiveResult, EnterprisingResult, GrahamResult

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


def _verdict_str(verdict: str) -> str:
    if verdict == "Qualifies":
        return f"{_GREEN}{_BOLD}✓ QUALIFIES{_RESET}"
    return f"{_RED}{_BOLD}✗ DOES NOT QUALIFY{_RESET}"


def _bar(met: int, total: int) -> str:
    ratio_str = f"{met} / {total} criteria"
    if met == total:
        color = _GREEN
    elif met >= total // 2:
        color = _YELLOW
    else:
        color = _RED
    return f"{color}[{ratio_str}]{_RESET}"


def _fmt_criterion(c: CriterionResult, indent: int = 2) -> str:
    pad = " " * indent
    check = _check(c.passed)
    # Align name in 22 chars
    name_col = (c.name + " ").ljust(22, "·")
    line = f"{pad}{check} {name_col} {c.label}"
    if c.note:
        line += f"\n{pad}  {_DIM}↳ {c.note}{_RESET}"
    return line


def _fmt_graham_number(result: DefensiveResult, price: float) -> str:
    gn = result.graham_number
    if gn is None:
        return f"  {_DIM}Graham Number: N/A (insufficient data){_RESET}"
    pct = result.price_vs_graham_pct
    direction = "above" if pct and pct > 0 else "below"
    pct_str = f"{abs(pct):.0f}%" if pct is not None else "N/A"
    color = _RED if (pct or 0) > 0 else _GREEN
    return (
        f"  Graham Number: {_BOLD}${gn:.2f}{_RESET}  |  "
        f"Price (${price:.2f}) is {color}{pct_str} {direction} Graham Number{_RESET}"
    )


def _fmt_ncav(result: EnterprisingResult, price: float) -> str:
    ncav = result.ncav_per_share
    if ncav is None:
        return f"  {_DIM}NCAV/share: N/A (insufficient data){_RESET}"
    pct = result.price_vs_ncav_pct
    if ncav < 0:
        return f"  NCAV/share: {_RED}${ncav:.2f}{_RESET}  (negative — liabilities exceed current assets)"
    pct_str = f"{abs(pct):.0f}%" if pct is not None else "N/A"
    if price <= ncav:
        status = f"{_GREEN}Price ≤ NCAV — qualifies for net-net screen{_RESET}"
        if price <= ncav * (2 / 3):
            status = f"{_GREEN}Price ≤ ⅔ NCAV — deep value net-net{_RESET}"
    else:
        status = f"Price is {_RED}{pct_str} above NCAV{_RESET}"
    return f"  NCAV/share: {_BOLD}${ncav:.2f}{_RESET}  |  {status}"


def _wide_sep(char: str = "═", width: int = 56) -> str:
    return char * width


def format_terminal(result: GrahamResult) -> str:
    lines: list[str] = []
    price = result.price

    # Header
    lines.append("")
    header = f"{_BOLD}{result.ticker}{_RESET} — Graham Evaluation ({result.as_of})  Price: {_BOLD}${price:.2f}{_RESET}"
    if result.company_name and result.company_name != result.ticker:
        header += f"  {_DIM}{result.company_name}{_RESET}"
    if result.eps_source != "yfinance":
        header += f"  {_DIM}[EPS: {result.eps_source}]{_RESET}"
    lines.append(header)
    lines.append(_wide_sep("═"))

    # ── Defensive ────────────────────────────────────────────────────────────
    d = result.defensive
    lines.append(f"{_BOLD}DEFENSIVE INVESTOR{_RESET}  {_wide_sep('─', 20)}  {_bar(d.criteria_met, d.total_criteria)}")
    lines.append(_wide_sep("─"))

    for c in d.criteria:
        # P/E × P/B is a sub-row — indent it extra
        if c.name == "P/E × P/B":
            pad = " " * 4
            check = _check(c.passed)
            name_col = (c.name + " ").ljust(22, "·")
            lines.append(f"{pad}{check} {name_col} {c.label}  {_DIM}(combined constraint){_RESET}")
        else:
            lines.append(_fmt_criterion(c))

    lines.append(_fmt_graham_number(d, price))
    verdict_line = f"  Verdict: {_verdict_str(d.verdict)}"
    lines.append(verdict_line)

    lines.append("")

    # ── Enterprising ─────────────────────────────────────────────────────────
    e = result.enterprising
    lines.append(f"{_BOLD}ENTERPRISING INVESTOR{_RESET}  {_wide_sep('─', 18)}  {_bar(e.criteria_met, e.total_criteria)}")
    lines.append(_wide_sep("─"))

    for c in e.criteria:
        lines.append(_fmt_criterion(c))

    lines.append(_fmt_ncav(e, price))
    lines.append(f"  Verdict: {_verdict_str(e.verdict)}")
    lines.append("")

    return "\n".join(lines)


def format_json(result: GrahamResult) -> str:
    return json.dumps(result.to_dict(), indent=2, default=str)


def format_markdown(result: GrahamResult) -> str:
    price = result.price
    lines: list[str] = []

    lines.append(f"# {result.ticker} — Graham Evaluation ({result.as_of})")
    lines.append(f"**Price:** ${price:.2f}  |  **Company:** {result.company_name}")
    lines.append("")

    # Defensive
    d = result.defensive
    lines.append(f"## Defensive Investor  [{d.criteria_met} / {d.total_criteria} criteria]")
    lines.append("")
    lines.append("| # | Criterion | Value | Pass? |")
    lines.append("|---|-----------|-------|-------|")
    for i, c in enumerate(d.criteria, 1):
        mark = "✓" if c.passed else "✗"
        lines.append(f"| {i} | {c.name} | {c.label} | {mark} |")
    lines.append("")
    gn = d.graham_number
    if gn:
        pct = d.price_vs_graham_pct
        direction = "above" if (pct or 0) > 0 else "below"
        lines.append(f"**Graham Number:** ${gn:.2f}  |  Price is {abs(pct or 0):.0f}% {direction} Graham Number")
    lines.append(f"**Verdict: {d.verdict}**")
    lines.append("")

    # Enterprising
    e = result.enterprising
    lines.append(f"## Enterprising Investor  [{e.criteria_met} / {e.total_criteria} criteria]")
    lines.append("")
    lines.append("| # | Criterion | Value | Pass? |")
    lines.append("|---|-----------|-------|-------|")
    for i, c in enumerate(e.criteria, 1):
        mark = "✓" if c.passed else "✗"
        lines.append(f"| {i} | {c.name} | {c.label} | {mark} |")
    lines.append("")
    ncav = e.ncav_per_share
    if ncav is not None:
        lines.append(f"**NCAV/share:** ${ncav:.2f}")
    lines.append(f"**Verdict: {e.verdict}**")

    return "\n".join(lines)


def format_summary(results: Sequence[GrahamResult]) -> str:
    """Condensed table: one row per stock, sorted by defensive criteria met desc."""
    lines: list[str] = []
    lines.append("")
    header = (
        f"{'Ticker':<7} {'Company':<28} {'Price':>8}  "
        f"{'Def':>5}  {'Ent':>5}  {'Qualifies'}"
    )
    lines.append(f"{_BOLD}{header}{_RESET}")
    lines.append("─" * 70)

    sorted_results = sorted(
        results,
        key=lambda r: (r.defensive.criteria_met, r.enterprising.criteria_met),
        reverse=True,
    )

    for r in sorted_results:
        d = r.defensive
        e = r.enterprising

        # Color the criteria count
        def _score_color(met: int, total: int) -> str:
            s = f"{met}/{total}"
            if met == total:
                return f"{_GREEN}{s}{_RESET}"
            if met >= total // 2:
                return f"{_YELLOW}{s}{_RESET}"
            return f"{_RED}{s}{_RESET}"

        d_both = d.verdict == "Qualifies" and e.verdict == "Qualifies"
        d_only = d.verdict == "Qualifies"
        e_only = e.verdict == "Qualifies"

        if d_both:
            verdict = f"{_GREEN}Both{_RESET}"
        elif d_only:
            verdict = f"{_GREEN}Defensive{_RESET}"
        elif e_only:
            verdict = f"{_YELLOW}Enterprising{_RESET}"
        else:
            verdict = f"{_DIM}Neither{_RESET}"

        name = (r.company_name[:26] + "..") if len(r.company_name) > 28 else r.company_name
        row = (
            f"{r.ticker:<7} {name:<28} ${r.price:>7.2f}  "
            f"{_score_color(d.criteria_met, d.total_criteria):>5}  "
            f"{_score_color(e.criteria_met, e.total_criteria):>5}  "
            f"{verdict}"
        )
        lines.append(row)

    lines.append("")
    qualifiers = [r for r in results if r.defensive.verdict == "Qualifies" or r.enterprising.verdict == "Qualifies"]
    lines.append(f"  {_BOLD}{len(qualifiers)} qualifier(s){_RESET} out of {len(results)} stocks screened.")
    lines.append("")
    return "\n".join(lines)


def print_results(results: Sequence[GrahamResult], output: str = "terminal") -> None:
    for result in results:
        if output == "json":
            print(format_json(result))
        elif output == "markdown":
            print(format_markdown(result))
        elif output == "summary":
            pass  # handled separately — caller uses format_summary()
        else:
            print(format_terminal(result))
