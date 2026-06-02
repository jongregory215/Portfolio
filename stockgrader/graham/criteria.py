"""
Graham criteria evaluation.

Implements all pass/fail tests for the Defensive and Enterprising investor
exactly as specified in The Intelligent Investor (Chapters 14 & 15).
"""
from __future__ import annotations

import math
from typing import Any

from stockgrader.graham.models import CriterionResult, DefensiveResult, EnterprisingResult


def _fmt_currency(v: float) -> str:
    """Format a dollar amount with appropriate suffix."""
    abs_v = abs(v)
    sign = "-" if v < 0 else ""
    if abs_v >= 1e12:
        return f"{sign}${abs_v / 1e12:.2f}T"
    if abs_v >= 1e9:
        return f"{sign}${abs_v / 1e9:.2f}B"
    if abs_v >= 1e6:
        return f"{sign}${abs_v / 1e6:.2f}M"
    return f"{sign}${abs_v:.2f}"


def _pct(v: float) -> str:
    return f"{v:.1f}%"


def _missing(name: str) -> CriterionResult:
    return CriterionResult(name=name, passed=False, label="N/A", note="Data unavailable")


# ── Defensive Investor ───────────────────────────────────────────────────────


def _d1_size(data: dict) -> CriterionResult:
    mc = data.get("market_cap")
    threshold = 2e9
    if mc is None:
        return _missing("Adequate Size")
    passed = mc >= threshold
    return CriterionResult(
        name="Adequate Size",
        passed=passed,
        label=f"{_fmt_currency(mc)} market cap  (≥ $2B)",
    )


def _d2_current_ratio(data: dict) -> CriterionResult:
    ca = data.get("current_assets")
    cl = data.get("current_liabilities")
    if ca is None or cl is None or cl == 0:
        return _missing("Current Ratio")
    ratio = ca / cl
    passed = ratio >= 2.0
    return CriterionResult(
        name="Current Ratio",
        passed=passed,
        label=f"{ratio:.2f}  (≥ 2.0)",
    )


def _d3_debt_vs_nca(data: dict) -> CriterionResult:
    ca  = data.get("current_assets")
    cl  = data.get("current_liabilities")
    ltd = data.get("long_term_debt")
    if ca is None or cl is None or ltd is None:
        return _missing("Debt vs. NCA")
    nca = ca - cl
    if nca <= 0:
        return CriterionResult(
            name="Debt vs. NCA",
            passed=False,
            label=f"NCA negative ({_fmt_currency(nca)})  (LT debt ≤ NCA required)",
        )
    ratio_pct = (ltd / nca) * 100
    passed = ltd <= nca
    return CriterionResult(
        name="Debt vs. NCA",
        passed=passed,
        label=f"LT debt is {_pct(ratio_pct)} of NCA  (≤ 100%)",
    )


def _d4_earnings_stability(data: dict, years_required: int = 10) -> CriterionResult:
    eps_hist = data.get("annual_eps", [])  # [(year, eps), ...] newest-first
    if not eps_hist:
        return _missing("Earnings Stability (10y)")

    available = len(eps_hist)
    insufficient = available < years_required
    if insufficient:
        return CriterionResult(
            name="Earnings Stability (10y)",
            passed=False,
            label=f"Only {available} year(s) of data  ({years_required} required)",
            note=f"{available} of {years_required} years available",
        )

    last_10 = eps_hist[:years_required]
    negative = [y for y, e in last_10 if e < 0]
    passed = len(negative) == 0
    label = "Positive all 10 years" if passed else f"Deficit in: {', '.join(str(y) for y in negative)}"
    return CriterionResult(name="Earnings Stability (10y)", passed=passed, label=label)


def _d5_dividend_record(data: dict, years_required: int = 20) -> CriterionResult:
    div_years = data.get("dividend_years", set())
    current_year = int(data.get("as_of", "2026")[:4])
    required_years = set(range(current_year - years_required, current_year))

    if not div_years:
        return CriterionResult(
            name="Dividend Record (20y)",
            passed=False,
            label="No dividend history found",
        )

    missing_years = required_years - div_years
    passed = len(missing_years) == 0
    earliest = min(div_years) if div_years else None

    years_with_div = len(required_years & div_years)

    if passed:
        label = f"Paid every year since ≤ {current_year - years_required}"
    else:
        label = f"{years_with_div} of {years_required} required years had dividends"

    return CriterionResult(
        name="Dividend Record (20y)",
        passed=passed,
        label=label,
        note="" if passed else f"Missing payments in {len(missing_years)} year(s) of required window",
    )


def _d6_eps_growth(data: dict) -> CriterionResult:
    eps_hist = data.get("annual_eps", [])
    if len(eps_hist) < 10:
        available = len(eps_hist)
        return CriterionResult(
            name="EPS Growth (10y)",
            passed=False,
            label=f"Only {available} year(s) of data  (10 required for 33% growth test)",
        )

    # Newest-first: index 0 = most recent, index 9 = 10 years ago
    early_avg = sum(e for _, e in eps_hist[7:10]) / 3   # years 8-10 (oldest)
    late_avg  = sum(e for _, e in eps_hist[0:3])  / 3   # years 1-3  (newest)

    if early_avg <= 0:
        return CriterionResult(
            name="EPS Growth (10y)",
            passed=False,
            label=f"Early 3yr avg EPS was negative ({early_avg:.2f}) — growth undefined",
        )

    growth_pct = ((late_avg - early_avg) / early_avg) * 100
    passed = growth_pct >= 33.0
    label = f"{_pct(growth_pct)} over 10 years  (≥ 33%)"
    return CriterionResult(name="EPS Growth (10y)", passed=passed, label=label)


def _d7_valuation(data: dict) -> tuple[CriterionResult, CriterionResult, CriterionResult]:
    """Returns three sub-criteria: P/E, P/B, and P/E × P/B."""
    price = data.get("price")
    bvps  = data.get("book_value_per_share")
    eps_hist = data.get("annual_eps", [])

    # ─ 7a: P/E using 3-year average EPS ─
    if price is None or len(eps_hist) < 3:
        pe_result = _missing("P/E (3yr avg EPS)")
        pe_val = None
    else:
        eps_3yr = sum(e for _, e in eps_hist[:3]) / 3
        if eps_3yr <= 0:
            pe_result = CriterionResult(
                name="P/E (3yr avg EPS)",
                passed=False,
                label=f"3yr avg EPS non-positive ({eps_3yr:.2f}) — P/E undefined",
            )
            pe_val = None
        else:
            pe_val = price / eps_3yr
            pe_result = CriterionResult(
                name="P/E (3yr avg EPS)",
                passed=pe_val <= 15.0,
                label=f"{pe_val:.1f}×  (≤ 15×)",
            )

    # ─ 7b: P/B ─
    if price is None or bvps is None or bvps <= 0:
        pb_result = _missing("Price / Book Value")
        pb_val = None
    else:
        pb_val = price / bvps
        pb_result = CriterionResult(
            name="Price / Book Value",
            passed=pb_val <= 1.5,
            label=f"{pb_val:.1f}×  (≤ 1.5×)",
        )

    # ─ 7c: P/E × P/B product ─
    if pe_val is not None and pb_val is not None:
        product = pe_val * pb_val
        product_result = CriterionResult(
            name="P/E × P/B",
            passed=product <= 22.5,
            label=f"{product:.1f}  (≤ 22.5)",
            note="Graham's combined valuation constraint",
        )
    else:
        product_result = _missing("P/E × P/B")

    return pe_result, pb_result, product_result


def _graham_number(data: dict) -> float | None:
    eps_hist = data.get("annual_eps", [])
    bvps = data.get("book_value_per_share")
    if len(eps_hist) < 3 or bvps is None or bvps <= 0:
        return None
    eps_3yr = sum(e for _, e in eps_hist[:3]) / 3
    if eps_3yr <= 0:
        return None
    return math.sqrt(22.5 * eps_3yr * bvps)


def evaluate_defensive(data: dict) -> DefensiveResult:
    notes: list[str] = []

    c1 = _d1_size(data)
    c2 = _d2_current_ratio(data)
    c3 = _d3_debt_vs_nca(data)
    c4 = _d4_earnings_stability(data)
    c5 = _d5_dividend_record(data)
    c6 = _d6_eps_growth(data)
    c7a, c7b, c7c = _d7_valuation(data)

    # The 7 Graham criteria: 7a + 7b + 7c all count as criterion #7
    # We display all three but gate pass on all three being true
    c7_passed = c7a.passed and c7b.passed and c7c.passed

    criteria = [c1, c2, c3, c4, c5, c6, c7a, c7b, c7c]

    # Count: criteria 1-6 individually, criterion 7 as one gate
    individual = [c1.passed, c2.passed, c3.passed, c4.passed, c5.passed, c6.passed, c7_passed]
    criteria_met = sum(individual)

    gn = _graham_number(data)
    price = data.get("price")
    gn_pct = None
    if gn and price:
        gn_pct = ((price - gn) / gn) * 100

    verdict = "Qualifies" if all(individual) else "Does Not Qualify"

    # Surface data-gap notes
    for c in criteria:
        if c.note:
            notes.append(c.note)

    return DefensiveResult(
        criteria=criteria,
        criteria_met=criteria_met,
        total_criteria=7,
        graham_number=gn,
        price_vs_graham_pct=gn_pct,
        verdict=verdict,
        notes=notes,
    )


# ── Enterprising Investor ────────────────────────────────────────────────────


def _e1_financial_condition(data: dict) -> CriterionResult:
    ca  = data.get("current_assets")
    cl  = data.get("current_liabilities")
    ltd = data.get("long_term_debt")
    if ca is None or cl is None:
        return _missing("Financial Condition")

    ratio = ca / cl if cl else 0.0
    ratio_ok = ratio >= 1.5

    nca = ca - (cl or 0)
    if ltd is not None and nca > 0:
        debt_pct = (ltd / nca) * 100
        debt_ok = debt_pct <= 110.0
        label = f"Ratio {ratio:.2f} / Debt {_pct(debt_pct)} of NCA  (≥1.5 / ≤110%)"
        passed = ratio_ok and debt_ok
    elif ltd is not None and nca <= 0:
        label = f"Ratio {ratio:.2f} / NCA negative  (≥1.5 / ≤110%)"
        passed = False
    else:
        label = f"Ratio {ratio:.2f}  (≥ 1.5)"
        passed = ratio_ok

    return CriterionResult(name="Financial Condition", passed=passed, label=label)


def _e2_earnings_stability(data: dict) -> CriterionResult:
    eps_hist = data.get("annual_eps", [])
    if not eps_hist:
        return _missing("Earnings Stability (5y)")

    last_5 = eps_hist[:5]
    available = len(last_5)
    if available < 5:
        return CriterionResult(
            name="Earnings Stability (5y)",
            passed=False,
            label=f"Only {available} year(s) of data  (5 required)",
        )

    deficits = [y for y, e in last_5 if e < 0]
    passed = len(deficits) == 0
    label = "No deficits in last 5 years" if passed else f"Deficit in: {', '.join(str(y) for y in deficits)}"
    return CriterionResult(name="Earnings Stability (5y)", passed=passed, label=label)


def _e3_dividend(data: dict) -> CriterionResult:
    div_years = data.get("dividend_years", set())
    current_year = int(data.get("as_of", "2026")[:4])
    recent_years = {current_year, current_year - 1}
    has_recent = bool(div_years & recent_years)
    label = "Current dividend paid" if has_recent else "No dividend in last 2 years"
    return CriterionResult(name="Dividend (current)", passed=has_recent, label=label)


def _e4_eps_growth(data: dict) -> CriterionResult:
    eps_hist = data.get("annual_eps", [])
    if len(eps_hist) < 2:
        return _missing("EPS Growth (YoY)")
    yr0, eps0 = eps_hist[0]
    yr1, eps1 = eps_hist[1]
    passed = eps0 > eps1
    label = f"${eps0:.2f} ({yr0}) > ${eps1:.2f} ({yr1})" if passed else f"${eps0:.2f} ({yr0}) ≤ ${eps1:.2f} ({yr1})"
    return CriterionResult(name="EPS Growth (YoY)", passed=passed, label=label)


def _e5_price_to_book(data: dict) -> CriterionResult:
    price = data.get("price")
    bvps  = data.get("book_value_per_share")
    if price is None or bvps is None or bvps <= 0:
        return _missing("Price / Book Value")
    pb = price / bvps
    passed = pb < 1.2
    return CriterionResult(
        name="Price / Book Value",
        passed=passed,
        label=f"{pb:.1f}×  (< 1.2×)",
    )


def _ncav(data: dict) -> float | None:
    ca = data.get("current_assets")
    tl = data.get("total_liabilities")
    shares = data.get("shares")
    if ca is None or tl is None or not shares or shares <= 0:
        return None
    return (ca - tl) / shares


def evaluate_enterprising(data: dict) -> EnterprisingResult:
    notes: list[str] = []

    c1 = _e1_financial_condition(data)
    c2 = _e2_earnings_stability(data)
    c3 = _e3_dividend(data)
    c4 = _e4_eps_growth(data)
    c5 = _e5_price_to_book(data)

    criteria = [c1, c2, c3, c4, c5]
    criteria_met = sum(c.passed for c in criteria)

    ncav_ps = _ncav(data)
    price   = data.get("price")
    ncav_pct = None
    if ncav_ps is not None and price is not None and ncav_ps != 0:
        ncav_pct = ((price - ncav_ps) / abs(ncav_ps)) * 100

    verdict = "Qualifies" if all(c.passed for c in criteria) else "Does Not Qualify"

    for c in criteria:
        if c.note:
            notes.append(c.note)

    return EnterprisingResult(
        criteria=criteria,
        criteria_met=criteria_met,
        total_criteria=5,
        ncav_per_share=ncav_ps,
        price_vs_ncav_pct=ncav_pct,
        verdict=verdict,
        notes=notes,
    )
