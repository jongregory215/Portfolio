"""
Howard Marks criteria evaluation.

Operationalizes themes from "The Most Important Thing" and "Mastering the
Market Cycle": avoiding permanent capital loss, durability through cycles,
the pendulum of sentiment, price vs. estimated value (Earnings Power Value),
balance-sheet resilience, and risk/reward asymmetry.

Unlike Graham's strict all-must-pass checklist, Marks distrusts mechanical
formulas — so results are reported as graded bands (criteria_met / total)
rather than a binary qualify/disqualify verdict.
"""
from __future__ import annotations

from stockgrader.data.normalizer import compute_altman_z
from stockgrader.data.fred_provider import FREDProvider
from stockgrader.howard_marks.models import CriterionResult, CycleReading, MarksResult

_EQUITY_RISK_PREMIUM = 0.055   # matches config.yaml fair_value.dcf.equity_risk_premium
_INTEREST_COVERAGE_MIN = 3.0
_RANGE_POSITION_MAX = 40.0      # pendulum criterion: pass if in bottom 40% of 52wk range


def _missing(name: str) -> CriterionResult:
    return CriterionResult(name=name, passed=False, label="N/A", note="Data unavailable")


def _pct(v: float) -> str:
    return f"{v:.1f}%"


# ── 1. Avoid the Losers — Financial Strength (Altman Z) ──────────────────────

def _c1_altman_z(data: dict) -> CriterionResult:
    z = compute_altman_z(data)
    if z is None:
        return _missing("Avoid the Losers (Altman Z)")

    if z >= 2.99:
        zone = "Safe zone"
    elif z >= 1.81:
        zone = "Grey zone"
    elif z >= 1.10:
        zone = "Distress zone"
    else:
        zone = "Bankruptcy zone"

    passed = z >= 1.81
    return CriterionResult(
        name="Avoid the Losers (Altman Z)",
        passed=passed,
        label=f"Z = {z:.2f}  ({zone}, pass requires ≥ 1.81)",
    )


# ── 2. Earnings Durability Through the Cycle ──────────────────────────────────

def _c2_earnings_durability(data: dict) -> CriterionResult:
    eps_hist = data.get("annual_eps", [])
    if not eps_hist:
        return _missing("Earnings Durability Through the Cycle")

    negative = [y for y, e in eps_hist if e < 0]
    passed = len(negative) == 0
    label = (
        f"Profitable in all {len(eps_hist)} available year(s)"
        if passed
        else f"Loss in: {', '.join(str(y) for y in negative)}"
    )
    return CriterionResult(name="Earnings Durability Through the Cycle", passed=passed, label=label)


# ── 3. Pendulum Position — 52-Week Range ───────────────────────────────────────

def _range_position_pct(data: dict) -> float | None:
    price = data.get("price")
    hi    = data.get("fifty_two_week_high")
    lo    = data.get("fifty_two_week_low")
    if price is None or hi is None or lo is None or hi <= lo:
        return None
    return (price - lo) / (hi - lo) * 100.0


def _c3_pendulum_position(data: dict, range_pos: float | None) -> CriterionResult:
    if range_pos is None:
        return _missing("Pendulum Position (52-Week Range)")

    passed = range_pos <= _RANGE_POSITION_MAX
    return CriterionResult(
        name="Pendulum Position (52-Week Range)",
        passed=passed,
        label=f"{range_pos:.0f}% of 52-week range  (pass if ≤ {_RANGE_POSITION_MAX:.0f}%)",
        note="Lower = closer to its 52-week low — sentiment has turned negative.",
    )


# ── 4. Sentiment vs. Fundamentals ──────────────────────────────────────────────

def _c4_sentiment_vs_fundamentals(data: dict) -> CriterionResult:
    price = data.get("price")
    hi    = data.get("fifty_two_week_high")
    eps_hist = data.get("annual_eps", [])

    if price is None or hi is None or hi <= 0 or len(eps_hist) < 3:
        return _missing("Sentiment vs. Fundamentals")

    price_decline_pct = (hi - price) / hi * 100.0

    # Normalized earnings: most-recent 3yr avg vs. oldest available 3yr avg
    recent_avg = sum(e for _, e in eps_hist[:3]) / 3
    oldest_avg = sum(e for _, e in eps_hist[-3:]) / 3

    if oldest_avg == 0:
        return _missing("Sentiment vs. Fundamentals")

    earnings_change_pct = (recent_avg - oldest_avg) / abs(oldest_avg) * 100.0
    earnings_decline_pct = max(0.0, -earnings_change_pct)

    passed = price_decline_pct > earnings_decline_pct
    label = (
        f"Price down {_pct(price_decline_pct)} from 52wk high vs. "
        f"earnings change of {earnings_change_pct:+.1f}%"
    )
    note = (
        "Price has fallen more than earnings power — the market may be pricing in "
        "more pessimism than the business has delivered."
        if passed else
        "Price decline does not exceed the decline in earnings power."
    )
    return CriterionResult(name="Sentiment vs. Fundamentals", passed=passed, label=label, note=note)


# ── 5. Margin of Safety — Earnings Power Value (EPV) ───────────────────────────

def _cost_of_equity(data: dict) -> float:
    beta = data.get("beta")
    if beta is None:
        beta = 1.0
    rf = FREDProvider().get_risk_free_rate("10yr")
    return rf + beta * _EQUITY_RISK_PREMIUM


def _compute_epv(data: dict) -> float | None:
    eps_hist = data.get("annual_eps", [])
    if len(eps_hist) < 3:
        return None
    eps_3yr = sum(e for _, e in eps_hist[:3]) / 3
    if eps_3yr <= 0:
        return None
    coe = _cost_of_equity(data)
    if coe <= 0:
        return None
    return eps_3yr / coe


def _c5_margin_of_safety(data: dict, epv: float | None, mos_multiplier: float) -> CriterionResult:
    price = data.get("price")
    if epv is None or price is None:
        return _missing("Margin of Safety (Earnings Power Value)")

    threshold = epv * mos_multiplier
    passed = price <= threshold
    pct = (price - epv) / epv * 100.0

    direction = "above" if pct > 0 else "below"
    label = (
        f"Price ${price:.2f} is {abs(pct):.0f}% {direction} EPV (${epv:.2f}); "
        f"pass if price ≤ ${threshold:.2f} (EPV × {mos_multiplier:.2f})"
    )
    return CriterionResult(name="Margin of Safety (Earnings Power Value)", passed=passed, label=label)


# ── 6. Balance Sheet Resilience — Interest Coverage ────────────────────────────

def _c6_interest_coverage(data: dict) -> CriterionResult:
    oi = data.get("operating_income")
    ie = data.get("interest_expense")

    if oi is None:
        return _missing("Balance Sheet Resilience (Interest Coverage)")

    if ie is None or abs(ie) < 1e-6:
        return CriterionResult(
            name="Balance Sheet Resilience (Interest Coverage)",
            passed=True,
            label="No meaningful interest expense — survives any downturn",
        )

    coverage = oi / abs(ie)
    passed = coverage >= _INTEREST_COVERAGE_MIN
    return CriterionResult(
        name="Balance Sheet Resilience (Interest Coverage)",
        passed=passed,
        label=f"{coverage:.1f}×  (≥ {_INTEREST_COVERAGE_MIN:.0f}× required)",
    )


# ── 7. Risk/Reward Asymmetry ───────────────────────────────────────────────────

def _c7_risk_reward(data: dict, epv: float | None) -> tuple[CriterionResult, float | None]:
    price = data.get("price")
    lo    = data.get("fifty_two_week_low")

    if epv is None or price is None or lo is None or price <= 0:
        return _missing("Risk/Reward Asymmetry"), None

    upside_pct   = max(0.0, (epv - price) / price)
    downside_pct = max(0.0, (price - lo) / price)

    if downside_pct == 0:
        return CriterionResult(
            name="Risk/Reward Asymmetry",
            passed=True,
            label="Already at/near 52-week low — limited further downside by recent range",
        ), None

    ratio = upside_pct / downside_pct
    passed = ratio >= 1.0
    label = (
        f"Upside to EPV {_pct(upside_pct * 100)} vs. downside to 52wk low {_pct(downside_pct * 100)}  "
        f"→ ratio {ratio:.2f}  (pass if ≥ 1.0)"
    )
    return CriterionResult(name="Risk/Reward Asymmetry", passed=passed, label=label), ratio


# ── Main entry point ───────────────────────────────────────────────────────────

def evaluate_marks(data: dict, cycle: CycleReading | None = None) -> MarksResult:
    mos_multiplier = cycle.mos_multiplier if cycle is not None else 1.00

    range_pos = _range_position_pct(data)
    epv = _compute_epv(data)

    c1 = _c1_altman_z(data)
    c2 = _c2_earnings_durability(data)
    c3 = _c3_pendulum_position(data, range_pos)
    c4 = _c4_sentiment_vs_fundamentals(data)
    c5 = _c5_margin_of_safety(data, epv, mos_multiplier)
    c6 = _c6_interest_coverage(data)
    c7, reward_risk_ratio = _c7_risk_reward(data, epv)

    criteria = [c1, c2, c3, c4, c5, c6, c7]
    criteria_met = sum(c.passed for c in criteria)
    total = len(criteria)

    if criteria_met >= 6:
        verdict = "Compelling Opportunity"
    elif criteria_met >= 4:
        verdict = "Worth a Closer Look"
    else:
        verdict = "Pass"

    price = data.get("price")
    price_vs_epv_pct = None
    if epv is not None and price is not None and epv != 0:
        price_vs_epv_pct = (price - epv) / epv * 100.0

    notes = [c.note for c in criteria if c.note]

    return MarksResult(
        ticker=data["ticker"],
        as_of=data["as_of"],
        price=price,
        company_name=data.get("company_name", data["ticker"]),
        criteria=criteria,
        criteria_met=criteria_met,
        total_criteria=total,
        verdict=verdict,
        epv=epv,
        price_vs_epv_pct=price_vs_epv_pct,
        range_position_pct=range_pos,
        reward_risk_ratio=reward_risk_ratio,
        notes=notes,
    )
