"""
Warren Buffett criteria evaluation.

Operationalizes Buffett's "wonderful company at a fair price" framework from
his shareholder letters, Poor Charlie's Almanack, and Buffettology:
durable competitive advantage (moat), high returns on capital, consistent and
growing earnings, owner-earnings quality, conservative financing, and a
meaningful margin of safety to intrinsic value.

Results are graded (criteria_met / total) rather than binary, since Buffett
explicitly weighs business quality and does not use mechanical formulas.
"""
from __future__ import annotations

from stockgrader.data.fred_provider import FREDProvider
from stockgrader.warren_buffett.models import BuffettResult, CriterionResult

_EQUITY_RISK_PREMIUM = 0.055   # consistent with marks criteria
_TAX_RATE            = 0.21
_MOS_MULTIPLIER      = 0.85    # require price ≤ IV × 0.85 (15% margin of safety)
_MAX_GROWTH_RATE     = 0.06    # cap perpetuity growth at 6%


def _missing(name: str) -> CriterionResult:
    return CriterionResult(name=name, passed=False, label="N/A", note="Data unavailable")


def _pct(v: float) -> str:
    return f"{v:.1f}%"


# ── 1. Durable Moat — Return on Equity ───────────────────────────────────────

def _c1_roe(data: dict) -> CriterionResult:
    roe = data.get("roe")
    if roe is None:
        return _missing("Durable Moat (Return on Equity)")

    passed = roe >= 0.15
    return CriterionResult(
        name="Durable Moat (Return on Equity)",
        passed=passed,
        label=f"{roe*100:.1f}%  (≥ 15% required)",
        note="High sustained ROE signals a durable competitive advantage.",
    )


# ── 2. Economic Value Creation — ROIC vs. Required Return ────────────────────

def _compute_roic(data: dict) -> tuple[float | None, float | None, float | None]:
    """Return (roic, required_return, spread) or (None, None, None)."""
    oi = data.get("operating_income")
    ta = data.get("total_assets")
    cl = data.get("current_liabilities")
    beta = data.get("beta") or 1.0

    if oi is None or ta is None or cl is None:
        return None, None, None

    invested_capital = ta - cl
    if invested_capital <= 0:
        return None, None, None

    nopat = oi * (1 - _TAX_RATE)
    roic = nopat / invested_capital

    rf = FREDProvider().get_risk_free_rate("10yr")
    required_return = rf + beta * _EQUITY_RISK_PREMIUM
    spread = roic - required_return

    return roic, required_return, spread


def _c2_roic_vs_wacc(data: dict) -> tuple[CriterionResult, float | None]:
    roic, req, spread = _compute_roic(data)
    if roic is None:
        return _missing("Economic Value Creation (ROIC vs. Required Return)"), None

    passed = spread > 0
    direction = "above" if spread >= 0 else "below"
    return CriterionResult(
        name="Economic Value Creation (ROIC vs. Required Return)",
        passed=passed,
        label=(
            f"ROIC {roic*100:.1f}% vs required return {req*100:.1f}%  "
            f"→ {abs(spread)*100:.1f} ppts {direction} hurdle"
        ),
        note="ROIC > required return means the business compounds capital above its cost.",
    ), spread


# ── 3. Pricing Power — Gross Margin ──────────────────────────────────────────

def _c3_gross_margin(data: dict) -> CriterionResult:
    gm = data.get("gross_margin")
    if gm is None:
        return _missing("Pricing Power (Gross Margin)")

    passed = gm >= 0.40
    return CriterionResult(
        name="Pricing Power (Gross Margin)",
        passed=passed,
        label=f"{gm*100:.1f}%  (≥ 40% required)",
        note="High gross margins indicate pricing power and a defensible market position.",
    )


# ── 4. Earnings Consistency ───────────────────────────────────────────────────

def _c4_earnings_consistency(data: dict) -> CriterionResult:
    eps_hist = data.get("annual_eps", [])
    if not eps_hist:
        return _missing("Earnings Consistency")

    loss_years = [y for y, e in eps_hist if e < 0]
    passed = len(loss_years) == 0
    label = (
        f"Profitable in all {len(eps_hist)} available year(s)"
        if passed
        else f"Loss year(s): {', '.join(str(y) for y in loss_years)}"
    )
    return CriterionResult(
        name="Earnings Consistency",
        passed=passed,
        label=label,
        note="Buffett avoids businesses with erratic or cyclical earnings.",
    )


# ── 5. Earnings Growth — EPS CAGR ────────────────────────────────────────────

def _eps_cagr(annual_eps: list, years: int) -> float | None:
    """CAGR over `years` intervals from a newest-first (year, eps) list."""
    if len(annual_eps) <= years:
        return None
    eps_new = annual_eps[0][1]
    eps_old = annual_eps[years][1]
    if eps_old <= 0 or eps_new <= 0:
        return None
    return (eps_new / eps_old) ** (1.0 / years) - 1.0


def _c5_earnings_growth(data: dict) -> CriterionResult:
    eps_hist = data.get("annual_eps", [])
    if not eps_hist:
        return _missing("Earnings Growth (EPS CAGR)")

    cagr = _eps_cagr(eps_hist, 5)
    periods = 5
    if cagr is None:
        cagr = _eps_cagr(eps_hist, 3)
        periods = 3
    if cagr is None:
        return CriterionResult(
            name="Earnings Growth (EPS CAGR)",
            passed=False,
            label="Cannot compute (negative base EPS or < 4 years of data)",
            note="Need at least 4 years of positive EPS to calculate growth.",
        )

    passed = cagr > 0.05
    return CriterionResult(
        name="Earnings Growth (EPS CAGR)",
        passed=passed,
        label=f"{cagr*100:.1f}% {periods}-yr CAGR  (> 5% required)",
        note="Consistent EPS growth compounds intrinsic value over time.",
    )


# ── 6. Owner Earnings Quality — OCF / Net Income ─────────────────────────────

def _c6_owner_earnings_quality(data: dict) -> CriterionResult:
    ocf    = data.get("operating_cf")
    shares = data.get("shares")

    # Net income: prefer trailing_eps * shares, fall back to latest annual EPS * shares
    eps_ttm    = data.get("trailing_eps")
    annual_eps = data.get("annual_eps", [])
    eps = eps_ttm or (annual_eps[0][1] if annual_eps else None)

    if ocf is None:
        return _missing("Owner Earnings Quality (OCF / Net Income)")
    if eps is None or shares is None or shares <= 0:
        return _missing("Owner Earnings Quality (OCF / Net Income)")

    net_income = eps * shares
    if abs(net_income) < 1:
        return CriterionResult(
            name="Owner Earnings Quality (OCF / Net Income)",
            passed=False,
            label="Net income near zero — ratio undefined",
        )

    ratio = ocf / net_income
    passed = ratio > 0.75

    def _fmt(v: float) -> str:
        billions = v / 1e9
        return f"${billions:.1f}B" if abs(billions) >= 1 else f"${v/1e6:.0f}M"

    return CriterionResult(
        name="Owner Earnings Quality (OCF / Net Income)",
        passed=passed,
        label=(
            f"OCF {_fmt(ocf)} / NI {_fmt(net_income)}  = {ratio:.2f}  (> 0.75 required)"
        ),
        note="High OCF/NI ratio means reported earnings are backed by real cash.",
    )


# ── 7. Conservative Financing — LT Debt vs. FCF ──────────────────────────────

def _c7_conservative_financing(data: dict) -> CriterionResult:
    ltd = data.get("long_term_debt")
    fcf = data.get("free_cf")

    if ltd is None or fcf is None:
        return _missing("Conservative Financing (LT Debt / FCF)")

    def _fmt(v: float) -> str:
        billions = v / 1e9
        return f"${billions:.1f}B" if abs(billions) >= 1 else f"${v/1e6:.0f}M"

    if fcf <= 0:
        return CriterionResult(
            name="Conservative Financing (LT Debt / FCF)",
            passed=False,
            label=f"LT debt {_fmt(ltd)}, FCF {_fmt(fcf)} — negative FCF cannot service debt",
        )

    ratio = ltd / fcf
    passed = ratio < 5.0
    return CriterionResult(
        name="Conservative Financing (LT Debt / FCF)",
        passed=passed,
        label=f"LT debt {_fmt(ltd)} / FCF {_fmt(fcf)}  = {ratio:.1f}×  (< 5× required)",
        note="Buffett prefers businesses that could retire all debt within ~5 years of FCF.",
    )


# ── 8. Margin of Safety — Intrinsic Value (Owner Earnings DCF) ───────────────

def _compute_intrinsic_value(data: dict) -> float | None:
    """
    Buffett-style IV: owner_earnings / (discount_rate - growth_rate).

    Owner earnings ≈ FCF. Falls back to 3yr avg EPS × shares when FCF is
    unavailable or negative.
    """
    shares = data.get("shares")
    if not shares or shares <= 0:
        return None

    # Owner earnings (total, not per-share)
    owner_earnings = data.get("free_cf")
    if owner_earnings is None or owner_earnings <= 0:
        annual_eps = data.get("annual_eps", [])
        if len(annual_eps) >= 3:
            eps_3yr = sum(e for _, e in annual_eps[:3]) / 3
            if eps_3yr > 0:
                owner_earnings = eps_3yr * shares
    if owner_earnings is None or owner_earnings <= 0:
        return None

    # Conservative growth rate: half of 5yr EPS CAGR, capped at 6%
    annual_eps = data.get("annual_eps", [])
    cagr = _eps_cagr(annual_eps, 5) or _eps_cagr(annual_eps, 3)
    growth_rate = min(max((cagr or 0.0) / 2, 0.0), _MAX_GROWTH_RATE)

    # Discount rate = 10yr risk-free + equity risk premium
    rf = FREDProvider().get_risk_free_rate("10yr")
    beta = data.get("beta") or 1.0
    discount_rate = rf + beta * _EQUITY_RISK_PREMIUM

    # Guard against discount_rate ≤ growth_rate (perpetuity formula undefined)
    if discount_rate <= growth_rate:
        discount_rate = growth_rate + 0.02

    iv_total = owner_earnings * (1 + growth_rate) / (discount_rate - growth_rate)
    return iv_total / shares


def _c8_margin_of_safety(data: dict) -> tuple[CriterionResult, float | None]:
    price = data.get("price")
    if price is None:
        return _missing("Margin of Safety (Intrinsic Value)"), None

    iv = _compute_intrinsic_value(data)
    if iv is None:
        return CriterionResult(
            name="Margin of Safety (Intrinsic Value)",
            passed=False,
            label="Cannot compute (insufficient FCF / EPS data)",
            note="Need positive FCF or 3yr+ EPS history to estimate intrinsic value.",
        ), None

    threshold = iv * _MOS_MULTIPLIER
    passed    = price <= threshold
    pct       = (price - iv) / iv * 100.0
    direction = "above" if pct > 0 else "below"

    return CriterionResult(
        name="Margin of Safety (Intrinsic Value)",
        passed=passed,
        label=(
            f"Price ${price:.2f} is {abs(pct):.0f}% {direction} intrinsic value (${iv:.2f}); "
            f"pass if price ≤ ${threshold:.2f} (IV × {_MOS_MULTIPLIER:.2f})"
        ),
        note="IV = owner earnings / (discount rate − growth rate). Growth capped at 6%, discount = rf + β×ERP.",
    ), iv


# ── Main evaluator ────────────────────────────────────────────────────────────

def evaluate_buffett(data: dict) -> BuffettResult:
    c1           = _c1_roe(data)
    c2, roic_spread = _c2_roic_vs_wacc(data)
    c3           = _c3_gross_margin(data)
    c4           = _c4_earnings_consistency(data)
    c5           = _c5_earnings_growth(data)
    c6           = _c6_owner_earnings_quality(data)
    c7           = _c7_conservative_financing(data)
    c8, iv       = _c8_margin_of_safety(data)

    criteria     = [c1, c2, c3, c4, c5, c6, c7, c8]
    criteria_met = sum(c.passed for c in criteria)
    total        = len(criteria)

    if criteria_met >= 7:
        verdict = "Exceptional Business at a Fair Price"
    elif criteria_met >= 5:
        verdict = "Strong Candidate"
    else:
        verdict = "Pass"

    price = data.get("price") or 0.0
    price_vs_iv: float | None = None
    if iv is not None and iv != 0:
        price_vs_iv = (price - iv) / iv * 100.0

    return BuffettResult(
        ticker=data["ticker"],
        as_of=data["as_of"],
        price=price,
        company_name=data.get("company_name", data["ticker"]),
        criteria=criteria,
        criteria_met=criteria_met,
        total_criteria=total,
        verdict=verdict,
        intrinsic_value=iv,
        price_vs_intrinsic_pct=price_vs_iv,
        roe=data.get("roe"),
        roic_spread=roic_spread,
    )
