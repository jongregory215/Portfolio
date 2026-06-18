"""
Charlie Munger criteria evaluation.

Operationalizes Munger's "Poor Charlie's Almanack" and Berkshire meeting
philosophy: exceptional (not merely adequate) returns on capital, asset-light
models that compound without heavy reinvestment, fortress-level pricing power,
extreme debt aversion, and a larger margin of safety than Buffett's baseline.

Like the Buffett screen, results are graded. Thresholds are deliberately
stricter — Munger's filter is designed to find the rare "inevitable" franchise,
not a merely good business.
"""
from __future__ import annotations

from stockgrader.data.fred_provider import FREDProvider
from stockgrader.charlie_munger.models import CriterionResult, MungerResult

_EQUITY_RISK_PREMIUM  = 0.055
_TAX_RATE             = 0.21
_MOS_MULTIPLIER       = 0.80   # require 20% margin of safety (vs Buffett's 15%)
_MAX_GROWTH_RATE      = 0.06


def _missing(name: str) -> CriterionResult:
    return CriterionResult(name=name, passed=False, label="N/A", note="Data unavailable")


# ── 1. Exceptional ROIC (> 20%) ───────────────────────────────────────────────

def _compute_roic(data: dict) -> float | None:
    oi = data.get("operating_income")
    ta = data.get("total_assets")
    cl = data.get("current_liabilities")
    if oi is None or ta is None or cl is None:
        return None
    ic = ta - cl
    if ic <= 0:
        return None
    return oi * (1 - _TAX_RATE) / ic


def _c1_exceptional_roic(data: dict) -> tuple[CriterionResult, float | None]:
    roic = _compute_roic(data)
    if roic is None:
        return _missing("Exceptional ROIC (> 20%)"), None

    passed = roic > 0.20
    direction = "above" if roic >= 0.20 else "below"
    return CriterionResult(
        name="Exceptional ROIC (> 20%)",
        passed=passed,
        label=f"{roic*100:.1f}%  (> 20% required)",
        note="Munger demands exceptional, not merely adequate, returns on invested capital.",
    ), roic


# ── 2. Asset-Light Model (Capex / Revenue < 5%) ───────────────────────────────

def _c2_asset_light(data: dict) -> tuple[CriterionResult, float | None]:
    ocf     = data.get("operating_cf")
    fcf     = data.get("free_cf")
    revenue = data.get("revenue")

    if ocf is None or fcf is None or revenue is None or revenue <= 0:
        return _missing("Asset-Light Model (Capex / Revenue)"), None

    capex = ocf - fcf   # capex ≈ OCF − FCF
    if capex < 0:
        capex = 0.0     # negative capex (proceeds > spend) → treat as zero

    capex_pct = capex / revenue
    passed = capex_pct < 0.05

    def _fmt(v: float) -> str:
        b = v / 1e9
        return f"${b:.1f}B" if abs(b) >= 1 else f"${v/1e6:.0f}M"

    return CriterionResult(
        name="Asset-Light Model (Capex / Revenue)",
        passed=passed,
        label=f"Capex {_fmt(capex)} / Revenue {_fmt(revenue)}  = {capex_pct*100:.1f}%  (< 5% required)",
        note="Asset-light businesses compound value without constant reinvestment — Munger's ideal.",
    ), capex_pct


# ── 3. Fortress Pricing Power (Gross Margin ≥ 45%) ───────────────────────────

def _c3_pricing_power(data: dict) -> CriterionResult:
    gm = data.get("gross_margin")
    if gm is None:
        return _missing("Fortress Pricing Power (Gross Margin)")

    passed = gm >= 0.45
    return CriterionResult(
        name="Fortress Pricing Power (Gross Margin)",
        passed=passed,
        label=f"{gm*100:.1f}%  (≥ 45% required)",
        note="A 45%+ gross margin signals near-impenetrable pricing power.",
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
        note="Munger avoids businesses with erratic earnings — invert: avoid the losers.",
    )


# ── 5. Strong Earnings Growth (EPS 5yr CAGR > 8%) ────────────────────────────

def _eps_cagr(annual_eps: list, years: int) -> float | None:
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
        return _missing("Strong Earnings Growth (EPS CAGR)")

    cagr = _eps_cagr(eps_hist, 5)
    periods = 5
    if cagr is None:
        cagr = _eps_cagr(eps_hist, 3)
        periods = 3
    if cagr is None:
        return CriterionResult(
            name="Strong Earnings Growth (EPS CAGR)",
            passed=False,
            label="Cannot compute (negative base EPS or < 4 years of data)",
        )

    passed = cagr > 0.08
    return CriterionResult(
        name="Strong Earnings Growth (EPS CAGR)",
        passed=passed,
        label=f"{cagr*100:.1f}% {periods}-yr CAGR  (> 8% required)",
        note="High-quality compounders grow earnings materially faster than the market.",
    )


# ── 6. Owner Earnings Quality (OCF / Net Income > 0.80) ──────────────────────

def _c6_owner_earnings_quality(data: dict) -> CriterionResult:
    ocf    = data.get("operating_cf")
    shares = data.get("shares")
    eps    = data.get("trailing_eps") or (data.get("annual_eps") or [(None, None)])[0][1]

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
    passed = ratio > 0.80

    def _fmt(v: float) -> str:
        b = v / 1e9
        return f"${b:.1f}B" if abs(b) >= 1 else f"${v/1e6:.0f}M"

    return CriterionResult(
        name="Owner Earnings Quality (OCF / Net Income)",
        passed=passed,
        label=f"OCF {_fmt(ocf)} / NI {_fmt(net_income)}  = {ratio:.2f}  (> 0.80 required)",
        note="Reported earnings must be firmly backed by cash — Munger is skeptical of accrual accounting.",
    )


# ── 7. Debt Aversion (LT Debt < 3× FCF) ─────────────────────────────────────

def _c7_debt_aversion(data: dict) -> CriterionResult:
    ltd = data.get("long_term_debt")
    fcf = data.get("free_cf")

    if ltd is None or fcf is None:
        return _missing("Debt Aversion (LT Debt / FCF)")

    def _fmt(v: float) -> str:
        b = v / 1e9
        return f"${b:.1f}B" if abs(b) >= 1 else f"${v/1e6:.0f}M"

    if fcf <= 0:
        return CriterionResult(
            name="Debt Aversion (LT Debt / FCF)",
            passed=False,
            label=f"LT debt {_fmt(ltd)}, FCF {_fmt(fcf)} — negative FCF",
        )

    ratio = ltd / fcf
    passed = ratio < 3.0
    return CriterionResult(
        name="Debt Aversion (LT Debt / FCF)",
        passed=passed,
        label=f"LT debt {_fmt(ltd)} / FCF {_fmt(fcf)}  = {ratio:.1f}×  (< 3× required)",
        note="Munger deeply distrusts leverage — wants debt retired well within 3 years of FCF.",
    )


# ── 8. Margin of Safety (IV × 0.80 — 20% MoS required) ──────────────────────

def _compute_intrinsic_value(data: dict) -> float | None:
    shares = data.get("shares")
    if not shares or shares <= 0:
        return None

    owner_earnings = data.get("free_cf")
    if owner_earnings is None or owner_earnings <= 0:
        annual_eps = data.get("annual_eps", [])
        if len(annual_eps) >= 3:
            eps_3yr = sum(e for _, e in annual_eps[:3]) / 3
            if eps_3yr > 0:
                owner_earnings = eps_3yr * shares
    if owner_earnings is None or owner_earnings <= 0:
        return None

    annual_eps = data.get("annual_eps", [])
    cagr = _eps_cagr(annual_eps, 5) or _eps_cagr(annual_eps, 3)
    growth_rate = min(max((cagr or 0.0) / 2, 0.0), _MAX_GROWTH_RATE)

    rf = FREDProvider().get_risk_free_rate("10yr")
    beta = data.get("beta") or 1.0
    discount_rate = rf + beta * _EQUITY_RISK_PREMIUM
    if discount_rate <= growth_rate:
        discount_rate = growth_rate + 0.02

    return owner_earnings * (1 + growth_rate) / (discount_rate - growth_rate) / shares


def _c8_margin_of_safety(data: dict) -> tuple[CriterionResult, float | None]:
    price = data.get("price")
    if price is None:
        return _missing("Margin of Safety (Intrinsic Value, 20% required)"), None

    iv = _compute_intrinsic_value(data)
    if iv is None:
        return CriterionResult(
            name="Margin of Safety (Intrinsic Value, 20% required)",
            passed=False,
            label="Cannot compute (insufficient FCF / EPS data)",
        ), None

    threshold = iv * _MOS_MULTIPLIER
    passed    = price <= threshold
    pct       = (price - iv) / iv * 100.0
    direction = "above" if pct > 0 else "below"

    return CriterionResult(
        name="Margin of Safety (Intrinsic Value, 20% required)",
        passed=passed,
        label=(
            f"Price ${price:.2f} is {abs(pct):.0f}% {direction} IV (${iv:.2f}); "
            f"pass if price ≤ ${threshold:.2f} (IV × {_MOS_MULTIPLIER:.2f})"
        ),
        note="Munger demands a wider margin of safety than Buffett's baseline — patience is a virtue.",
    ), iv


# ── Main evaluator ────────────────────────────────────────────────────────────

def evaluate_munger(data: dict) -> MungerResult:
    c1, roic        = _c1_exceptional_roic(data)
    c2, capex_pct   = _c2_asset_light(data)
    c3              = _c3_pricing_power(data)
    c4              = _c4_earnings_consistency(data)
    c5              = _c5_earnings_growth(data)
    c6              = _c6_owner_earnings_quality(data)
    c7              = _c7_debt_aversion(data)
    c8, iv          = _c8_margin_of_safety(data)

    criteria     = [c1, c2, c3, c4, c5, c6, c7, c8]
    criteria_met = sum(c.passed for c in criteria)
    total        = len(criteria)

    if criteria_met >= 7:
        verdict = "Munger-Grade Business"
    elif criteria_met >= 5:
        verdict = "Strong Candidate"
    else:
        verdict = "Pass"

    price = data.get("price") or 0.0
    price_vs_iv: float | None = None
    if iv is not None and iv != 0:
        price_vs_iv = (price - iv) / iv * 100.0

    return MungerResult(
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
        roic=roic,
        capex_intensity=capex_pct,
    )
