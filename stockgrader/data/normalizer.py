"""
Normalizer — maps raw provider data into the canonical data dict consumed
by all three scoring engines.

Two phases:
  1. Field extraction: FMP / yfinance field names → canonical names
  2. Derived metrics: Altman Z, Piotroski F, ROIC, WACC, CAGRs, etc.

All functions are pure (same inputs → same output) so the backtest engine
can replay them on historical data without modification.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────
# FMP field-name mappings
# ──────────────────────────────────────────────────────────────

_IS_MAP = {          # income statement
    "revenue":                      "revenue",
    "grossProfit":                  "gross_profit",
    "operatingIncome":              "operating_income",
    "netIncome":                    "net_income",
    "epsdiluted":                   "eps",
    "ebitda":                       "ebitda",
    "interestExpense":              "interest_expense",
    "depreciationAndAmortization":  "d_and_a",
    "incomeTaxExpense":             "tax_expense",
    "incomeBeforeTax":              "ebt",
    "weightedAverageShsOutDil":     "shares_diluted",
}

_BS_MAP = {          # balance sheet
    "totalAssets":                  "total_assets",
    "totalLiabilities":             "total_liabilities",
    "totalStockholdersEquity":      "total_equity",
    "retainedEarnings":             "retained_earnings",
    "cashAndCashEquivalents":       "cash",
    "totalDebt":                    "total_debt",
    "totalCurrentAssets":           "current_assets",
    "totalCurrentLiabilities":      "current_liabilities",
    "shortTermDebt":                "short_term_debt",
    "longTermDebt":                 "long_term_debt",
    "goodwillAndIntangibleAssets":  "goodwill_intangibles",
}

_CF_MAP = {          # cash flow
    "operatingCashFlow":            "operating_cf",
    "capitalExpenditure":           "capex",
    "freeCashFlow":                 "free_cf",
    "dividendsPaid":                "dividends_paid",
    "netIncome":                    "net_income_cf",   # for quality-of-earnings
}

_KM_MAP = {          # key-metrics-ttm — handles both v3 and stable API field names
    # v3 names
    "revenuePerShareTTM":           "revenue_per_share",
    "currentRatioTTM":              "current_ratio",
    "debtToEquityTTM":              "debt_equity",
    "interestCoverageTTM":          "interest_coverage",
    "payoutRatioTTM":               "payout_ratio",
    "priceEarningsRatioTTM":        "pe_trailing",
    "priceToBookRatioTTM":          "pb",
    "priceToSalesRatioTTM":         "ps",
    "enterpriseValueOverEBITDATTM": "ev_ebitda",
    "priceEarningsToGrowthRatioTTM":"peg",
    "dividendYieldTTM":             "dividend_yield",
    "earningsYieldTTM":             "earnings_yield",
    "freeCashFlowYieldTTM":         "fcf_yield",
    "returnOnEquityTTM":            "roe",
    "returnOnAssetsTTM":            "roa",
    "roicTTM":                      "roic_fmp",
    "investedCapitalTTM":           "invested_capital",
    "netDebtTTM":                   "net_debt",
    "marketCapTTM":                 "market_cap_km",
    "enterpriseValueTTM":           "enterprise_value",
    "grahamNumberTTM":              "graham_number",
    "pegRatioTTM":                  "peg",
    # stable API names (different from v3)
    "evToEBITDATTM":                "ev_ebitda",
    "returnOnInvestedCapitalTTM":   "roic_fmp",
    "marketCap":                    "market_cap_km",  # stable km-ttm uses marketCap not marketCapTTM
}

_RATIOS_MAP = {      # ratios-ttm — handles both v3 and stable API field names
    # v3 names
    "grossProfitMarginTTM":         "gross_margin",
    "operatingProfitMarginTTM":     "operating_margin",
    "netProfitMarginTTM":           "net_margin",
    "returnOnEquityTTM":            "roe",
    "returnOnAssetsTTM":            "roa",
    "debtRatioTTM":                 "debt_to_assets",
    "quickRatioTTM":                "quick_ratio",
    "currentRatioTTM":              "current_ratio",
    "dividendYielTTM":              "dividend_yield",   # v3 has a typo (no 'd')
    "payoutRatioTTM":               "payout_ratio",
    "priceEarningsRatioTTM":        "pe_trailing",
    "priceToBookRatioTTM":          "pb",
    "priceToSalesRatioTTM":         "ps",
    # stable API names (different from v3)
    "dividendYieldTTM":             "dividend_yield",   # stable fixes the typo
    "dividendPayoutRatioTTM":       "payout_ratio",
    "priceToEarningsRatioTTM":      "pe_trailing",
    "debtToAssetsRatioTTM":         "debt_to_assets",
    "debtToEquityRatioTTM":         "debt_equity",
    "interestCoverageRatioTTM":     "interest_coverage",
    "revenuePerShareTTM":           "revenue_per_share",
    "enterpriseValueMultipleTTM":   "ev_ebitda",
}

_YF_MAP = {          # yfinance .info
    "trailingPE":                   "pe_trailing",
    "forwardPE":                    "pe_forward",
    "priceToBook":                  "pb",
    "priceToSalesTrailing12Months": "ps",
    "trailingEps":                  "eps_ttm",
    "forwardEps":                   "forward_eps",
    "beta":                         "beta",
    "marketCap":                    "market_cap",
    "enterpriseValue":              "enterprise_value",
    "enterpriseToEbitda":           "ev_ebitda",
    "enterpriseToRevenue":          "ev_revenue",
    "currentRatio":                 "current_ratio",
    "quickRatio":                   "quick_ratio",
    "debtToEquity":                 "debt_equity",
    "grossMargins":                 "gross_margin",
    "operatingMargins":             "operating_margin",
    "profitMargins":                "net_margin",
    "returnOnEquity":               "roe",
    "returnOnAssets":               "roa",
    "dividendYield":                "dividend_yield",
    "payoutRatio":                  "payout_ratio",
    "revenueGrowth":                "forward_revenue_growth",
    "earningsGrowth":               "forward_eps_growth",
    "earningsQuarterlyGrowth":      "earnings_quarterly_growth",
    "pegRatio":                     "peg",
    "totalDebt":                    "total_debt_spot",   # spot check vs BS series
    "totalCash":                    "cash_spot",
    "totalRevenue":                 "revenue_spot",
    "operatingCashflow":            "operating_cf_spot",
    "freeCashflow":                 "free_cf_spot",
    "averageVolume":                "avg_volume",
    "sharesOutstanding":            "shares_outstanding",
    "sector":                       "sector",
    "industry":                     "industry",
    "exchange":                     "exchange",
    "currency":                     "currency",
    "country":                      "country",
    "shortName":                    "company_name",
    "longName":                     "company_name",      # fallback if shortName absent
}


# ──────────────────────────────────────────────────────────────
# Pure helper functions
# ──────────────────────────────────────────────────────────────

def _safe(val: Any) -> float | None:
    """Return float or None; skip zeros that represent missing data."""
    if val is None:
        return None
    try:
        f = float(val)
        return None if (np.isnan(f) or np.isinf(f)) else f
    except (TypeError, ValueError):
        return None


def _extract_series(statements: list[dict], field_map: dict[str, str], src_key: str) -> dict[str, list[float | None]]:
    """
    Extract time-series lists (newest-first) from a list of FMP statement dicts.

    Returns: {canonical_name: [v_year0, v_year1, ...], ...}
    """
    series: dict[str, list[float | None]] = {canon: [] for canon in field_map.values()}
    for stmt in statements:
        for fmp_key, canon in field_map.items():
            series[canon].append(_safe(stmt.get(fmp_key)))
    return series


def compute_cagr(series: list[float | None], years: int) -> float | None:
    """CAGR from a newest-first list over `years` periods."""
    if len(series) <= years:
        return None
    v0, vn = _safe(series[0]), _safe(series[years])
    if v0 is None or vn is None or vn == 0:
        return None
    ratio = v0 / vn
    if ratio <= 0:
        return None
    return ratio ** (1.0 / years) - 1.0


def compute_altman_z(data: dict) -> float | None:
    """
    Altman Z-score for publicly traded companies:
      Z = 1.2*X1 + 1.4*X2 + 3.3*X3 + 0.6*X4 + 1.0*X5
    """
    ca  = _safe(data.get("current_assets",    [None])[0] if isinstance(data.get("current_assets"), list) else data.get("current_assets"))
    cl  = _safe(data.get("current_liabilities",[None])[0] if isinstance(data.get("current_liabilities"), list) else data.get("current_liabilities"))
    ta  = _safe(data.get("total_assets",       [None])[0] if isinstance(data.get("total_assets"), list) else data.get("total_assets"))
    re  = _safe(data.get("retained_earnings",  [None])[0] if isinstance(data.get("retained_earnings"), list) else data.get("retained_earnings"))
    oi  = _safe(data.get("operating_income",   [None])[0] if isinstance(data.get("operating_income"), list) else data.get("operating_income"))
    mc  = _safe(data.get("market_cap"))
    tl  = _safe(data.get("total_liabilities",  [None])[0] if isinstance(data.get("total_liabilities"), list) else data.get("total_liabilities"))
    rev = _safe(data.get("revenue",            [None])[0] if isinstance(data.get("revenue"), list) else data.get("revenue"))

    if any(v is None for v in [ca, cl, ta, re, oi, mc, tl, rev]):
        return None
    if ta == 0 or tl == 0:
        return None

    x1 = (ca - cl) / ta
    x2 = re / ta
    x3 = oi / ta
    x4 = mc / tl
    x5 = rev / ta
    return 1.2 * x1 + 1.4 * x2 + 3.3 * x3 + 0.6 * x4 + 1.0 * x5


def compute_piotroski_f(data: dict) -> int | None:
    """
    Piotroski F-score (0–9). Requires two years of balance/income/cf data.

    Profitability (4): ROA>0, OCF>0, ROA improved, OCF>net income (accruals)
    Leverage/Liquidity (3): lower leverage, higher current ratio, no dilution
    Efficiency (2): higher gross margin, higher asset turnover
    """
    def _series_val(key: str, idx: int) -> float | None:
        v = data.get(key)
        if isinstance(v, list) and len(v) > idx:
            return _safe(v[idx])
        return None

    ta0  = _series_val("total_assets", 0)
    ta1  = _series_val("total_assets", 1)
    ni0  = _series_val("net_income", 0)
    ocf0 = _series_val("operating_cf", 0)
    rev0 = _series_val("revenue", 0)
    rev1 = _series_val("revenue", 1)
    gp0  = _series_val("gross_profit", 0)
    gp1  = _series_val("gross_profit", 1)
    cl0  = _series_val("current_liabilities", 0)
    ca0  = _series_val("current_assets", 0)
    cl1  = _series_val("current_liabilities", 1)
    ca1  = _series_val("current_assets", 1)
    td0  = _series_val("total_debt", 0)
    td1  = _series_val("total_debt", 1)
    sh0  = _series_val("shares_diluted", 0)
    sh1  = _series_val("shares_diluted", 1)

    if ta0 is None or ta0 == 0:
        return None

    roa0 = ni0  / ta0 if ni0  is not None else None
    roa1 = (ni0 / ta1) if (ta1 and ta1 != 0 and ni0 is not None) else None

    f = 0
    # F1: ROA > 0
    if roa0 is not None and roa0 > 0:
        f += 1
    # F2: Operating cash flow > 0
    if ocf0 is not None and ocf0 > 0:
        f += 1
    # F3: ROA improved year-over-year
    if roa0 is not None and roa1 is not None and roa0 > roa1:
        f += 1
    # F4: Cash flow > net income (low accruals = quality earnings)
    if ocf0 is not None and ni0 is not None and ocf0 > ni0:
        f += 1
    # F5: Leverage (long-term debt / avg assets) decreased
    if td0 is not None and td1 is not None and ta1 is not None and ta1 != 0:
        lev0 = td0 / ta0
        lev1 = td1 / ta1
        if lev0 < lev1:
            f += 1
    # F6: Current ratio improved
    if ca0 is not None and cl0 is not None and cl0 != 0 and ca1 is not None and cl1 is not None and cl1 != 0:
        if (ca0 / cl0) > (ca1 / cl1):
            f += 1
    # F7: No share dilution
    if sh0 is not None and sh1 is not None and sh0 <= sh1:
        f += 1
    # F8: Gross margin improved
    if rev0 is not None and gp0 is not None and rev1 is not None and gp1 is not None:
        if rev0 != 0 and rev1 != 0:
            if (gp0 / rev0) > (gp1 / rev1):
                f += 1
    # F9: Asset turnover improved (revenue / assets)
    if rev0 is not None and rev1 is not None and ta1 is not None and ta1 != 0:
        if (rev0 / ta0) > (rev1 / ta1):
            f += 1

    return f


def compute_roic(data: dict, tax_rate: float = 0.21) -> float | None:
    """
    ROIC = NOPAT / Invested Capital
    NOPAT = Operating Income * (1 - tax_rate)
    Invested Capital = Total Equity + Total Debt - Cash
    """
    oi  = _safe(data.get("operating_income", [None])[0] if isinstance(data.get("operating_income"), list) else data.get("operating_income"))
    eq  = _safe(data.get("total_equity",     [None])[0] if isinstance(data.get("total_equity"), list) else data.get("total_equity"))
    td  = _safe(data.get("total_debt",       [None])[0] if isinstance(data.get("total_debt"), list) else data.get("total_debt"))
    csh = _safe(data.get("cash",             [None])[0] if isinstance(data.get("cash"), list) else data.get("cash"))

    if any(v is None for v in [oi, eq, td, csh]):
        return None

    invested_capital = eq + td - csh
    if invested_capital == 0:
        return None
    nopat = oi * (1.0 - tax_rate)
    return nopat / invested_capital


def compute_wacc(
    data: dict,
    rf: float,
    erp: float = 0.055,
    tax_rate: float = 0.21,
) -> float | None:
    """
    WACC = (E/V)*CoE + (D/V)*CoD*(1-t)
    CoE  = rf + beta * ERP
    CoD  = interest_expense / total_debt   (pre-tax cost of debt)
    """
    mc  = _safe(data.get("market_cap"))
    td  = _safe(data.get("total_debt",      [None])[0] if isinstance(data.get("total_debt"), list) else data.get("total_debt"))
    ie  = _safe(data.get("interest_expense",[None])[0] if isinstance(data.get("interest_expense"), list) else data.get("interest_expense"))
    beta = _safe(data.get("beta"))

    if mc is None or td is None:
        return None
    if beta is None:
        beta = 1.0   # market-neutral default

    cost_equity = rf + beta * erp

    if td <= 0:
        return cost_equity

    pre_tax_cod = (abs(ie) / td) if (ie is not None and td != 0) else (rf + 0.015)
    cost_debt   = pre_tax_cod * (1.0 - tax_rate)

    v = mc + td
    return (mc / v) * cost_equity + (td / v) * cost_debt


def compute_interest_coverage(data: dict) -> float | None:
    """Interest Coverage = EBIT / Interest Expense. EBIT ≈ Operating Income."""
    oi = _safe(data.get("operating_income", [None])[0] if isinstance(data.get("operating_income"), list) else data.get("operating_income"))
    ie = _safe(data.get("interest_expense", [None])[0] if isinstance(data.get("interest_expense"), list) else data.get("interest_expense"))
    if oi is None or ie is None or ie == 0:
        return None
    return oi / abs(ie)


def compute_quality_of_earnings(data: dict) -> float | None:
    """
    OCF / Net Income.  >1.0 = cash-backed earnings; <0.5 = accrual-heavy, flag it.
    """
    ocf = _safe(data.get("operating_cf", [None])[0] if isinstance(data.get("operating_cf"), list) else data.get("operating_cf"))
    ni  = _safe(data.get("net_income",   [None])[0] if isinstance(data.get("net_income"), list) else data.get("net_income"))
    if ocf is None or ni is None or ni == 0:
        return None
    return ocf / ni


def compute_margin_trajectory(data: dict, years: int = 3) -> float | None:
    """
    Linear slope of operating margin over the last `years` annual observations.
    Returns percentage-point change per year (e.g. +0.5 = margin expanding 0.5 ppt/yr).
    """
    rev_series = data.get("revenue", [])
    oi_series  = data.get("operating_income", [])

    margins = []
    for i in range(min(years, len(rev_series), len(oi_series))):
        rev = _safe(rev_series[i])
        oi  = _safe(oi_series[i])
        if rev and rev != 0 and oi is not None:
            margins.append(oi / rev * 100.0)
        else:
            margins.append(None)

    # Need at least 2 valid points
    valid = [(i, m) for i, m in enumerate(margins) if m is not None]
    if len(valid) < 2:
        return None

    # Oldest to newest (series is newest-first → reverse)
    xs = np.array([len(valid) - 1 - i for i, _ in valid], dtype=float)
    ys = np.array([m for _, m in valid], dtype=float)

    # Least-squares slope (ppts/yr, positive = expanding margin)
    if len(xs) >= 2:
        slope = float(np.polyfit(xs, ys, 1)[0])
        return slope
    return None


def compute_fcf_conversion(data: dict) -> float | None:
    """FCF / Net Income — measures earnings-to-cash conversion quality."""
    fcf = _safe(data.get("free_cf",   [None])[0] if isinstance(data.get("free_cf"), list) else data.get("free_cf"))
    ni  = _safe(data.get("net_income",[None])[0] if isinstance(data.get("net_income"), list) else data.get("net_income"))
    if fcf is None or ni is None or ni == 0:
        return None
    return fcf / ni


# ──────────────────────────────────────────────────────────────
# Main normalize() function
# ──────────────────────────────────────────────────────────────

def normalize(
    ticker: str,
    price_df: Any,
    yf_info: dict,
    fmp_data: dict,
    estimates: dict,
    peers: list[str],
    peer_metrics: dict[str, dict],
    rf_rates: dict[str, float],
    cfg: dict,
) -> dict:
    """
    Merge raw provider data into the canonical data dict.

    Returns a flat dict consumed by all three scoring engines. Every key
    is either a float, list[float|None], or other primitive. DataFrames
    are passed through untouched (price_history key).

    Missing fields are tracked in data["missing_fields"]; imputed fields
    (filled with a fallback) in data["imputed_fields"].
    """
    missing:  list[str] = []
    imputed:  list[str] = []
    sources:  dict[str, str] = {}
    warnings: list[str] = []

    # ── 1. Extract FMP time series ─────────────────────────────
    is_raw  = fmp_data.get("income_statements", [])
    bs_raw  = fmp_data.get("balance_sheets", [])
    cf_raw  = fmp_data.get("cash_flows", [])
    km_ttm  = fmp_data.get("key_metrics_ttm", {})
    rat_ttm = fmp_data.get("ratios_ttm", {})
    profile = fmp_data.get("profile", {})

    is_series = _extract_series(is_raw, _IS_MAP, "income_statement")
    bs_series = _extract_series(bs_raw, _BS_MAP, "balance_sheet")
    cf_series = _extract_series(cf_raw, _CF_MAP, "cash_flow")

    # ── 2. Build data dict with time series ─────────────────────
    data: dict[str, Any] = {
        "ticker":        ticker,
        "price_history": price_df,
        "peers":         peers,
        "peer_metrics":  peer_metrics,
    }

    # Time series (newest-first lists)
    for key, series in {**is_series, **bs_series, **cf_series}.items():
        if any(v is not None for v in series):
            data[key] = series
            if is_series or bs_series or cf_series:
                sources[key] = "fmp"
        else:
            missing.append(key)

    # ── 3. Point-in-time: key metrics TTM ──────────────────────
    for fmp_key, canon in _KM_MAP.items():
        val = _safe(km_ttm.get(fmp_key))
        if val is not None:
            data.setdefault(canon, val)
            sources[canon] = "fmp_km_ttm"

    for fmp_key, canon in _RATIOS_MAP.items():
        val = _safe(rat_ttm.get(fmp_key))
        if val is not None:
            data.setdefault(canon, val)
            sources.setdefault(canon, "fmp_ratios_ttm")

    # ── 4. Profile / metadata ───────────────────────────────────
    for src_key, canon in [
        ("mktCap",       "market_cap"),
        ("beta",         "beta"),
        ("sector",       "sector"),
        ("industry",     "industry"),
        ("exchange",     "exchange"),
        ("currency",     "currency"),
        ("country",      "country"),
        ("companyName",  "company_name"),
        ("price",        "price"),
        ("volAvg",       "avg_volume"),
    ]:
        val = profile.get(src_key)
        if val is not None:
            data.setdefault(canon, val)
            sources[canon] = "fmp_profile"

    # ── 5. yfinance info — fills gaps ───────────────────────────
    for yf_key, canon in _YF_MAP.items():
        val = yf_info.get(yf_key)
        if val is not None:
            if canon not in data or data[canon] is None:
                data[canon] = val
                sources[canon] = "yfinance"

    # Current price from yfinance if not in profile
    for price_key in ("currentPrice", "regularMarketPrice", "previousClose"):
        if "price" not in data or data["price"] is None:
            val = _safe(yf_info.get(price_key))
            if val:
                data["price"] = val
                sources["price"] = "yfinance"
                break

    # ── 6. Forward estimates ─────────────────────────────────────
    for key in ("forward_eps", "forward_eps_growth", "forward_revenue_growth", "num_analysts"):
        val = estimates.get(key)
        if val is not None:
            data[key] = val
            sources[key] = "fmp_estimates"

    # FMP forward PE from estimates if not already present
    if "pe_forward" not in data and data.get("forward_eps") and data.get("price"):
        fwd_pe = _safe(data["price"]) / _safe(data["forward_eps"]) if _safe(data.get("forward_eps", 0)) else None
        if fwd_pe:
            data["pe_forward"] = fwd_pe
            sources["pe_forward"] = "computed"

    # ── 7. Risk-free rates ───────────────────────────────────────
    data["risk_free_rate_3mo"]  = rf_rates.get("3mo",  0.05)
    data["risk_free_rate_10yr"] = rf_rates.get("10yr", 0.045)

    # ── 8. Derived metrics ──────────────────────────────────────
    dcf_cfg  = cfg.get("fair_value", {}).get("dcf", {})
    tax_rate = float(dcf_cfg.get("tax_rate", 0.21))
    erp      = float(dcf_cfg.get("equity_risk_premium", 0.055))
    rf       = data["risk_free_rate_3mo"]

    data["revenue_cagr_3yr"] = compute_cagr(data.get("revenue", []), 3)
    data["revenue_cagr_5yr"] = compute_cagr(data.get("revenue", []), 5)
    data["eps_cagr_3yr"]     = compute_cagr(data.get("eps", []), 3)
    data["eps_cagr_5yr"]     = compute_cagr(data.get("eps", []), 5)

    if data.get("revenue_cagr_3yr") is not None: sources["revenue_cagr_3yr"] = "computed"
    if data.get("eps_cagr_3yr") is not None:     sources["eps_cagr_3yr"]     = "computed"

    data["roic"]                 = compute_roic(data, tax_rate)
    data["wacc"]                 = compute_wacc(data, rf, erp, tax_rate)
    data["altman_z"]             = compute_altman_z(data)
    data["piotroski_f"]          = compute_piotroski_f(data)
    data["interest_coverage_c"]  = compute_interest_coverage(data)  # computed version
    data["quality_of_earnings"]  = compute_quality_of_earnings(data)
    data["margin_trajectory"]    = compute_margin_trajectory(data)
    data["fcf_conversion"]       = compute_fcf_conversion(data)

    if data["roic"] is not None and data["wacc"] is not None:
        data["roic_vs_wacc"] = data["roic"] - data["wacc"]
        sources["roic_vs_wacc"] = "computed"

    for k in ("roic", "wacc", "altman_z", "piotroski_f", "quality_of_earnings",
              "margin_trajectory", "fcf_conversion"):
        if data.get(k) is not None:
            sources[k] = "computed"

    # ── 9. Quality-of-earnings flag (accruals > 20% of net income) ─
    qoe = _safe(data.get("quality_of_earnings"))
    data["quality_of_earnings_flag"] = (qoe is not None and qoe < 0.8)

    # ── 10. Current ratio from balance sheet if not in KM ───────
    if "current_ratio" not in data:
        ca = _safe(data.get("current_assets",     [None])[0] if isinstance(data.get("current_assets"), list) else data.get("current_assets"))
        cl = _safe(data.get("current_liabilities",[None])[0] if isinstance(data.get("current_liabilities"), list) else data.get("current_liabilities"))
        if ca is not None and cl and cl != 0:
            data["current_ratio"] = ca / cl
            sources["current_ratio"] = "computed"

    if "interest_coverage" not in data and data.get("interest_coverage_c") is not None:
        data["interest_coverage"] = data["interest_coverage_c"]

    # ── 11. Track required fields that are truly missing ────────
    required_fields = [
        "price", "market_cap", "revenue", "net_income", "total_assets",
        "total_equity", "total_debt", "operating_cf", "pe_trailing", "pb",
        "gross_margin", "operating_margin", "roe", "roa", "current_ratio",
        "debt_equity", "beta", "sector",
    ]
    for field in required_fields:
        val = data.get(field)
        is_empty = (val is None) or (isinstance(val, list) and all(v is None for v in val))
        if is_empty:
            missing.append(field)

    # ── 12. Data completeness score ─────────────────────────────
    completeness = 1.0 - (len(set(missing)) / max(len(required_fields), 1))

    # ── 13. Assemble metadata ───────────────────────────────────
    data["missing_fields"]  = sorted(set(missing))
    data["imputed_fields"]  = sorted(set(imputed))
    data["sources"]         = sources
    data["warnings"]        = warnings
    data["data_completeness"] = round(completeness, 3)

    return data
