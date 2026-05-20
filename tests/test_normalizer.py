"""
Unit tests for the normalizer — derived metric calculations.
All tests use fixture data; no live API calls.
"""
import pytest
from tests.fixtures.sample_fmp import (
    FMP_DATA, YF_INFO, ESTIMATES, RF_RATES,
    INCOME_STATEMENTS, BALANCE_SHEETS, CASH_FLOWS,
)
from stockgrader.data.normalizer import (
    normalize,
    compute_cagr,
    compute_altman_z,
    compute_piotroski_f,
    compute_roic,
    compute_wacc,
    compute_interest_coverage,
    compute_quality_of_earnings,
    compute_margin_trajectory,
    compute_fcf_conversion,
    _safe,
)
from stockgrader.config import get_config


@pytest.fixture
def cfg():
    return get_config()


@pytest.fixture
def normalized(cfg):
    return normalize(
        ticker       = "TEST",
        price_df     = None,
        yf_info      = YF_INFO,
        fmp_data     = FMP_DATA,
        estimates    = ESTIMATES,
        peers        = ["PEER1", "PEER2"],
        peer_metrics = {},
        rf_rates     = RF_RATES,
        cfg          = cfg,
    )


# ── _safe ────────────────────────────────────────────────────

def test_safe_none():
    assert _safe(None) is None

def test_safe_nan():
    import math
    assert _safe(float("nan")) is None

def test_safe_inf():
    assert _safe(float("inf")) is None

def test_safe_int():
    assert _safe(42) == 42.0

def test_safe_string_number():
    assert _safe("3.14") == pytest.approx(3.14)

def test_safe_bad_string():
    assert _safe("hello") is None


# ── CAGR ─────────────────────────────────────────────────────

def test_cagr_3yr():
    # Revenue: 391, 383, 394, 366 (newest first)
    series = [391e9, 383e9, 394e9, 366e9]
    cagr = compute_cagr(series, 3)
    assert cagr is not None
    expected = (391 / 366) ** (1/3) - 1
    assert cagr == pytest.approx(expected, rel=1e-4)

def test_cagr_insufficient_data():
    assert compute_cagr([100.0, 90.0], 3) is None

def test_cagr_zero_base():
    assert compute_cagr([100.0, 0.0, 0.0, 0.0], 3) is None

def test_cagr_negative_ratio():
    # Negative base value → ratio is negative → undefined CAGR
    assert compute_cagr([100.0, 50.0, 80.0, -90.0], 3) is None


# ── Altman Z ─────────────────────────────────────────────────

def test_altman_z_computed(normalized):
    z = normalized.get("altman_z")
    assert z is not None
    # Fixture data is large-cap tech with strong financials → Z well above 3
    assert z > 2.0

def test_altman_z_missing_field():
    data = {
        "current_assets":     [100.0],
        "current_liabilities":[50.0],
        # total_assets missing
        "retained_earnings":  [20.0],
        "operating_income":   [30.0],
        "market_cap":         500.0,
        "total_liabilities":  [100.0],
        "revenue":            [200.0],
    }
    assert compute_altman_z(data) is None


# ── Piotroski F ──────────────────────────────────────────────

def test_piotroski_f_range(normalized):
    f = normalized.get("piotroski_f")
    assert f is not None
    assert 0 <= f <= 9

def test_piotroski_f_strong_company(normalized):
    # Fixture: profitable, growing FCF, no dilution → expect high score
    assert normalized["piotroski_f"] >= 5

def test_piotroski_f_needs_two_years():
    # Only one year of data → can't compute most signals → returns None or low score
    data = {
        "total_assets":       [100.0],
        "net_income":         [10.0],
        "operating_cf":       [15.0],
        "revenue":            [200.0],
        "gross_profit":       [80.0],
        "current_assets":     [50.0],
        "current_liabilities":[30.0],
        "total_debt":         [40.0],
        "shares_diluted":     [1000.0],
    }
    result = compute_piotroski_f(data)
    # With only one year, most YoY signals can't fire → expect None or <= 4
    assert result is None or result <= 4


# ── ROIC ────────────────────────────────────────────────────

def test_roic_positive(normalized):
    roic = normalized.get("roic")
    assert roic is not None
    # Fixture: 120B operating income, 132B invested capital (57+105-30)
    assert roic > 0

def test_roic_formula():
    data = {
        "operating_income":  [100.0],
        "total_equity":      [200.0],
        "total_debt":        [100.0],
        "cash":              [50.0],
    }
    roic = compute_roic(data, tax_rate=0.21)
    # NOPAT = 100 * 0.79 = 79; IC = 200+100-50 = 250
    assert roic == pytest.approx(79.0 / 250.0, rel=1e-4)


# ── WACC ────────────────────────────────────────────────────

def test_wacc_range(normalized):
    wacc = normalized.get("wacc")
    assert wacc is not None
    assert 0.03 < wacc < 0.20   # reasonable range for a large-cap

def test_wacc_no_debt():
    data = {
        "market_cap":        1000.0,
        "total_debt":        0.0,
        "interest_expense":  [0.0],
        "beta":              1.0,
    }
    wacc = compute_wacc(data, rf=0.05, erp=0.055)
    # Pure equity: WACC = CoE = 0.05 + 1.0*0.055 = 0.105
    assert wacc == pytest.approx(0.105, rel=1e-4)


# ── Interest coverage ────────────────────────────────────────

def test_interest_coverage(normalized):
    ic = normalized.get("interest_coverage") or normalized.get("interest_coverage_c")
    assert ic is not None
    assert ic > 10   # fixture: 120B / 3.8B ≈ 31.6x


# ── Quality of earnings ──────────────────────────────────────

def test_quality_of_earnings(normalized):
    qoe = normalized.get("quality_of_earnings")
    assert qoe is not None
    # Fixture: OCF 118B / NI 94B ≈ 1.26 → high quality
    assert qoe > 1.0

def test_quality_of_earnings_flag(normalized):
    # High QoE ratio → flag should be False
    assert normalized["quality_of_earnings_flag"] is False


# ── Margin trajectory ────────────────────────────────────────

def test_margin_trajectory(normalized):
    mt = normalized.get("margin_trajectory")
    assert mt is not None
    # Fixture has roughly flat margins → slope near 0
    assert abs(mt) < 5.0   # within ±5 ppt/yr


# ── FCF conversion ───────────────────────────────────────────

def test_fcf_conversion(normalized):
    fc = normalized.get("fcf_conversion")
    assert fc is not None
    # Fixture: FCF 109B / NI 94B ≈ 1.16
    assert fc > 1.0


# ── Full normalize() output ──────────────────────────────────

def test_normalize_returns_ticker(normalized):
    assert normalized["ticker"] == "TEST"

def test_normalize_has_required_keys(normalized):
    for key in ["price", "market_cap", "beta", "sector", "pe_trailing",
                "gross_margin", "roe", "roic", "wacc", "altman_z", "piotroski_f",
                "revenue_cagr_3yr", "data_completeness"]:
        assert key in normalized, f"Missing key: {key}"

def test_normalize_completeness_high(normalized):
    # Fixture provides most fields → completeness should be high
    assert normalized["data_completeness"] >= 0.70

def test_normalize_peers(normalized):
    assert normalized["peers"] == ["PEER1", "PEER2"]

def test_normalize_rf_rates(normalized):
    assert normalized["risk_free_rate_3mo"] == pytest.approx(0.053)

def test_normalize_revenue_cagr(normalized):
    cagr = normalized["revenue_cagr_3yr"]
    assert cagr is not None
    # 391B growing from 366B over 3 years
    expected = (391 / 366) ** (1/3) - 1
    assert cagr == pytest.approx(expected, rel=1e-3)

def test_normalize_roic_vs_wacc(normalized):
    spread = normalized.get("roic_vs_wacc")
    assert spread is not None
    # Strong company → ROIC should exceed WACC
    assert spread > 0
