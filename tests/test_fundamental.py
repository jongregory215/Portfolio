"""
Unit tests for the Fundamental Engine.
All tests use fixture or hand-crafted data — no live API calls.
"""
import pytest
from stockgrader.engines.fundamental import FundamentalEngine, _hi_score, _lo_score, _peer_vals
from stockgrader.models import FundamentalResult, Grade
from stockgrader.config import get_config
from tests.fixtures.sample_fmp import FMP_DATA, YF_INFO, ESTIMATES, RF_RATES


@pytest.fixture
def cfg():
    return get_config()


@pytest.fixture
def engine(cfg):
    return FundamentalEngine(config=cfg)


@pytest.fixture
def good_data():
    """A fundamentally strong company: high quality, reasonable valuation."""
    return {
        # Valuation
        "pe_trailing":    18.0,
        "pe_forward":     15.0,
        "pb":              3.5,
        "ps":              4.0,
        "ev_ebitda":      12.0,
        "peg":             1.2,
        # Profitability
        "gross_margin":    0.55,
        "operating_margin":0.28,
        "net_margin":      0.20,
        "roe":             0.22,
        "roa":             0.12,
        "roic":            0.22,
        "wacc":            0.09,
        "roic_vs_wacc":    0.13,
        "margin_trajectory": 1.2,
        # Growth
        "revenue_cagr_3yr":    0.10,
        "revenue_cagr_5yr":    0.09,
        "eps_cagr_3yr":        0.12,
        "eps_cagr_5yr":        0.11,
        "forward_eps_growth":  0.10,
        "forward_revenue_growth": 0.09,
        # Health
        "current_ratio":   1.8,
        "quick_ratio":     1.3,
        "debt_equity":     0.4,
        "interest_coverage": 12.0,
        "altman_z":        4.5,
        "piotroski_f":     8,
        # Capital allocation
        "fcf_conversion":  1.15,
        "fcf_yield":       0.04,
        "payout_ratio":    0.30,
        "shares_diluted":  [950_000_000, 1_000_000_000],  # buybacks
        "roic_fmp":        0.22,
        # Misc
        "peer_metrics": {},
        "quality_of_earnings_flag": False,
    }


@pytest.fixture
def weak_data():
    """A fundamentally weak company: distressed, overvalued, poor growth."""
    return {
        "pe_trailing":     55.0,
        "pe_forward":      48.0,
        "pb":              12.0,
        "ps":              10.0,
        "ev_ebitda":       35.0,
        "peg":              4.5,
        "gross_margin":     0.08,
        "operating_margin": -0.02,
        "net_margin":      -0.05,
        "roe":             -0.10,
        "roa":             -0.03,
        "roic":            -0.02,
        "wacc":             0.09,
        "roic_vs_wacc":    -0.11,
        "margin_trajectory": -2.5,
        "revenue_cagr_3yr":  -0.02,
        "revenue_cagr_5yr":  -0.01,
        "eps_cagr_3yr":      -0.05,
        "eps_cagr_5yr":      -0.03,
        "forward_eps_growth": -0.03,
        "forward_revenue_growth": 0.01,
        "current_ratio":    0.6,
        "quick_ratio":      0.4,
        "debt_equity":      3.5,
        "interest_coverage": 1.2,
        "altman_z":         1.1,
        "piotroski_f":      2,
        "fcf_conversion":   0.2,
        "fcf_yield":        0.005,
        "payout_ratio":     1.2,
        "shares_diluted":   [1_100_000_000, 1_000_000_000],  # dilution
        "roic_fmp":        -0.02,
        "peer_metrics": {},
        "quality_of_earnings_flag": True,
    }


# ── Score bounds ─────────────────────────────────────────────

def test_score_range_good(engine, good_data):
    result = engine.score(good_data)
    assert isinstance(result, FundamentalResult)
    assert 0 <= result.score <= 100
    assert result.score >= 60, f"Expected ≥60 for good company, got {result.score:.1f}"


def test_score_range_weak(engine, weak_data):
    result = engine.score(weak_data)
    assert isinstance(result, FundamentalResult)
    assert 0 <= result.score <= 100
    assert result.score <= 40, f"Expected ≤40 for weak company, got {result.score:.1f}"


def test_good_beats_weak(engine, good_data, weak_data):
    assert engine.score(good_data).score > engine.score(weak_data).score


# ── Pillar scores ─────────────────────────────────────────────

def test_valuation_pillar_good(engine, good_data):
    result = engine.score(good_data)
    assert result.pillars.valuation >= 50


def test_valuation_pillar_weak(engine, weak_data):
    result = engine.score(weak_data)
    assert result.pillars.valuation <= 40


def test_health_pillar_good(engine, good_data):
    result = engine.score(good_data)
    assert result.pillars.financial_health >= 70


def test_health_pillar_distressed(engine, weak_data):
    result = engine.score(weak_data)
    assert result.pillars.financial_health <= 40


def test_growth_pillar(engine, good_data, weak_data):
    assert engine.score(good_data).pillars.growth > engine.score(weak_data).pillars.growth


def test_profitability_pillar(engine, good_data, weak_data):
    assert engine.score(good_data).pillars.profitability > engine.score(weak_data).pillars.profitability


def test_capital_allocation_pillar_buybacks(engine, good_data):
    result = engine.score(good_data)
    assert result.pillars.capital_allocation >= 60


# ── Key derived signals in result ────────────────────────────

def test_roic_vs_wacc_reported(engine, good_data):
    result = engine.score(good_data)
    assert result.roic_vs_wacc == pytest.approx(0.13)


def test_altman_z_reported(engine, good_data):
    result = engine.score(good_data)
    assert result.altman_z == pytest.approx(4.5)


def test_piotroski_f_reported(engine, good_data):
    result = engine.score(good_data)
    assert result.piotroski_f == 8


# ── Missing data handling ────────────────────────────────────

def test_missing_valuation_metrics(engine):
    """Engine should return a valid result when valuation data is absent."""
    data = {
        "gross_margin": 0.30, "operating_margin": 0.12, "net_margin": 0.08,
        "roe": 0.10, "roa": 0.05, "roic": 0.10, "wacc": 0.09,
        "roic_vs_wacc": 0.01, "revenue_cagr_3yr": 0.05, "eps_cagr_3yr": 0.06,
        "current_ratio": 1.2, "debt_equity": 0.8, "altman_z": 2.5, "piotroski_f": 5,
        "fcf_conversion": 0.8, "peer_metrics": {},
        "quality_of_earnings_flag": False,
    }
    result = engine.score(data)
    assert isinstance(result, FundamentalResult)
    assert 0 <= result.score <= 100
    assert "pe_trailing" in result.missing_fields


def test_completely_empty_data(engine):
    """Engine must not crash on an empty data dict."""
    result = engine.score({"peer_metrics": {}, "quality_of_earnings_flag": False})
    assert isinstance(result, FundamentalResult)
    assert 0 <= result.score <= 100


def test_missing_fields_tracked(engine, good_data):
    del good_data["altman_z"]
    del good_data["piotroski_f"]
    result = engine.score(good_data)
    assert "altman_z" in result.missing_fields
    assert "piotroski_f" in result.missing_fields


# ── Peer-relative scoring ─────────────────────────────────────

def test_peer_relative_low_pe_scores_well(engine, good_data):
    """A stock with the lowest P/E in its peer group should score near 100 on that metric."""
    good_data["pe_trailing"] = 8.0
    good_data["peer_metrics"] = {
        f"PEER{i}": {"priceEarningsRatioTTM": 20.0 + i * 2} for i in range(10)
    }
    result = engine.score(good_data)
    # Valuation pillar should be strong — lowest PE in peer group
    assert result.pillars.valuation >= 65


def test_peer_relative_high_pe_scores_poorly(engine, good_data):
    """A stock with the highest P/E in its peer group should score low on valuation."""
    good_data["pe_trailing"] = 60.0
    good_data["peer_metrics"] = {
        f"PEER{i}": {"priceEarningsRatioTTM": 10.0 + i * 2} for i in range(10)
    }
    result = engine.score(good_data)
    assert result.pillars.valuation <= 60


def test_fallback_to_absolute_with_few_peers(engine, good_data):
    """With fewer than MIN_PEERS (5), absolute thresholds are used."""
    good_data["pe_trailing"] = 12.0
    good_data["peer_metrics"] = {
        "P1": {"priceEarningsRatioTTM": 20.0},
        "P2": {"priceEarningsRatioTTM": 25.0},
    }
    result = engine.score(good_data)
    assert isinstance(result, FundamentalResult)   # should not crash


# ── Breakpoint helpers ────────────────────────────────────────

def test_hi_score_above_top_break():
    from stockgrader.engines.fundamental import _CURRENT_RATIO
    assert _hi_score(3.0, _CURRENT_RATIO) == 100

def test_hi_score_below_bottom_break():
    from stockgrader.engines.fundamental import _CURRENT_RATIO
    assert _hi_score(0.0, _CURRENT_RATIO) == 15

def test_lo_score_below_top_break():
    from stockgrader.engines.fundamental import _PE_ABS
    assert _lo_score(8.0, _PE_ABS) == 100   # P/E = 8 → cheapest bucket

def test_lo_score_above_bottom_break():
    from stockgrader.engines.fundamental import _PE_ABS
    assert _lo_score(55.0, _PE_ABS) == 12   # P/E = 55 → most expensive bucket


# ── Fixture-based full-pipeline test ─────────────────────────

def test_score_with_fixture_data(engine, cfg):
    """Run normalize() → engine.score() on the sample fixture to ensure E2E works."""
    from stockgrader.data.normalizer import normalize
    data = normalize(
        ticker="TEST", price_df=None,
        yf_info=YF_INFO, fmp_data=FMP_DATA, estimates=ESTIMATES,
        peers=[], peer_metrics={}, rf_rates=RF_RATES, cfg=cfg,
    )
    result = engine.score(data)
    assert isinstance(result, FundamentalResult)
    assert 0 <= result.score <= 100
    assert result.altman_z is not None
    assert result.piotroski_f is not None
    assert result.roic_vs_wacc is not None
    # Fixture is a strong company → expect decent fundamental score
    assert result.score >= 40
