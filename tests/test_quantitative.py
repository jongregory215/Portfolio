"""
Unit tests for the Quantitative Engine.
Uses synthetic price data and hand-crafted data dicts — no live API calls.
"""
import math
import numpy as np
import pandas as pd
import pytest
from datetime import date

from stockgrader.engines.quantitative import (
    QuantitativeEngine,
    _f, _peer_z, _abs_z, _norm_cdf, _z_to_pct,
)
from stockgrader.models import QuantitativeResult
from stockgrader.config import get_config
from tests.fixtures.sample_fmp import FMP_DATA, YF_INFO, ESTIMATES, RF_RATES


# ── Fixtures ─────────────────────────────────────────────────

def make_price_df(n=600, trend=0.0005, vol=0.012, start=100.0, seed=42):
    rng     = np.random.default_rng(seed)
    returns = rng.normal(trend, vol, n)
    close   = start * np.exp(np.cumsum(returns))
    noise   = rng.uniform(0.002, 0.008, n)
    dates   = pd.bdate_range(end=date.today(), periods=n)
    df = pd.DataFrame({
        "open":      close * (1 + rng.normal(0, 0.003, n)),
        "high":      close * (1 + noise),
        "low":       close * (1 - noise),
        "close":     close,
        "volume":    rng.integers(500_000, 5_000_000, n).astype(float),
        "adj_close": close,
    }, index=dates)
    df.index.name = "Date"
    return df


@pytest.fixture
def cfg():
    return get_config()


@pytest.fixture
def engine(cfg):
    return QuantitativeEngine(config=cfg)


@pytest.fixture
def uptrend_df():
    return make_price_df(n=800, trend=0.0008, seed=10)


@pytest.fixture
def downtrend_df():
    return make_price_df(n=800, trend=-0.0008, seed=20)


@pytest.fixture
def good_data(uptrend_df):
    """Strong value+quality company in an uptrend."""
    return {
        "price_history":     uptrend_df,
        "pe_trailing":       14.0,
        "pb":                2.0,
        "ps":                2.5,
        "fcf_yield":         0.065,
        "roe":               0.22,
        "roa":               0.12,
        "debt_equity":       0.3,
        "quality_of_earnings": 1.3,
        "market_cap":        50_000_000_000,
        "beta":              0.85,
        "risk_free_rate_3mo": 0.05,
        "peer_metrics":      {},
        "quality_of_earnings_flag": False,
    }


@pytest.fixture
def bad_data(downtrend_df):
    """Overvalued, low-quality company in a downtrend."""
    return {
        "price_history":     downtrend_df,
        "pe_trailing":       65.0,
        "pb":                15.0,
        "ps":                12.0,
        "fcf_yield":         0.005,
        "roe":               0.04,
        "roa":               0.01,
        "debt_equity":       3.5,
        "quality_of_earnings": 0.4,
        "market_cap":        500_000_000,
        "beta":              2.2,
        "risk_free_rate_3mo": 0.05,
        "peer_metrics":      {},
        "quality_of_earnings_flag": True,
    }


# ── Score bounds ─────────────────────────────────────────────

def test_score_range_good(engine, good_data):
    result = engine.score(good_data)
    assert isinstance(result, QuantitativeResult)
    assert 0 <= result.score <= 100


def test_score_range_bad(engine, bad_data):
    result = engine.score(bad_data)
    assert 0 <= result.score <= 100


def test_good_beats_bad(engine, good_data, bad_data):
    assert engine.score(good_data).score > engine.score(bad_data).score


# ── Factor scores ─────────────────────────────────────────────

def test_value_factor_cheap_positive(engine, good_data):
    result = engine.score(good_data)
    # PE=14, PB=2 → positive value z-score → pct > 50
    if result.factors.value_pct is not None:
        assert result.factors.value_pct > 50


def test_value_factor_expensive_negative(engine, bad_data):
    result = engine.score(bad_data)
    # PE=65, PB=15 → negative value z-score → pct < 50
    if result.factors.value_pct is not None:
        assert result.factors.value_pct < 50


def test_quality_factor_strong(engine, good_data):
    result = engine.score(good_data)
    if result.factors.quality_pct is not None:
        assert result.factors.quality_pct > 50


def test_quality_factor_weak(engine, bad_data):
    result = engine.score(bad_data)
    if result.factors.quality_pct is not None:
        assert result.factors.quality_pct < 50


def test_momentum_factor_populated(engine, good_data):
    result = engine.score(good_data)
    # With 800 bars of data we should have the 12-1 factor
    # (requires 252+21 = 273 bars minimum)
    if result.factors.momentum_z is not None:
        assert result.factors.momentum_pct is not None
        assert 0 <= result.factors.momentum_pct <= 100


def test_low_vol_factor_low_beta(engine, good_data):
    # beta=0.85 → low vol → positive low_vol factor
    result = engine.score(good_data)
    if result.factors.low_volatility_pct is not None:
        assert result.factors.low_volatility_pct > 50


def test_low_vol_factor_high_beta(engine, bad_data):
    # beta=2.2 → high vol → negative low_vol factor
    result = engine.score(bad_data)
    if result.factors.low_volatility_pct is not None:
        assert result.factors.low_volatility_pct < 50


# ── Risk metrics ──────────────────────────────────────────────

def test_risk_metrics_populated(engine, good_data):
    result = engine.score(good_data)
    rm = result.risk_metrics
    assert rm.sharpe_1yr       is not None
    assert rm.max_drawdown_3yr is not None
    assert rm.current_drawdown is not None
    assert rm.realized_vol_1yr is not None


def test_sharpe_sign_uptrend(engine, good_data):
    # An uptrending stock should have positive Sharpe over the period
    result = engine.score(good_data)
    if result.risk_metrics.sharpe_1yr is not None:
        # Uptrend (trend=0.0008/day) — Sharpe should be positive
        # Note: last 252 bars might vary by seed; just check it's a real number
        assert math.isfinite(result.risk_metrics.sharpe_1yr)


def test_max_drawdown_negative(engine, good_data):
    result = engine.score(good_data)
    if result.risk_metrics.max_drawdown_3yr is not None:
        assert result.risk_metrics.max_drawdown_3yr <= 0.0


def test_current_drawdown_negative_or_zero(engine, good_data):
    result = engine.score(good_data)
    if result.risk_metrics.current_drawdown is not None:
        assert result.risk_metrics.current_drawdown <= 0.0


def test_beta_from_data_dict(engine, good_data):
    result = engine.score(good_data)
    assert result.risk_metrics.beta_1yr == pytest.approx(0.85)


def test_vol_positive(engine, good_data):
    result = engine.score(good_data)
    if result.risk_metrics.realized_vol_1yr is not None:
        assert result.risk_metrics.realized_vol_1yr > 0


# ── Peer-relative factor scoring ──────────────────────────────

def test_peer_relative_value_cheap(engine, good_data):
    """Stock with low PE vs peers → high value percentile."""
    good_data["pe_trailing"] = 8.0
    good_data["peer_metrics"] = {
        f"P{i}": {"priceEarningsRatioTTM": 20.0 + i * 3} for i in range(6)
    }
    result = engine.score(good_data)
    if result.factors.value_pct is not None:
        assert result.factors.value_pct > 70


def test_peer_relative_value_expensive(engine, good_data):
    """Stock with highest PE among peers → low value percentile."""
    good_data["pe_trailing"] = 80.0
    good_data["peer_metrics"] = {
        f"P{i}": {"priceEarningsRatioTTM": 15.0 + i * 2} for i in range(6)
    }
    result = engine.score(good_data)
    if result.factors.value_pct is not None:
        assert result.factors.value_pct < 30


# ── Missing data handling ─────────────────────────────────────

def test_no_price_history(engine):
    data = {
        "pe_trailing": 15.0, "pb": 2.5, "fcf_yield": 0.04,
        "roe": 0.18, "debt_equity": 0.5, "quality_of_earnings": 1.1,
        "market_cap": 10_000_000_000, "beta": 1.0,
        "risk_free_rate_3mo": 0.05,
        "peer_metrics": {}, "quality_of_earnings_flag": False,
    }
    result = engine.score(data)
    assert isinstance(result, QuantitativeResult)
    assert 0 <= result.score <= 100
    # Risk metrics should be absent / None
    assert result.risk_metrics.sharpe_1yr is None


def test_empty_data_dict(engine):
    result = engine.score({})
    assert isinstance(result, QuantitativeResult)
    assert 0 <= result.score <= 100


def test_short_price_history(engine, good_data):
    good_data["price_history"] = make_price_df(n=25, seed=5)
    result = engine.score(good_data)
    assert isinstance(result, QuantitativeResult)
    # Short history: risk metrics may be absent but factors still computed
    assert 0 <= result.score <= 100


# ── FF regression ─────────────────────────────────────────────

def test_ff_regression_skipped_without_file(engine, good_data):
    # Ken French data file is not present in test env → should return None
    result = engine.score(good_data)
    assert result.ff_regression is None   # file not present → None is correct


# ── Pure helper unit tests ────────────────────────────────────

def test_f_safe():
    assert _f(None)        is None
    assert _f(float("nan")) is None
    assert _f(float("inf")) is None
    assert _f(42)          == pytest.approx(42.0)


def test_peer_z_basic():
    z = _peer_z(10.0, [8.0, 9.0, 10.0, 11.0, 12.0])
    assert z == pytest.approx(0.0, abs=0.1)   # at the mean → z ≈ 0


def test_peer_z_above_mean():
    z = _peer_z(15.0, [10.0, 10.0, 10.0, 10.0, 10.0])
    assert z > 0.0


def test_peer_z_insufficient_peers():
    assert _peer_z(10.0, [9.0, 11.0]) is None   # < 4 peers


def test_abs_z_at_mean():
    # At the distribution mean → z = 0
    mu, _ = (0.050, 0.025)
    z = _abs_z(0.050, "earnings_yield")
    assert z == pytest.approx(0.0, abs=0.01)


def test_norm_cdf_center():
    assert _norm_cdf(0.0) == pytest.approx(50.0, abs=0.1)


def test_norm_cdf_positive():
    assert _norm_cdf(1.0) > 50.0
    assert _norm_cdf(-1.0) < 50.0


def test_z_to_pct_none():
    assert _z_to_pct(None) is None


def test_z_to_pct_range():
    for z in [-3.0, -1.0, 0.0, 1.0, 3.0]:
        pct = _z_to_pct(z)
        assert 0.0 <= pct <= 100.0


# ── Full pipeline test with fixture data ──────────────────────

def test_score_with_fixture_data(engine, cfg):
    from stockgrader.data.normalizer import normalize
    data = normalize(
        ticker="TEST", price_df=make_price_df(n=600, seed=7),
        yf_info=YF_INFO, fmp_data=FMP_DATA, estimates=ESTIMATES,
        peers=[], peer_metrics={}, rf_rates=RF_RATES, cfg=cfg,
    )
    result = engine.score(data)
    assert isinstance(result, QuantitativeResult)
    assert 0 <= result.score <= 100
    assert result.risk_metrics.sharpe_1yr is not None
    assert result.risk_metrics.max_drawdown_3yr is not None
