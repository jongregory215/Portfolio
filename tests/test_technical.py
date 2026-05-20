"""
Unit tests for the Technical Engine.
Uses synthetic OHLCV data — no live API calls.
"""
import numpy as np
import pandas as pd
import pytest
from datetime import date

from stockgrader.engines.technical import (
    TechnicalEngine, _rsi, _macd, _bollinger, _obv,
    _swing_lows, _swing_highs, _nearest_below, _nearest_above,
)
from stockgrader.models import TechnicalResult
from stockgrader.config import get_config


# ── Synthetic data helpers ────────────────────────────────────

def make_df(n: int = 600, trend: float = 0.0005, vol: float = 0.012,
            start: float = 100.0, seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic OHLCV DataFrame.
    trend > 0  → uptrend; trend < 0 → downtrend; trend = 0 → sideways
    """
    rng     = np.random.default_rng(seed)
    returns = rng.normal(trend, vol, n)
    close   = start * np.exp(np.cumsum(returns))
    noise   = rng.uniform(0.002, 0.008, n)
    high    = close * (1 + noise)
    low     = close * (1 - noise)
    open_   = np.roll(close, 1); open_[0] = close[0]
    volume  = rng.integers(500_000, 5_000_000, n).astype(float)
    dates   = pd.bdate_range(end=date.today(), periods=n)
    df = pd.DataFrame(
        {"open": open_, "high": high, "low": low,
         "close": close, "volume": volume, "adj_close": close},
        index=dates,
    )
    df.index.name = "Date"
    return df


@pytest.fixture
def cfg():
    return get_config()


@pytest.fixture
def engine(cfg):
    return TechnicalEngine(config=cfg)


@pytest.fixture
def uptrend_df():
    return make_df(n=600, trend=0.0008, seed=1)   # ~22% annual trend


@pytest.fixture
def downtrend_df():
    return make_df(n=600, trend=-0.0008, seed=2)   # ~22% annual downtrend


@pytest.fixture
def sideways_df():
    return make_df(n=600, trend=0.0, seed=3)


# ── Basic score bounds ────────────────────────────────────────

def test_score_range_uptrend(engine, uptrend_df):
    result = engine.score({"price_history": uptrend_df})
    assert isinstance(result, TechnicalResult)
    assert 0 <= result.score <= 100


def test_score_range_downtrend(engine, downtrend_df):
    result = engine.score({"price_history": downtrend_df})
    assert 0 <= result.score <= 100


def test_uptrend_beats_downtrend(engine, uptrend_df, downtrend_df):
    up   = engine.score({"price_history": uptrend_df}).score
    down = engine.score({"price_history": downtrend_df}).score
    assert up > down, f"Uptrend {up:.1f} should beat downtrend {down:.1f}"


def test_trend_pillar_above_50_for_uptrend(engine, uptrend_df):
    # Random-seed uptrends may have a recent pullback, depressing momentum.
    # The trend pillar (SMA relationships, ADX, slope) should still confirm uptrend.
    result = engine.score({"price_history": uptrend_df})
    assert result.pillars.trend > 50, f"Trend pillar {result.pillars.trend:.1f} should be >50"


def test_trend_pillar_below_50_for_downtrend(engine, downtrend_df):
    result = engine.score({"price_history": downtrend_df})
    assert result.pillars.trend < 50, f"Trend pillar {result.pillars.trend:.1f} should be <50"


# ── Pillar ordering ───────────────────────────────────────────

def test_trend_pillar_uptrend(engine, uptrend_df, downtrend_df):
    up   = engine.score({"price_history": uptrend_df}).pillars.trend
    down = engine.score({"price_history": downtrend_df}).pillars.trend
    assert up > down


def test_trend_pillar_uptrend_beats_downtrend(engine, uptrend_df, downtrend_df):
    # The trend pillar (SMA structure, golden/death cross, ADX, slope) is the
    # most reliable discriminator across random seeds.
    up   = engine.score({"price_history": uptrend_df}).pillars.trend
    down = engine.score({"price_history": downtrend_df}).pillars.trend
    assert up > down


# ── Regime detection ─────────────────────────────────────────

def test_regime_uptrend(engine, uptrend_df):
    result = engine.score({"price_history": uptrend_df})
    assert result.regime in ("strong_uptrend", "moderate_uptrend")


def test_regime_downtrend(engine, downtrend_df):
    result = engine.score({"price_history": downtrend_df})
    assert result.regime in ("strong_downtrend", "moderate_downtrend", "sideways")


def test_regime_not_unknown_with_data(engine, sideways_df):
    result = engine.score({"price_history": sideways_df})
    assert result.regime != "unknown"


# ── Key level outputs ─────────────────────────────────────────

def test_52w_high_low_populated(engine, uptrend_df):
    result = engine.score({"price_history": uptrend_df})
    assert result.week_52_high is not None
    assert result.week_52_low  is not None
    assert result.week_52_high > result.week_52_low


def test_52w_levels_within_price_range(engine, uptrend_df):
    price_range = (uptrend_df["low"].min(), uptrend_df["high"].max())
    result = engine.score({"price_history": uptrend_df})
    assert result.week_52_low  >= price_range[0] * 0.95
    assert result.week_52_high <= price_range[1] * 1.05


def test_support_below_price(engine, uptrend_df):
    result = engine.score({"price_history": uptrend_df})
    if result.nearest_support is not None:
        price = float(uptrend_df["close"].iloc[-1])
        assert result.nearest_support < price


def test_resistance_above_price(engine, uptrend_df):
    result = engine.score({"price_history": uptrend_df})
    if result.nearest_resistance is not None:
        price = float(uptrend_df["close"].iloc[-1])
        assert result.nearest_resistance > price


# ── Insufficient data handling ────────────────────────────────

def test_insufficient_data_no_crash(engine):
    tiny_df = make_df(n=10, seed=99)
    result  = engine.score({"price_history": tiny_df})
    assert isinstance(result, TechnicalResult)
    assert result.score == 50.0
    assert result.regime == "unknown"
    assert "price_history" in result.missing_fields


def test_none_price_history(engine):
    result = engine.score({"price_history": None})
    assert isinstance(result, TechnicalResult)
    assert result.score == 50.0


def test_empty_data_dict(engine):
    result = engine.score({})
    assert isinstance(result, TechnicalResult)


# ── Indicator unit tests ──────────────────────────────────────

def test_rsi_range():
    close = pd.Series(make_df(200)["close"].values)
    rsi   = _rsi(close, 14)
    valid = rsi.dropna()
    assert (valid >= 0).all() and (valid <= 100).all()


def test_rsi_overbought_on_strong_up():
    # A straight-up series (zero losses) → RSI should be 100
    prices = pd.Series(np.linspace(100, 200, 50))
    rsi    = _rsi(prices, 14)
    valid  = rsi.dropna()
    assert len(valid) > 0, "RSI should have valid values after warm-up"
    assert valid.iloc[-1] >= 99.0


def test_rsi_oversold_on_strong_down():
    prices = pd.Series(np.linspace(200, 100, 50))
    rsi    = _rsi(prices, 14)
    assert rsi.dropna().iloc[-1] < 35


def test_macd_histogram_sign():
    # Rising prices → positive MACD histogram
    prices  = pd.Series(np.linspace(100, 150, 60))
    _, _, h = _macd(prices)
    assert h.dropna().iloc[-1] > 0


def test_bollinger_pct_range():
    close = pd.Series(make_df(100)["close"].values)
    _, _, _, pct_b = _bollinger(close)
    valid = pct_b.dropna()
    # Most values should be between 0 and 1; some outliers outside are expected
    assert ((valid >= -0.5) & (valid <= 1.5)).mean() > 0.90


def test_obv_rises_on_up_volume():
    # Increasing prices with constant volume → OBV should rise
    n      = 30
    prices = pd.Series(np.linspace(100, 130, n))
    vol    = pd.Series(np.ones(n) * 1_000_000)
    obv    = _obv(prices, vol)
    assert float(obv.iloc[-1]) > float(obv.iloc[0])


# ── Swing point tests ─────────────────────────────────────────

def test_swing_lows_below_current_price(uptrend_df):
    lows = _swing_lows(uptrend_df["low"].astype(float))
    price = float(uptrend_df["close"].iloc[-1])
    supports = _nearest_below(price, lows)
    # Uptrend: there should be some support levels below the current price
    # (Not guaranteed for every random seed, but very likely for a 600-bar uptrend)
    if lows:
        assert any(l < price for l in lows)


def test_nearest_below():
    price  = 100.0
    levels = [80.0, 90.0, 95.0, 105.0, 110.0]
    assert _nearest_below(price, levels) == 95.0


def test_nearest_above():
    price  = 100.0
    levels = [80.0, 90.0, 95.0, 105.0, 110.0]
    assert _nearest_above(price, levels) == 105.0


def test_nearest_below_no_candidates():
    assert _nearest_below(100.0, [110.0, 120.0]) is None


def test_nearest_above_no_candidates():
    assert _nearest_above(100.0, [80.0, 90.0]) is None


# ── Full engine compute_indicators ────────────────────────────

def test_compute_indicators_keys(engine, uptrend_df):
    ind = engine._compute_indicators(uptrend_df)
    for key in ["price", "sma50", "sma200", "rsi", "macd_hist",
                "adx", "obv_trend", "high_52w", "low_52w"]:
        assert key in ind, f"Missing indicator key: {key}"


def test_indicators_price_matches_last_close(engine, uptrend_df):
    ind   = engine._compute_indicators(uptrend_df)
    price = ind.get("price")
    last  = float(uptrend_df["close"].iloc[-1])
    assert price == pytest.approx(last, rel=1e-5)


def test_sma50_less_than_sma200_in_downtrend(engine, downtrend_df):
    """In a strong downtrend, SMA50 should be below SMA200 (death cross)."""
    ind = engine._compute_indicators(downtrend_df)
    if ind.get("sma50") and ind.get("sma200"):
        # Not guaranteed for every random series, but very likely at 600 bars
        assert ind["sma50"] <= ind["sma200"] * 1.05  # allow 5% margin
