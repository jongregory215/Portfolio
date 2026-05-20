"""
Technical Engine — scores a stock across three pillars.

  Trend (40%):            Price vs SMA50/200, golden/death cross, ADX, SMA200 slope
  Momentum (35%):         RSI, MACD, Bollinger %, rate-of-change, Stochastic
  Volume / Structure (25%): OBV trend, volume ratio, support/resistance proximity

Regime-aware scoring: an RSI of 75 is a positive momentum signal in a
strong uptrend (ADX > 25, price > SMA200) but an overbought warning in a
sideways market.  Every momentum metric is re-weighted by the detected regime.

All indicators are computed from pandas/numpy without external TA libraries
(except Stochastic and ADX where ta is used with a graceful fallback) so the
engine is portable and robust to library version changes.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from stockgrader.engines.base import BaseEngine
from stockgrader.models import TechnicalPillars, TechnicalResult

logger = logging.getLogger(__name__)

MIN_BARS        = 50    # minimum bars required to produce a useful score
SWING_LOOKBACK  = 8     # bars each side for swing-point detection
MAX_SWING_LEVELS= 10    # keep this many most-recent swing points


# ──────────────────────────────────────────────────────────────
# Pure indicator functions (pandas / numpy only)
# ──────────────────────────────────────────────────────────────

def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(window, min_periods=window).mean()
    loss  = (-delta.clip(upper=0)).rolling(window, min_periods=window).mean()
    # When loss is zero (all gains, no losses) → RSI = 100
    rs    = gain / loss.where(loss != 0, np.nan)
    rsi   = 100.0 - 100.0 / (1.0 + rs)
    return rsi.where(loss != 0, 100.0)


def _macd(close: pd.Series, fast: int = 12, slow: int = 26, sig: int = 9):
    ema_fast  = close.ewm(span=fast, adjust=False).mean()
    ema_slow  = close.ewm(span=slow, adjust=False).mean()
    line      = ema_fast - ema_slow
    signal    = line.ewm(span=sig, adjust=False).mean()
    histogram = line - signal
    return line, signal, histogram


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([high - low,
                    (high - prev_close).abs(),
                    (low  - prev_close).abs()], axis=1).max(axis=1)
    return tr.rolling(window, min_periods=window).mean()


def _bollinger(close: pd.Series, window: int = 20, n_std: float = 2.0):
    mid   = close.rolling(window, min_periods=window).mean()
    std   = close.rolling(window, min_periods=window).std()
    upper = mid + n_std * std
    lower = mid - n_std * std
    pct_b = (close - lower) / (upper - lower).replace(0, np.nan)
    return mid, upper, lower, pct_b


def _obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    direction = np.sign(close.diff().fillna(0))
    return (direction * volume).cumsum()


def _adx_ta(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14):
    """ADX via ta library; returns (adx, plus_di, minus_di) or (None, None, None)."""
    try:
        from ta.trend import ADXIndicator  # type: ignore[import]
        ind = ADXIndicator(high, low, close, window=window, fillna=False)
        return ind.adx(), ind.adx_pos(), ind.adx_neg()
    except Exception as exc:
        logger.debug("ta.ADXIndicator unavailable: %s — ADX will be None.", exc)
        return None, None, None


def _stochastic_ta(high: pd.Series, low: pd.Series, close: pd.Series, k: int = 14, d: int = 3):
    """Stochastic K/D via ta library; returns (k_series, d_series) or (None, None)."""
    try:
        from ta.momentum import StochasticOscillator  # type: ignore[import]
        ind = StochasticOscillator(high, low, close, window=k, smooth_window=d, fillna=False)
        return ind.stoch(), ind.stoch_signal()
    except Exception as exc:
        logger.debug("ta.Stochastic unavailable: %s — Stochastic will be None.", exc)
        return None, None


def _swing_lows(low: pd.Series, n: int = SWING_LOOKBACK) -> list[float]:
    """Return the most-recent swing-low price levels."""
    levels = []
    vals   = low.values
    for i in range(n, len(vals) - n):
        window = vals[i - n: i + n + 1]
        if vals[i] == window.min():
            levels.append(float(vals[i]))
    return levels[-MAX_SWING_LEVELS:]


def _swing_highs(high: pd.Series, n: int = SWING_LOOKBACK) -> list[float]:
    """Return the most-recent swing-high price levels."""
    levels = []
    vals   = high.values
    for i in range(n, len(vals) - n):
        window = vals[i - n: i + n + 1]
        if vals[i] == window.max():
            levels.append(float(vals[i]))
    return levels[-MAX_SWING_LEVELS:]


def _nearest_below(price: float, levels: list[float]) -> float | None:
    candidates = [l for l in levels if l < price]
    return max(candidates) if candidates else None


def _nearest_above(price: float, levels: list[float]) -> float | None:
    candidates = [l for l in levels if l > price]
    return min(candidates) if candidates else None


# ──────────────────────────────────────────────────────────────
# Technical Engine
# ──────────────────────────────────────────────────────────────

class TechnicalEngine(BaseEngine):

    def score(self, data: dict[str, Any]) -> TechnicalResult:
        df: pd.DataFrame | None = data.get("price_history")

        if df is None or len(df) < MIN_BARS:
            return self._insufficient(len(df) if df is not None else 0)

        ind    = self._compute_indicators(df)
        regime = self._detect_regime(ind)

        pw = self.config.get("technical", {}).get("pillar_weights", {
            "trend":            0.40,
            "momentum":         0.35,
            "volume_structure": 0.25,
        })

        trend_score, trend_det, trend_miss = self._score_trend(ind, regime)
        mom_score,   mom_det,   mom_miss   = self._score_momentum(ind, regime)
        vol_score,   vol_det,   vol_miss   = self._score_volume_structure(ind, df)

        pillar_scores = {
            "trend":            trend_score,
            "momentum":         mom_score,
            "volume_structure": vol_score,
        }
        composite = self.weighted_average(
            {k: v for k, v in pillar_scores.items() if v is not None},
            pw,
        )

        pillars = TechnicalPillars(
            trend            = trend_score or 50.0,
            momentum         = mom_score   or 50.0,
            volume_structure = vol_score   or 50.0,
            details={
                "trend":            trend_det,
                "momentum":         mom_det,
                "volume_structure": vol_det,
            },
        )

        return TechnicalResult(
            score               = self.clamp(composite),
            pillars             = pillars,
            regime              = regime,
            nearest_support     = ind.get("nearest_support"),
            nearest_resistance  = ind.get("nearest_resistance"),
            week_52_high        = ind.get("high_52w"),
            week_52_low         = ind.get("low_52w"),
            missing_fields      = sorted(set(trend_miss + mom_miss + vol_miss)),
        )

    # ── Indicator computation ─────────────────────────────────

    def _compute_indicators(self, df: pd.DataFrame) -> dict[str, Any]:
        close  = df["close"].astype(float)
        high   = df["high"].astype(float)
        low    = df["low"].astype(float)
        volume = df["volume"].astype(float)
        n      = len(df)

        def last(s: pd.Series) -> float | None:
            v = s.iloc[-1] if len(s) else np.nan
            return None if (pd.isna(v) or np.isinf(v)) else float(v)

        price = last(close)

        # Moving averages
        sma = {p: close.rolling(p, min_periods=p).mean() for p in [20, 50, 100, 200]}
        ema = {p: close.ewm(span=p, adjust=False).mean()  for p in [20, 50]}

        # RSI
        rsi_series  = _rsi(close, 14)
        rsi_prev    = rsi_series.iloc[-2] if n > 1 else np.nan

        # MACD
        macd_line, macd_sig, macd_hist = _macd(close)
        macd_hist_prev = macd_hist.iloc[-2] if n > 1 else np.nan

        # ATR
        atr_val = last(_atr(high, low, close, 14))

        # Bollinger
        bb_mid, bb_upper, bb_lower, bb_pct = _bollinger(close)

        # OBV
        obv_series = _obv(close, volume)
        obv_20_ago = obv_series.iloc[-21] if n >= 21 else None

        # Volume
        vol_avg20 = volume.rolling(20, min_periods=10).mean()

        # 52-week levels
        days_yr = min(252, n)
        high_52w = float(high.iloc[-days_yr:].max())
        low_52w  = float(low.iloc[-days_yr:].min())

        # SMA200 slope: % change over last 20 bars
        sma200_now  = last(sma[200])
        sma200_20   = sma[200].iloc[-20] if n >= 220 else None
        sma200_slope = (
            (sma200_now - float(sma200_20)) / float(sma200_20)
            if (sma200_now and sma200_20 and float(sma200_20) != 0) else None
        )

        # Rate of change
        roc_20 = ((price / float(close.iloc[-21])) - 1) if (price and n >= 21) else None
        roc_60 = ((price / float(close.iloc[-61])) - 1) if (price and n >= 61) else None

        # ADX (ta library)
        adx_s, plus_di_s, minus_di_s = _adx_ta(high, low, close, 14)
        adx_val    = last(adx_s)    if adx_s    is not None else None
        plus_di    = last(plus_di_s)  if plus_di_s  is not None else None
        minus_di   = last(minus_di_s) if minus_di_s is not None else None

        # Stochastic (ta library)
        stoch_k_s, stoch_d_s = _stochastic_ta(high, low, close)
        stoch_k = last(stoch_k_s) if stoch_k_s is not None else None
        stoch_d = last(stoch_d_s) if stoch_d_s is not None else None

        # Swing levels (last 252 bars)
        look_df = df.iloc[-252:] if n >= 252 else df
        s_lows  = _swing_lows(look_df["low"].astype(float))
        s_highs = _swing_highs(look_df["high"].astype(float))

        nearest_support    = _nearest_below(price, s_lows)  if price else None
        nearest_resistance = _nearest_above(price, s_highs) if price else None

        # Breakout / breakdown: did price cross a level in the last 5 bars?
        recent_high = float(high.iloc[-5:].max()) if n >= 5 else None
        recent_low  = float(low.iloc[-5:].min())  if n >= 5 else None
        breakout    = (nearest_resistance and recent_high and
                       recent_high > nearest_resistance)
        breakdown   = (nearest_support and recent_low and
                       recent_low < nearest_support)

        obv_now = last(obv_series)
        obv_trend = (
            (obv_now - float(obv_20_ago)) / abs(float(obv_20_ago))
            if (obv_now is not None and obv_20_ago is not None
                and float(obv_20_ago) != 0) else None
        )

        vol_ratio = (
            float(volume.iloc[-1]) / float(vol_avg20.iloc[-1])
            if (not pd.isna(vol_avg20.iloc[-1]) and float(vol_avg20.iloc[-1]) > 0) else None
        )

        return {
            "price":            price,
            "sma20":            last(sma[20]),
            "sma50":            last(sma[50]),
            "sma100":           last(sma[100]),
            "sma200":           last(sma[200]),
            "ema20":            last(ema[20]),
            "ema50":            last(ema[50]),
            "rsi":              last(rsi_series),
            "rsi_prev":         float(rsi_prev) if not pd.isna(rsi_prev) else None,
            "macd_line":        last(macd_line),
            "macd_signal":      last(macd_sig),
            "macd_hist":        last(macd_hist),
            "macd_hist_prev":   float(macd_hist_prev) if not pd.isna(macd_hist_prev) else None,
            "atr":              atr_val,
            "bb_mid":           last(bb_mid),
            "bb_upper":         last(bb_upper),
            "bb_lower":         last(bb_lower),
            "bb_pct":           last(bb_pct),
            "adx":              adx_val,
            "plus_di":          plus_di,
            "minus_di":         minus_di,
            "stoch_k":          stoch_k,
            "stoch_d":          stoch_d,
            "obv_trend":        obv_trend,
            "vol_ratio":        vol_ratio,
            "high_52w":         high_52w,
            "low_52w":          low_52w,
            "sma200_slope":     sma200_slope,
            "roc_20":           roc_20,
            "roc_60":           roc_60,
            "nearest_support":  nearest_support,
            "nearest_resistance": nearest_resistance,
            "breakout":         breakout,
            "breakdown":        breakdown,
        }

    # ── Regime detection ──────────────────────────────────────

    def _detect_regime(self, ind: dict) -> str:
        price   = ind.get("price")
        sma50   = ind.get("sma50")
        sma200  = ind.get("sma200")
        adx     = ind.get("adx") or 20.0

        if price is None:
            return "unknown"

        above_200    = (sma200 is not None) and (price > sma200)
        above_50     = (sma50  is not None) and (price > sma50)
        golden_cross = (sma50 is not None and sma200 is not None and sma50 > sma200)
        strong_trend = adx > 25.0

        if above_200 and golden_cross and strong_trend:
            return "strong_uptrend"
        elif above_200 and (above_50 or golden_cross):
            return "moderate_uptrend"
        elif not above_200 and not golden_cross and strong_trend:
            return "strong_downtrend"
        elif not above_200 and (not above_50 or not golden_cross):
            return "moderate_downtrend"
        else:
            return "sideways"

    # ── Trend pillar ──────────────────────────────────────────

    def _score_trend(
        self, ind: dict, regime: str
    ) -> tuple[float, dict, list[str]]:
        scores:  dict[str, float] = {}
        details: dict[str, Any]   = {}
        missing: list[str]        = []

        price   = ind.get("price")
        sma50   = ind.get("sma50")
        sma200  = ind.get("sma200")
        adx     = ind.get("adx")
        slope   = ind.get("sma200_slope")

        # Price vs SMA50
        if price and sma50:
            pct = (price - sma50) / sma50
            details["pct_above_sma50"] = round(pct * 100, 1)
            if pct >  0.10: scores["vs_sma50"] = 92.0
            elif pct > 0.04: scores["vs_sma50"] = 75.0
            elif pct > 0.00: scores["vs_sma50"] = 60.0
            elif pct > -0.05: scores["vs_sma50"] = 42.0
            elif pct > -0.12: scores["vs_sma50"] = 28.0
            else:             scores["vs_sma50"] = 12.0
        else:
            missing.append("sma50")

        # Price vs SMA200
        if price and sma200:
            pct = (price - sma200) / sma200
            details["pct_above_sma200"] = round(pct * 100, 1)
            if pct >  0.15: scores["vs_sma200"] = 90.0
            elif pct > 0.05: scores["vs_sma200"] = 75.0
            elif pct > 0.00: scores["vs_sma200"] = 62.0
            elif pct > -0.05: scores["vs_sma200"] = 42.0
            elif pct > -0.15: scores["vs_sma200"] = 25.0
            else:              scores["vs_sma200"] = 10.0
        else:
            missing.append("sma200")

        # Golden / death cross
        if sma50 and sma200:
            cross_pct = (sma50 - sma200) / sma200
            details["sma50_vs_sma200_pct"] = round(cross_pct * 100, 1)
            if cross_pct >  0.05: scores["cross"] = 88.0   # established golden cross
            elif cross_pct > 0.00: scores["cross"] = 68.0  # just crossed above
            elif cross_pct > -0.05: scores["cross"] = 35.0 # just crossed below
            else:                   scores["cross"] = 15.0  # established death cross
        else:
            missing.append("cross_state")

        # ADX — direction-aware
        if adx is not None:
            details["adx"] = round(adx, 1)
            above_200 = (price and sma200 and price > sma200)
            if adx > 35:
                scores["adx"] = 90.0 if above_200 else 15.0
            elif adx > 25:
                scores["adx"] = 78.0 if above_200 else 25.0
            elif adx > 18:
                scores["adx"] = 58.0 if above_200 else 42.0
            else:
                scores["adx"] = 50.0   # weak/choppy trend — neutral
        else:
            missing.append("adx")

        # SMA200 slope
        if slope is not None:
            details["sma200_slope_pct"] = round(slope * 100, 2)
            if slope >  0.03:  scores["sma200_slope"] = 95.0
            elif slope > 0.01: scores["sma200_slope"] = 78.0
            elif slope > 0.00: scores["sma200_slope"] = 60.0
            elif slope > -0.01: scores["sma200_slope"] = 40.0
            elif slope > -0.03: scores["sma200_slope"] = 22.0
            else:               scores["sma200_slope"] = 8.0
        else:
            missing.append("sma200_slope")

        composite = self.weighted_average(scores, {k: 1.0 for k in scores}) if scores else 50.0
        details["regime"] = regime
        return composite, details, missing

    # ── Momentum pillar ───────────────────────────────────────

    def _score_momentum(
        self, ind: dict, regime: str
    ) -> tuple[float, dict, list[str]]:
        scores:  dict[str, float] = {}
        details: dict[str, Any]   = {}
        missing: list[str]        = []

        rsi       = ind.get("rsi")
        rsi_prev  = ind.get("rsi_prev")
        macd_hist = ind.get("macd_hist")
        macd_prev = ind.get("macd_hist_prev")
        bb_pct    = ind.get("bb_pct")
        stoch_k   = ind.get("stoch_k")
        roc_20    = ind.get("roc_20")
        roc_60    = ind.get("roc_60")

        is_uptrend   = regime in ("strong_uptrend",   "moderate_uptrend")
        is_downtrend = regime in ("strong_downtrend", "moderate_downtrend")

        # RSI — regime-aware
        if rsi is not None:
            details["rsi"] = round(rsi, 1)
            if is_uptrend:
                # In uptrends, higher RSI is generally confirmatory
                if rsi >= 80:    scores["rsi"] = 65.0   # extended; still bullish
                elif rsi >= 60:  scores["rsi"] = 88.0   # sweet spot for uptrend
                elif rsi >= 45:  scores["rsi"] = 62.0   # momentum cooling
                elif rsi >= 30:  scores["rsi"] = 38.0   # losing momentum
                else:            scores["rsi"] = 20.0   # oversold in uptrend; unusual
            elif is_downtrend:
                # In downtrends, any elevated RSI is likely a dead-cat bounce
                if rsi >= 65:    scores["rsi"] = 25.0   # overbought in downtrend
                elif rsi >= 50:  scores["rsi"] = 32.0
                elif rsi >= 35:  scores["rsi"] = 42.0   # slight oversold bounce possible
                else:            scores["rsi"] = 22.0   # deeply oversold downtrend
            else:  # sideways
                if rsi >= 70:    scores["rsi"] = 28.0   # overbought in range
                elif rsi >= 55:  scores["rsi"] = 58.0
                elif rsi >= 45:  scores["rsi"] = 62.0   # neutral zone = good
                elif rsi >= 30:  scores["rsi"] = 48.0
                else:            scores["rsi"] = 32.0   # oversold (possible bounce)

            # RSI momentum: is it rising or falling?
            if rsi_prev is not None:
                rsi_direction = rsi - rsi_prev
                details["rsi_direction"] = round(rsi_direction, 1)
                if rsi_direction > 2 and is_uptrend:
                    scores["rsi"] = min(scores.get("rsi", 50) + 8, 100)
                elif rsi_direction < -2 and is_uptrend:
                    scores["rsi"] = max(scores.get("rsi", 50) - 8, 0)
        else:
            missing.append("rsi")

        # MACD histogram — direction and momentum
        if macd_hist is not None:
            details["macd_hist"]  = round(macd_hist, 4)
            rising = (macd_prev is not None and macd_hist > macd_prev)
            if macd_hist > 0 and rising:
                scores["macd"] = 88.0   # bullish and strengthening
            elif macd_hist > 0:
                scores["macd"] = 68.0   # bullish but weakening
            elif macd_hist < 0 and not rising:
                scores["macd"] = 18.0   # bearish and strengthening downside
            else:
                scores["macd"] = 42.0   # bearish but turning up (early recovery)
        else:
            missing.append("macd")

        # Bollinger Band % position (0=lower, 0.5=mid, 1=upper)
        if bb_pct is not None:
            details["bb_pct"] = round(bb_pct, 2)
            if is_uptrend:
                # Riding the upper band is healthy in uptrends
                if bb_pct >= 0.8:  scores["bb"] = 75.0
                elif bb_pct >= 0.5: scores["bb"] = 70.0
                elif bb_pct >= 0.2: scores["bb"] = 52.0
                else:               scores["bb"] = 30.0
            else:
                # Neutral or downtrend: upper band = overbought
                if bb_pct >= 0.9:  scores["bb"] = 25.0
                elif bb_pct >= 0.6: scores["bb"] = 52.0
                elif bb_pct >= 0.4: scores["bb"] = 62.0
                elif bb_pct >= 0.1: scores["bb"] = 48.0
                else:               scores["bb"] = 35.0   # oversold
        else:
            missing.append("bb_pct")

        # Stochastic K
        if stoch_k is not None:
            details["stoch_k"] = round(stoch_k, 1)
            if is_uptrend:
                if stoch_k >= 80:  scores["stoch"] = 72.0
                elif stoch_k >= 50: scores["stoch"] = 80.0
                elif stoch_k >= 20: scores["stoch"] = 48.0
                else:               scores["stoch"] = 28.0
            elif is_downtrend:
                if stoch_k >= 80:  scores["stoch"] = 22.0
                elif stoch_k >= 50: scores["stoch"] = 35.0
                elif stoch_k >= 20: scores["stoch"] = 42.0
                else:               scores["stoch"] = 25.0
            else:  # sideways
                if stoch_k >= 80:  scores["stoch"] = 28.0
                elif stoch_k >= 50: scores["stoch"] = 60.0
                elif stoch_k >= 20: scores["stoch"] = 52.0
                else:               scores["stoch"] = 38.0
        else:
            missing.append("stochastic")

        # Rate of change (3-month = 60 trading days)
        if roc_60 is not None:
            details["roc_60_pct"] = round(roc_60 * 100, 1)
            if roc_60 >  0.20:  scores["roc"] = 90.0
            elif roc_60 > 0.10: scores["roc"] = 78.0
            elif roc_60 > 0.05: scores["roc"] = 65.0
            elif roc_60 > 0.00: scores["roc"] = 52.0
            elif roc_60 > -0.05: scores["roc"] = 38.0
            elif roc_60 > -0.15: scores["roc"] = 22.0
            else:                scores["roc"] = 10.0
        elif roc_20 is not None:
            details["roc_20_pct"] = round(roc_20 * 100, 1)
            if roc_20 >  0.08:  scores["roc"] = 80.0
            elif roc_20 > 0.03: scores["roc"] = 65.0
            elif roc_20 > 0.00: scores["roc"] = 52.0
            elif roc_20 > -0.03: scores["roc"] = 38.0
            else:                scores["roc"] = 20.0
        else:
            missing.append("rate_of_change")

        composite = self.weighted_average(scores, {k: 1.0 for k in scores}) if scores else 50.0
        return composite, details, missing

    # ── Volume / Structure pillar ─────────────────────────────

    def _score_volume_structure(
        self, ind: dict, df: pd.DataFrame
    ) -> tuple[float, dict, list[str]]:
        scores:  dict[str, float] = {}
        details: dict[str, Any]   = {}
        missing: list[str]        = []

        obv_trend = ind.get("obv_trend")
        vol_ratio = ind.get("vol_ratio")
        price     = ind.get("price")
        support   = ind.get("nearest_support")
        resistance= ind.get("nearest_resistance")
        breakout  = ind.get("breakout")
        breakdown = ind.get("breakdown")

        # OBV trend (volume confirming price action?)
        if obv_trend is not None:
            details["obv_trend_pct"] = round(obv_trend * 100, 1)
            if obv_trend >  0.05:  scores["obv"] = 88.0
            elif obv_trend > 0.01: scores["obv"] = 72.0
            elif obv_trend > -0.01: scores["obv"] = 52.0
            elif obv_trend > -0.05: scores["obv"] = 32.0
            else:                   scores["obv"] = 15.0
        else:
            missing.append("obv")

        # Volume ratio (current bar vs 20-day average)
        if vol_ratio is not None:
            details["vol_ratio"] = round(vol_ratio, 2)
            # High volume on a rising day = bullish confirmation
            # High volume on a falling day = bearish
            # Low volume = lack of conviction (neutral)
            close = df["close"].astype(float)
            recent_positive = (len(close) >= 2 and close.iloc[-1] > close.iloc[-2])
            if vol_ratio >= 2.0:
                scores["volume"] = 80.0 if recent_positive else 20.0
            elif vol_ratio >= 1.3:
                scores["volume"] = 72.0 if recent_positive else 30.0
            elif vol_ratio >= 0.7:
                scores["volume"] = 55.0
            else:
                scores["volume"] = 45.0   # below-average volume = lack of conviction
        else:
            missing.append("volume_ratio")

        # Support / resistance proximity
        if price and support:
            dist_sup = (price - support) / price
            details["dist_to_support_pct"] = round(dist_sup * 100, 1)
            # Near support = potential entry; bouncing off = bullish
            if dist_sup <= 0.02:   scores["support"] = 78.0  # at support
            elif dist_sup <= 0.05: scores["support"] = 68.0
            elif dist_sup <= 0.10: scores["support"] = 58.0
            else:                   scores["support"] = 50.0  # far from support
        else:
            missing.append("nearest_support")

        if price and resistance:
            dist_res = (resistance - price) / price
            details["dist_to_resistance_pct"] = round(dist_res * 100, 1)
            # Near resistance = upside limited; just broke through = very bullish
            if dist_res <= 0.02:   scores["resistance"] = 35.0  # about to be capped
            elif dist_res <= 0.05: scores["resistance"] = 48.0
            elif dist_res <= 0.10: scores["resistance"] = 58.0
            else:                   scores["resistance"] = 62.0  # clear air above
        else:
            missing.append("nearest_resistance")

        # Breakout / breakdown
        if breakout:
            scores["breakout"] = 92.0
            details["pattern"] = "breakout"
        elif breakdown:
            scores["breakdown"] = 10.0
            details["pattern"] = "breakdown"

        # 52-week position
        high_52w = ind.get("high_52w")
        low_52w  = ind.get("low_52w")
        if price and high_52w and low_52w and high_52w > low_52w:
            pct_of_range = (price - low_52w) / (high_52w - low_52w)
            details["52w_range_pct"] = round(pct_of_range * 100, 1)
            if pct_of_range >= 0.90:   scores["52w_pos"] = 80.0   # near highs
            elif pct_of_range >= 0.70: scores["52w_pos"] = 70.0
            elif pct_of_range >= 0.50: scores["52w_pos"] = 58.0
            elif pct_of_range >= 0.30: scores["52w_pos"] = 45.0
            else:                       scores["52w_pos"] = 30.0   # near lows
        else:
            missing.append("52w_position")

        composite = self.weighted_average(scores, {k: 1.0 for k in scores}) if scores else 50.0
        return composite, details, missing

    # ── Fallback result ───────────────────────────────────────

    def _insufficient(self, n_bars: int) -> TechnicalResult:
        msg = f"price_history has {n_bars} bars; need ≥{MIN_BARS}"
        logger.warning("TechnicalEngine: %s", msg)
        return TechnicalResult(
            score   = 50.0,
            pillars = TechnicalPillars(
                trend=50.0, momentum=50.0, volume_structure=50.0
            ),
            regime  = "unknown",
            missing_fields = ["price_history"],
        )
