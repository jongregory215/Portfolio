"""
Market cycle gauge — a one-time, market-wide "where are we in the cycle?"
reading, independent of any single ticker.

Combines three widely-watched indicators of investor psychology and credit
conditions:
  - Yield curve spread (10yr - 2yr treasuries, FRED T10Y2Y): inversion is a
    classic late-cycle / complacency signal.
  - High-yield credit spread (ICE BofA US HY OAS, FRED BAMLH0A0HYM2): tight
    spreads signal complacency ("greed"); wide spreads signal risk-aversion
    ("fear").
  - VIX: low readings signal complacency; elevated readings signal fear.

The resulting zone adjusts ``mos_multiplier``, which loosens or tightens the
margin-of-safety bar used by criterion #5 (Earnings Power Value) in
``criteria.py`` — per Marks, the discount you should demand varies with where
the pendulum currently sits.
"""
from __future__ import annotations

import logging
from datetime import date

import yfinance as yf

from stockgrader.data.fred_provider import FREDProvider
from stockgrader.howard_marks.models import CycleReading

logger = logging.getLogger(__name__)

_YIELD_CURVE_SERIES = "T10Y2Y"
_HY_SPREAD_SERIES   = "BAMLH0A0HYM2"

_HY_SPREAD_TIGHT = 3.5   # %, below this = complacency
_HY_SPREAD_WIDE  = 6.0   # %, above this = fear / risk-off
_VIX_LOW  = 15.0
_VIX_HIGH = 25.0


def _get_vix() -> float | None:
    try:
        hist = yf.Ticker("^VIX").history(period="5d")
        if hist is None or hist.empty:
            return None
        return float(hist["Close"].dropna().iloc[-1])
    except Exception as exc:
        logger.warning("VIX fetch failed: %s", exc)
        return None


def get_cycle_reading() -> CycleReading:
    """Compute a one-time market-cycle reading. Degrades to 'Neutral' if macro data is unavailable."""
    fred = FREDProvider()

    yield_curve_spread = fred.get_latest(_YIELD_CURVE_SERIES)
    high_yield_spread  = fred.get_latest(_HY_SPREAD_SERIES)
    vix                = _get_vix()

    fear_signals  = 0
    greed_signals = 0
    notes: list[str] = []

    if yield_curve_spread is not None:
        if yield_curve_spread < 0:
            greed_signals += 1
            notes.append(f"yield curve inverted ({yield_curve_spread:+.2f} pts, 10yr-2yr)")
        else:
            notes.append(f"yield curve {yield_curve_spread:+.2f} pts (10yr-2yr), not inverted")
    else:
        notes.append("yield curve spread unavailable")

    if high_yield_spread is not None:
        if high_yield_spread < _HY_SPREAD_TIGHT:
            greed_signals += 1
            notes.append(f"high-yield credit spreads tight ({high_yield_spread:.2f}%)")
        elif high_yield_spread > _HY_SPREAD_WIDE:
            fear_signals += 1
            notes.append(f"high-yield credit spreads wide ({high_yield_spread:.2f}%)")
        else:
            notes.append(f"high-yield credit spreads in normal range ({high_yield_spread:.2f}%)")
    else:
        notes.append("high-yield credit spread unavailable")

    if vix is not None:
        if vix < _VIX_LOW:
            greed_signals += 1
            notes.append(f"VIX low ({vix:.1f})")
        elif vix > _VIX_HIGH:
            fear_signals += 1
            notes.append(f"VIX elevated ({vix:.1f})")
        else:
            notes.append(f"VIX in normal range ({vix:.1f})")
    else:
        notes.append("VIX unavailable")

    have_any_signal = (yield_curve_spread is not None or high_yield_spread is not None or vix is not None)

    if not have_any_signal:
        zone = "Neutral"
        mos_multiplier = 1.00
        commentary = (
            "No macro data available (FRED_API_KEY not set, or providers unreachable) — "
            "treating the cycle as neutral. " + "; ".join(notes) + "."
        )
    elif fear_signals > greed_signals:
        zone = "Fear / Capitulation"
        mos_multiplier = 1.10
        commentary = (
            "Indicators skew toward fear/risk-aversion (" + "; ".join(notes) + "). "
            "Marks would lean toward being more aggressive — willing to pay closer to "
            "estimated value."
        )
    elif greed_signals > fear_signals:
        zone = "Greed / Late-Cycle"
        mos_multiplier = 0.90
        commentary = (
            "Indicators skew toward complacency/late-cycle (" + "; ".join(notes) + "). "
            "Marks would urge caution — demand a bigger discount to estimated value."
        )
    else:
        zone = "Neutral"
        mos_multiplier = 1.00
        commentary = "Indicators are mixed/neutral (" + "; ".join(notes) + ")."

    return CycleReading(
        as_of=date.today().isoformat(),
        yield_curve_spread=yield_curve_spread,
        high_yield_spread=high_yield_spread,
        vix=vix,
        zone=zone,
        commentary=commentary,
        mos_multiplier=mos_multiplier,
    )
