"""
FRED data provider — risk-free rates for WACC and Sharpe calculations.

Reads FRED_API_KEY from environment. Falls back to config hardcoded value
if the key is absent or FRED is unreachable, and logs a warning.
"""
from __future__ import annotations

import logging
import os
from typing import Any

from stockgrader.data.cache import DiskCache
from stockgrader.config import get_cache_config, get_config

logger = logging.getLogger(__name__)

# FRED series IDs
_SERIES = {
    "3mo":  "DGS3MO",
    "10yr": "DGS10",
    "2yr":  "DGS2",
}


class FREDProvider:
    """Provides current risk-free rates via the FRED API."""

    def __init__(self, cache: DiskCache | None = None):
        self.api_key = os.environ.get("FRED_API_KEY", "")
        cfg = get_cache_config()
        self.cache = cache or DiskCache(cache_dir=cfg.get("directory"))
        self._macro_ttl = cfg["ttl"]["macro"]

        if not self.api_key:
            logger.warning(
                "FRED_API_KEY not set — risk-free rate will use config fallback (%.2f).",
                self._fallback_rate(),
            )

    def _fallback_rate(self) -> float:
        cfg = get_config()
        return float(cfg.get("optimizer", {}).get("risk_free_fallback", 0.05))

    def get_risk_free_rate(self, tenor: str = "3mo") -> float:
        """
        Return the current annualized risk-free rate as a decimal (e.g. 0.053 = 5.3%).

        tenor: "3mo" | "2yr" | "10yr"
        Falls back to config value if FRED is unavailable.
        """
        if not self.api_key:
            return self._fallback_rate()

        series_id = _SERIES.get(tenor, "DGS3MO")
        cache_key = self.cache.ticker_key("__fred__", f"rf_{tenor}")
        cached = self.cache.get(cache_key, ttl=self._macro_ttl)
        if cached is not None:
            return float(cached)

        try:
            from fredapi import Fred  # type: ignore[import]
            fred = Fred(api_key=self.api_key)
            series = fred.get_series(series_id)
            # Drop NaN and take most recent value
            val = series.dropna().iloc[-1]
            rate = float(val) / 100.0   # FRED returns in percent (e.g. 5.30)
            self.cache.set(cache_key, rate)
            return rate
        except Exception as exc:
            logger.warning("FRED request failed for %s: %s — using fallback.", series_id, exc)
            return self._fallback_rate()

    def get_rates(self) -> dict[str, float]:
        """Return 3-month and 10-year risk-free rates."""
        return {
            "3mo":  self.get_risk_free_rate("3mo"),
            "10yr": self.get_risk_free_rate("10yr"),
        }

    def get_latest(self, series_id: str) -> float | None:
        """
        Return the most recent value of an arbitrary FRED series (raw units,
        not divided by 100), or None if FRED is unavailable.

        Used for macro indicators (e.g. yield-curve spreads, credit spreads)
        that don't fit the risk-free-rate-specific helpers above.
        """
        if not self.api_key:
            return None

        cache_key = self.cache.ticker_key("__fred__", f"latest_{series_id}")
        cached = self.cache.get(cache_key, ttl=self._macro_ttl)
        if cached is not None:
            return float(cached)

        try:
            from fredapi import Fred  # type: ignore[import]
            fred = Fred(api_key=self.api_key)
            series = fred.get_series(series_id)
            val = series.dropna().iloc[-1]
            value = float(val)
            self.cache.set(cache_key, value)
            return value
        except Exception as exc:
            logger.warning("FRED request failed for %s: %s", series_id, exc)
            return None
