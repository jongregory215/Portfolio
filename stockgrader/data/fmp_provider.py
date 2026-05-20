"""
Financial Modeling Prep data provider.

Primary source for fundamentals, estimates, sector peers, and the full
tradable-universe list. Reads FMP_API_KEY from environment. Degrades
gracefully (returns empty dicts / lists) when the key is absent or a
request fails, so the normalizer can flag missing fields.
"""
from __future__ import annotations

import logging
import os
import time
from datetime import date
from typing import Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from stockgrader.data.base import DataProvider, DataProviderError
from stockgrader.data.cache import DiskCache
from stockgrader.config import get_cache_config

logger = logging.getLogger(__name__)

_BASE_V3 = "https://financialmodelingprep.com/api/v3"
_BASE_V4 = "https://financialmodelingprep.com/api/v4"


def _build_session() -> requests.Session:
    session = requests.Session()
    retry   = Retry(
        total=3,
        backoff_factor=1.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    return session


class FMPProvider(DataProvider):
    name = "fmp"

    def __init__(self, cache: DiskCache | None = None):
        self.api_key = os.environ.get("FMP_API_KEY", "")
        if not self.api_key:
            logger.warning(
                "FMP_API_KEY not set — fundamental data unavailable; "
                "falling back to yfinance where possible."
            )

        cfg = get_cache_config()
        self.cache = cache or DiskCache(cache_dir=cfg.get("directory"))
        self._fund_ttl     = cfg["ttl"]["fundamentals"]
        self._est_ttl      = cfg["ttl"]["estimates"]
        self._universe_ttl = cfg["ttl"]["universe"]
        self._peer_ttl     = cfg["ttl"]["peer_list"]

        self._session = _build_session()

    @property
    def _available(self) -> bool:
        return bool(self.api_key)

    # ── Core request method ────────────────────────────────────

    def _get(
        self,
        base: str,
        endpoint: str,
        params: dict | None = None,
        cache_key: str | None = None,
        ttl: int | None = None,
    ) -> Any:
        """
        GET {base}/{endpoint}?apikey=...&{params}.

        Returns parsed JSON payload or None on any failure.
        Results cached by cache_key when provided.
        """
        if not self._available:
            return None

        if cache_key:
            cached = self.cache.get(cache_key, ttl=ttl or self._fund_ttl)
            if cached is not None:
                return cached

        p = {"apikey": self.api_key}
        if params:
            p.update(params)

        url = f"{base}/{endpoint}"
        try:
            resp = self._session.get(url, params=p, timeout=20)
        except requests.RequestException as exc:
            logger.warning("FMP request error for %s: %s", endpoint, exc)
            return None

        if resp.status_code == 429:
            logger.warning("FMP rate-limit hit; sleeping 60s before continuing.")
            time.sleep(60)
            return None
        if resp.status_code == 404:
            logger.debug("FMP 404 for %s", endpoint)
            return None
        if not resp.ok:
            logger.warning("FMP %s → HTTP %s", endpoint, resp.status_code)
            return None

        try:
            data = resp.json()
        except Exception:
            logger.warning("FMP non-JSON response for %s", endpoint)
            return None

        # FMP returns {"Error Message": "..."} for invalid tickers / missing data
        if isinstance(data, dict) and "Error Message" in data:
            logger.debug("FMP error for %s: %s", endpoint, data["Error Message"])
            return None

        if cache_key and data:
            self.cache.set(cache_key, data)
        return data

    # ── Individual endpoint helpers ────────────────────────────

    def _profile(self, ticker: str) -> dict[str, Any]:
        key    = self.cache.ticker_key(ticker, "fmp_profile")
        result = self._get(_BASE_V3, f"profile/{ticker}", cache_key=key, ttl=self._fund_ttl)
        if isinstance(result, list) and result:
            return result[0]
        return {}

    def _income_statements(self, ticker: str, limit: int = 6) -> list[dict]:
        key    = self.cache.ticker_key(ticker, f"fmp_income_{limit}")
        result = self._get(
            _BASE_V3, f"income-statement/{ticker}",
            params={"limit": limit, "period": "annual"},
            cache_key=key, ttl=self._fund_ttl,
        )
        return result if isinstance(result, list) else []

    def _balance_sheets(self, ticker: str, limit: int = 6) -> list[dict]:
        key    = self.cache.ticker_key(ticker, f"fmp_balance_{limit}")
        result = self._get(
            _BASE_V3, f"balance-sheet-statement/{ticker}",
            params={"limit": limit, "period": "annual"},
            cache_key=key, ttl=self._fund_ttl,
        )
        return result if isinstance(result, list) else []

    def _cash_flows(self, ticker: str, limit: int = 6) -> list[dict]:
        key    = self.cache.ticker_key(ticker, f"fmp_cashflow_{limit}")
        result = self._get(
            _BASE_V3, f"cash-flow-statement/{ticker}",
            params={"limit": limit, "period": "annual"},
            cache_key=key, ttl=self._fund_ttl,
        )
        return result if isinstance(result, list) else []

    def _key_metrics_ttm(self, ticker: str) -> dict[str, Any]:
        key    = self.cache.ticker_key(ticker, "fmp_km_ttm")
        result = self._get(_BASE_V3, f"key-metrics-ttm/{ticker}", cache_key=key, ttl=self._fund_ttl)
        if isinstance(result, list) and result:
            return result[0]
        return {}

    def _ratios_ttm(self, ticker: str) -> dict[str, Any]:
        key    = self.cache.ticker_key(ticker, "fmp_ratios_ttm")
        result = self._get(_BASE_V3, f"ratios-ttm/{ticker}", cache_key=key, ttl=self._fund_ttl)
        if isinstance(result, list) and result:
            return result[0]
        return {}

    def _analyst_estimates(self, ticker: str, limit: int = 2) -> list[dict]:
        key    = self.cache.ticker_key(ticker, f"fmp_estimates_{limit}")
        result = self._get(
            _BASE_V3, f"analyst-estimates/{ticker}",
            params={"limit": limit},
            cache_key=key, ttl=self._est_ttl,
        )
        return result if isinstance(result, list) else []

    def _stock_peers(self, ticker: str) -> list[str]:
        key    = self.cache.ticker_key(ticker, "fmp_peers")
        result = self._get(
            _BASE_V4, "stock_peers",
            params={"symbol": ticker},
            cache_key=key, ttl=self._peer_ttl,
        )
        if isinstance(result, list) and result:
            return [p for p in result[0].get("peersList", []) if p != ticker]
        return []

    # ── DataProvider interface ────────────────────────────────

    def get_fundamentals(self, ticker: str) -> dict[str, Any]:
        """
        Return a combined raw FMP data dict consumed by the normalizer.

        Structure:
          profile, income_statements, balance_sheets, cash_flows,
          key_metrics_ttm, ratios_ttm
        """
        return {
            "profile":           self._profile(ticker),
            "income_statements": self._income_statements(ticker),
            "balance_sheets":    self._balance_sheets(ticker),
            "cash_flows":        self._cash_flows(ticker),
            "key_metrics_ttm":   self._key_metrics_ttm(ticker),
            "ratios_ttm":        self._ratios_ttm(ticker),
        }

    def get_estimates(self, ticker: str) -> dict[str, Any]:
        raw = self._analyst_estimates(ticker)
        if not raw:
            return {}
        latest = raw[0]
        return {
            "forward_eps":            latest.get("estimatedEpsAvg"),
            "forward_revenue":        latest.get("estimatedRevenueAvg"),
            "forward_eps_growth":     latest.get("estimatedEpsGrowth"),
            "forward_revenue_growth": latest.get("estimatedRevenueGrowth"),
            "num_analysts":           latest.get("numberAnalystEstimatedEps"),
        }

    def get_sector_peers(self, ticker: str, method: str = "gics_sub_industry") -> list[str]:
        return self._stock_peers(ticker)

    def get_price_history(self, ticker: str, start: date, end: date, interval: str = "1d") -> Any:
        """FMP price history — used as a fallback when yfinance fails."""
        import pandas as pd
        key    = self.cache.ticker_key(ticker, f"fmp_price_{start}_{end}")
        result = self._get(
            _BASE_V3, f"historical-price-full/{ticker}",
            params={"from": str(start), "to": str(end)},
            cache_key=key, ttl=self._fund_ttl,
        )
        if not result:
            raise DataProviderError(f"FMP returned no price data for {ticker}")
        bars = result.get("historical", [])
        if not bars:
            raise DataProviderError(f"FMP empty price history for {ticker}")
        df = pd.DataFrame(bars)
        df = df.rename(columns={
            "date": "Date", "open": "open", "high": "high",
            "low": "low", "close": "close", "volume": "volume",
            "adjClose": "adj_close",
        })
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.set_index("Date").sort_index(ascending=True)
        return df

    def get_universe(self, exchanges: list[str] | None = None) -> list[str]:
        """
        Return all tradable tickers from FMP's available-traded/list endpoint.

        Applies exchange filter and excludes tickers with dots (ADR/international format).
        """
        key    = self.cache.ticker_key("__universe__", "fmp_universe")
        result = self._get(_BASE_V3, "available-traded/list", cache_key=key, ttl=self._universe_ttl)
        if not result or not isinstance(result, list):
            logger.warning("FMP universe list unavailable.")
            return []

        allowed  = set(exchanges) if exchanges else set()
        tickers  = []
        for item in result:
            sym = item.get("symbol", "")
            ex  = item.get("exchangeShortName", "")
            # Skip non-equity types and international-format tickers
            if not sym or "." in sym or "-" in sym:
                continue
            if allowed and ex not in allowed:
                continue
            tickers.append(sym)
        return tickers

    def get_peer_metrics(self, peers: list[str]) -> dict[str, dict]:
        """
        Fetch key-metrics-ttm for each peer.
        Used for peer-relative percentile scoring in the fundamental engine.
        """
        result = {}
        for peer in peers:
            metrics = self._key_metrics_ttm(peer)
            if metrics:
                result[peer] = metrics
        return result
