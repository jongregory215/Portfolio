"""
Financial Modeling Prep data provider — stable API edition.

Uses https://financialmodelingprep.com/stable/* endpoints which work on
the free tier (no subscription required beyond email verification).

Available on free tier:
  profile, income-statement, balance-sheet-statement, cash-flow-statement,
  key-metrics-ttm, ratios-ttm, key-metrics

Not available on free tier (handled gracefully with empty returns):
  stock_peers, analyst-estimates, available-traded/list

Reads FMP_API_KEY from the environment (or .env file via python-dotenv).
Degrades gracefully when the key is absent.
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

_BASE_STABLE = "https://financialmodelingprep.com/stable"


def _build_session() -> requests.Session:
    session = requests.Session()
    retry   = Retry(
        total=3,
        backoff_factor=1.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        raise_on_status=False,
    )
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session


class FMPProvider(DataProvider):
    name = "fmp"

    def __init__(self, cache: DiskCache | None = None):
        self.api_key = os.environ.get("FMP_API_KEY", "")
        if not self.api_key:
            logger.warning("FMP_API_KEY not set — FMP data unavailable.")

        cfg = get_cache_config()
        self.cache         = cache or DiskCache(cache_dir=cfg.get("directory"))
        self._fund_ttl     = cfg["ttl"]["fundamentals"]
        self._est_ttl      = cfg["ttl"]["estimates"]
        self._universe_ttl = cfg["ttl"]["universe"]
        self._peer_ttl     = cfg["ttl"]["peer_list"]
        self._session      = _build_session()

    @property
    def _available(self) -> bool:
        return bool(self.api_key)

    # ── Core request ──────────────────────────────────────────

    def _get(
        self,
        endpoint: str,
        params:    dict | None = None,
        cache_key: str  | None = None,
        ttl:       int  | None = None,
    ) -> Any:
        """GET {_BASE_STABLE}/{endpoint}?apikey=...&{params}."""
        if not self._available:
            return None

        if cache_key:
            cached = self.cache.get(cache_key, ttl=ttl or self._fund_ttl)
            if cached is not None:
                return cached

        p = {"apikey": self.api_key}
        if params:
            p.update(params)

        url = f"{_BASE_STABLE}/{endpoint}"
        try:
            resp = self._session.get(url, params=p, timeout=20)
        except requests.RequestException as exc:
            logger.warning("FMP request error (%s): %s", endpoint, exc)
            return None

        if resp.status_code == 429:
            logger.warning("FMP rate-limit — sleeping 60s.")
            time.sleep(60)
            return None
        if resp.status_code == 404:
            logger.debug("FMP 404: %s", endpoint)
            return None
        if not resp.ok:
            logger.warning("FMP %s → HTTP %s", endpoint, resp.status_code)
            return None

        try:
            data = resp.json()
        except Exception:
            logger.warning("FMP non-JSON response: %s", endpoint)
            return None

        if isinstance(data, dict) and "Error Message" in data:
            logger.debug("FMP error (%s): %s", endpoint, data["Error Message"])
            return None

        if cache_key and data:
            self.cache.set(cache_key, data)
        return data

    # ── Endpoint helpers ──────────────────────────────────────

    def _profile(self, ticker: str) -> dict[str, Any]:
        key    = self.cache.ticker_key(ticker, "fmp_stable_profile")
        result = self._get("profile", {"symbol": ticker}, cache_key=key, ttl=self._fund_ttl)
        if isinstance(result, list) and result:
            raw = result[0]
            # Remap stable field names → normalizer-expected v3 names
            return {
                "mktCap":      raw.get("marketCap"),
                "beta":        raw.get("beta"),
                "sector":      raw.get("sector"),
                "industry":    raw.get("industry"),
                "exchange":    raw.get("exchange"),
                "currency":    raw.get("currency"),
                "country":     raw.get("country"),
                "companyName": raw.get("companyName"),
                "price":       raw.get("price"),
                "volAvg":      raw.get("averageVolume"),
            }
        return {}

    def _income_statements(self, ticker: str, limit: int = 6) -> list[dict]:
        key    = self.cache.ticker_key(ticker, f"fmp_stable_income_{limit}")
        result = self._get(
            "income-statement",
            {"symbol": ticker, "limit": limit, "period": "annual"},
            cache_key=key, ttl=self._fund_ttl,
        )
        if not isinstance(result, list):
            return []
        # Remap epsDiluted → epsdiluted so _IS_MAP picks it up
        for row in result:
            if "epsDiluted" in row and "epsdiluted" not in row:
                row["epsdiluted"] = row["epsDiluted"]
            if "netIncomeFromContinuingOperations" in row and "netIncome" not in row:
                row["netIncome"] = row["netIncomeFromContinuingOperations"]
        return result

    def _balance_sheets(self, ticker: str, limit: int = 6) -> list[dict]:
        key    = self.cache.ticker_key(ticker, f"fmp_stable_balance_{limit}")
        result = self._get(
            "balance-sheet-statement",
            {"symbol": ticker, "limit": limit, "period": "annual"},
            cache_key=key, ttl=self._fund_ttl,
        )
        return result if isinstance(result, list) else []

    def _cash_flows(self, ticker: str, limit: int = 6) -> list[dict]:
        key    = self.cache.ticker_key(ticker, f"fmp_stable_cashflow_{limit}")
        result = self._get(
            "cash-flow-statement",
            {"symbol": ticker, "limit": limit, "period": "annual"},
            cache_key=key, ttl=self._fund_ttl,
        )
        return result if isinstance(result, list) else []

    def _key_metrics_ttm(self, ticker: str) -> dict[str, Any]:
        key    = self.cache.ticker_key(ticker, "fmp_stable_km_ttm")
        result = self._get(
            "key-metrics-ttm", {"symbol": ticker},
            cache_key=key, ttl=self._fund_ttl,
        )
        if isinstance(result, list) and result:
            return result[0]
        return {}

    def _ratios_ttm(self, ticker: str) -> dict[str, Any]:
        key    = self.cache.ticker_key(ticker, "fmp_stable_ratios_ttm")
        result = self._get(
            "ratios-ttm", {"symbol": ticker},
            cache_key=key, ttl=self._fund_ttl,
        )
        if isinstance(result, list) and result:
            return result[0]
        return {}

    # ── DataProvider interface ────────────────────────────────

    def get_fundamentals(self, ticker: str) -> dict[str, Any]:
        return {
            "profile":           self._profile(ticker),
            "income_statements": self._income_statements(ticker),
            "balance_sheets":    self._balance_sheets(ticker),
            "cash_flows":        self._cash_flows(ticker),
            "key_metrics_ttm":   self._key_metrics_ttm(ticker),
            "ratios_ttm":        self._ratios_ttm(ticker),
        }

    def get_estimates(self, ticker: str) -> dict[str, Any]:
        # analyst-estimates not available on free stable tier → empty
        return {}

    def get_sector_peers(self, ticker: str, method: str = "gics_sub_industry") -> list[str]:
        # stock_peers not available on free stable tier → empty
        return []

    def get_price_history(self, ticker: str, start: date, end: date, interval: str = "1d") -> Any:
        """Price history via stable API — used only if yfinance fails."""
        import pandas as pd
        key    = self.cache.ticker_key(ticker, f"fmp_stable_price_{start}_{end}")
        result = self._get(
            f"historical-price-full/{ticker}",
            {"from": str(start), "to": str(end)},
            cache_key=key, ttl=self._fund_ttl,
        )
        if not result:
            raise DataProviderError(f"FMP returned no price data for {ticker}")
        bars = result.get("historical", [])
        if not bars:
            raise DataProviderError(f"FMP empty price history for {ticker}")
        df = pd.DataFrame(bars)
        df = df.rename(columns={"date": "Date", "adjClose": "adj_close"})
        df["Date"] = pd.to_datetime(df["Date"])
        return df.set_index("Date").sort_index(ascending=True)

    def get_universe(self, exchanges: list[str] | None = None) -> list[str]:
        # available-traded/list not on free stable tier → caller uses universe.yaml
        return []

    def get_peer_metrics(self, peers: list[str]) -> dict[str, dict]:
        # peers not available on free tier → empty
        return {}
