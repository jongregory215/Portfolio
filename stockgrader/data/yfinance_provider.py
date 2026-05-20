"""
yFinance data provider — price history and supplementary info.

Primary role: OHLCV price history (two-plus years of daily bars).
Supplementary role: fills fundamental gaps when FMP is unavailable.
"""
from __future__ import annotations

import logging
from datetime import date, datetime, timedelta
from typing import Any

import pandas as pd
import yfinance as yf

from stockgrader.data.base import DataProvider, DataProviderError
from stockgrader.data.cache import DiskCache
from stockgrader.config import get_cache_config

logger = logging.getLogger(__name__)

# yfinance column → canonical column name
_COL_MAP = {
    "Open": "open", "High": "high", "Low": "low",
    "Close": "close", "Volume": "volume",
    "Dividends": "dividends", "Stock Splits": "stock_splits",
}


class YFinanceProvider(DataProvider):
    name = "yfinance"

    def __init__(self, cache: DiskCache | None = None):
        cfg = get_cache_config()
        self.cache = cache or DiskCache(
            cache_dir=cfg.get("directory"),
        )
        self._price_ttl = cfg["ttl"]["intraday_price"]
        self._eod_ttl   = cfg["ttl"]["eod_price"]

    # ── Price history ──────────────────────────────────────────

    def get_price_history(
        self,
        ticker: str,
        start: date,
        end: date,
        interval: str = "1d",
    ) -> pd.DataFrame:
        cache_key = self.cache.ticker_key(ticker, f"yf_price_{interval}_{start}_{end}")
        cached = self.cache.get(cache_key, ttl=self._eod_ttl)
        if cached is not None:
            return _deserialize_df(cached)

        try:
            t   = yf.Ticker(ticker)
            raw = t.history(
                start=str(start),
                end=str(end),
                interval=interval,
                auto_adjust=True,
                actions=True,
            )
        except Exception as exc:
            raise DataProviderError(f"yfinance history failed for {ticker}: {exc}") from exc

        if raw is None or raw.empty:
            raise DataProviderError(f"yfinance returned empty price history for {ticker}")

        # Normalize columns
        raw.index = raw.index.tz_localize(None)          # strip timezone
        raw = raw.rename(columns=_COL_MAP)
        raw["adj_close"] = raw["close"]                  # auto_adjust=True → close already adjusted

        required = {"open", "high", "low", "close", "volume"}
        missing  = required - set(raw.columns)
        if missing:
            raise DataProviderError(f"yfinance missing price columns for {ticker}: {missing}")

        self.cache.set(cache_key, _serialize_df(raw))
        return raw

    # ── Basic info (supplementary fundamentals) ─────────────────

    def get_fundamentals(self, ticker: str) -> dict[str, Any]:
        """
        Return yfinance .info dict. Used as a fallback / supplement when FMP
        is unavailable. Contains ratios, margins, beta, sector, etc.
        """
        cache_key = self.cache.ticker_key(ticker, "yf_info")
        cached = self.cache.get(cache_key, ttl=self._eod_ttl)
        if cached is not None:
            return cached

        try:
            t    = yf.Ticker(ticker)
            info = t.info or {}
        except Exception as exc:
            logger.warning("yfinance .info failed for %s: %s", ticker, exc)
            return {}

        if not info:
            return {}

        self.cache.set(cache_key, info)
        return info

    def get_yf_statements(self, ticker: str) -> dict[str, Any]:
        """
        Fetch annual financial statements from yfinance.
        Used as a fallback when FMP is not available.

        Returns: {income, balance_sheet, cash_flow} as DataFrames serialised to records.
        """
        cache_key = self.cache.ticker_key(ticker, "yf_statements")
        cached = self.cache.get(cache_key, ttl=self._eod_ttl)
        if cached is not None:
            return cached

        try:
            t = yf.Ticker(ticker)
            income   = t.financials
            balance  = t.balance_sheet
            cashflow = t.cashflow
        except Exception as exc:
            logger.warning("yfinance statements failed for %s: %s", ticker, exc)
            return {}

        result: dict[str, Any] = {}
        for name, df in [("income", income), ("balance_sheet", balance), ("cash_flow", cashflow)]:
            if df is not None and not df.empty:
                # DataFrames are (fields × dates); transpose to (dates × fields) for easier handling
                df = df.T.sort_index(ascending=False)
                df.index = df.index.strftime("%Y-%m-%d")
                result[name] = df.to_dict(orient="index")

        self.cache.set(cache_key, result)
        return result

    # ── Estimates (from .info) ──────────────────────────────────

    def get_estimates(self, ticker: str) -> dict[str, Any]:
        info = self.get_fundamentals(ticker)
        return {
            "forward_eps":            info.get("forwardEps"),
            "forward_pe":             info.get("forwardPE"),
            "forward_eps_growth":     info.get("earningsGrowth"),
            "forward_revenue_growth": info.get("revenueGrowth"),
            "peg":                    info.get("pegRatio"),
            "num_analysts":           info.get("numberOfAnalystOpinions"),
        }

    def get_sector_peers(self, ticker: str, method: str = "gics_sub_industry") -> list[str]:
        return []   # yfinance does not provide peer lists


# ── Serialization helpers ──────────────────────────────────────

def _serialize_df(df: pd.DataFrame) -> dict:
    """Convert a DataFrame to a JSON-safe dict (preserves DatetimeIndex)."""
    return {
        "index":   [str(i.date()) if hasattr(i, "date") else str(i) for i in df.index],
        "columns": list(df.columns),
        "data":    df.values.tolist(),
    }


def _deserialize_df(payload: dict) -> pd.DataFrame:
    """Reconstruct a DataFrame from a serialized dict."""
    df = pd.DataFrame(
        data=payload["data"],
        columns=payload["columns"],
        index=pd.to_datetime(payload["index"]),
    )
    df.index.name = "Date"
    return df
