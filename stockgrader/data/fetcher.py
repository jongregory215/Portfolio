"""
DataFetcher — orchestrates providers and returns the canonical data dict.

Usage:
    fetcher = DataFetcher()
    data = fetcher.fetch("AAPL")
    # data is the normalized dict consumed by all three scoring engines

Provider chain:
  - Price history  : YFinanceProvider → FMPProvider (fallback)
  - Fundamentals   : FMPProvider → yfinance statements (fallback)
  - Estimates      : FMPProvider → YFinanceProvider (fallback)
  - Risk-free rate : FREDProvider
  - Peers          : FMPProvider
"""
from __future__ import annotations

import logging
import os
from datetime import date, timedelta
from typing import Any

from stockgrader.config import get_config, get_config_hash
from stockgrader.data.cache import DiskCache
from stockgrader.data.base import DataProviderError
from stockgrader.data.yfinance_provider import YFinanceProvider
from stockgrader.data.fmp_provider import FMPProvider
from stockgrader.data.fred_provider import FREDProvider
from stockgrader.data.normalizer import normalize

logger = logging.getLogger(__name__)


class DataFetcher:
    """
    Orchestrates all data providers and returns the fully normalized data
    dict that scoring engines consume.
    """

    def __init__(
        self,
        yf_provider:   YFinanceProvider | None = None,
        fmp_provider:  FMPProvider       | None = None,
        fred_provider: FREDProvider      | None = None,
        cache:         DiskCache         | None = None,
    ):
        cfg = get_config()
        cache_cfg = cfg["data"]["cache"]

        self._cache = cache or DiskCache(
            cache_dir=cache_cfg.get("directory"),
        )
        self._yf   = yf_provider   or YFinanceProvider(cache=self._cache)
        self._fmp  = fmp_provider  or FMPProvider(cache=self._cache)
        self._fred = fred_provider or FREDProvider(cache=self._cache)
        self._cfg  = cfg

    # ── Public API ────────────────────────────────────────────

    def fetch(self, ticker: str, use_cache: bool = True) -> dict[str, Any]:
        """
        Fetch all data for `ticker` and return the normalized data dict.

        Parameters
        ----------
        ticker:    Equity ticker symbol (case-insensitive).
        use_cache: If False, bypass disk cache and force live fetches.

        Returns
        -------
        Normalized data dict. On data failures, returns a partial dict with
        data["missing_fields"] populated and data["warnings"] explaining gaps.
        """
        ticker = ticker.upper().strip()
        logger.info("Fetching data for %s …", ticker)

        cfg          = self._cfg
        history_yrs  = int(cfg["data"].get("history_years", 3))
        peer_cfg     = cfg["fundamental"].get("peer_set", {})
        min_peers    = int(peer_cfg.get("min_peers", 5))

        end   = date.today()
        start = end - timedelta(days=history_yrs * 365 + 30)   # +30 buffer for weekends

        # ── 1. Price history ──────────────────────────────────
        price_df = None
        try:
            price_df = self._yf.get_price_history(ticker, start, end)
            logger.debug("yfinance price history: %d bars", len(price_df))
        except DataProviderError as exc:
            logger.warning("yfinance price history failed for %s: %s — trying FMP.", ticker, exc)
            try:
                price_df = self._fmp.get_price_history(ticker, start, end)
                logger.debug("FMP price history: %d bars", len(price_df))
            except DataProviderError as exc2:
                logger.error("Both price providers failed for %s: %s", ticker, exc2)

        # ── 2. yfinance supplementary info ───────────────────
        yf_info = {}
        try:
            yf_info = self._yf.get_fundamentals(ticker)
        except Exception as exc:
            logger.warning("yfinance .info failed for %s: %s", ticker, exc)

        # ── 3. FMP fundamentals ───────────────────────────────
        fmp_data: dict = {}
        if self._fmp._available:
            try:
                fmp_data = self._fmp.get_fundamentals(ticker)
                logger.debug(
                    "FMP: %d IS, %d BS, %d CF rows",
                    len(fmp_data.get("income_statements", [])),
                    len(fmp_data.get("balance_sheets", [])),
                    len(fmp_data.get("cash_flows", [])),
                )
            except Exception as exc:
                logger.warning("FMP fundamentals failed for %s: %s", ticker, exc)
        else:
            # yfinance statement fallback when FMP key absent
            stmts = {}
            try:
                stmts = self._yf.get_yf_statements(ticker)
            except Exception:
                pass
            if stmts:
                fmp_data = _yf_stmts_to_fmp_format(stmts, yf_info)

        # ── 4. Estimates ──────────────────────────────────────
        estimates: dict = {}
        if self._fmp._available:
            try:
                estimates = self._fmp.get_estimates(ticker)
            except Exception as exc:
                logger.warning("FMP estimates failed for %s: %s", ticker, exc)
        if not estimates:
            try:
                estimates = self._yf.get_estimates(ticker)
            except Exception:
                pass

        # ── 5. Sector peers ───────────────────────────────────
        peers: list[str] = []
        if self._fmp._available:
            try:
                peers = self._fmp.get_sector_peers(ticker)
            except Exception as exc:
                logger.warning("FMP peers failed for %s: %s", ticker, exc)

        if len(peers) < min_peers:
            logger.info(
                "Only %d peers found for %s (need %d); peer-relative scoring may be limited.",
                len(peers), ticker, min_peers,
            )

        # ── 6. Peer key metrics (for percentile scoring) ──────
        peer_metrics: dict[str, dict] = {}
        if peers and self._fmp._available:
            try:
                peer_metrics = self._fmp.get_peer_metrics(peers[:30])  # cap at 30 peers
                logger.debug("Fetched metrics for %d peers", len(peer_metrics))
            except Exception as exc:
                logger.warning("Peer metrics fetch failed for %s: %s", ticker, exc)

        # ── 7. Risk-free rate ─────────────────────────────────
        rf_rates: dict[str, float] = {}
        try:
            rf_rates = self._fred.get_rates()
        except Exception as exc:
            logger.warning("FRED rate fetch failed: %s — using fallback.", exc)

        # ── 8. Normalize everything into canonical dict ────────
        data = normalize(
            ticker       = ticker,
            price_df     = price_df,
            yf_info      = yf_info,
            fmp_data     = fmp_data,
            estimates    = estimates,
            peers        = peers,
            peer_metrics = peer_metrics,
            rf_rates     = rf_rates,
            cfg          = self._cfg,
        )

        data["config_hash"] = get_config_hash()

        if data["missing_fields"]:
            logger.info(
                "%s: %d required fields missing: %s",
                ticker,
                len(data["missing_fields"]),
                ", ".join(data["missing_fields"][:5]),
            )

        return data


# ── yfinance statement format → FMP-like format ───────────────

_YF_IS_FIELDS = {
    "Total Revenue":        "revenue",
    "Gross Profit":         "gross_profit",
    "Operating Income":     "operating_income",
    "Net Income":           "net_income",
    "Basic EPS":            "eps",
    "EBITDA":               "ebitda",
    "Interest Expense":     "interest_expense",
    "Reconciled Depreciation": "d_and_a",
    "Tax Provision":        "tax_expense",
    "Diluted Average Shares":  "shares_diluted",
}
_YF_BS_FIELDS = {
    "Total Assets":            "total_assets",
    "Total Liabilities Net Minority Interest": "total_liabilities",
    "Stockholders Equity":     "total_equity",
    "Retained Earnings":       "retained_earnings",
    "Cash And Cash Equivalents": "cash",
    "Total Debt":              "total_debt",
    "Current Assets":          "current_assets",
    "Current Liabilities":     "current_liabilities",
}
_YF_CF_FIELDS = {
    "Operating Cash Flow":    "operating_cf",
    "Capital Expenditure":    "capex",
    "Free Cash Flow":         "free_cf",
    "Cash Dividends Paid":    "dividends_paid",
    "Net Income":             "net_income_cf",
}


def _yf_stmts_to_fmp_format(stmts: dict, yf_info: dict) -> dict:
    """
    Convert yfinance statement dicts (date→field→value) into FMP-like
    list-of-dicts format so the normalizer can process them uniformly.
    """
    def _to_list(stmt_dict: dict, field_map: dict) -> list[dict]:
        if not stmt_dict:
            return []
        rows = []
        for date_str in sorted(stmt_dict.keys(), reverse=True):   # newest first
            row_raw = stmt_dict[date_str]
            row = {"date": date_str}
            for yf_field, canon in field_map.items():
                # yfinance uses the actual field name as the key
                val = row_raw.get(yf_field)
                row[canon] = val
            rows.append(row)
        return rows

    # yfinance statements use canonical field names after our transpose step
    # We re-map them to FMP-like names so the normalizer's _IS_MAP works
    income_list  = _to_list(stmts.get("income", {}),       _YF_IS_FIELDS)
    balance_list = _to_list(stmts.get("balance_sheet", {}), _YF_BS_FIELDS)
    cf_list      = _to_list(stmts.get("cash_flow", {}),     _YF_CF_FIELDS)

    # Rebuild into FMP keys so the normalizer's _IS_MAP / _BS_MAP / _CF_MAP work
    def _remap(rows: list[dict], inv_map: dict) -> list[dict]:
        result = []
        for row in rows:
            new_row = {"date": row.get("date")}
            for canon_name, val in row.items():
                if canon_name == "date":
                    continue
                # Find the FMP key that maps to this canon name
                for fmp_key, cn in inv_map.items():
                    if cn == canon_name:
                        new_row[fmp_key] = val
            result.append(new_row)
        return result

    from stockgrader.data.normalizer import _IS_MAP, _BS_MAP, _CF_MAP
    return {
        "profile":           {
            "mktCap":     yf_info.get("marketCap"),
            "beta":       yf_info.get("beta"),
            "sector":     yf_info.get("sector"),
            "industry":   yf_info.get("industry"),
            "exchange":   yf_info.get("exchange"),
            "currency":   yf_info.get("currency"),
            "country":    yf_info.get("country"),
            "companyName":yf_info.get("shortName") or yf_info.get("longName"),
            "price":      yf_info.get("currentPrice") or yf_info.get("regularMarketPrice"),
        },
        "income_statements": _remap(income_list,  _IS_MAP),
        "balance_sheets":    _remap(balance_list, _BS_MAP),
        "cash_flows":        _remap(cf_list,      _CF_MAP),
        "key_metrics_ttm":   {},
        "ratios_ttm":        {},
    }
