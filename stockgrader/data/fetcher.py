"""
DataFetcher — orchestrates providers and returns the canonical data dict.

Provider chain (yfinance-only; no paid API required):
  Price history  : YFinanceProvider (primary)
  Fundamentals   : YFinanceProvider (.info + statements)
  Estimates      : YFinanceProvider (.info forward fields)
  Risk-free rate : FREDProvider (free, requires FRED_API_KEY)
                   Falls back to 5 % when key is absent.

Peer-relative scoring (percentile vs GICS peers) is disabled in this
configuration — valuation and profitability pillars use absolute thresholds
instead. Peer support can be re-enabled later by wiring in a peer provider.
"""
from __future__ import annotations

import logging
from datetime import date, timedelta
from typing import Any

from stockgrader.config import get_config, get_config_hash
from stockgrader.data.cache import DiskCache
from stockgrader.data.base import DataProviderError
from stockgrader.data.yfinance_provider import YFinanceProvider
from stockgrader.data.fred_provider import FREDProvider
from stockgrader.data.normalizer import normalize, _IS_MAP, _BS_MAP, _CF_MAP

logger = logging.getLogger(__name__)


class DataFetcher:
    """
    Orchestrates yfinance + FRED and returns the fully normalized data dict
    consumed by all three scoring engines.
    """

    def __init__(
        self,
        yf_provider:   YFinanceProvider | None = None,
        fred_provider: FREDProvider     | None = None,
        cache:         DiskCache        | None = None,
    ):
        cfg       = get_config()
        cache_cfg = cfg["data"]["cache"]

        self._cache = cache or DiskCache(cache_dir=cache_cfg.get("directory"))
        self._yf    = yf_provider   or YFinanceProvider(cache=self._cache)
        self._fred  = fred_provider or FREDProvider(cache=self._cache)
        self._cfg   = cfg

    # ── Public API ────────────────────────────────────────────

    def fetch(self, ticker: str, use_cache: bool = True) -> dict[str, Any]:
        """
        Fetch all data for `ticker` and return the normalized data dict.

        Parameters
        ----------
        ticker:    Equity ticker symbol (case-insensitive).
        use_cache: If False, bypass disk cache and force live fetches.
        """
        ticker = ticker.upper().strip()
        logger.info("Fetching data for %s …", ticker)

        cfg         = self._cfg
        history_yrs = int(cfg["data"].get("history_years", 3))

        end   = date.today()
        start = end - timedelta(days=history_yrs * 365 + 30)

        # ── 1. Price history ──────────────────────────────────
        price_df = None
        try:
            price_df = self._yf.get_price_history(ticker, start, end)
            logger.debug("yfinance price history: %d bars", len(price_df))
        except DataProviderError as exc:
            logger.error("Price history unavailable for %s: %s", ticker, exc)

        # ── 2. yfinance .info (ratios, metadata, estimates) ───
        yf_info: dict = {}
        try:
            yf_info = self._yf.get_fundamentals(ticker)
        except Exception as exc:
            logger.warning("yfinance .info failed for %s: %s", ticker, exc)

        # ── 3. Financial statements (IS / BS / CF) ────────────
        fmp_data: dict = {}
        try:
            stmts = self._yf.get_yf_statements(ticker)
            if stmts:
                fmp_data = _yf_stmts_to_fmp_format(stmts, yf_info)
        except Exception as exc:
            logger.warning("yfinance statements failed for %s: %s", ticker, exc)

        # ── 4. Forward estimates (from .info) ─────────────────
        estimates: dict = {}
        try:
            estimates = self._yf.get_estimates(ticker)
        except Exception as exc:
            logger.warning("yfinance estimates failed for %s: %s", ticker, exc)

        # ── 5. Risk-free rate (FRED, free) ─────────────────────
        rf_rates: dict[str, float] = {}
        try:
            rf_rates = self._fred.get_rates()
        except Exception as exc:
            logger.warning("FRED rate fetch failed: %s — using 5%% fallback.", exc)

        # ── 6. Normalize into canonical dict ──────────────────
        data = normalize(
            ticker       = ticker,
            price_df     = price_df,
            yf_info      = yf_info,
            fmp_data     = fmp_data,
            estimates    = estimates,
            peers        = [],          # no peer source in yfinance-only mode
            peer_metrics = {},
            rf_rates     = rf_rates,
            cfg          = self._cfg,
        )

        data["config_hash"] = get_config_hash()

        if data["missing_fields"]:
            logger.info(
                "%s: %d field(s) missing: %s",
                ticker,
                len(data["missing_fields"]),
                ", ".join(data["missing_fields"][:5]),
            )

        return data


# ──────────────────────────────────────────────────────────────
# yfinance statement format → FMP-compatible format
# ──────────────────────────────────────────────────────────────

# Maps yfinance statement row keys to the FMP keys the normalizer expects.
_YF_IS_FIELDS: dict[str, str] = {
    "Total Revenue":                "revenue",
    "Gross Profit":                 "grossProfit",
    "Operating Income":             "operatingIncome",
    "Net Income":                   "netIncome",
    "Basic EPS":                    "epsdiluted",
    "EBITDA":                       "ebitda",
    "Interest Expense":             "interestExpense",
    "Reconciled Depreciation":      "depreciationAndAmortization",
    "Tax Provision":                "incomeTaxExpense",
    "Pretax Income":                "incomeBeforeTax",
    "Diluted Average Shares":       "weightedAverageShsOutDil",
    # alternate yfinance field names
    "Net Income From Continuing Operations": "netIncome",
    "Total Revenue":                "revenue",
}

_YF_BS_FIELDS: dict[str, str] = {
    "Total Assets":                                     "totalAssets",
    "Total Liabilities Net Minority Interest":          "totalLiabilities",
    "Stockholders Equity":                              "totalStockholdersEquity",
    "Retained Earnings":                                "retainedEarnings",
    "Cash And Cash Equivalents":                        "cashAndCashEquivalents",
    "Total Debt":                                       "totalDebt",
    "Current Assets":                                   "totalCurrentAssets",
    "Current Liabilities":                              "totalCurrentLiabilities",
    "Current Debt":                                     "shortTermDebt",
    "Long Term Debt":                                   "longTermDebt",
    "Goodwill And Other Intangible Assets":             "goodwillAndIntangibleAssets",
    # alternate names
    "Cash Cash Equivalents And Short Term Investments": "cashAndCashEquivalents",
    "Total Equity Gross Minority Interest":             "totalStockholdersEquity",
}

_YF_CF_FIELDS: dict[str, str] = {
    "Operating Cash Flow":   "operatingCashFlow",
    "Capital Expenditure":   "capitalExpenditure",
    "Free Cash Flow":        "freeCashFlow",
    "Cash Dividends Paid":   "dividendsPaid",
    "Net Income":            "netIncome",
}


def _yf_stmts_to_fmp_format(stmts: dict, yf_info: dict) -> dict:
    """
    Convert yfinance statement dicts (date → field → value) into the
    list-of-dicts format the normalizer expects (same shape as FMP responses).
    """
    def _to_list(stmt_dict: dict, field_map: dict[str, str]) -> list[dict]:
        if not stmt_dict:
            return []
        rows = []
        for date_str in sorted(stmt_dict.keys(), reverse=True):   # newest first
            row_raw = stmt_dict[date_str]
            row: dict = {"date": date_str}
            for yf_field, fmp_key in field_map.items():
                val = row_raw.get(yf_field)
                if val is not None and fmp_key not in row:
                    row[fmp_key] = val
            rows.append(row)
        return rows

    income_list  = _to_list(stmts.get("income", {}),        _YF_IS_FIELDS)
    balance_list = _to_list(stmts.get("balance_sheet", {}), _YF_BS_FIELDS)
    cf_list      = _to_list(stmts.get("cash_flow", {}),     _YF_CF_FIELDS)

    # Build profile from yf_info
    profile = {
        "mktCap":      yf_info.get("marketCap"),
        "beta":        yf_info.get("beta"),
        "sector":      yf_info.get("sector"),
        "industry":    yf_info.get("industry"),
        "exchange":    yf_info.get("exchange"),
        "currency":    yf_info.get("currency"),
        "country":     yf_info.get("country"),
        "companyName": yf_info.get("shortName") or yf_info.get("longName"),
        "price":       yf_info.get("currentPrice") or yf_info.get("regularMarketPrice"),
        "volAvg":      yf_info.get("averageVolume"),
    }

    return {
        "profile":           profile,
        "income_statements": income_list,
        "balance_sheets":    balance_list,
        "cash_flows":        cf_list,
        "key_metrics_ttm":   {},   # covered by _YF_MAP in normalizer
        "ratios_ttm":        {},   # covered by _YF_MAP in normalizer
    }
