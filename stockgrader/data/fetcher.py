"""
DataFetcher — orchestrates providers and returns the canonical data dict.

Default (no API key required):
  Price history  : YFinanceProvider
  Fundamentals   : YFinanceProvider (.info + annual statements)
  Estimates      : YFinanceProvider (.info forward fields)
  Peers          : none — valuation/profitability use absolute thresholds
  Risk-free rate : FREDProvider (free key) or 5 % fallback

Deep mode (FMP_API_KEY set, --deep flag on analyze.py):
  Fundamentals   : FMPProvider (quarterly, TTM ratios, analyst estimates)
  Peers + metrics: FMPProvider (GICS peer list + peer key-metrics-ttm)
  Price history  : YFinanceProvider (unchanged — yfinance is better here)

Pass deep=True (or inject fmp_provider) to enable the richer path.
Everything else is identical — the normalizer, engines, and reporters
see the same canonical dict either way.
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
    Orchestrates data providers and returns the fully normalized data dict
    consumed by all three scoring engines.

    Parameters
    ----------
    deep : bool
        When True and FMP_API_KEY is set in the environment, uses FMP for
        fundamentals, TTM ratios, analyst estimates, and peer-relative metrics.
        Falls back silently to yfinance when the key is absent.
    yf_provider, fmp_provider, fred_provider : optional overrides for testing.
    """

    def __init__(
        self,
        deep:          bool            = False,
        yf_provider:   YFinanceProvider | None = None,
        fmp_provider:  FMPProvider      | None = None,
        fred_provider: FREDProvider     | None = None,
        cache:         DiskCache        | None = None,
    ):
        cfg       = get_config()
        cache_cfg = cfg["data"]["cache"]

        self._cache = cache or DiskCache(cache_dir=cache_cfg.get("directory"))
        self._yf    = yf_provider   or YFinanceProvider(cache=self._cache)
        self._fred  = fred_provider or FREDProvider(cache=self._cache)
        self._cfg   = cfg

        # FMP: use injected provider, or create one when deep=True + key present
        if fmp_provider is not None:
            self._fmp = fmp_provider
        elif deep and os.environ.get("FMP_API_KEY"):
            self._fmp = FMPProvider(cache=self._cache)
        else:
            self._fmp = None

        if deep and self._fmp is None:
            logger.warning(
                "Deep mode requested but FMP_API_KEY is not set — "
                "falling back to yfinance-only mode."
            )

    @property
    def _deep(self) -> bool:
        return self._fmp is not None and self._fmp._available

    # ── Public API ────────────────────────────────────────────

    def fetch(self, ticker: str, use_cache: bool = True) -> dict[str, Any]:
        """
        Fetch all data for `ticker` and return the normalized data dict.

        When deep=True and FMP_API_KEY is set, the result includes:
        - Proper quarterly TTM ratios (EV/EBITDA, ROIC, interest coverage …)
        - GICS peer list + peer key-metrics-ttm for percentile scoring
        - Analyst consensus estimates from FMP
        """
        ticker = ticker.upper().strip()
        logger.info("Fetching %s [%s] …", ticker, "deep/FMP" if self._deep else "yfinance")

        cfg         = self._cfg
        history_yrs = int(cfg["data"].get("history_years", 3))
        min_peers   = int(cfg.get("fundamental", {}).get("peer_set", {}).get("min_peers", 5))

        end   = date.today()
        start = end - timedelta(days=history_yrs * 365 + 30)

        # ── 1. Price history (always yfinance) ────────────────
        price_df = None
        try:
            price_df = self._yf.get_price_history(ticker, start, end)
        except DataProviderError as exc:
            logger.error("Price history unavailable for %s: %s", ticker, exc)

        # ── 2. yfinance .info (metadata + spot ratios) ────────
        yf_info: dict = {}
        try:
            yf_info = self._yf.get_fundamentals(ticker)
        except Exception as exc:
            logger.warning("yfinance .info failed for %s: %s", ticker, exc)

        # ── 3. Fundamentals ───────────────────────────────────
        fmp_data: dict = {}
        if self._deep:
            try:
                fmp_data = self._fmp.get_fundamentals(ticker)
                logger.debug("FMP fundamentals: %d IS rows",
                             len(fmp_data.get("income_statements", [])))
            except Exception as exc:
                logger.warning("FMP fundamentals failed for %s: %s — falling back.", ticker, exc)

        if not fmp_data.get("income_statements"):
            # yfinance fallback (or primary when not deep)
            try:
                stmts = self._yf.get_yf_statements(ticker)
                if stmts:
                    fmp_data = _yf_stmts_to_fmp_format(stmts, yf_info)
            except Exception as exc:
                logger.warning("yfinance statements failed for %s: %s", ticker, exc)

        # ── 4. Estimates ──────────────────────────────────────
        estimates: dict = {}
        if self._deep:
            try:
                estimates = self._fmp.get_estimates(ticker)
            except Exception as exc:
                logger.warning("FMP estimates failed for %s: %s", ticker, exc)
        if not estimates:
            try:
                estimates = self._yf.get_estimates(ticker)
            except Exception:
                pass

        # ── 5. Peers + peer metrics (FMP deep only) ───────────
        peers:        list[str]       = []
        peer_metrics: dict[str, dict] = {}
        if self._deep:
            try:
                peers = self._fmp.get_sector_peers(ticker)
            except Exception as exc:
                logger.warning("FMP peers failed for %s: %s", ticker, exc)

            if len(peers) < min_peers:
                logger.info("%s: only %d peers found (need %d).",
                            ticker, len(peers), min_peers)

            if peers:
                try:
                    peer_metrics = self._fmp.get_peer_metrics(peers[:30])
                except Exception as exc:
                    logger.warning("FMP peer metrics failed for %s: %s", ticker, exc)

        # ── 6. Risk-free rate (FRED, free) ─────────────────────
        rf_rates: dict[str, float] = {}
        try:
            rf_rates = self._fred.get_rates()
        except Exception as exc:
            logger.warning("FRED failed: %s — using 5%% fallback.", exc)

        # ── 7. Normalize ──────────────────────────────────────
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
        data["deep_mode"]   = self._deep

        if data["missing_fields"]:
            logger.info("%s: %d field(s) missing — %s",
                        ticker, len(data["missing_fields"]),
                        ", ".join(data["missing_fields"][:5]))

        return data


# ──────────────────────────────────────────────────────────────
# yfinance statement format → FMP-compatible format
# ──────────────────────────────────────────────────────────────

_YF_IS_FIELDS: dict[str, str] = {
    "Total Revenue":                         "revenue",
    "Gross Profit":                          "grossProfit",
    "Operating Income":                      "operatingIncome",
    "Net Income":                            "netIncome",
    "Basic EPS":                             "epsdiluted",
    "EBITDA":                                "ebitda",
    "Interest Expense":                      "interestExpense",
    "Reconciled Depreciation":               "depreciationAndAmortization",
    "Tax Provision":                         "incomeTaxExpense",
    "Pretax Income":                         "incomeBeforeTax",
    "Diluted Average Shares":                "weightedAverageShsOutDil",
    "Net Income From Continuing Operations": "netIncome",
}

_YF_BS_FIELDS: dict[str, str] = {
    "Total Assets":                                     "totalAssets",
    "Total Liabilities Net Minority Interest":          "totalLiabilities",
    "Stockholders Equity":                              "totalStockholdersEquity",
    "Retained Earnings":                                "retainedEarnings",
    "Cash And Cash Equivalents":                        "cashAndCashEquivalents",
    "Cash Cash Equivalents And Short Term Investments": "cashAndCashEquivalents",
    "Total Debt":                                       "totalDebt",
    "Current Assets":                                   "totalCurrentAssets",
    "Current Liabilities":                              "totalCurrentLiabilities",
    "Current Debt":                                     "shortTermDebt",
    "Long Term Debt":                                   "longTermDebt",
    "Goodwill And Other Intangible Assets":             "goodwillAndIntangibleAssets",
    "Total Equity Gross Minority Interest":             "totalStockholdersEquity",
}

_YF_CF_FIELDS: dict[str, str] = {
    "Operating Cash Flow": "operatingCashFlow",
    "Capital Expenditure": "capitalExpenditure",
    "Free Cash Flow":      "freeCashFlow",
    "Cash Dividends Paid": "dividendsPaid",
    "Net Income":          "netIncome",
}


def _yf_stmts_to_fmp_format(stmts: dict, yf_info: dict) -> dict:
    """Convert yfinance statement dicts into the list-of-dicts the normalizer expects."""
    def _to_list(stmt_dict: dict, field_map: dict[str, str]) -> list[dict]:
        if not stmt_dict:
            return []
        rows = []
        for date_str in sorted(stmt_dict.keys(), reverse=True):
            row_raw = stmt_dict[date_str]
            row: dict = {"date": date_str}
            for yf_field, fmp_key in field_map.items():
                val = row_raw.get(yf_field)
                if val is not None and fmp_key not in row:
                    row[fmp_key] = val
            rows.append(row)
        return rows

    return {
        "profile": {
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
        },
        "income_statements": _to_list(stmts.get("income", {}),        _YF_IS_FIELDS),
        "balance_sheets":    _to_list(stmts.get("balance_sheet", {}), _YF_BS_FIELDS),
        "cash_flows":        _to_list(stmts.get("cash_flow", {}),     _YF_CF_FIELDS),
        "key_metrics_ttm":   {},   # filled from yf_info via _YF_MAP in normalizer
        "ratios_ttm":        {},
    }
