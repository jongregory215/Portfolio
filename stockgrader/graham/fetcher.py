"""
Graham-specific data fetcher.

Primary source: yfinance (price, dividends, balance sheet, ~4yr income history).
Extended source: FMP when FMP_API_KEY is set (up to 12yr annual EPS history,
                 more accurate balance sheet snapshot).
"""
from __future__ import annotations

import logging
import math
import os
import warnings
from datetime import date
from typing import Any

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


class GrahamFetcher:
    """Fetch raw data for a single ticker and return a normalized dict."""

    def __init__(self) -> None:
        self._fmp: Any = None
        if os.environ.get("FMP_API_KEY"):
            try:
                from stockgrader.data.fmp_provider import FMPProvider
                self._fmp = FMPProvider()
                logger.info("FMP available — will use for extended EPS history.")
            except Exception as exc:
                logger.warning("FMP init failed: %s — falling back to yfinance only.", exc)

    def fetch(self, ticker: str) -> dict[str, Any]:
        t = yf.Ticker(ticker)
        info      = self._safe_info(t, ticker)
        stmts     = self._safe_statements(t, ticker)
        dividends = self._safe_dividends(t, ticker)

        price  = (info.get("currentPrice")
                  or info.get("regularMarketPrice")
                  or info.get("previousClose"))
        shares = info.get("sharesOutstanding") or info.get("impliedSharesOutstanding")
        bvps   = info.get("bookValue")

        # Balance sheet — prefer FMP snapshot, fall back to yfinance
        bs_data = self._balance_sheet_data(ticker, stmts, shares)

        # EPS history — FMP gives up to 12 years; yfinance gives ~4
        annual_eps = self._eps_history(ticker, stmts, shares, t)

        # Revenue (latest year from yfinance income stmt)
        latest_inc = self._latest_row(stmts.get("income", {}))
        revenue = self._field(latest_inc, ["Total Revenue", "totalRevenue", "Revenue"])

        # Dividend history — yfinance has the best long-term dividend series
        div_years = self._dividend_years(dividends)

        return {
            "ticker":              ticker.upper(),
            "company_name":        info.get("longName") or info.get("shortName") or ticker.upper(),
            "price":               price,
            "market_cap":          info.get("marketCap"),
            "shares":              shares,
            "revenue":             revenue,
            "current_assets":      bs_data.get("current_assets"),
            "current_liabilities": bs_data.get("current_liabilities"),
            "long_term_debt":      bs_data.get("long_term_debt"),
            "total_liabilities":   bs_data.get("total_liabilities"),
            "book_value_per_share": bvps,
            "trailing_eps":        info.get("trailingEps"),
            "forward_eps":         info.get("forwardEps"),
            "annual_eps":          annual_eps,   # [(year, eps), ...] newest-first
            "dividend_years":      div_years,    # set of calendar years with payments
            "as_of":               date.today().isoformat(),
            "eps_source":          "fmp" if self._fmp else "yfinance",
        }

    # ── Balance sheet ────────────────────────────────────────────────────────

    def _balance_sheet_data(self, ticker: str, yf_stmts: dict, shares: float | None) -> dict:
        """Return balance sheet fields, preferring FMP over yfinance."""
        # Try FMP first
        if self._fmp:
            try:
                rows = self._fmp._balance_sheets(ticker, limit=1)
                if rows:
                    r = rows[0]
                    ca  = self._coerce(r.get("totalCurrentAssets"))
                    cl  = self._coerce(r.get("totalCurrentLiabilities"))
                    ltd = self._coerce(r.get("longTermDebt"))
                    tl  = self._coerce(r.get("totalLiabilities"))
                    if ca is not None:   # at least partial FMP data
                        return {"current_assets": ca, "current_liabilities": cl,
                                "long_term_debt": ltd, "total_liabilities": tl}
            except Exception as exc:
                logger.debug("FMP balance sheet failed for %s: %s", ticker, exc)

        # Fall back to yfinance
        latest = self._latest_row(yf_stmts.get("balance_sheet", {}))
        return {
            "current_assets":      self._field(latest, ["Current Assets", "Total Current Assets", "currentAssets"]),
            "current_liabilities": self._field(latest, ["Current Liabilities", "Total Current Liabilities", "currentLiabilities"]),
            "long_term_debt":      self._field(latest, ["Long Term Debt", "longTermDebt", "Long-Term Debt"]),
            "total_liabilities":   self._field(latest, ["Total Liabilities Net Minority Interest", "Total Liabilities", "totalLiab", "totalLiabilities"]),
        }

    # ── EPS history ──────────────────────────────────────────────────────────

    def _eps_history(
        self,
        ticker: str,
        yf_stmts: dict,
        shares: float | None,
        ticker_obj: yf.Ticker,
    ) -> list[tuple[int, float]]:
        """
        Build (year, eps) list, newest-first, up to 10 entries.

        Priority: FMP (up to 12yr) → yfinance income stmt → yfinance .earnings.
        """
        eps_by_year: dict[int, float] = {}

        # 1. FMP — 5 years max on free tier; direct EPS fields
        if self._fmp:
            try:
                rows = self._fmp._income_statements(ticker, limit=5)
                for row in rows:
                    year = self._year_from_row(row)
                    if year is None:
                        continue
                    # stable API returns 'eps' and 'epsDiluted'
                    eps = self._coerce(row.get("epsDiluted") or row.get("eps")
                                       or row.get("epsdiluted"))
                    if eps is None:
                        ni = self._coerce(row.get("netIncome"))
                        sh = (self._coerce(row.get("weightedAverageShsOutDil"))
                              or self._coerce(row.get("weightedAverageShsOut"))
                              or shares)
                        if ni is not None and sh and sh > 0:
                            eps = ni / sh
                    if eps is not None and year not in eps_by_year:
                        eps_by_year[year] = eps
            except Exception as exc:
                logger.debug("FMP EPS history failed for %s: %s", ticker, exc)

        # 2. yfinance income statement (fills gaps / serves as sole source if no FMP)
        income = yf_stmts.get("income", {})
        for date_str, row in sorted(income.items(), reverse=True):
            year = int(date_str[:4])
            if year in eps_by_year:
                continue
            eps = self._field(row, ["Basic EPS", "Diluted EPS", "basicEps", "dilutedEps",
                                    "Earnings Per Share", "EPS"])
            if eps is None:
                ni = self._field(row, ["Net Income", "netIncome",
                                       "Net Income Common Stockholders",
                                       "Net Income Applicable To Common Shares"])
                if ni is not None and shares and shares > 0:
                    eps = ni / shares
            if eps is not None:
                eps_by_year[year] = eps

        # 3. yfinance .earnings (deprecated but sometimes adds older years)
        if len(eps_by_year) < 5:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", DeprecationWarning)
                    earnings_df = ticker_obj.earnings
                if earnings_df is not None and not earnings_df.empty:
                    for idx_val, row in earnings_df.iterrows():
                        year = int(str(idx_val)[:4])
                        if year in eps_by_year:
                            continue
                        eps_val = row.get("Earnings") or row.get("EPS")
                        if eps_val is not None:
                            if shares and shares > 0 and abs(float(eps_val)) > 1e6:
                                eps_by_year[year] = float(eps_val) / shares
                            else:
                                eps_by_year[year] = float(eps_val)
            except Exception:
                pass

        return [(y, e) for y, e in sorted(eps_by_year.items(), reverse=True)][:10]

    # ── yfinance helpers ─────────────────────────────────────────────────────

    def _safe_info(self, t: yf.Ticker, ticker: str) -> dict:
        try:
            return t.info or {}
        except Exception as exc:
            logger.warning("yfinance .info failed for %s: %s", ticker, exc)
            return {}

    def _safe_statements(self, t: yf.Ticker, ticker: str) -> dict:
        try:
            result: dict[str, Any] = {}
            for name, attrs in [
                ("income",        ["financials", "income_stmt"]),
                ("balance_sheet", ["balance_sheet"]),
                ("cash_flow",     ["cashflow", "cash_flow"]),
            ]:
                df = None
                for attr in attrs:
                    try:
                        df = getattr(t, attr, None)
                        if df is not None and not df.empty:
                            break
                    except Exception:
                        continue
                if df is not None and not df.empty:
                    df_t = df.T.sort_index(ascending=False)
                    df_t.index = df_t.index.strftime("%Y-%m-%d")
                    result[name] = df_t.to_dict(orient="index")
            return result
        except Exception as exc:
            logger.warning("yfinance statements failed for %s: %s", ticker, exc)
            return {}

    def _safe_dividends(self, t: yf.Ticker, ticker: str) -> pd.Series:
        try:
            divs = t.dividends
            return divs if divs is not None else pd.Series(dtype=float)
        except Exception as exc:
            logger.warning("yfinance dividends failed for %s: %s", ticker, exc)
            return pd.Series(dtype=float)

    # ── Shared utilities ─────────────────────────────────────────────────────

    def _latest_row(self, stmts: dict) -> dict:
        if not stmts:
            return {}
        return stmts[sorted(stmts.keys(), reverse=True)[0]]

    def _field(self, row: dict, keys: list[str]) -> float | None:
        for k in keys:
            v = row.get(k)
            if v is not None and not (isinstance(v, float) and math.isnan(v)):
                try:
                    return float(v)
                except (TypeError, ValueError):
                    continue
        return None

    def _coerce(self, v: Any) -> float | None:
        if v is None:
            return None
        try:
            f = float(v)
            return None if math.isnan(f) else f
        except (TypeError, ValueError):
            return None

    def _year_from_row(self, row: dict) -> int | None:
        """Extract fiscal year from an FMP statement row."""
        for key in ("fiscalYear", "calendarYear"):
            cy = row.get(key)
            if cy:
                try:
                    return int(cy)
                except (TypeError, ValueError):
                    pass
        d = row.get("date") or row.get("filingDate") or row.get("fillingDate")
        if d:
            try:
                return int(str(d)[:4])
            except (TypeError, ValueError):
                pass
        return None

    def _dividend_years(self, dividends: pd.Series) -> set[int]:
        """Return the set of calendar years in which at least one dividend was paid."""
        if dividends is None or dividends.empty:
            return set()
        try:
            years: set[int] = set()
            for ts, val in dividends.items():
                if val and float(val) > 0:
                    year = ts.year if hasattr(ts, "year") else pd.Timestamp(ts).year
                    years.add(int(year))
            return years
        except Exception as exc:
            logger.warning("Error processing dividend history: %s", exc)
            return set()
