"""
Howard Marks-specific data fetcher.

Extends GrahamFetcher with the extra fields needed for Marks-style
evaluation: 52-week range, beta (for cost of equity), and the Altman Z /
interest-coverage inputs (total assets, retained earnings, operating income,
interest expense). Reuses GrahamFetcher's yfinance/FMP plumbing so a Marks
fetch costs the same number of requests as a Graham fetch.
"""
from __future__ import annotations

import logging
from typing import Any

import yfinance as yf

from stockgrader.graham.fetcher import GrahamFetcher

logger = logging.getLogger(__name__)


class MarksFetcher(GrahamFetcher):
    """Fetch raw data for a single ticker, extended for Howard Marks criteria."""

    def fetch(self, ticker: str) -> dict[str, Any]:
        data = super().fetch(ticker)

        t    = yf.Ticker(ticker)
        info = self._safe_info(t, ticker)
        stmts = self._safe_statements(t, ticker)

        data["fifty_two_week_high"] = info.get("fiftyTwoWeekHigh")
        data["fifty_two_week_low"]  = info.get("fiftyTwoWeekLow")
        data["beta"]                = info.get("beta")

        extra = self._altman_and_coverage_inputs(ticker, stmts)
        data.update(extra)

        return data

    # ── Altman Z / interest-coverage inputs ─────────────────────────────────

    def _altman_and_coverage_inputs(self, ticker: str, yf_stmts: dict) -> dict[str, float | None]:
        """
        Return total_assets, retained_earnings, operating_income, and
        interest_expense — preferring FMP, falling back to yfinance.
        """
        total_assets: float | None = None
        retained_earnings: float | None = None
        operating_income: float | None = None
        interest_expense: float | None = None

        if self._fmp:
            try:
                bs_rows = self._fmp._balance_sheets(ticker, limit=1)
                if bs_rows:
                    total_assets      = self._coerce(bs_rows[0].get("totalAssets"))
                    retained_earnings = self._coerce(bs_rows[0].get("retainedEarnings"))
            except Exception as exc:
                logger.debug("FMP balance sheet (Marks extras) failed for %s: %s", ticker, exc)

            try:
                inc_rows = self._fmp._income_statements(ticker, limit=1)
                if inc_rows:
                    operating_income = self._coerce(inc_rows[0].get("operatingIncome"))
                    interest_expense = self._coerce(inc_rows[0].get("interestExpense"))
            except Exception as exc:
                logger.debug("FMP income statement (Marks extras) failed for %s: %s", ticker, exc)

        latest_bs  = self._latest_row(yf_stmts.get("balance_sheet", {}))
        latest_inc = self._latest_row(yf_stmts.get("income", {}))

        if total_assets is None:
            total_assets = self._field(latest_bs, ["Total Assets", "totalAssets"])
        if retained_earnings is None:
            retained_earnings = self._field(latest_bs, ["Retained Earnings", "retainedEarnings"])
        if operating_income is None:
            operating_income = self._field(latest_inc, ["Operating Income", "EBIT", "operatingIncome"])
        if interest_expense is None:
            interest_expense = self._field(latest_inc, [
                "Interest Expense", "Interest Expense Non Operating", "interestExpense",
            ])

        return {
            "total_assets":      total_assets,
            "retained_earnings": retained_earnings,
            "operating_income":  operating_income,
            "interest_expense":  interest_expense,
        }
