"""
Warren Buffett-specific data fetcher.

Extends MarksFetcher with the extra fields needed for Buffett-style evaluation:
gross margin (pricing power), ROE (moat proxy), operating cash flow and free
cash flow (owner earnings quality and intrinsic value), and debt/equity
(balance sheet conservatism). Reuses all parent plumbing so this fetch costs
the same number of requests as a Marks fetch.
"""
from __future__ import annotations

import logging
from typing import Any

import yfinance as yf

from stockgrader.howard_marks.fetcher import MarksFetcher

logger = logging.getLogger(__name__)


class BuffettFetcher(MarksFetcher):
    """Fetch raw data for a single ticker, extended for Warren Buffett criteria."""

    def fetch(self, ticker: str) -> dict[str, Any]:
        data = super().fetch(ticker)

        t    = yf.Ticker(ticker)
        info = self._safe_info(t, ticker)

        data["gross_margin"]  = self._coerce(info.get("grossMargins"))
        data["roe"]           = self._coerce(info.get("returnOnEquity"))
        data["debt_equity"]   = self._coerce(info.get("debtToEquity"))
        data["operating_cf"]  = self._coerce(info.get("operatingCashflow"))
        data["free_cf"]       = self._coerce(info.get("freeCashflow"))

        return data
