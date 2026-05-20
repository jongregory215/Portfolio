"""
DataProvider abstract interface.

All data sources (yfinance, FMP, Alpha Vantage, etc.) implement this
contract so the rest of the system can swap providers without touching
scoring or grading code.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date
from typing import Any

import pandas as pd


class DataProviderError(Exception):
    """Raised when a provider cannot fulfill a request (rate limit, bad ticker, etc.)."""


class DataProvider(ABC):
    """Abstract base for all market-data providers."""

    name: str = "base"   # overridden by each concrete provider

    # ── Price / volume ─────────────────────────────────────────

    @abstractmethod
    def get_price_history(
        self,
        ticker: str,
        start: date,
        end: date,
        interval: str = "1d",
    ) -> pd.DataFrame:
        """
        Return OHLCV DataFrame indexed by date (ascending).

        Required columns: open, high, low, close, volume, adj_close
        All prices in USD. Volume in shares.
        """

    # ── Fundamentals ───────────────────────────────────────────

    @abstractmethod
    def get_fundamentals(self, ticker: str) -> dict[str, Any]:
        """
        Return a dict of fundamental data points.

        Expected top-level keys (all optional — missing fields are handled
        by the engine's missing-data policy):
          income:      revenue, gross_profit, operating_income, net_income, eps_ttm,
                       eps_history (list of annual values, newest first)
          balance:     total_assets, total_liabilities, total_equity, cash,
                       total_debt, current_assets, current_liabilities
          cashflow:    operating_cf, free_cf, capex, dividends_paid
          per_share:   pe_trailing, pe_forward, pb, ps, ev_ebitda, peg,
                       dividend_yield, payout_ratio, market_cap, enterprise_value
          growth:      revenue_cagr_3yr, revenue_cagr_5yr, eps_cagr_3yr, eps_cagr_5yr,
                       forward_revenue_growth, forward_eps_growth
          margins:     gross_margin, operating_margin, net_margin
          returns:     roe, roa, roic, wacc
          health:      current_ratio, quick_ratio, debt_equity, interest_coverage
          meta:        sector, industry, gics_sector, gics_industry, gics_sub_industry,
                       exchange, currency, description, country
        """

    @abstractmethod
    def get_estimates(self, ticker: str) -> dict[str, Any]:
        """
        Return forward analyst estimates.

        Expected keys: forward_eps, forward_revenue, eps_next_yr,
                       eps_next_5yr_growth, num_analysts
        """

    # ── Peer set ───────────────────────────────────────────────

    @abstractmethod
    def get_sector_peers(self, ticker: str, method: str = "gics_sub_industry") -> list[str]:
        """
        Return peer tickers for peer-relative percentile scoring.

        method: gics_sector | gics_industry | gics_sub_industry | market_cap_band
        Returns tickers sorted by market cap descending; at most 50 names.
        """

    # ── Universe ───────────────────────────────────────────────

    def get_universe(self, exchanges: list[str] | None = None) -> list[str]:
        """
        Return the full tradable universe for Stage-1 portfolio screening.

        Not all providers support this; raises NotImplementedError by default.
        """
        raise NotImplementedError(f"{self.name} does not support universe listing.")

    # ── Macro ──────────────────────────────────────────────────

    def get_risk_free_rate(self, tenor: str = "3mo") -> float | None:
        """
        Return the current annualized risk-free rate (e.g. 3-month T-bill yield).

        tenor: "3mo" | "10yr"
        Returns None if unavailable; caller should use config fallback.
        """
        return None
