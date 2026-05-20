"""
Point-in-time data adapter layer.

CRITICAL: the backtest engine MUST refuse to run silently on non-PIT data.
Using today's restated financials or a survivorship-biased universe produces
look-ahead contaminated results that are actively misleading.

Hierarchy:
  PITAdapter (ABC)
    ├── SharadarAdapter     — true PIT, requires Nasdaq Data Link subscription
    ├── YFinancePITAdapter  — NOT PIT; raises in strict mode, labels in warn mode
    └── SyntheticPITAdapter — deterministic test data; always PIT-safe

Usage:
    adapter = SharadarAdapter(api_key=os.environ["NASDAQ_DATA_LINK_API_KEY"])
    require_pit(adapter, config)          # raises if contaminated + strict mode
    universe = adapter.get_universe(as_of_date)
    data     = adapter.get_fundamentals_pit("AAPL", as_of_date)
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from datetime import date, timedelta
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class BacktestContaminationError(RuntimeError):
    """Raised when a non-PIT data source is used in strict contamination mode."""


# ──────────────────────────────────────────────────────────────
# Abstract base
# ──────────────────────────────────────────────────────────────

class PITAdapter(ABC):
    """Abstract point-in-time data adapter for the backtest engine."""

    name: str = "base"

    @abstractmethod
    def is_pit_guaranteed(self) -> bool:
        """Return True only if this source guarantees PIT and survivorship-free semantics."""

    @abstractmethod
    def contamination_label(self) -> str:
        """Human-readable description of any contamination risk."""

    @abstractmethod
    def get_universe(self, as_of_date: date) -> list[str]:
        """
        Return all tickers in the universe on as_of_date.
        MUST include delisted / acquired / bankrupt names that traded on that date.
        """

    @abstractmethod
    def get_fundamentals_pit(
        self,
        ticker:      str,
        as_of_date:  date,
        lag_days:    int = 45,
    ) -> dict[str, Any] | None:
        """
        Return fundamentals knowable on as_of_date.

        lag_days: reporting lag applied to each filing period.
                  Default 45 days (10-Q assumed filed 45 days after quarter-end).
        Returns None if no data available.
        """

    @abstractmethod
    def get_price_history_pit(
        self,
        ticker:    str,
        as_of_date: date,
        years:     int = 3,
    ) -> pd.DataFrame | None:
        """
        Return daily OHLCV history ending on as_of_date (no future prices).
        """

    def get_forward_returns(
        self,
        ticker:    str,
        as_of_date: date,
        horizons:  list[int] = (21, 63, 126, 252),
    ) -> dict[int, float | None]:
        """
        Compute forward returns at each trading-day horizon.
        Default horizons: 21 (≈1m), 63 (≈3m), 126 (≈6m), 252 (≈12m).

        This is called AFTER the scoring date to generate labels for
        grade validation.  It does NOT introduce look-ahead into the
        scoring (scores are frozen at as_of_date).
        """
        raise NotImplementedError


# ──────────────────────────────────────────────────────────────
# Contamination enforcement
# ──────────────────────────────────────────────────────────────

def require_pit(adapter: PITAdapter, config: dict) -> str:
    """
    Enforce PIT data requirement based on config.backtest.contamination_check.

    Returns a label string that must be prepended to all backtest output.
    In strict mode, raises BacktestContaminationError for non-PIT sources.
    In warn mode, returns a contamination label.
    In off mode, silently continues (not recommended).
    """
    mode = config.get("backtest", {}).get("contamination_check", "strict")

    if adapter.is_pit_guaranteed():
        return "OUT-OF-SAMPLE (PIT)"

    msg = (
        f"\n{'='*70}\n"
        f"⚠️  NON-PIT DATA SOURCE DETECTED: {adapter.name}\n"
        f"   {adapter.contamination_label()}\n"
        f"   Running a backtest on this data WILL produce look-ahead contaminated\n"
        f"   results.  The IC and return spreads will be inflated.\n"
        f"   Use Sharadar Core US Equities (Nasdaq Data Link) for valid backtesting.\n"
        f"   See README §Backtesting for recommended data tiers.\n"
        f"{'='*70}\n"
    )

    if mode == "strict":
        raise BacktestContaminationError(msg)
    elif mode == "warn":
        logger.warning(msg)
        return "INDICATIVE / LOOK-AHEAD-CONTAMINATED — DO NOT TRUST"
    else:
        return "CONTAMINATED (mode=off)"


# ──────────────────────────────────────────────────────────────
# yfinance adapter (NOT PIT — for testing only in warn mode)
# ──────────────────────────────────────────────────────────────

class YFinancePITAdapter(PITAdapter):
    """
    Thin yfinance wrapper.

    NOT point-in-time.  yfinance returns restated/adjusted historical data
    and today's survivor universe.  Using it for backtesting introduces:
      - Look-ahead bias via restated fundamentals
      - Survivorship bias (delisted names missing)
      - No filing-date lag enforcement

    This adapter always fails require_pit() in strict mode.
    """
    name = "yfinance"

    def is_pit_guaranteed(self) -> bool:
        return False

    def contamination_label(self) -> str:
        return (
            "yfinance returns today's restated fundamentals and today's "
            "surviving universe — both are forms of look-ahead bias."
        )

    def get_universe(self, as_of_date: date) -> list[str]:
        raise NotImplementedError("yfinance does not expose historical universes.")

    def get_fundamentals_pit(self, ticker, as_of_date, lag_days=45):
        raise NotImplementedError("yfinance data is not point-in-time.")

    def get_price_history_pit(self, ticker, as_of_date, years=3):
        raise NotImplementedError("Use this only for price data (no fundamental PIT needed).")


# ──────────────────────────────────────────────────────────────
# Sharadar adapter (true PIT)
# ──────────────────────────────────────────────────────────────

class SharadarAdapter(PITAdapter):
    """
    Sharadar Core US Equities via Nasdaq Data Link.

    Provides:
      - SF1: point-in-time annual and quarterly fundamentals (as-reported)
      - SEP: daily price data including delisted tickers (survivorship-free)

    Requires:
      - Nasdaq Data Link API key (NASDAQ_DATA_LINK_API_KEY env var)
      - Sharadar Core US Equities subscription (~$50/month)

    If the subscription is unavailable, this adapter raises ImportError.
    """
    name = "sharadar"

    def __init__(self, api_key: str):
        self._api_key = api_key
        try:
            import nasdaqdatalink  # type: ignore[import]
            nasdaqdatalink.ApiConfig.api_key = api_key
            self._ndl = nasdaqdatalink
        except ImportError as exc:
            raise ImportError(
                "nasdaqdatalink package required for Sharadar adapter. "
                "Install with: pip install nasdaq-data-link"
            ) from exc

    def is_pit_guaranteed(self) -> bool:
        return True

    def contamination_label(self) -> str:
        return "Sharadar SF1/SEP — true PIT with survivorship-bias-free universe."

    def get_universe(self, as_of_date: date) -> list[str]:
        """Return all US equities listed on as_of_date from SEP."""
        try:
            df = self._ndl.get_table(
                "SHARADAR/SEP",
                date=str(as_of_date),
                paginate=True,
            )
            return sorted(df["ticker"].unique().tolist())
        except Exception as exc:
            logger.error("Sharadar universe fetch failed for %s: %s", as_of_date, exc)
            return []

    def get_fundamentals_pit(self, ticker, as_of_date, lag_days=45):
        """
        Return SF1 fundamentals with reporting-lag enforcement.

        Filters to rows where datekey (filing date) <= as_of_date - lag_days.
        Returns the most recent qualifying row, or None.
        """
        try:
            cutoff = as_of_date - timedelta(days=lag_days)
            df = self._ndl.get_table(
                "SHARADAR/SF1",
                ticker=ticker,
                dimension="ARQ",   # as-reported quarterly
                paginate=True,
            )
            if df.empty:
                return None
            # Filter: only rows filed on or before the cutoff
            df["datekey"] = pd.to_datetime(df["datekey"])
            available = df[df["datekey"] <= pd.Timestamp(cutoff)]
            if available.empty:
                return None
            return available.sort_values("datekey").iloc[-1].to_dict()
        except Exception as exc:
            logger.error("Sharadar fundamentals fetch failed for %s: %s", ticker, exc)
            return None

    def get_price_history_pit(self, ticker, as_of_date, years=3):
        """Return SEP price history up to as_of_date (no future prices)."""
        try:
            start = as_of_date - timedelta(days=years * 365 + 30)
            df = self._ndl.get_table(
                "SHARADAR/SEP",
                ticker=ticker,
                date={"gte": str(start), "lte": str(as_of_date)},
                paginate=True,
            )
            if df.empty:
                return None
            df = df.sort_values("date").rename(columns={
                "open": "open", "high": "high", "low": "low",
                "close": "close", "volume": "volume",
                "closeadj": "adj_close",
            })
            df["date"] = pd.to_datetime(df["date"])
            return df.set_index("date")
        except Exception as exc:
            logger.error("Sharadar price fetch failed for %s: %s", ticker, exc)
            return None


# ──────────────────────────────────────────────────────────────
# Synthetic adapter (unit tests only)
# ──────────────────────────────────────────────────────────────

class SyntheticPITAdapter(PITAdapter):
    """
    Deterministic synthetic PIT data for unit testing.

    Generates N tickers with monthly observations.  Composite scores are
    positively correlated with 12-month forward returns (true IC ≈ 0.08–0.12)
    so the grade-validation tests can verify monotonicity and IC computation.

    Never use in production — this is test scaffolding only.
    """
    name = "synthetic"

    def __init__(
        self,
        n_tickers:  int  = 30,
        start_date: date = date(2018, 1, 1),
        end_date:   date = date(2022, 12, 31),
        seed:       int  = 42,
        true_ic:    float = 0.10,
    ):
        self._rng       = np.random.default_rng(seed)
        self._tickers   = [f"SYN{i:02d}" for i in range(n_tickers)]
        self._start     = start_date
        self._end       = end_date
        self._true_ic   = true_ic
        self._data      = self._generate()

    def is_pit_guaranteed(self) -> bool:
        return True

    def contamination_label(self) -> str:
        return "Synthetic deterministic data — no real-world contamination."

    def get_universe(self, as_of_date: date) -> list[str]:
        return list(self._tickers)

    def get_fundamentals_pit(self, ticker, as_of_date, lag_days=45):
        key = (ticker, as_of_date)
        return self._data.get("fundamentals", {}).get(key)

    def get_price_history_pit(self, ticker, as_of_date, years=3):
        df = self._data.get("prices", {}).get(ticker)
        if df is None:
            return None
        return df[df.index <= pd.Timestamp(as_of_date)]

    def get_all_scores(self) -> pd.DataFrame:
        """
        Return the full pre-computed score/return table for testing.
        Columns: date, ticker, composite, fund_score, tech_score, quant_score,
                 grade, fwd_1m, fwd_3m, fwd_6m, fwd_12m
        """
        return self._data["scores"].copy()

    def _generate(self) -> dict:
        """Generate synthetic scores and forward returns with known IC."""
        months = pd.date_range(self._start, self._end, freq="MS")
        n_m    = len(months)
        n_t    = len(self._tickers)

        # Latent quality factor → drives both scores and returns
        quality = self._rng.standard_normal((n_m, n_t))  # (months, tickers)

        # Composite score: quality + noise, mapped to [20, 85]
        noise_score = self._rng.standard_normal((n_m, n_t)) * 0.8
        raw_scores  = quality * (1 - self._true_ic) + noise_score
        scores_01   = (raw_scores - raw_scores.min()) / (raw_scores.max() - raw_scores.min() + 1e-9)
        composites  = (scores_01 * 65 + 20).clip(20, 85)

        # Forward returns: IC * composite + independent noise
        ic_signal   = quality * self._true_ic
        noise_ret   = self._rng.standard_normal((n_m, n_t)) * 0.20
        fwd_12m_raw = ic_signal + noise_ret  # z-scores approximately

        # Convert to annualized returns (mean ~8%, std ~20%)
        fwd_12m     = 0.08 + fwd_12m_raw * 0.15

        rows = []
        for mi, dt in enumerate(months):
            for ti, tkr in enumerate(self._tickers):
                comp = float(composites[mi, ti])
                f_sc = float(composites[mi, ti] * 0.5 + self._rng.uniform(-5, 5))
                t_sc = float(composites[mi, ti] * 0.3 + self._rng.uniform(-8, 8))
                q_sc = float(composites[mi, ti] * 0.2 + self._rng.uniform(-5, 5))
                f12  = float(fwd_12m[mi, ti])
                f6   = f12 * 0.5 + self._rng.standard_normal() * 0.08
                f3   = f12 * 0.25 + self._rng.standard_normal() * 0.06
                f1   = f12 * 0.08 + self._rng.standard_normal() * 0.04
                rows.append({
                    "date":        dt.date(),
                    "ticker":      tkr,
                    "composite":   comp,
                    "fund_score":  float(np.clip(f_sc, 0, 100)),
                    "tech_score":  float(np.clip(t_sc, 0, 100)),
                    "quant_score": float(np.clip(q_sc, 0, 100)),
                    "grade":       _composite_to_grade_str(comp),
                    "fwd_1m":      f1,
                    "fwd_3m":      f3,
                    "fwd_6m":      f6,
                    "fwd_12m":     f12,
                })

        scores_df = pd.DataFrame(rows)
        return {"scores": scores_df, "prices": {}, "fundamentals": {}}


def _composite_to_grade_str(comp: float) -> str:
    if comp >= 80: return "Gotta Have"
    if comp >= 60: return "Buy"
    if comp >= 40: return "Hold"
    if comp >= 20: return "Sell"
    return "Stay Away"
