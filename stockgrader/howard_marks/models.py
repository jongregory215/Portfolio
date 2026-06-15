"""Pydantic result models for Howard Marks evaluation."""
from __future__ import annotations

from typing import Any
from pydantic import BaseModel

from stockgrader.graham.models import CriterionResult

__all__ = ["CriterionResult", "CycleReading", "MarksResult"]


class CycleReading(BaseModel):
    """A one-time, market-wide 'where are we in the cycle' reading."""

    as_of: str
    yield_curve_spread: float | None   # 10yr - 2yr treasury (FRED T10Y2Y)
    high_yield_spread: float | None    # ICE BofA US High Yield OAS (FRED BAMLH0A0HYM2)
    vix: float | None                  # CBOE Volatility Index (^VIX)
    zone: str                          # "Fear / Capitulation" | "Neutral" | "Greed / Late-Cycle"
    commentary: str
    mos_multiplier: float              # adjusts the EPV margin-of-safety threshold


class MarksResult(BaseModel):
    ticker: str
    as_of: str
    price: float
    company_name: str
    criteria: list[CriterionResult]
    criteria_met: int
    total_criteria: int
    verdict: str   # "Compelling Opportunity" | "Worth a Closer Look" | "Pass"
    epv: float | None
    price_vs_epv_pct: float | None       # ((price - epv) / epv) * 100
    range_position_pct: float | None     # 0-100, position within 52wk range
    reward_risk_ratio: float | None
    notes: list[str] = []

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()
