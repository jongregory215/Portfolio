"""Pydantic result models for Charlie Munger evaluation."""
from __future__ import annotations

from typing import Any
from pydantic import BaseModel

from stockgrader.graham.models import CriterionResult

__all__ = ["CriterionResult", "MungerResult"]


class MungerResult(BaseModel):
    ticker: str
    as_of: str
    price: float
    company_name: str
    criteria: list[CriterionResult]
    criteria_met: int
    total_criteria: int
    verdict: str   # "Munger-Grade Business" | "Strong Candidate" | "Pass"
    intrinsic_value: float | None
    price_vs_intrinsic_pct: float | None   # ((price - iv) / iv) * 100; negative = undervalued
    roic: float | None
    capex_intensity: float | None          # capex / revenue

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()
