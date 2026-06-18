"""Pydantic result models for Warren Buffett evaluation."""
from __future__ import annotations

from typing import Any
from pydantic import BaseModel

from stockgrader.graham.models import CriterionResult

__all__ = ["CriterionResult", "BuffettResult"]


class BuffettResult(BaseModel):
    ticker: str
    as_of: str
    price: float
    company_name: str
    criteria: list[CriterionResult]
    criteria_met: int
    total_criteria: int
    verdict: str   # "Exceptional Business at a Fair Price" | "Strong Candidate" | "Pass"
    intrinsic_value: float | None
    price_vs_intrinsic_pct: float | None   # ((price - iv) / iv) * 100; negative = undervalued
    roe: float | None
    roic_spread: float | None              # ROIC - required return (ppts)
    notes: list[str] = []

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()
