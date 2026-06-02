"""Pydantic result models for Graham evaluation."""
from __future__ import annotations

from typing import Any
from pydantic import BaseModel


class CriterionResult(BaseModel):
    name: str
    passed: bool
    label: str        # displayed row, e.g. "0.95  (≥ 2.0)"
    note: str = ""    # optional clarification, e.g. "14 of 20 years available"


class DefensiveResult(BaseModel):
    criteria: list[CriterionResult]
    criteria_met: int
    total_criteria: int
    graham_number: float | None
    price_vs_graham_pct: float | None   # ((price - graham_number) / graham_number) * 100
    verdict: str                         # "Qualifies" or "Does Not Qualify"
    notes: list[str] = []


class EnterprisingResult(BaseModel):
    criteria: list[CriterionResult]
    criteria_met: int
    total_criteria: int
    ncav_per_share: float | None
    price_vs_ncav_pct: float | None     # ((price - ncav) / abs(ncav)) * 100 when ncav > 0
    verdict: str
    notes: list[str] = []


class GrahamResult(BaseModel):
    ticker: str
    as_of: str
    price: float
    company_name: str
    eps_source: str = "yfinance"   # "fmp" when FMP provided extended history
    defensive: DefensiveResult
    enterprising: EnterprisingResult

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()
