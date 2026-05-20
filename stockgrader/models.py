"""
Pydantic output schema for the Stock Analysis Engine.

Every analysis run produces an AnalysisResult. All sub-objects are typed
so downstream consumers (reporters, backtester, portfolio optimizer) can
rely on a stable contract.
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


# ──────────────────────────────────────────────────────────────
# Grade scale
# ──────────────────────────────────────────────────────────────

class Grade(str, Enum):
    STAY_AWAY  = "Stay Away"
    SELL       = "Sell"
    HOLD       = "Hold"
    BUY        = "Buy"
    GOTTA_HAVE = "Gotta Have"


# Maps each grade to its (inclusive) composite score band.
GRADE_BANDS: dict[Grade, tuple[int, int]] = {
    Grade.STAY_AWAY:  (0,  19),
    Grade.SELL:       (20, 39),
    Grade.HOLD:       (40, 59),
    Grade.BUY:        (60, 79),
    Grade.GOTTA_HAVE: (80, 100),
}

_GRADE_ORDER = [Grade.STAY_AWAY, Grade.SELL, Grade.HOLD, Grade.BUY, Grade.GOTTA_HAVE]


def composite_to_grade(composite: float) -> Grade:
    """Map a 0–100 composite score to its grade bucket."""
    c = max(0.0, min(100.0, composite))
    for grade, (lo, hi) in GRADE_BANDS.items():
        if lo <= c <= hi:
            return grade
    return Grade.STAY_AWAY


def cap_grade(current: Grade, cap: Grade) -> Grade:
    """Return the lower of current and cap (e.g. circuit breaker caps)."""
    return current if _GRADE_ORDER.index(current) <= _GRADE_ORDER.index(cap) else cap


# ──────────────────────────────────────────────────────────────
# Price ladder
# ──────────────────────────────────────────────────────────────

class FairValueSensitivity(BaseModel):
    """DCF fair value under low / base / high growth+discount-rate scenarios."""
    low:  float
    base: float
    high: float


class SensitivityGrid(BaseModel):
    """
    2-D sensitivity table: growth_delta × discount_rate_delta → fair value.
    Keys are string representations of the delta pairs, e.g. "-0.02/-0.01".
    """
    cells: dict[str, float] = Field(default_factory=dict)
    growth_deltas:        list[float] = Field(default_factory=list)
    discount_rate_deltas: list[float] = Field(default_factory=list)


class PriceLadder(BaseModel):
    """Grade-boundary price levels and fair-value estimates."""
    fair_value:             float
    fair_value_sensitivity: FairValueSensitivity
    sensitivity_grid:       SensitivityGrid = Field(default_factory=SensitivityGrid)

    gotta_have_at:  float          # price ≤ this → Gotta Have
    buy_at:         float          # price ≤ this → Buy
    hold_low:       float          # lower bound of Hold range (= buy_at)
    hold_high:      float          # upper bound of Hold range (= sell_above)
    sell_above:     float          # price > this → Sell
    stay_away_above: float         # price > this → Stay Away

    upside_to_fv_pct:        float   # (fv - price) / price
    implied_growth_rate:     Optional[float] = None  # reverse-DCF implied growth


# ──────────────────────────────────────────────────────────────
# Fundamental engine
# ──────────────────────────────────────────────────────────────

class FundamentalPillars(BaseModel):
    valuation:          float = Field(..., ge=0, le=100)
    profitability:      float = Field(..., ge=0, le=100)
    growth:             float = Field(..., ge=0, le=100)
    financial_health:   float = Field(..., ge=0, le=100)
    capital_allocation: float = Field(..., ge=0, le=100)
    details:            dict[str, Any] = Field(default_factory=dict)


class FundamentalResult(BaseModel):
    score:   float = Field(..., ge=0, le=100)
    pillars: FundamentalPillars

    # Key derived signals surfaced explicitly
    roic_vs_wacc:          Optional[float] = None   # ROIC - WACC in percentage points
    wacc:                  Optional[float] = None
    roic:                  Optional[float] = None
    altman_z:              Optional[float] = None
    piotroski_f:           Optional[int]   = None   # 0–9
    margin_trajectory_3yr: Optional[float] = None   # operating margin slope (ppts/yr)
    quality_of_earnings_flag: bool = False           # True = accruals > 20% of net income

    peer_count: int = 0
    missing_fields: list[str] = Field(default_factory=list)
    imputed_fields: list[str] = Field(default_factory=list)


# ──────────────────────────────────────────────────────────────
# Technical engine
# ──────────────────────────────────────────────────────────────

class TechnicalPillars(BaseModel):
    trend:            float = Field(..., ge=0, le=100)
    momentum:         float = Field(..., ge=0, le=100)
    volume_structure: float = Field(..., ge=0, le=100)
    details:          dict[str, Any] = Field(default_factory=dict)


class TechnicalResult(BaseModel):
    score:   float = Field(..., ge=0, le=100)
    pillars: TechnicalPillars

    regime: str = "unknown"    # e.g. "strong_uptrend", "sideways", "downtrend"

    # Key levels for price-target input
    nearest_support:    Optional[float] = None
    nearest_resistance: Optional[float] = None
    week_52_high:       Optional[float] = None
    week_52_low:        Optional[float] = None

    missing_fields: list[str] = Field(default_factory=list)


# ──────────────────────────────────────────────────────────────
# Quantitative engine
# ──────────────────────────────────────────────────────────────

class FactorScores(BaseModel):
    """Cross-sectional factor z-scores and universe percentiles."""
    # Z-scores (mean=0, std=1 vs. factor universe)
    value_z:          Optional[float] = None
    quality_z:        Optional[float] = None
    momentum_z:       Optional[float] = None
    size_z:           Optional[float] = None
    low_volatility_z: Optional[float] = None

    # Percentile ranks (0–100) within the factor universe
    value_pct:          Optional[float] = None
    quality_pct:        Optional[float] = None
    momentum_pct:       Optional[float] = None
    size_pct:           Optional[float] = None
    low_volatility_pct: Optional[float] = None


class RiskMetrics(BaseModel):
    beta_1yr:          Optional[float] = None
    beta_3yr:          Optional[float] = None
    sharpe_1yr:        Optional[float] = None
    sharpe_3yr:        Optional[float] = None
    sortino_1yr:       Optional[float] = None
    sortino_3yr:       Optional[float] = None
    max_drawdown_3yr:  Optional[float] = None   # expressed as negative fraction, e.g. -0.35
    current_drawdown:  Optional[float] = None   # drawdown from most recent peak
    downside_deviation_1yr: Optional[float] = None
    corr_spy:          Optional[float] = None
    corr_agg:          Optional[float] = None
    realized_vol_1yr:  Optional[float] = None   # annualized


class FFRegression(BaseModel):
    """Fama-French factor regression loadings."""
    model: str = "FF5"   # FF3 | FF5

    # FF3
    mkt_rf_beta: Optional[float] = None
    smb_beta:    Optional[float] = None
    hml_beta:    Optional[float] = None

    # FF5 additions
    rmw_beta: Optional[float] = None
    cma_beta: Optional[float] = None

    # Intercept
    alpha_annualized: Optional[float] = None
    alpha_t_stat:     Optional[float] = None
    r_squared:        Optional[float] = None
    period_years:     Optional[float] = None


class QuantitativeResult(BaseModel):
    score:        float = Field(..., ge=0, le=100)
    factors:      FactorScores
    risk_metrics: RiskMetrics
    ff_regression: Optional[FFRegression] = None

    missing_fields: list[str] = Field(default_factory=list)


# ──────────────────────────────────────────────────────────────
# Combined engine outputs
# ──────────────────────────────────────────────────────────────

class EngineResults(BaseModel):
    fundamental:  FundamentalResult
    technical:    TechnicalResult
    quantitative: QuantitativeResult


# ──────────────────────────────────────────────────────────────
# Overall grade
# ──────────────────────────────────────────────────────────────

class OverallGrade(BaseModel):
    grade:     Grade
    composite: float = Field(..., ge=0, le=100)
    confidence: float = Field(..., ge=0.0, le=1.0)

    # Explainability — required; no grade ships without this
    drivers_positive: list[str] = Field(default_factory=list)   # top 3 positive
    drivers_negative: list[str] = Field(default_factory=list)   # top 3 negative
    circuit_breakers: list[str] = Field(default_factory=list)   # any breakers fired

    # Engine sub-scores used in the composite (for audit trail)
    fundamental_score:  float = 0.0
    technical_score:    float = 0.0
    quantitative_score: float = 0.0
    weights_used: dict[str, float] = Field(default_factory=dict)


# ──────────────────────────────────────────────────────────────
# Portfolio sub-grades
# ──────────────────────────────────────────────────────────────

class GateResult(BaseModel):
    gate:   str
    passed: bool
    value:  Optional[Any] = None     # actual value tested
    limit:  Optional[Any] = None     # threshold value


class PortfolioGrade(BaseModel):
    grade:      Grade
    composite:  float = Field(..., ge=0, le=100)
    gate_results: list[GateResult] = Field(default_factory=list)
    failed_gates: list[str] = Field(default_factory=list)
    rationale:  str = ""

    @model_validator(mode="after")
    def sync_failed_gates(self) -> "PortfolioGrade":
        self.failed_gates = [g.gate for g in self.gate_results if not g.passed]
        return self


class PortfolioGrades(BaseModel):
    very_conservative: PortfolioGrade
    conservative:      PortfolioGrade
    balanced:          PortfolioGrade
    aggressive:        PortfolioGrade
    very_aggressive:   PortfolioGrade

    def as_dict(self) -> dict[str, PortfolioGrade]:
        return {
            "very_conservative": self.very_conservative,
            "conservative":      self.conservative,
            "balanced":          self.balanced,
            "aggressive":        self.aggressive,
            "very_aggressive":   self.very_aggressive,
        }


# ──────────────────────────────────────────────────────────────
# Data quality metadata
# ──────────────────────────────────────────────────────────────

class DataQuality(BaseModel):
    missing_fields:  list[str] = Field(default_factory=list)
    imputed_fields:  list[str] = Field(default_factory=list)
    sources: dict[str, str]   = Field(default_factory=dict)   # field → provider name
    as_of_timestamps: dict[str, datetime] = Field(default_factory=dict)
    warnings: list[str]       = Field(default_factory=list)
    data_completeness: float  = 1.0    # fraction of expected fields actually available


# ──────────────────────────────────────────────────────────────
# Top-level result
# ──────────────────────────────────────────────────────────────

class AnalysisResult(BaseModel):
    """Canonical output of a single-ticker analysis run."""
    ticker:   str
    as_of:    datetime
    price:    float

    overall:      OverallGrade
    price_ladder: PriceLadder
    engines:      EngineResults
    portfolios:   PortfolioGrades
    data_quality: DataQuality

    # Reproducibility
    config_hash: str = ""   # SHA-256[:16] of config.yaml at run time
    run_id:      str = ""   # UUID assigned per run

    def to_json_dict(self) -> dict[str, Any]:
        """Serialize to the canonical JSON format documented in spec §13.1."""
        pg = self.portfolios
        return {
            "ticker": self.ticker,
            "as_of": self.as_of.isoformat(),
            "price": self.price,
            "overall": {
                "grade":            self.overall.grade.value,
                "composite":        round(self.overall.composite, 2),
                "confidence":       round(self.overall.confidence, 3),
                "drivers_positive": self.overall.drivers_positive,
                "drivers_negative": self.overall.drivers_negative,
                "circuit_breakers": self.overall.circuit_breakers,
            },
            "price_ladder": {
                "gotta_have_at":        self.price_ladder.gotta_have_at,
                "buy_at":               self.price_ladder.buy_at,
                "hold_range":           [self.price_ladder.hold_low, self.price_ladder.hold_high],
                "sell_above":           self.price_ladder.sell_above,
                "stay_away_above":      self.price_ladder.stay_away_above,
                "fair_value":           self.price_ladder.fair_value,
                "fair_value_sensitivity": {
                    "low":  self.price_ladder.fair_value_sensitivity.low,
                    "base": self.price_ladder.fair_value_sensitivity.base,
                    "high": self.price_ladder.fair_value_sensitivity.high,
                },
                "upside_to_fv_pct": round(self.price_ladder.upside_to_fv_pct, 4),
            },
            "engines": {
                "fundamental": {
                    "score":   round(self.engines.fundamental.score, 2),
                    "pillars": {
                        "valuation":          round(self.engines.fundamental.pillars.valuation, 2),
                        "profitability":      round(self.engines.fundamental.pillars.profitability, 2),
                        "growth":             round(self.engines.fundamental.pillars.growth, 2),
                        "financial_health":   round(self.engines.fundamental.pillars.financial_health, 2),
                        "capital_allocation": round(self.engines.fundamental.pillars.capital_allocation, 2),
                    },
                    "roic_vs_wacc": self.engines.fundamental.roic_vs_wacc,
                    "altman_z":     self.engines.fundamental.altman_z,
                    "piotroski_f":  self.engines.fundamental.piotroski_f,
                },
                "technical": {
                    "score":  round(self.engines.technical.score, 2),
                    "pillars": {
                        "trend":            round(self.engines.technical.pillars.trend, 2),
                        "momentum":         round(self.engines.technical.pillars.momentum, 2),
                        "volume_structure": round(self.engines.technical.pillars.volume_structure, 2),
                    },
                    "regime": self.engines.technical.regime,
                },
                "quantitative": {
                    "score": round(self.engines.quantitative.score, 2),
                    "factors": {
                        "value_z":          self.engines.quantitative.factors.value_z,
                        "quality_z":        self.engines.quantitative.factors.quality_z,
                        "momentum_z":       self.engines.quantitative.factors.momentum_z,
                        "size_z":           self.engines.quantitative.factors.size_z,
                        "low_volatility_z": self.engines.quantitative.factors.low_volatility_z,
                    },
                    "risk_metrics": {
                        "beta_1yr":         self.engines.quantitative.risk_metrics.beta_1yr,
                        "sharpe_1yr":       self.engines.quantitative.risk_metrics.sharpe_1yr,
                        "max_drawdown_3yr": self.engines.quantitative.risk_metrics.max_drawdown_3yr,
                        "corr_spy":         self.engines.quantitative.risk_metrics.corr_spy,
                    },
                },
            },
            "portfolios": {
                name: {
                    "grade":        pg_obj.grade.value,
                    "composite":    round(pg_obj.composite, 2),
                    "failed_gates": pg_obj.failed_gates,
                    "rationale":    pg_obj.rationale,
                }
                for name, pg_obj in pg.as_dict().items()
            },
            "data_quality": {
                "missing_fields":    self.data_quality.missing_fields,
                "imputed_fields":    self.data_quality.imputed_fields,
                "data_completeness": round(self.data_quality.data_completeness, 3),
                "sources":           self.data_quality.sources,
                "warnings":          self.data_quality.warnings,
            },
            "meta": {
                "config_hash": self.config_hash,
                "run_id":      self.run_id,
            },
        }
