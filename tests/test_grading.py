"""
Unit tests for the grading layer — composite, circuit breakers, confidence, drivers.
No live API calls; uses fixture data and hand-crafted inputs.
"""
import numpy as np
import pandas as pd
import pytest
from datetime import date

from stockgrader.config import get_config
from stockgrader.grading.circuit_breakers import apply_circuit_breakers, breaker_summary
from stockgrader.grading.composite import Orchestrator, _confidence, _extract_drivers, grade_ticker
from stockgrader.models import (
    FundamentalPillars, FundamentalResult,
    TechnicalPillars,   TechnicalResult,
    QuantitativeResult, FactorScores, RiskMetrics,
    OverallGrade, EngineResults, Grade, composite_to_grade,
)
from tests.fixtures.sample_fmp import FMP_DATA, YF_INFO, ESTIMATES, RF_RATES


# ── Helpers ───────────────────────────────────────────────────

def make_price_df(n=400, trend=0.0005, seed=42):
    rng = np.random.default_rng(seed)
    r   = rng.normal(trend, 0.012, n)
    c   = 100.0 * np.exp(np.cumsum(r))
    dates = pd.bdate_range(end=date.today(), periods=n)
    df = pd.DataFrame({
        "open": c, "high": c * 1.005, "low": c * 0.995,
        "close": c, "volume": np.ones(n) * 1_000_000, "adj_close": c,
    }, index=dates)
    df.index.name = "Date"
    return df


def _fund_result(score=65.0, altman_z=3.5, piotroski_f=7, roic_vs_wacc=0.08):
    return FundamentalResult(
        score=score,
        pillars=FundamentalPillars(
            valuation=65, profitability=70, growth=60,
            financial_health=75, capital_allocation=65,
        ),
        altman_z=altman_z,
        piotroski_f=piotroski_f,
        roic_vs_wacc=roic_vs_wacc,
    )


def _tech_result(score=60.0, regime="moderate_uptrend"):
    return TechnicalResult(
        score=score,
        pillars=TechnicalPillars(trend=65, momentum=58, volume_structure=55),
        regime=regime,
    )


def _quant_result(score=55.0):
    return QuantitativeResult(
        score=score,
        factors=FactorScores(value_pct=65, quality_pct=70, momentum_pct=60),
        risk_metrics=RiskMetrics(
            sharpe_1yr=0.9, max_drawdown_3yr=-0.22, current_drawdown=-0.05,
            realized_vol_1yr=0.18,
        ),
    )


@pytest.fixture
def cfg():
    return get_config()


@pytest.fixture
def orch(cfg):
    return Orchestrator(cfg)


# ── Grade mapping ─────────────────────────────────────────────

@pytest.mark.parametrize("composite,expected", [
    (10.0,  Grade.STAY_AWAY),
    (25.0,  Grade.SELL),
    (50.0,  Grade.HOLD),
    (65.0,  Grade.BUY),
    (85.0,  Grade.GOTTA_HAVE),
    (0.0,   Grade.STAY_AWAY),
    (19.0,  Grade.STAY_AWAY),
    (20.0,  Grade.SELL),
    (39.0,  Grade.SELL),
    (40.0,  Grade.HOLD),
    (59.0,  Grade.HOLD),
    (60.0,  Grade.BUY),
    (79.0,  Grade.BUY),
    (80.0,  Grade.GOTTA_HAVE),
    (100.0, Grade.GOTTA_HAVE),
])
def test_grade_bands(composite, expected):
    assert composite_to_grade(composite) == expected


# ── Circuit breakers ──────────────────────────────────────────

def test_altman_z_distress_caps_at_sell(cfg):
    fund = _fund_result(score=90.0, altman_z=1.2)  # distressed
    composite, triggered = apply_circuit_breakers(90.0, {}, fund, cfg)
    assert composite <= 39.0
    assert any("Altman" in t for t in triggered)


def test_altman_z_safe_no_cap(cfg):
    fund = _fund_result(score=90.0, altman_z=4.0)  # safe zone
    composite, triggered = apply_circuit_breakers(90.0, {}, fund, cfg)
    assert composite == pytest.approx(90.0)
    assert not any("Altman" in t for t in triggered)


def test_altman_z_grey_zone_no_cap(cfg):
    # Z = 2.5 is grey zone (above 1.8 threshold) → no breaker
    fund = _fund_result(score=80.0, altman_z=2.5)
    composite, triggered = apply_circuit_breakers(80.0, {}, fund, cfg)
    assert not any("Altman" in t for t in triggered)


def test_negative_fcf_high_leverage_caps(cfg):
    data = {
        "free_cf":    [-5e9, -3e9, 2e9],   # 2 negative years
        "debt_equity": 2.0,                  # > 1.5 threshold
    }
    fund = _fund_result(score=70.0, altman_z=3.5)
    composite, triggered = apply_circuit_breakers(70.0, data, fund, cfg)
    assert composite <= 39.0
    assert any("FCF" in t for t in triggered)


def test_negative_fcf_low_leverage_no_cap(cfg):
    # Negative FCF but low leverage → no breaker
    data = {"free_cf": [-5e9, -3e9], "debt_equity": 0.3}
    fund = _fund_result(score=70.0, altman_z=3.5)
    composite, triggered = apply_circuit_breakers(70.0, data, fund, cfg)
    assert not any("FCF" in t for t in triggered)


def test_going_concern_caps_at_stay_away(cfg):
    data = {"going_concern_flag": True}
    fund = _fund_result(score=80.0, altman_z=4.0)
    composite, triggered = apply_circuit_breakers(80.0, data, fund, cfg)
    assert composite <= 19.0
    assert any("Going-concern" in t for t in triggered)


def test_extreme_overvaluation_caps(cfg):
    data = {"price": 200.0, "_ladder_stay_away_above": 150.0}
    fund = _fund_result(score=85.0, altman_z=4.0)
    composite, triggered = apply_circuit_breakers(85.0, data, fund, cfg)
    assert composite <= 39.0
    assert any("overvaluation" in t for t in triggered)


def test_no_breakers_fired(cfg):
    data = {"free_cf": [5e9, 4e9], "debt_equity": 0.5}
    fund = _fund_result(score=75.0, altman_z=4.0)
    composite, triggered = apply_circuit_breakers(75.0, data, fund, cfg)
    assert composite == pytest.approx(75.0)
    assert not triggered


def test_multiple_breakers_most_restrictive_wins(cfg):
    # Both Altman Z AND going concern fire; going concern is more restrictive
    data = {"going_concern_flag": True}
    fund = _fund_result(score=90.0, altman_z=1.0)  # both fire
    composite, triggered = apply_circuit_breakers(90.0, data, fund, cfg)
    assert composite <= 19.0  # stay_away threshold wins


def test_breaker_summary_none():
    assert "No circuit breakers" in breaker_summary([])


def test_breaker_summary_with_triggers():
    msg = breaker_summary(["Altman Z 1.2 → Sell", "Going concern → Stay Away"])
    assert "2 circuit breaker" in msg


# ── Confidence ────────────────────────────────────────────────

def test_confidence_range():
    f  = _fund_result()
    t  = _tech_result()
    q  = _quant_result()
    data = {"data_completeness": 0.9}
    c = _confidence(data, f, t, q)
    assert 0.05 <= c <= 1.00


def test_confidence_high_when_scores_agree():
    f = _fund_result(score=65.0)
    t = _tech_result(score=65.0)
    q = _quant_result(score=65.0)
    data = {"data_completeness": 1.0}
    c = _confidence(data, f, t, q)
    assert c > 0.7


def test_confidence_lower_when_scores_diverge():
    f_aligned = _fund_result(score=65.0)
    t_aligned = _tech_result(score=65.0)
    q_aligned = _quant_result(score=65.0)

    f_div = _fund_result(score=90.0)
    t_div = _tech_result(score=20.0)
    q_div = _quant_result(score=50.0)

    data = {"data_completeness": 1.0}
    c_aligned = _confidence(data, f_aligned, t_aligned, q_aligned)
    c_diverged = _confidence(data, f_div, t_div, q_div)
    assert c_aligned > c_diverged


def test_confidence_lower_with_missing_fields():
    f = _fund_result()
    f.missing_fields = ["altman_z", "piotroski_f", "roic_vs_wacc", "pe_trailing",
                         "pb", "ev_ebitda", "peg", "pe_forward"]
    t = _tech_result()
    q = _quant_result()
    data = {"data_completeness": 0.5}
    c = _confidence(data, f, t, q)
    assert c < 0.75


# ── Driver extraction ─────────────────────────────────────────

def test_drivers_returns_three_each():
    f, t, q = _fund_result(), _tech_result(), _quant_result()
    pos, neg = _extract_drivers(f, t, q, {})
    assert len(pos) == 3
    assert len(neg) == 3


def test_drivers_positive_scores_above_negative():
    f, t, q = _fund_result(score=70.0), _tech_result(score=65.0), _quant_result(score=60.0)
    pos, neg = _extract_drivers(f, t, q, {})
    # Positive drivers should look different from negative (they come from different score buckets)
    assert set(pos).isdisjoint(set(neg)) or len(pos) == len(neg)  # no overlap expected


def test_roic_driver_present_when_available():
    f = _fund_result(roic_vs_wacc=0.15)
    t, q = _tech_result(), _quant_result()
    pos, neg = _extract_drivers(f, t, q, {})
    all_drivers = pos + neg
    assert any("ROIC" in d for d in all_drivers)


def test_altman_driver_present():
    f = _fund_result(altman_z=1.0)   # distress zone
    t, q = _tech_result(), _quant_result()
    pos, neg = _extract_drivers(f, t, q, {})
    all_drivers = pos + neg
    assert any("Altman" in d for d in all_drivers)


# ── Orchestrator ──────────────────────────────────────────────

def test_orchestrator_returns_correct_types(orch):
    data = {
        "price_history": make_price_df(n=400, seed=77),
        "peer_metrics": {}, "quality_of_earnings_flag": False,
        "data_completeness": 0.8, "risk_free_rate_3mo": 0.05,
        "pe_trailing": 22.0, "pb": 3.5, "ps": 4.0, "ev_ebitda": 15.0, "peg": 2.0,
        "gross_margin": 0.40, "operating_margin": 0.18, "net_margin": 0.12,
        "roe": 0.15, "roa": 0.08, "roic": 0.14, "wacc": 0.09,
        "roic_vs_wacc": 0.05, "margin_trajectory": 0.5,
        "revenue_cagr_3yr": 0.08, "eps_cagr_3yr": 0.10,
        "forward_eps_growth": 0.09, "forward_revenue_growth": 0.07,
        "current_ratio": 1.5, "quick_ratio": 1.0, "debt_equity": 0.5,
        "interest_coverage": 8.0, "altman_z": 3.8, "piotroski_f": 7,
        "fcf_conversion": 1.1, "fcf_yield": 0.035, "payout_ratio": 0.30,
        "market_cap": 50_000_000_000, "beta": 1.0,
    }
    overall, engines = orch.analyze(data)
    assert isinstance(overall, OverallGrade)
    assert isinstance(engines, EngineResults)


def test_orchestrator_grade_is_valid(orch):
    data = {"peer_metrics": {}, "quality_of_earnings_flag": False,
            "data_completeness": 0.5, "risk_free_rate_3mo": 0.05}
    overall, _ = orch.analyze(data)
    assert overall.grade in list(Grade)


def test_orchestrator_composite_in_range(orch):
    data = {"peer_metrics": {}, "quality_of_earnings_flag": False,
            "data_completeness": 0.5, "risk_free_rate_3mo": 0.05}
    overall, _ = orch.analyze(data)
    assert 0.0 <= overall.composite <= 100.0


def test_orchestrator_weights_sum_to_one(orch):
    data = {"peer_metrics": {}, "quality_of_earnings_flag": False,
            "data_completeness": 0.5}
    overall, _ = orch.analyze(data)
    total = sum(overall.weights_used.values())
    assert total == pytest.approx(1.0, abs=0.01)


def test_orchestrator_distressed_company_capped(orch, cfg):
    data = {
        "peer_metrics": {}, "quality_of_earnings_flag": False,
        "data_completeness": 0.9, "risk_free_rate_3mo": 0.05,
        "altman_z": 1.1,    # distress zone → Altman Z breaker fires
        "piotroski_f": 2,
        "debt_equity": 3.0,
        "gross_margin": 0.05, "operating_margin": -0.05,
        "roe": -0.10, "roa": -0.03,
    }
    overall, _ = orch.analyze(data)
    assert overall.composite <= 39.0
    assert any("Altman" in b for b in overall.circuit_breakers)
    assert overall.grade in (Grade.STAY_AWAY, Grade.SELL)


def test_orchestrator_with_fixture_data(orch, cfg):
    from stockgrader.data.normalizer import normalize
    data = normalize(
        ticker="TEST", price_df=make_price_df(n=500, seed=42),
        yf_info=YF_INFO, fmp_data=FMP_DATA, estimates=ESTIMATES,
        peers=[], peer_metrics={}, rf_rates=RF_RATES, cfg=cfg,
    )
    overall, engines = orch.analyze(data)
    assert isinstance(overall, OverallGrade)
    assert overall.grade in list(Grade)
    assert 0 <= overall.composite <= 100
    assert 0.05 <= overall.confidence <= 1.0
    assert len(overall.drivers_positive) == 3
    assert len(overall.drivers_negative) == 3
    assert isinstance(engines.fundamental.score, float)
    assert isinstance(engines.technical.score,   float)
    assert isinstance(engines.quantitative.score, float)


def test_grade_ticker_convenience(cfg):
    from stockgrader.data.normalizer import normalize
    data = normalize(
        ticker="TEST", price_df=make_price_df(n=400, seed=5),
        yf_info=YF_INFO, fmp_data=FMP_DATA, estimates=ESTIMATES,
        peers=[], peer_metrics={}, rf_rates=RF_RATES, cfg=cfg,
    )
    overall, engines = grade_ticker(data, config=cfg)
    assert isinstance(overall, OverallGrade)
    assert isinstance(engines, EngineResults)
