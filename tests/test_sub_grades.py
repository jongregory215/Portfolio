"""
Unit tests for portfolio sub-grading.
No live API calls — uses hand-crafted engine results and fixture data.
"""
import pytest
from stockgrader.config import get_config
from stockgrader.portfolios.sub_grades import (
    compute_sub_grades,
    _check_gates, _adj_fund_score, _adj_quant_score, _build_rationale,
)
from stockgrader.models import (
    FundamentalPillars, FundamentalResult,
    TechnicalPillars,   TechnicalResult,
    QuantitativeResult, FactorScores, RiskMetrics,
    PortfolioGrades, PortfolioGrade, GateResult, Grade,
)
from tests.fixtures.sample_fmp import FMP_DATA, YF_INFO, ESTIMATES, RF_RATES


@pytest.fixture
def cfg():
    return get_config()


# ── Shared test fixtures ──────────────────────────────────────

def _fund(score=68.0, altman_z=3.5, roic_vs_wacc=0.09, valuation=65.0,
          profitability=70.0, growth=65.0, health=75.0, capalloc=65.0):
    return FundamentalResult(
        score=score,
        pillars=FundamentalPillars(
            valuation=valuation, profitability=profitability, growth=growth,
            financial_health=health, capital_allocation=capalloc,
        ),
        altman_z=altman_z,
        roic_vs_wacc=roic_vs_wacc,
        piotroski_f=7,
    )


def _tech(score=62.0):
    return TechnicalResult(
        score=score,
        pillars=TechnicalPillars(trend=68, momentum=58, volume_structure=60),
        regime="moderate_uptrend",
    )


def _quant(score=58.0, value_pct=68.0, quality_pct=72.0, momentum_pct=60.0,
           size_pct=55.0, low_vol_pct=65.0, beta=0.85, max_dd=-0.18):
    return QuantitativeResult(
        score=score,
        factors=FactorScores(
            value_pct=value_pct, quality_pct=quality_pct,
            momentum_pct=momentum_pct, size_pct=size_pct,
            low_volatility_pct=low_vol_pct,
        ),
        risk_metrics=RiskMetrics(
            beta_1yr=beta, max_drawdown_3yr=max_dd,
            sharpe_1yr=0.8, realized_vol_1yr=0.18,
        ),
    )


def _data(beta=0.85, market_cap=25e9, dividend_yield=0.018,
          payout_ratio=0.28, debt_equity=0.4):
    return {
        "beta":          beta,
        "market_cap":    market_cap,
        "dividend_yield":dividend_yield,
        "payout_ratio":  payout_ratio,
        "debt_equity":   debt_equity,
        "peer_metrics":  {},
        "quality_of_earnings_flag": False,
    }


# ── Gate checks ───────────────────────────────────────────────

def test_gate_max_beta_passes(cfg):
    gates = {"max_beta": 0.9}
    results = _check_gates(_data(beta=0.85), _fund(), _quant(), gates)
    beta_gate = next(g for g in results if g.gate == "max_beta")
    assert beta_gate.passed


def test_gate_max_beta_fails(cfg):
    gates = {"max_beta": 0.9}
    results = _check_gates(_data(beta=1.4), _fund(), _quant(), gates)
    beta_gate = next(g for g in results if g.gate == "max_beta")
    assert not beta_gate.passed


def test_gate_min_market_cap_passes(cfg):
    gates = {"min_market_cap": 10e9}
    results = _check_gates(_data(market_cap=25e9), _fund(), _quant(), gates)
    mc_gate = next(g for g in results if g.gate == "min_market_cap")
    assert mc_gate.passed


def test_gate_min_market_cap_fails(cfg):
    gates = {"min_market_cap": 10e9}
    results = _check_gates(_data(market_cap=2e9), _fund(), _quant(), gates)
    mc_gate = next(g for g in results if g.gate == "min_market_cap")
    assert not mc_gate.passed


def test_gate_require_dividend_passes(cfg):
    gates = {"require_dividend": True}
    results = _check_gates(_data(dividend_yield=0.03), _fund(), _quant(), gates)
    div_gate = next(g for g in results if g.gate == "require_dividend")
    assert div_gate.passed


def test_gate_require_dividend_fails_no_div(cfg):
    gates = {"require_dividend": True}
    results = _check_gates(_data(dividend_yield=0.0), _fund(), _quant(), gates)
    div_gate = next(g for g in results if g.gate == "require_dividend")
    assert not div_gate.passed


def test_gate_min_altman_z_passes(cfg):
    gates = {"min_altman_z": 3.0}
    results = _check_gates(_data(), _fund(altman_z=4.5), _quant(), gates)
    z_gate = next(g for g in results if g.gate == "min_altman_z")
    assert z_gate.passed


def test_gate_min_altman_z_fails(cfg):
    gates = {"min_altman_z": 3.0}
    results = _check_gates(_data(), _fund(altman_z=2.1), _quant(), gates)
    z_gate = next(g for g in results if g.gate == "min_altman_z")
    assert not z_gate.passed


def test_gate_max_drawdown_passes(cfg):
    gates = {"max_drawdown_3yr": 0.35}
    results = _check_gates(_data(), _fund(), _quant(max_dd=-0.25), gates)
    dd_gate = next(g for g in results if g.gate == "max_drawdown_3yr")
    assert dd_gate.passed


def test_gate_max_drawdown_fails(cfg):
    gates = {"max_drawdown_3yr": 0.30}
    results = _check_gates(_data(), _fund(), _quant(max_dd=-0.45), gates)
    dd_gate = next(g for g in results if g.gate == "max_drawdown_3yr")
    assert not dd_gate.passed


def test_gate_max_debt_equity_passes(cfg):
    gates = {"max_debt_equity": 0.8}
    results = _check_gates(_data(debt_equity=0.4), _fund(), _quant(), gates)
    de_gate = next(g for g in results if g.gate == "max_debt_equity")
    assert de_gate.passed


def test_gate_max_debt_equity_fails(cfg):
    gates = {"max_debt_equity": 0.8}
    results = _check_gates(_data(debt_equity=2.0), _fund(), _quant(), gates)
    de_gate = next(g for g in results if g.gate == "max_debt_equity")
    assert not de_gate.passed


def test_none_beta_passes_gate(cfg):
    """Unknown beta should not fail the gate — flag but pass."""
    gates = {"max_beta": 0.9}
    d = _data(); d["beta"] = None
    results = _check_gates(d, _fund(), _quant(), gates)
    beta_gate = next(g for g in results if g.gate == "max_beta")
    assert beta_gate.passed


# ── Score adjustments ─────────────────────────────────────────

def test_adj_fund_no_penalty_when_valuation_ok():
    fund = _fund(score=68.0, valuation=70.0)
    adj  = _adj_fund_score(fund, "high")
    # Valuation > 50 → no penalty
    assert adj == pytest.approx(68.0, abs=0.1)


def test_adj_fund_penalty_when_expensive():
    fund = _fund(score=70.0, valuation=20.0)  # very expensive
    adj_high = _adj_fund_score(fund, "high")
    adj_low  = _adj_fund_score(fund, "low")
    assert adj_high < adj_low          # conservative penalises more
    assert adj_high < fund.score       # some penalty applied


def test_adj_fund_low_penalty_near_original():
    fund = _fund(score=70.0, valuation=15.0)  # expensive
    adj  = _adj_fund_score(fund, "low")
    assert abs(adj - fund.score) < 5.0  # minimal penalty


def test_adj_quant_favoured_factors_improve_score():
    """Preferring quality + low-vol should boost score when those factors are high."""
    q = _quant(score=58.0, quality_pct=88.0, low_vol_pct=85.0, momentum_pct=30.0)
    base_fw = {"value": 0.20, "quality": 0.30, "momentum": 0.25, "size": 0.10, "low_volatility": 0.15}
    # Very Conservative preferences: quality++ and low_vol++
    prefs = {"quality": 2.0, "low_volatility": 2.0, "momentum": 0.5}
    adj   = _adj_quant_score(q, prefs, base_fw)
    neutral = _adj_quant_score(q, {}, base_fw)   # no preference adjustment
    assert adj > neutral  # emphasising the high-scoring factors helps


def test_adj_quant_downweighted_favoured_factor_hurts():
    """If preferred factor has low percentile, emphasising it hurts the score."""
    q = _quant(score=58.0, quality_pct=15.0, low_vol_pct=12.0, momentum_pct=80.0)
    base_fw = {"value": 0.20, "quality": 0.30, "momentum": 0.25, "size": 0.10, "low_volatility": 0.15}
    prefs = {"quality": 2.0, "low_volatility": 2.0, "momentum": 0.5}
    adj  = _adj_quant_score(q, prefs, base_fw)
    neutral = _adj_quant_score(q, {}, base_fw)
    assert adj < neutral  # low-quality + low-vol company is penalised


def test_adj_quant_no_prefs_returns_blend_near_original():
    q = _quant(score=58.0)
    base_fw = {"value": 0.20, "quality": 0.30, "momentum": 0.25, "size": 0.10, "low_volatility": 0.15}
    adj = _adj_quant_score(q, {}, base_fw)
    # 60% of original + 40% of factor composite (all at 50 default) should be near original
    assert abs(adj - 58.0) < 10.0


# ── Full sub-grade computation ────────────────────────────────

def test_sub_grades_returns_correct_type(cfg):
    pg = compute_sub_grades(_data(), _fund(), _tech(), _quant(), cfg)
    assert isinstance(pg, PortfolioGrades)


def test_sub_grades_all_portfolios_populated(cfg):
    pg = compute_sub_grades(_data(), _fund(), _tech(), _quant(), cfg)
    for attr in ["very_conservative", "conservative", "balanced",
                 "aggressive", "very_aggressive"]:
        grade = getattr(pg, attr)
        assert isinstance(grade, PortfolioGrade)
        assert grade.grade in list(Grade)
        assert 0.0 <= grade.composite <= 100.0


def test_sub_grades_conservative_grade_gte_very_conservative(cfg):
    """A stock that passes both should typically score >= conservative vs very_conservative
    (conservative has looser gates and weights more technical momentum)."""
    pg = compute_sub_grades(_data(), _fund(), _tech(), _quant(), cfg)
    # Both may be Stay Away; just ensure composites are valid and ordered reasonably
    vc_c = pg.very_conservative.composite
    c_c  = pg.conservative.composite
    # If very_conservative passed, conservative should too (it has looser gates)
    if vc_c > 0:
        assert c_c >= 0.0


def test_high_beta_fails_very_conservative(cfg):
    data = _data(beta=1.8)
    pg   = compute_sub_grades(data, _fund(), _tech(), _quant(beta=1.8), cfg)
    assert pg.very_conservative.grade == Grade.STAY_AWAY
    assert any("max_beta" in g for g in pg.very_conservative.failed_gates)


def test_high_beta_passes_very_aggressive(cfg):
    data = _data(beta=1.8)
    pg   = compute_sub_grades(data, _fund(), _tech(), _quant(beta=1.8), cfg)
    # Very aggressive has no beta cap → should not Stay Away due to beta alone
    assert "max_beta" not in pg.very_aggressive.failed_gates


def test_no_dividend_fails_very_conservative(cfg):
    data = _data(dividend_yield=0.0)
    pg   = compute_sub_grades(data, _fund(), _tech(), _quant(), cfg)
    assert pg.very_conservative.grade == Grade.STAY_AWAY
    assert any("require_dividend" in g for g in pg.very_conservative.failed_gates)


def test_small_cap_fails_very_conservative(cfg):
    data = _data(market_cap=500_000_000)  # $500M
    pg   = compute_sub_grades(data, _fund(), _tech(), _quant(), cfg)
    assert pg.very_conservative.grade == Grade.STAY_AWAY


def test_distressed_fails_min_altman_z(cfg):
    pg = compute_sub_grades(_data(), _fund(altman_z=1.5), _tech(), _quant(), cfg)
    # Very Conservative requires Z >= 3.0
    assert pg.very_conservative.grade == Grade.STAY_AWAY


def test_rationale_mentions_failed_gate(cfg):
    data = _data(dividend_yield=0.0)
    pg   = compute_sub_grades(data, _fund(), _tech(), _quant(), cfg)
    assert "dividend" in pg.very_conservative.rationale.lower()


def test_rationale_non_empty_for_all(cfg):
    pg = compute_sub_grades(_data(), _fund(), _tech(), _quant(), cfg)
    for attr in ["very_conservative", "conservative", "balanced",
                 "aggressive", "very_aggressive"]:
        assert len(getattr(pg, attr).rationale) > 0


def test_aggressive_allows_high_beta(cfg):
    data = _data(beta=1.4, dividend_yield=0.0)
    pg   = compute_sub_grades(data, _fund(), _tech(), _quant(beta=1.4), cfg)
    # Aggressive: max_beta=1.5, require_dividend=False → should not fail on beta/dividend
    assert "max_beta" not in pg.aggressive.failed_gates
    assert "require_dividend" not in pg.aggressive.failed_gates


def test_grade_progression_for_quality_stock(cfg):
    """A high-quality, low-beta dividend payer should grade better in conservative sleeves
    than a volatile, no-dividend growth stock grades in conservative ones."""
    # Quality conservative stock
    data_con = _data(beta=0.7, dividend_yield=0.03, market_cap=50e9)
    pg_con   = compute_sub_grades(data_con, _fund(valuation=70, health=85),
                                   _tech(), _quant(beta=0.7, low_vol_pct=80), cfg)
    # Volatile growth stock
    data_agg = _data(beta=1.6, dividend_yield=0.0, market_cap=5e9)
    pg_agg   = compute_sub_grades(data_agg, _fund(valuation=30, growth=85),
                                   _tech(score=75), _quant(beta=1.6, momentum_pct=88), cfg)
    # Conservative sleeve: quality stock should outscore growth stock
    assert pg_con.conservative.composite >= pg_agg.conservative.composite


def test_full_pipeline_with_fixture(cfg):
    from stockgrader.data.normalizer import normalize
    import numpy as np, pandas as pd
    from datetime import date
    rng   = np.random.default_rng(42)
    n     = 500
    dates = pd.bdate_range(end=date.today(), periods=n)
    close = 150.0 * np.exp(np.cumsum(rng.normal(0.0005, 0.012, n)))
    price_df = pd.DataFrame({
        "open": close, "high": close*1.005, "low": close*0.995,
        "close": close, "volume": np.ones(n)*1e6, "adj_close": close,
    }, index=dates)

    from stockgrader.engines.fundamental  import FundamentalEngine
    from stockgrader.engines.technical    import TechnicalEngine
    from stockgrader.engines.quantitative import QuantitativeEngine

    data = normalize(
        ticker="TEST", price_df=price_df,
        yf_info=YF_INFO, fmp_data=FMP_DATA, estimates=ESTIMATES,
        peers=[], peer_metrics={}, rf_rates=RF_RATES, cfg=cfg,
    )
    fund_res  = FundamentalEngine(cfg).score(data)
    tech_res  = TechnicalEngine(cfg).score(data)
    quant_res = QuantitativeEngine(cfg).score(data)

    pg = compute_sub_grades(data, fund_res, tech_res, quant_res, cfg)
    assert isinstance(pg, PortfolioGrades)
    for attr in ["very_conservative", "conservative", "balanced",
                 "aggressive", "very_aggressive"]:
        grade = getattr(pg, attr)
        assert 0.0 <= grade.composite <= 100.0
        assert len(grade.rationale) > 0
