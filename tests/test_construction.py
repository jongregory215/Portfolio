"""
Unit tests for the portfolio construction funnel.
Tests each stage independently with synthetic data — no live API calls.
"""
import math
import numpy as np
import pandas as pd
import pytest
from datetime import date, timedelta

from stockgrader.config import get_config
from stockgrader.portfolios.construction import (
    screen_universe, filter_and_rank, build_portfolio, build_bond_sleeve,
    _build_returns_matrix, _compute_cov, _compute_expected_returns,
    _optimize_weights, _mandate_check, _bond_mandate_score,
    PortfolioResult, PortfolioAnalytics, _ScoredCandidate,
)
from stockgrader.models import Grade, PortfolioGrade, GateResult


@pytest.fixture
def cfg():
    return get_config()


# ── Helpers ───────────────────────────────────────────────────

def _make_pg(composite=65.0, grade=Grade.BUY, failed_gates=None):
    failed_gates = failed_gates or []
    gate_results = [GateResult(gate=g, passed=False) for g in failed_gates]
    return PortfolioGrade(grade=grade, composite=composite, gate_results=gate_results)


def _make_candidate(ticker="AAPL", composite=65.0, grade=Grade.BUY,
                    beta=1.0, sector="Technology"):
    data = {
        "ticker":        ticker,
        "beta":          beta,
        "sector":        sector,
        "dividend_yield":0.01,
        "debt_equity":   0.5,
        "revenue_cagr_3yr": 0.08,
    }
    return _ScoredCandidate(
        ticker=ticker, composite=composite, grade=grade,
        data=data, pg=_make_pg(composite, grade),
    )


def _make_price_df(n=300, trend=0.0005, vol=0.012, start=100.0, seed=None):
    rng   = np.random.default_rng(seed)
    rets  = rng.normal(trend, vol, n)
    close = start * np.exp(np.cumsum(rets))
    dates = pd.bdate_range(end=date.today(), periods=n)
    df = pd.DataFrame({
        "open": close, "high": close*1.005, "low": close*0.995,
        "close": close, "volume": np.ones(n)*1e6, "adj_close": close,
    }, index=dates)
    df.index.name = "Date"
    return df


def _make_candidates_with_prices(n_tickers=10, n_bars=300, seed=42):
    """Generate N candidates each with synthetic price history."""
    rng = np.random.default_rng(seed)
    candidates = []
    sectors = ["Technology", "Healthcare", "Financials", "Consumer", "Energy"]
    for i in range(n_tickers):
        ticker = f"T{i:02d}"
        df     = _make_price_df(n_bars, trend=rng.uniform(0, 0.001),
                                 vol=rng.uniform(0.008, 0.020), seed=seed+i)
        composite = float(rng.uniform(60, 85))
        grade     = Grade.GOTTA_HAVE if composite >= 80 else Grade.BUY
        c = _ScoredCandidate(
            ticker=ticker, composite=composite, grade=grade,
            data={
                "ticker": ticker,
                "price_history": df,
                "beta": rng.uniform(0.5, 1.5),
                "sector": sectors[i % len(sectors)],
                "dividend_yield": rng.uniform(0, 0.04),
                "debt_equity": rng.uniform(0, 1.5),
                "revenue_cagr_3yr": rng.uniform(0.02, 0.15),
            },
            pg=_make_pg(composite, grade),
        )
        candidates.append(c)
    return candidates


# ── Stage 1: screen_universe ──────────────────────────────────

def _universe_item(**kwargs):
    base = {"ticker": "X", "price": 50.0, "avg_volume": 2e6,
            "market_cap": 5e9, "beta": 1.0, "dividend_yield": 0.02,
            "debt_equity": 0.5, "altman_z": 3.5, "sector": "Tech"}
    base.update(kwargs)
    return base


def test_screen_universe_all_pass(cfg):
    universe = [_universe_item(ticker="A"), _universe_item(ticker="B")]
    pcfg = cfg["portfolios"]["balanced"]
    result = screen_universe(universe, pcfg)
    assert "A" in result and "B" in result


def test_screen_universe_low_price_filtered(cfg):
    universe = [_universe_item(ticker="CHEAP", price=2.0)]
    pcfg = cfg["portfolios"]["balanced"]
    liq  = cfg["universe"]["liquidity_floor"]
    result = screen_universe(universe, pcfg, liq)
    assert "CHEAP" not in result


def test_screen_universe_high_beta_filtered_vc(cfg):
    universe = [_universe_item(ticker="HIGHB", beta=1.8)]
    pcfg = cfg["portfolios"]["very_conservative"]
    result = screen_universe(universe, pcfg)
    assert "HIGHB" not in result


def test_screen_universe_high_beta_passes_aggressive(cfg):
    universe = [_universe_item(ticker="HIGHB", beta=1.8)]
    pcfg = cfg["portfolios"]["very_aggressive"]
    result = screen_universe(universe, pcfg)
    assert "HIGHB" in result


def test_screen_universe_no_dividend_filtered_vc(cfg):
    universe = [_universe_item(ticker="NODIV", dividend_yield=0.0)]
    pcfg = cfg["portfolios"]["very_conservative"]
    result = screen_universe(universe, pcfg)
    assert "NODIV" not in result


def test_screen_universe_small_cap_filtered_vc(cfg):
    universe = [_universe_item(ticker="SMALL", market_cap=500e6)]
    pcfg = cfg["portfolios"]["very_conservative"]
    result = screen_universe(universe, pcfg)
    assert "SMALL" not in result


def test_screen_universe_empty_universe(cfg):
    assert screen_universe([], cfg["portfolios"]["balanced"]) == []


def test_screen_universe_no_ticker_skipped(cfg):
    universe = [{"price": 50.0, "avg_volume": 2e6, "market_cap": 5e9}]
    result = screen_universe(universe, cfg["portfolios"]["balanced"])
    assert result == []


# ── Stage 2: filter_and_rank ──────────────────────────────────

def test_filter_keeps_buy_gotta_have():
    candidates = [
        _make_candidate("A", 82.0, Grade.GOTTA_HAVE),
        _make_candidate("B", 65.0, Grade.BUY),
        _make_candidate("C", 45.0, Grade.HOLD),
        _make_candidate("D", 25.0, Grade.SELL),
    ]
    result = filter_and_rank(candidates)
    tickers = [c.ticker for c in result]
    assert "A" in tickers and "B" in tickers
    assert "C" not in tickers and "D" not in tickers


def test_filter_ranks_by_composite_desc():
    candidates = [
        _make_candidate("LOW",  61.0, Grade.BUY),
        _make_candidate("HIGH", 79.0, Grade.BUY),
        _make_candidate("MID",  70.0, Grade.BUY),
    ]
    result = filter_and_rank(candidates)
    assert result[0].ticker == "HIGH"
    assert result[-1].ticker == "LOW"


def test_filter_caps_at_max_positions():
    # Keep composites in [60, 79] (Buy band) to satisfy pydantic le=100 constraint
    candidates = [_make_candidate(f"T{i}", 60.0 + i * 0.38, Grade.BUY) for i in range(50)]
    result = filter_and_rank(candidates, max_positions=10)
    assert len(result) <= 10


def test_filter_empty_returns_empty():
    assert filter_and_rank([]) == []


# ── Stage 3: Optimization ─────────────────────────────────────

@pytest.fixture
def synth_returns():
    rng = np.random.default_rng(42)
    n_assets, n_obs = 8, 300
    # Generate correlated returns via Cholesky
    cov_true = np.eye(n_assets) * 0.04 + 0.01
    L        = np.linalg.cholesky(cov_true)
    raw      = rng.standard_normal((n_obs, n_assets))
    data     = (raw @ L.T) / math.sqrt(252)
    tickers  = [f"T{i}" for i in range(n_assets)]
    dates    = pd.bdate_range(end=date.today(), periods=n_obs)
    return pd.DataFrame(data, columns=tickers, index=dates)


def test_compute_cov_shape(synth_returns):
    cov = _compute_cov(synth_returns)
    n = len(synth_returns.columns)
    assert cov.shape == (n, n)


def test_compute_cov_positive_definite(synth_returns):
    cov = _compute_cov(synth_returns)
    eigvals = np.linalg.eigvalsh(cov)
    assert (eigvals > 0).all()


def test_compute_expected_returns_shape(synth_returns):
    mu = _compute_expected_returns(synth_returns)
    assert mu.shape == (len(synth_returns.columns),)


def test_compute_expected_returns_shrinkage_bounds(synth_returns):
    mu0 = _compute_expected_returns(synth_returns, shrinkage=0.0)  # pure sample
    mu1 = _compute_expected_returns(synth_returns, shrinkage=1.0)  # all equal grand mean
    assert np.std(mu1) < np.std(mu0)   # shrinkage reduces dispersion


def test_optimize_weights_sum_to_one(synth_returns):
    cov = _compute_cov(synth_returns)
    mu  = _compute_expected_returns(synth_returns)
    w   = _optimize_weights(mu, cov, "min_variance", 0.05, 0.04, 0.01, 0.10, {}, 0.25)
    assert abs(np.sum(w) - 1.0) < 1e-5


def test_optimize_weights_within_bounds(synth_returns):
    cov = _compute_cov(synth_returns)
    mu  = _compute_expected_returns(synth_returns)
    # pos_max=0.20 is feasible for 8 assets (8 x 0.20 = 1.6 > 1)
    w   = _optimize_weights(mu, cov, "max_sharpe", 0.05, 0.04, 0.01, 0.20, {}, 0.50)
    assert (w >= -1e-6).all()
    assert (w <= 0.20 + 1e-4).all()
    assert abs(np.sum(w) - 1.0) < 1e-4


def test_optimize_min_variance_lower_vol_than_equal(synth_returns):
    cov = _compute_cov(synth_returns)
    mu  = _compute_expected_returns(synth_returns)
    n   = len(mu)
    w_mv  = _optimize_weights(mu, cov, "min_variance", 0.05, 0.04, 0.01, 0.20, {}, 1.0)
    w_eq  = np.ones(n) / n
    vol_mv = math.sqrt(max(w_mv @ cov @ w_mv, 0))
    vol_eq = math.sqrt(max(w_eq @ cov @ w_eq, 0))
    assert vol_mv <= vol_eq + 1e-4   # min-var should have ≤ volatility


def test_optimize_max_sharpe_positive_weights(synth_returns):
    cov = _compute_cov(synth_returns)
    mu  = _compute_expected_returns(synth_returns)
    w   = _optimize_weights(mu, cov, "max_sharpe", 0.05, 0.04, 0.01, 0.15, {}, 0.30)
    assert np.all(w >= -1e-6)
    assert abs(np.sum(w) - 1.0) < 1e-4


# ── Bond sleeve ───────────────────────────────────────────────

def test_bond_sleeve_sums_to_one(cfg):
    bond_cfg = cfg.get("bond_candidates", {})
    for fund_name in ["very_conservative", "conservative", "balanced",
                      "aggressive", "very_aggressive"]:
        weights = build_bond_sleeve(fund_name, bond_cfg)
        if weights:
            assert abs(sum(weights.values()) - 1.0) < 1e-6


def test_bond_sleeve_no_position_exceeds_cap(cfg):
    bond_cfg = cfg.get("bond_candidates", {})
    weights  = build_bond_sleeve("conservative", bond_cfg, max_per_etf=0.30)
    for ticker, w in weights.items():
        assert w <= 0.30 + 1e-6, f"{ticker} weight {w:.3f} exceeds cap"


def test_bond_sleeve_vc_prefers_short_duration(cfg):
    bond_cfg = cfg.get("bond_candidates", {})
    vc_w  = build_bond_sleeve("very_conservative", bond_cfg)
    agg_w = build_bond_sleeve("very_aggressive",   bond_cfg)
    # Short-duration ETFs should have higher combined weight in Very Conservative
    short = ["SHY", "VGSH", "BSV", "VCSH"]
    vc_short  = sum(vc_w.get(t, 0) for t in short)
    agg_short = sum(agg_w.get(t, 0) for t in short)
    assert vc_short >= agg_short


def test_bond_mandate_score_treasury_preferred_vc():
    short_treasury = {"ticker": "SHY",  "duration_yrs": 1.9, "credit": "treasury"}
    long_corp      = {"ticker": "LQD",  "duration_yrs": 8.6, "credit": "ig_corp"}
    assert (_bond_mandate_score(short_treasury, "very_conservative") >
            _bond_mandate_score(long_corp,      "very_conservative"))


def test_bond_mandate_score_corp_preferred_va():
    long_corp  = {"ticker": "LQD", "duration_yrs": 8.6, "credit": "ig_corp"}
    short_tsy  = {"ticker": "SHY", "duration_yrs": 1.9, "credit": "treasury"}
    assert (_bond_mandate_score(long_corp,  "very_aggressive") >
            _bond_mandate_score(short_tsy, "very_aggressive"))


# ── Mandate check ─────────────────────────────────────────────

def test_mandate_check_passes_within_band(cfg):
    analytics = PortfolioAnalytics(0.04, 0.08, 0.025, 0.85)
    pcfg = cfg["portfolios"]["conservative"]   # target [0.03, 0.05]
    mc   = _mandate_check(analytics, pcfg)
    assert mc.passed


def test_mandate_check_fails_below_band(cfg):
    analytics = PortfolioAnalytics(0.01, 0.06, 0.02, 0.70)
    pcfg = cfg["portfolios"]["conservative"]
    mc   = _mandate_check(analytics, pcfg)
    assert not mc.passed
    assert mc.gap < 0


def test_mandate_check_gap_quantified(cfg):
    analytics = PortfolioAnalytics(0.04, 0.08, 0.02, 0.90)
    pcfg = cfg["portfolios"]["balanced"]   # target [0.05, 0.07]
    mc   = _mandate_check(analytics, pcfg)
    assert not mc.passed
    assert abs(mc.gap - (0.04 - 0.05)) < 1e-4


# ── build_portfolio integration ───────────────────────────────

def test_build_portfolio_returns_result(cfg):
    candidates = _make_candidates_with_prices(n_tickers=8, seed=7)
    result = build_portfolio("balanced", [], cfg, pre_scored=candidates)
    assert isinstance(result, PortfolioResult)


def test_build_portfolio_holdings_nonempty(cfg):
    candidates = _make_candidates_with_prices(n_tickers=8, seed=8)
    result = build_portfolio("balanced", [], cfg, pre_scored=candidates)
    assert len(result.holdings) > 0


def test_build_portfolio_equity_bond_split(cfg):
    candidates = _make_candidates_with_prices(n_tickers=8, seed=9)
    result = build_portfolio("conservative", [], cfg, pre_scored=candidates)
    eq_w   = sum(h.weight for h in result.holdings if h.sleeve == "equity")
    bond_w = sum(h.weight for h in result.holdings if h.sleeve == "bond")
    total  = eq_w + bond_w
    assert abs(total - 1.0) < 0.02   # weights sum to 1 (within rounding)


def test_build_portfolio_no_candidates_returns_empty(cfg):
    # All candidates are Hold grade → none survive Stage 2
    hold_candidates = [
        _ScoredCandidate("X", 45.0, Grade.HOLD, {}, _make_pg(45.0, Grade.HOLD))
    ]
    result = build_portfolio("balanced", [], cfg, pre_scored=hold_candidates)
    assert result.holdings == []


def test_build_portfolio_weights_within_position_limits(cfg):
    # Use 15 tickers so pos_max=0.07 is feasible (15 x 0.07 = 1.05 > 1)
    candidates = _make_candidates_with_prices(n_tickers=15, seed=11)
    result     = build_portfolio("balanced", [], cfg, pre_scored=candidates)
    pcfg       = cfg["portfolios"]["balanced"]
    eq_pct     = float(pcfg.get("equity_pct", 0.55))
    max_pos_sleeve = float(pcfg["position_limits"]["max"])
    # Allow +2% tolerance for optimizer fallback cases
    for h in result.holdings:
        if h.sleeve == "equity":
            sleeve_w = h.weight / eq_pct
            assert sleeve_w <= max_pos_sleeve + 0.02, (
                f"{h.ticker}: sleeve weight {sleeve_w:.3f} > {max_pos_sleeve + 0.02:.3f}"
            )


def test_build_portfolio_funnel_stats(cfg):
    candidates = _make_candidates_with_prices(n_tickers=8, seed=12)
    result = build_portfolio("balanced", [{"dummy": True}] * 15, cfg, pre_scored=candidates)
    assert result.funnel.stage1_survivors == 15
    assert result.funnel.stage2_candidates >= 1


def test_build_portfolio_has_analytics(cfg):
    candidates = _make_candidates_with_prices(n_tickers=6, seed=13)
    result = build_portfolio("aggressive", [], cfg, pre_scored=candidates)
    a = result.analytics
    assert math.isfinite(a.projected_return)
    assert a.projected_volatility >= 0
    assert a.effective_n_holdings >= 1


def test_build_returns_matrix_aligns_dates():
    candidates = _make_candidates_with_prices(n_tickers=5, n_bars=200, seed=14)
    ret = _build_returns_matrix(candidates)
    if ret is not None:
        assert ret.shape[1] <= len(candidates)
        assert ret.shape[0] >= 60
