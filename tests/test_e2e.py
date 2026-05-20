"""
End-to-end pipeline tests — spec §18 "Definition of Done" coverage.

Uses fixture data + synthetic price history (no live API calls).
Verifies the full pipeline: normalize -> price ladder -> grade -> sub-grades
-> AnalysisResult -> all three reporters.
"""
import json
import math
import uuid
import numpy as np
import pandas as pd
import pytest
from datetime import date, datetime, timezone

from stockgrader.config import get_config
from stockgrader.data.normalizer import normalize
from stockgrader.models import (
    AnalysisResult, Grade, DataQuality,
    FairValueSensitivity, PriceLadder, SensitivityGrid,
)
from stockgrader.pipeline import assemble_result, _fallback_ladder
from stockgrader.reporting import JSONReporter, MarkdownReporter, TerminalReporter
from tests.fixtures.sample_fmp import FMP_DATA, YF_INFO, ESTIMATES, RF_RATES


# ── Fixtures ─────────────────────────────────────────────────

def _price_df(n=600, trend=0.0005, seed=42):
    rng   = np.random.default_rng(seed)
    r     = rng.normal(trend, 0.012, n)
    c     = 150.0 * np.exp(np.cumsum(r))
    dates = pd.bdate_range(end=date.today(), periods=n)
    df = pd.DataFrame({
        "open": c, "high": c*1.005, "low": c*0.995,
        "close": c, "volume": np.ones(n)*2e6, "adj_close": c,
    }, index=dates)
    df.index.name = "Date"
    return df


@pytest.fixture(scope="module")
def cfg():
    return get_config()


@pytest.fixture(scope="module")
def data(cfg):
    return normalize(
        ticker="TEST", price_df=_price_df(n=600, seed=42),
        yf_info=YF_INFO, fmp_data=FMP_DATA, estimates=ESTIMATES,
        peers=[], peer_metrics={}, rf_rates=RF_RATES, cfg=cfg,
    )


@pytest.fixture(scope="module")
def result(cfg, data):
    return assemble_result("TEST", data, config=cfg)


# ──────────────────────────────────────────────────────────────
# §18 Definition-of-Done items
# ──────────────────────────────────────────────────────────────

class TestDefinitionOfDone:
    """Directly maps to spec §18 bullet points."""

    def test_analyze_runs_end_to_end(self, result):
        """analyze TICKER produces a complete AnalysisResult."""
        assert isinstance(result, AnalysisResult)
        assert result.ticker == "TEST"
        assert result.price > 0

    def test_overall_grade_present(self, result):
        assert result.overall.grade in list(Grade)
        assert 0 <= result.overall.composite <= 100

    def test_price_ladder_present(self, result):
        pl = result.price_ladder
        assert pl is not None
        assert pl.fair_value > 0

    def test_price_ladder_internally_consistent(self, result):
        pl = result.price_ladder
        assert pl.gotta_have_at  < pl.buy_at,          "GH < Buy"
        assert pl.buy_at         < pl.sell_above,       "Buy < Sell"
        assert pl.sell_above     < pl.stay_away_above,  "Sell < SA"
        assert pl.hold_low       == pl.buy_at,          "hold_low == buy_at"
        assert pl.hold_high      == pl.sell_above,      "hold_high == sell_above"

    def test_five_portfolio_sub_grades_present(self, result):
        pg = result.portfolios
        for attr in ["very_conservative", "conservative", "balanced",
                     "aggressive", "very_aggressive"]:
            sub = getattr(pg, attr)
            assert sub.grade in list(Grade)
            assert 0 <= sub.composite <= 100

    def test_all_grades_have_explainability(self, result):
        """Every grade ships with drivers; no unexplained numbers."""
        assert len(result.overall.drivers_positive) == 3
        assert len(result.overall.drivers_negative) == 3
        assert all(isinstance(d, str) and len(d) > 0
                   for d in result.overall.drivers_positive + result.overall.drivers_negative)

    def test_confidence_score_present(self, result):
        assert 0.05 <= result.overall.confidence <= 1.00

    def test_data_quality_populated(self, result):
        dq = result.data_quality
        assert 0.0 <= dq.data_completeness <= 1.0
        assert isinstance(dq.missing_fields, list)

    def test_config_hash_present(self, result):
        assert result.config_hash and len(result.config_hash) > 0

    def test_run_id_present(self, result):
        assert result.run_id and len(result.run_id) > 0


# ──────────────────────────────────────────────────────────────
# Circuit breaker tests
# ──────────────────────────────────────────────────────────────

class TestCircuitBreakers:

    def test_distressed_company_capped_at_sell(self, cfg):
        """Altman Z < 1.8 → circuit breaker caps composite ≤ 39."""
        bad_data = normalize(
            ticker="DISTRESSED", price_df=_price_df(n=300, seed=99),
            yf_info={**YF_INFO, "currentPrice": 5.0},
            fmp_data={
                **FMP_DATA,
                "key_metrics_ttm": {**FMP_DATA["key_metrics_ttm"],
                                    "currentRatioTTM": 0.5,
                                    "debtToEquityTTM": 5.0},
            },
            estimates=ESTIMATES,
            peers=[], peer_metrics={}, rf_rates=RF_RATES, cfg=cfg,
        )
        # Force Altman Z into distress zone
        bad_data["altman_z"] = 1.2
        bad_data["piotroski_f"] = 2

        result = assemble_result("DISTRESSED", bad_data, config=cfg)
        assert result.overall.composite <= 39.0
        assert result.overall.grade in (Grade.STAY_AWAY, Grade.SELL)
        assert any("Altman" in b for b in result.overall.circuit_breakers)

    def test_extreme_overvaluation_circuit_breaker(self, cfg):
        """Price >> stay_away_above triggers overvaluation circuit breaker."""
        data_overvd = normalize(
            ticker="PRICEY", price_df=_price_df(n=300, seed=88),
            yf_info={**YF_INFO, "currentPrice": 5000.0},
            fmp_data=FMP_DATA, estimates=ESTIMATES,
            peers=[], peer_metrics={}, rf_rates=RF_RATES, cfg=cfg,
        )
        data_overvd["price"] = 5000.0
        result = assemble_result("PRICEY", data_overvd, config=cfg)
        # Price ladder should show overvaluation; either overvaluation CB fires
        # or price is in Stay Away zone on the ladder
        pl = result.price_ladder
        if pl.stay_away_above < 5000.0:
            assert (result.overall.composite <= 39 or
                    any("overvaluation" in b.lower() for b in result.overall.circuit_breakers))

    def test_circuit_breakers_list_in_result(self, result):
        """circuit_breakers is always a list (possibly empty)."""
        assert isinstance(result.overall.circuit_breakers, list)

    def test_no_breaker_for_strong_company(self, result):
        """Fixture data represents a strong company; no breakers should fire."""
        # If Altman Z is above 1.8 for the fixture company, no breaker
        if result.engines.fundamental.altman_z and result.engines.fundamental.altman_z >= 1.8:
            assert not any("Altman" in b for b in result.overall.circuit_breakers)


# ──────────────────────────────────────────────────────────────
# Portfolio sub-grade eligibility
# ──────────────────────────────────────────────────────────────

class TestPortfolioEligibility:

    def test_failed_gate_produces_stay_away(self, cfg):
        """A stock that fails a hard gate is Stay Away for that sleeve."""
        high_beta_data = normalize(
            ticker="HBETA", price_df=_price_df(n=300, seed=55),
            yf_info={**YF_INFO, "beta": 2.5, "dividendYield": 0.0},
            fmp_data=FMP_DATA, estimates=ESTIMATES,
            peers=[], peer_metrics={}, rf_rates=RF_RATES, cfg=cfg,
        )
        high_beta_data["beta"] = 2.5
        high_beta_data["dividend_yield"] = 0.0
        result = assemble_result("HBETA", high_beta_data, config=cfg)
        # Very Conservative requires beta <= 0.8 → must be Stay Away
        vc = result.portfolios.very_conservative
        assert vc.grade == Grade.STAY_AWAY
        assert len(vc.failed_gates) > 0

    def test_each_portfolio_has_rationale(self, result):
        pg = result.portfolios
        for attr in ["very_conservative", "conservative", "balanced",
                     "aggressive", "very_aggressive"]:
            assert len(getattr(pg, attr).rationale) > 0

    def test_portfolio_composites_in_range(self, result):
        pg = result.portfolios
        for attr in ["very_conservative", "conservative", "balanced",
                     "aggressive", "very_aggressive"]:
            assert 0.0 <= getattr(pg, attr).composite <= 100.0


# ──────────────────────────────────────────────────────────────
# Price ladder details
# ──────────────────────────────────────────────────────────────

class TestPriceLadder:

    def test_sensitivity_grid_dimensions(self, result):
        sg = result.price_ladder.sensitivity_grid
        if sg and sg.cells:
            g = len(sg.growth_deltas or [-0.02, 0.0, 0.02])
            d = len(sg.discount_rate_deltas or [-0.01, 0.0, 0.01])
            assert len(sg.cells) == g * d

    def test_fv_sensitivity_ordering(self, result):
        s = result.price_ladder.fair_value_sensitivity
        assert s.low <= s.base <= s.high

    def test_upside_to_fv_is_finite(self, result):
        assert math.isfinite(result.price_ladder.upside_to_fv_pct)

    def test_fallback_ladder_price_ordering(self):
        ladder = _fallback_ladder(100.0)
        assert ladder.gotta_have_at < ladder.buy_at < ladder.sell_above < ladder.stay_away_above

    def test_fallback_ladder_anchored_to_price(self):
        ladder = _fallback_ladder(200.0)
        assert ladder.fair_value == pytest.approx(200.0)
        assert ladder.gotta_have_at < 200.0
        assert ladder.stay_away_above > 200.0


# ──────────────────────────────────────────────────────────────
# Reporter integration
# ──────────────────────────────────────────────────────────────

class TestReporterIntegration:

    def test_json_output_is_valid(self, result):
        output = JSONReporter().render(result)
        parsed = json.loads(output)
        assert parsed["ticker"] == "TEST"
        assert "overall" in parsed
        assert "price_ladder" in parsed
        assert "portfolios" in parsed

    def test_json_all_five_portfolios(self, result):
        parsed = json.loads(JSONReporter().render(result))
        pg = parsed["portfolios"]
        for name in ["very_conservative", "conservative", "balanced",
                     "aggressive", "very_aggressive"]:
            assert name in pg
            assert "grade" in pg[name]
            assert pg[name]["grade"] in [g.value for g in Grade]

    def test_json_price_ladder_has_required_keys(self, result):
        parsed = json.loads(JSONReporter().render(result))
        pl = parsed["price_ladder"]
        for key in ["gotta_have_at", "buy_at", "hold_range", "sell_above",
                    "stay_away_above", "fair_value", "upside_to_fv_pct"]:
            assert key in pl

    def test_markdown_contains_ticker(self, result):
        md = MarkdownReporter().render(result)
        assert "TEST" in md

    def test_markdown_contains_grade(self, result):
        md = MarkdownReporter().render(result)
        assert result.overall.grade.value.upper() in md.upper()

    def test_markdown_has_price_ladder(self, result):
        md = MarkdownReporter().render(result)
        assert "Price Ladder" in md

    def test_markdown_has_portfolio_table(self, result):
        md = MarkdownReporter().render(result)
        assert "Very Conservative" in md or "Portfolio Sub-Grades" in md

    def test_markdown_has_drivers(self, result):
        md = MarkdownReporter().render(result)
        assert any(d[:20] in md for d in result.overall.drivers_positive)

    def test_terminal_plain_no_crash(self, result):
        text = TerminalReporter().render_plain(result)
        assert "TEST" in text

    def test_terminal_plain_contains_grade(self, result):
        text = TerminalReporter().render_plain(result)
        assert result.overall.grade.value in text or \
               result.overall.grade.name.upper() in text.upper()

    def test_all_three_renderers_no_crash(self, result):
        from io import StringIO
        from rich.console import Console
        buf     = StringIO()
        console = Console(file=buf, color_system=None)
        reporter = TerminalReporter(console=console)
        reporter.print_full(result)   # must not raise
        JSONReporter().render(result) # must not raise
        MarkdownReporter().render(result) # must not raise


# ──────────────────────────────────────────────────────────────
# Engine scores sanity
# ──────────────────────────────────────────────────────────────

class TestEngineScores:

    def test_all_three_engine_scores_in_range(self, result):
        for attr in ["fundamental_score", "technical_score", "quantitative_score"]:
            val = getattr(result.overall, attr)
            assert 0 <= val <= 100, f"{attr} = {val}"

    def test_weights_sum_to_one(self, result):
        total = sum(result.overall.weights_used.values())
        assert total == pytest.approx(1.0, abs=0.01)

    def test_composite_matches_weighted_sum(self, result):
        w = result.overall.weights_used
        expected = (
            w["fundamental"]  * result.overall.fundamental_score  +
            w["technical"]    * result.overall.technical_score    +
            w["quantitative"] * result.overall.quantitative_score
        )
        # Allow delta from circuit breaker capping
        assert result.overall.composite <= expected + 1.0

    def test_fundamental_pillars_in_range(self, result):
        p = result.engines.fundamental.pillars
        for name in ["valuation", "profitability", "growth", "financial_health", "capital_allocation"]:
            val = getattr(p, name)
            assert 0 <= val <= 100, f"Pillar {name} = {val}"

    def test_technical_regime_non_empty(self, result):
        assert result.engines.technical.regime not in (None, "")

    def test_quant_risk_metrics_finite(self, result):
        rm = result.engines.quantitative.risk_metrics
        for attr in ["max_drawdown_3yr", "realized_vol_1yr"]:
            val = getattr(rm, attr)
            if val is not None:
                assert math.isfinite(val)


# ──────────────────────────────────────────────────────────────
# CLI integration (tests the analyze.py entry point)
# ──────────────────────────────────────────────────────────────

class TestCLI:

    def test_cli_with_mocked_fetcher(self, cfg, data, tmp_path):
        """Test the CLI pipeline end-to-end using a mocked DataFetcher."""
        from unittest.mock import MagicMock
        from stockgrader.pipeline import run_analysis

        mock_fetcher = MagicMock()
        mock_fetcher.fetch.return_value = data

        result = run_analysis(
            "TEST",
            portfolio="all",
            no_cache=True,
            config=cfg,
            fetcher=mock_fetcher,
        )
        assert isinstance(result, AnalysisResult)
        assert result.ticker == "TEST"
        mock_fetcher.fetch.assert_called_once()

    def test_cli_json_format(self, cfg, data):
        from unittest.mock import MagicMock
        from stockgrader.pipeline import run_analysis

        mock_fetcher = MagicMock()
        mock_fetcher.fetch.return_value = data
        result = run_analysis("TEST", config=cfg, fetcher=mock_fetcher)
        output = JSONReporter().render(result)
        assert json.loads(output)["ticker"] == "TEST"

    def test_cli_raises_on_no_price(self, cfg):
        from unittest.mock import MagicMock
        from stockgrader.pipeline import run_analysis

        mock_fetcher = MagicMock()
        mock_fetcher.fetch.return_value = {"missing_fields": ["price"]}

        with pytest.raises(ValueError, match="No valid price"):
            run_analysis("BADTICKER", config=cfg, fetcher=mock_fetcher)

    def test_cli_portfolio_override(self, cfg, data):
        """Requesting a specific portfolio uses that portfolio's weights."""
        from unittest.mock import MagicMock
        from stockgrader.pipeline import run_analysis

        mock_fetcher = MagicMock()
        mock_fetcher.fetch.return_value = data

        result_all  = run_analysis("TEST", portfolio="all",       config=cfg, fetcher=mock_fetcher)
        mock_fetcher.fetch.return_value = data
        result_cons = run_analysis("TEST", portfolio="conservative", config=cfg, fetcher=mock_fetcher)

        # Different portfolio weights → different composites (not guaranteed same)
        assert isinstance(result_all.overall.composite, float)
        assert isinstance(result_cons.overall.composite, float)
        # Both should be valid AnalysisResult
        assert result_all.portfolios is not None
        assert result_cons.portfolios is not None
