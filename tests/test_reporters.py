"""
Unit tests for all three reporters.
Builds a minimal AnalysisResult from fixtures — no live API calls.
"""
import json
import uuid
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path

import pytest

from stockgrader.config import get_config
from stockgrader.models import (
    AnalysisResult, OverallGrade, EngineResults,
    FundamentalResult, FundamentalPillars,
    TechnicalResult,   TechnicalPillars,
    QuantitativeResult, FactorScores, RiskMetrics,
    PortfolioGrades, PortfolioGrade, GateResult,
    PriceLadder, FairValueSensitivity, SensitivityGrid,
    DataQuality, Grade,
)
from stockgrader.reporting import JSONReporter, MarkdownReporter, TerminalReporter


# ── Sample AnalysisResult factory ────────────────────────────

def _make_analysis_result(ticker="AAPL", grade=Grade.BUY, composite=67.0) -> AnalysisResult:
    pg = lambda g, c, failed=None: PortfolioGrade(
        grade=g, composite=c,
        gate_results=[GateResult(gate=f, passed=False) for f in (failed or [])],
        rationale=f"Test rationale for {g.value}",
    )
    return AnalysisResult(
        ticker   = ticker,
        as_of    = datetime(2026, 5, 20, 14, 30, tzinfo=timezone.utc),
        price    = 182.74,
        run_id   = uuid.uuid4().hex[:8],
        config_hash = "abc123",
        overall  = OverallGrade(
            grade              = grade,
            composite          = composite,
            confidence         = 0.81,
            drivers_positive   = ["ROIC +9ppt above WACC", "Profitability 78/100", "Quality factor 72nd pct"],
            drivers_negative   = ["Valuation 52/100", "Momentum 4th pct", "Max drawdown -18%"],
            circuit_breakers   = [],
            fundamental_score  = 70.5,
            technical_score    = 60.2,
            quantitative_score = 58.0,
            weights_used       = {"fundamental": 0.50, "technical": 0.30, "quantitative": 0.20},
        ),
        price_ladder = PriceLadder(
            fair_value              = 183.00,
            gotta_have_at           = 128.10,
            buy_at                  = 155.55,
            hold_low                = 155.55,
            hold_high               = 210.45,
            sell_above              = 210.45,
            stay_away_above         = 256.20,
            upside_to_fv_pct        = 0.0014,
            implied_growth_rate     = 0.085,
            fair_value_sensitivity  = FairValueSensitivity(low=150.0, base=183.0, high=215.0),
            sensitivity_grid        = SensitivityGrid(
                cells = {
                    "-0.02/-0.01": 140.0, "-0.02/+0.00": 130.0, "-0.02/+0.01": 122.0,
                    "+0.00/-0.01": 196.0, "+0.00/+0.00": 183.0, "+0.00/+0.01": 171.0,
                    "+0.02/-0.01": 255.0, "+0.02/+0.00": 236.0, "+0.02/+0.01": 220.0,
                },
                growth_deltas        = [-0.02, 0.00, 0.02],
                discount_rate_deltas = [-0.01, 0.00, 0.01],
            ),
        ),
        engines = EngineResults(
            fundamental = FundamentalResult(
                score    = 70.5,
                pillars  = FundamentalPillars(
                    valuation=52, profitability=78, growth=68,
                    financial_health=82, capital_allocation=70,
                ),
                roic_vs_wacc          = 0.09,
                wacc                  = 0.09,
                roic                  = 0.18,
                altman_z              = 4.25,
                piotroski_f           = 7,
                margin_trajectory_3yr = 0.5,
                quality_of_earnings_flag = False,
                peer_count            = 8,
                missing_fields        = [],
            ),
            technical = TechnicalResult(
                score   = 60.2,
                pillars = TechnicalPillars(
                    trend=75, momentum=48, volume_structure=58,
                    details={
                        "trend":    {"pct_above_sma50": 4.2, "pct_above_sma200": 8.1, "adx": 28.5},
                        "momentum": {"rsi": 58.3, "macd_hist": 0.042, "roc_60_pct": 3.2},
                    },
                ),
                regime             = "moderate_uptrend",
                nearest_support    = 170.0,
                nearest_resistance = 195.0,
                week_52_high       = 199.0,
                week_52_low        = 143.0,
                missing_fields     = [],
            ),
            quantitative = QuantitativeResult(
                score   = 58.0,
                factors = FactorScores(
                    value_z=0.82, quality_z=0.60, momentum_z=-1.77, size_z=0.20,
                    low_volatility_z=0.39,
                    value_pct=79.0, quality_pct=72.0, momentum_pct=4.0,
                    size_pct=58.0, low_volatility_pct=65.0,
                ),
                risk_metrics = RiskMetrics(
                    beta_1yr=1.10, beta_3yr=1.08,
                    sharpe_1yr=0.92, sharpe_3yr=0.85,
                    sortino_1yr=1.25,
                    max_drawdown_3yr=-0.182, current_drawdown=-0.05,
                    realized_vol_1yr=0.198,
                ),
                missing_fields = [],
            ),
        ),
        portfolios = PortfolioGrades(
            very_conservative = pg(Grade.STAY_AWAY, 0.0, ["max_beta", "require_dividend"]),
            conservative      = pg(Grade.HOLD, 52.1),
            balanced          = pg(Grade.BUY,  65.8),
            aggressive        = pg(Grade.BUY,  69.3),
            very_aggressive   = pg(Grade.BUY,  71.2),
        ),
        data_quality = DataQuality(
            missing_fields    = ["pe_forward", "quick_ratio"],
            imputed_fields    = [],
            sources           = {"price": "yfinance", "fundamentals": "fmp"},
            data_completeness = 0.82,
            warnings          = [],
        ),
    )


@pytest.fixture
def result():
    return _make_analysis_result()


# ── JSONReporter ──────────────────────────────────────────────

def test_json_render_is_valid_json(result):
    reporter = JSONReporter()
    output   = reporter.render(result)
    parsed   = json.loads(output)   # must not raise
    assert isinstance(parsed, dict)


def test_json_has_required_top_level_keys(result):
    parsed = json.loads(JSONReporter().render(result))
    for key in ["ticker", "as_of", "price", "overall", "price_ladder",
                "engines", "portfolios", "data_quality"]:
        assert key in parsed, f"Missing key: {key}"


def test_json_ticker_matches(result):
    parsed = json.loads(JSONReporter().render(result))
    assert parsed["ticker"] == "AAPL"


def test_json_grade_is_string(result):
    parsed = json.loads(JSONReporter().render(result))
    assert isinstance(parsed["overall"]["grade"], str)


def test_json_price_ladder_has_all_keys(result):
    parsed = json.loads(JSONReporter().render(result))
    pl = parsed["price_ladder"]
    for key in ["gotta_have_at", "buy_at", "hold_range", "sell_above",
                "stay_away_above", "fair_value", "upside_to_fv_pct"]:
        assert key in pl, f"Price ladder missing key: {key}"


def test_json_five_portfolios(result):
    parsed = json.loads(JSONReporter().render(result))
    pg = parsed["portfolios"]
    for name in ["very_conservative", "conservative", "balanced",
                 "aggressive", "very_aggressive"]:
        assert name in pg


def test_json_save_writes_file(result, tmp_path):
    path = JSONReporter().save(result, runs_dir=tmp_path)
    assert path.exists()
    assert path.suffix == ".json"
    content = json.loads(path.read_text())
    assert content["ticker"] == "AAPL"


def test_json_save_custom_filename(result, tmp_path):
    path = JSONReporter().save(result, runs_dir=tmp_path, filename="custom.json")
    assert path.name == "custom.json"


# ── MarkdownReporter ──────────────────────────────────────────

def test_md_render_is_string(result):
    output = MarkdownReporter().render(result)
    assert isinstance(output, str) and len(output) > 100


def test_md_has_ticker_in_header(result):
    output = MarkdownReporter().render(result)
    assert "AAPL" in output


def test_md_has_grade_in_header(result):
    output = MarkdownReporter().render(result)
    assert "BUY" in output.upper()


def test_md_has_price_ladder_section(result):
    output = MarkdownReporter().render(result)
    assert "Price Ladder" in output


def test_md_has_fair_value(result):
    output = MarkdownReporter().render(result)
    assert "Fair Value" in output or "$183" in output


def test_md_has_sensitivity_grid(result):
    output = MarkdownReporter().render(result)
    assert "Sensitivity" in output


def test_md_has_drivers_section(result):
    output = MarkdownReporter().render(result)
    assert "Drivers" in output or "Positive" in output


def test_md_has_engine_breakdown(result):
    output = MarkdownReporter().render(result)
    assert "Engine Breakdown" in output or "Fundamental" in output


def test_md_has_portfolio_table(result):
    output = MarkdownReporter().render(result)
    assert "Portfolio Sub-Grades" in output or "Very Conservative" in output


def test_md_shows_failed_gates(result):
    output = MarkdownReporter().render(result)
    assert "max_beta" in output or "require_dividend" in output


def test_md_has_data_quality(result):
    output = MarkdownReporter().render(result)
    assert "Data Quality" in output or "Completeness" in output


def test_md_has_roic_info(result):
    output = MarkdownReporter().render(result)
    assert "ROIC" in output


def test_md_save_writes_file(result, tmp_path):
    path = MarkdownReporter().save(result, runs_dir=tmp_path)
    assert path.exists()
    assert path.suffix == ".md"
    assert len(path.read_text(encoding="utf-8")) > 200


def test_md_no_price_ladder_graceful():
    r = _make_analysis_result()
    r.price_ladder = None   # type: ignore[assignment]
    output = MarkdownReporter().render(r)
    assert "unavailable" in output or "Price Ladder" in output


def test_md_circuit_breaker_shown():
    r = _make_analysis_result(grade=Grade.SELL, composite=35.0)
    r.overall.circuit_breakers = ["Altman Z 1.2 < 1.8 → capped at Sell"]
    output = MarkdownReporter().render(r)
    assert "Circuit Breakers" in output or "Altman" in output


def test_md_sensitivity_grid_correct_size(result):
    output = MarkdownReporter().render(result)
    # 3 growth rows + 2 header rows = at least 5 rows in the grid section
    grid_lines = [l for l in output.split("\n") if "| G " in l]
    assert len(grid_lines) == 3


# ── TerminalReporter ──────────────────────────────────────────

def test_terminal_render_plain_no_crash(result):
    reporter = TerminalReporter()
    text = reporter.render_plain(result)
    assert isinstance(text, str) and len(text) > 20


def test_terminal_compact_line_no_crash(result):
    reporter = TerminalReporter()
    text = reporter.compact_line(result)
    assert "AAPL" in text
    assert "BUY" in text.upper() or "67" in text


def test_terminal_compact_contains_price(result):
    text = TerminalReporter().compact_line(result)
    assert "182" in text


def test_terminal_compact_contains_zone(result):
    text = TerminalReporter().compact_line(result)
    assert "zone" in text.lower()


def test_terminal_render_plain_contains_all_portfolios(result):
    text = TerminalReporter().render_plain(result)
    assert "VC:" in text
    assert "VA:" in text


def test_terminal_print_compact_with_mock_console(result):
    """Verify print_compact doesn't crash when rich is available."""
    from rich.console import Console
    buf = StringIO()
    console = Console(file=buf, color_system=None)
    reporter = TerminalReporter(console=console)
    reporter.print_compact(result)   # should not raise
    output = buf.getvalue()
    assert "AAPL" in output


def test_terminal_print_full_with_mock_console(result):
    from rich.console import Console
    buf = StringIO()
    console = Console(file=buf, color_system=None)
    reporter = TerminalReporter(console=console)
    reporter.print_full(result)
    output = buf.getvalue()
    assert "AAPL" in output
    assert "BUY" in output.upper() or "67" in output


def test_terminal_stay_away_shown_red(result):
    r = _make_analysis_result(grade=Grade.STAY_AWAY, composite=10.0)
    from rich.console import Console
    buf = StringIO()
    console = Console(file=buf, color_system=None)
    TerminalReporter(console=console).print_compact(r)
    output = buf.getvalue()
    assert "SA" in output or "Stay Away" in output


# ── Reporter consistency ──────────────────────────────────────

def test_json_and_md_agree_on_composite(result):
    parsed  = json.loads(JSONReporter().render(result))
    md_text = MarkdownReporter().render(result)
    assert str(round(result.overall.composite, 1)) in md_text
    assert parsed["overall"]["composite"] == pytest.approx(result.overall.composite, abs=0.1)


def test_all_reporters_handle_stay_away():
    r = _make_analysis_result(grade=Grade.STAY_AWAY, composite=12.0)
    r.overall.circuit_breakers = ["Altman Z 1.1 < 1.8 → capped at Sell"]
    # None should raise
    json_out = JSONReporter().render(r)
    md_out   = MarkdownReporter().render(r)
    plain    = TerminalReporter().render_plain(r)
    assert json.loads(json_out)["overall"]["grade"] == "Stay Away"
    assert "Stay Away" in md_out or "STAY" in md_out.upper()
    assert "SA" in plain or "Stay Away" in plain
