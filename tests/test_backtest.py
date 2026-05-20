"""
Tests for the backtesting & validation suite (Step 12).

All tests use SyntheticPITAdapter (deterministic, true-IC ≈ 0.10)
so they run offline without any API keys.
"""
from __future__ import annotations

import json
from datetime import date

import numpy as np
import pandas as pd
import pytest

from stockgrader.backtest.pit_adapter import (
    BacktestContaminationError,
    SyntheticPITAdapter,
    YFinancePITAdapter,
    require_pit,
)
from stockgrader.backtest.grade_validator import (
    ICStats,
    GradeValidationReport,
    compute_ic,
    compute_ic_series,
    compute_ic_stats,
    compute_grade_bucket_stats,
    check_monotonicity,
    compute_hit_rate,
    compute_per_engine_ic,
    split_oos,
    validate_grades,
)
from stockgrader.backtest.walk_forward import (
    CostModel,
    PeriodResult,
    BacktestResult,
    compute_period_return,
    compute_turnover,
    compute_equity_curve,
    compute_max_drawdown,
    annualize_stats,
    walk_forward_from_scores,
    get_rebalance_dates,
)
from stockgrader.backtest.calibrator import (
    CalibrationResult,
    GridPoint,
    _weight_grid,
    _blend_composite,
    _eval_weights,
    calibrate_weights,
)
from stockgrader.backtest.report import (
    render_validation_report,
    render_backtest_report,
    render_calibration_report,
    to_json,
    save_report,
)


# ──────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def synthetic_adapter():
    return SyntheticPITAdapter(
        n_tickers=20,
        start_date=date(2018, 1, 1),
        end_date=date(2022, 12, 31),
        seed=42,
        true_ic=0.10,
    )


@pytest.fixture(scope="module")
def scores(synthetic_adapter):
    return synthetic_adapter.get_all_scores()


@pytest.fixture(scope="module")
def validation_report(scores):
    return validate_grades(scores, contamination_label="OUT-OF-SAMPLE (PIT)")


@pytest.fixture(scope="module")
def backtest_result(scores):
    records = scores.copy()
    records["period_return"] = records["fwd_12m"]
    return walk_forward_from_scores(
        scored_records      = records,
        portfolio_name      = "test_fund",
        rebalance_freq      = "quarterly",
        cost_model          = CostModel(),
        config              = {},
        contamination_label = "OUT-OF-SAMPLE (PIT)",
    )


@pytest.fixture(scope="module")
def calibration_result(scores):
    return calibrate_weights(
        records   = scores,
        train_end = date(2020, 1, 1),
        val_end   = date(2021, 1, 1),
        contamination_label = "OUT-OF-SAMPLE (PIT)",
        weight_step = 0.20,    # coarser grid → fast tests
        min_weight  = 0.20,
    )


# ──────────────────────────────────────────────────────────────
# PIT Adapter tests
# ──────────────────────────────────────────────────────────────

class TestSyntheticAdapter:
    def test_is_pit_guaranteed(self, synthetic_adapter):
        assert synthetic_adapter.is_pit_guaranteed() is True

    def test_contamination_label_contains_synthetic(self, synthetic_adapter):
        assert "synthetic" in synthetic_adapter.contamination_label().lower()

    def test_get_universe_returns_all_tickers(self, synthetic_adapter):
        u = synthetic_adapter.get_universe(date(2020, 6, 1))
        assert len(u) == 20
        assert all(t.startswith("SYN") for t in u)

    def test_get_all_scores_columns(self, scores):
        expected = {"date", "ticker", "composite", "fund_score",
                    "tech_score", "quant_score", "grade",
                    "fwd_1m", "fwd_3m", "fwd_6m", "fwd_12m"}
        assert expected.issubset(scores.columns)

    def test_scores_grade_distribution(self, scores):
        # should have at least 3 distinct grades
        assert scores["grade"].nunique() >= 3

    def test_scores_composite_range(self, scores):
        assert scores["composite"].between(20, 85).all()

    def test_get_price_history_pit_returns_none_for_unknown(self, synthetic_adapter):
        result = synthetic_adapter.get_price_history_pit("UNKNOWN", date(2020, 1, 1))
        assert result is None


class TestRequirePit:
    def test_pit_adapter_returns_oos_label(self, synthetic_adapter):
        label = require_pit(synthetic_adapter, {})
        assert "OUT-OF-SAMPLE" in label

    def test_non_pit_strict_raises(self):
        adapter = YFinancePITAdapter()
        with pytest.raises(BacktestContaminationError):
            require_pit(adapter, {"backtest": {"contamination_check": "strict"}})

    def test_non_pit_warn_returns_label(self):
        adapter = YFinancePITAdapter()
        label = require_pit(adapter, {"backtest": {"contamination_check": "warn"}})
        assert "CONTAMINATED" in label

    def test_non_pit_off_returns_label(self):
        adapter = YFinancePITAdapter()
        label = require_pit(adapter, {"backtest": {"contamination_check": "off"}})
        assert label  # not empty


# ──────────────────────────────────────────────────────────────
# Grade Validator tests
# ──────────────────────────────────────────────────────────────

class TestComputeIC:
    def test_positive_ic_for_correlated_data(self):
        scores  = pd.Series([10, 20, 30, 40, 50, 60])
        returns = pd.Series([0.01, 0.02, 0.03, 0.04, 0.05, 0.06])
        ic = compute_ic(scores, returns)
        assert ic > 0.9

    def test_negative_ic_for_anticorrelated_data(self):
        scores  = pd.Series([10, 20, 30, 40, 50, 60])
        returns = pd.Series([0.06, 0.05, 0.04, 0.03, 0.02, 0.01])
        ic = compute_ic(scores, returns)
        assert ic < -0.9

    def test_returns_zero_for_few_obs(self):
        scores  = pd.Series([1, 2, 3])
        returns = pd.Series([4, 5, 6])
        assert compute_ic(scores, returns) == 0.0

    def test_handles_nan_values(self):
        scores  = pd.Series([1, 2, float("nan"), 4, 5, 6, 7])
        returns = pd.Series([1, 2, 3, float("nan"), 5, 6, 7])
        ic = compute_ic(scores, returns)
        assert -1.0 <= ic <= 1.0


class TestComputeICSeries:
    def test_returns_series_indexed_by_date(self, scores):
        ic_s = compute_ic_series(scores, return_col="fwd_12m")
        assert isinstance(ic_s, pd.Series)
        assert len(ic_s) > 0

    def test_excludes_dates_with_few_obs(self):
        # Build a 2-ticker panel (below min_obs=5)
        data = pd.DataFrame({
            "date": [date(2020, 1, 1)] * 3,
            "ticker": ["A", "B", "C"],
            "composite": [50, 60, 70],
            "fwd_12m": [0.01, 0.02, 0.03],
        })
        ic_s = compute_ic_series(data, min_obs_per_date=5)
        assert len(ic_s) == 0


class TestICStats:
    def test_ic_stats_information_ratio(self):
        ic_s = pd.Series([0.05, 0.10, 0.08, 0.06, 0.09])
        stats = compute_ic_stats(ic_s, horizon_months=12)
        assert stats.mean > 0
        assert stats.volatility > 0
        assert abs(stats.information_ratio - stats.mean / stats.volatility) < 1e-9

    def test_ic_stats_too_short(self):
        stats = compute_ic_stats(pd.Series([0.05]), horizon_months=12)
        assert stats.mean == 0.0
        assert stats.information_ratio == 0.0


class TestGradeBucketStats:
    def test_returns_all_five_grades(self, scores):
        buckets = compute_grade_bucket_stats(scores, "fwd_12m")
        grades = [b.grade for b in buckets]
        assert set(grades) == {"Stay Away", "Sell", "Hold", "Buy", "Gotta Have"}

    def test_hit_rate_between_zero_and_one(self, scores):
        for b in compute_grade_bucket_stats(scores, "fwd_12m"):
            assert 0.0 <= b.hit_rate <= 1.0

    def test_zero_obs_bucket_all_zeros(self):
        data = pd.DataFrame({"grade": ["Buy"] * 5, "fwd_12m": [0.05] * 5})
        buckets = compute_grade_bucket_stats(data, "fwd_12m")
        stay_away = next(b for b in buckets if b.grade == "Stay Away")
        assert stay_away.n_obs == 0
        assert stay_away.mean_return == 0.0


class TestMonotonicity:
    def test_perfect_monotonicity(self):
        from stockgrader.backtest.grade_validator import GradeBucketStats
        buckets = [
            GradeBucketStats("Stay Away", -0.05, -0.05, 0, 0, 10, 0.3),
            GradeBucketStats("Sell",      -0.02, -0.02, 0, 0, 10, 0.4),
            GradeBucketStats("Hold",       0.05,  0.05, 0, 0, 10, 0.5),
            GradeBucketStats("Buy",        0.10,  0.10, 0, 0, 10, 0.6),
            GradeBucketStats("Gotta Have", 0.15,  0.15, 0, 0, 10, 0.7),
        ]
        holds, note = check_monotonicity(buckets)
        assert holds is True

    def test_violation_detected(self):
        from stockgrader.backtest.grade_validator import GradeBucketStats
        buckets = [
            GradeBucketStats("Stay Away",  0.10,  0.10, 0, 0, 10, 0.3),
            GradeBucketStats("Sell",       0.05,  0.05, 0, 0, 10, 0.4),
            GradeBucketStats("Hold",       0.08,  0.08, 0, 0, 10, 0.5),
            GradeBucketStats("Buy",        0.09,  0.09, 0, 0, 10, 0.6),
            GradeBucketStats("Gotta Have", 0.07,  0.07, 0, 0, 10, 0.7),
        ]
        holds, note = check_monotonicity(buckets)
        assert holds is False
        assert "FAILS" in note


class TestValidateGrades:
    def test_report_type(self, validation_report):
        assert isinstance(validation_report, GradeValidationReport)

    def test_ic_keys_present(self, validation_report):
        assert 12 in validation_report.ic_stats

    def test_grade_returns_has_12m(self, validation_report):
        assert 12 in validation_report.grade_returns

    def test_positive_ic_synthetic(self, validation_report):
        # Synthetic data has true IC ≈ 0.10; mean should be > 0
        assert validation_report.ic_stats[12].mean > 0

    def test_oos_split(self, scores):
        oos_from = date(2021, 1, 1)
        report = validate_grades(scores, oos_from=oos_from)
        assert report.start_date >= oos_from

    def test_empty_after_oos_raises(self, scores):
        with pytest.raises(ValueError):
            validate_grades(scores, oos_from=date(2099, 1, 1))


# ──────────────────────────────────────────────────────────────
# Walk-forward tests
# ──────────────────────────────────────────────────────────────

class TestWalkForwardHelpers:
    def test_get_rebalance_dates_quarterly(self):
        dates = get_rebalance_dates(date(2020, 1, 1), date(2021, 12, 31), "quarterly")
        assert len(dates) >= 4

    def test_get_rebalance_dates_monthly(self):
        dates = get_rebalance_dates(date(2020, 1, 1), date(2020, 12, 31), "monthly")
        assert len(dates) >= 10

    def test_compute_period_return_single_ticker(self):
        df = pd.DataFrame({"ticker": ["A"], "start_price": [100.0], "end_price": [110.0]})
        ret = compute_period_return({"A": 1.0}, df)
        assert abs(ret - 0.10) < 1e-9

    def test_compute_period_return_missing_ticker(self):
        df = pd.DataFrame({"ticker": ["B"], "start_price": [100.0], "end_price": [110.0]})
        ret = compute_period_return({"A": 1.0}, df)
        assert ret == 0.0

    def test_compute_turnover_full_replacement(self):
        prev = {"A": 1.0}
        new  = {"B": 1.0}
        assert abs(compute_turnover(prev, new) - 1.0) < 1e-9

    def test_compute_turnover_no_change(self):
        w = {"A": 0.5, "B": 0.5}
        assert compute_turnover(w, w) < 1e-9

    def test_compute_turnover_partial(self):
        prev = {"A": 0.6, "B": 0.4}
        new  = {"A": 0.4, "B": 0.4, "C": 0.2}
        to = compute_turnover(prev, new)
        assert 0 < to <= 1

    def test_compute_equity_curve_starts_near_one(self):
        returns = pd.Series([0.05, -0.02, 0.03])
        curve = compute_equity_curve(returns)
        assert abs(curve.iloc[0] - 1.05) < 1e-9

    def test_compute_max_drawdown_negative(self):
        equity = pd.Series([1.0, 1.2, 0.9, 1.1])
        mdd = compute_max_drawdown(equity)
        assert mdd < 0

    def test_annualize_stats_keys(self):
        returns = pd.Series([0.02, -0.01, 0.03, 0.01, -0.005, 0.02])
        stats = annualize_stats(returns, periods_per_yr=12)
        assert all(k in stats for k in ["ann_return", "ann_vol", "sharpe", "sortino", "max_drawdown"])


class TestWalkForwardFromScores:
    def test_returns_backtest_result(self, backtest_result):
        assert isinstance(backtest_result, BacktestResult)

    def test_equity_curve_starts_above_zero(self, backtest_result):
        assert (backtest_result.equity_curve > 0).all()

    def test_equity_curve_len_at_least_periods(self, backtest_result):
        # equity_curve includes cash periods (no eligible stocks); periods only
        # records rebalance windows where Buy/Gotta Have holdings existed.
        assert len(backtest_result.equity_curve) >= len(backtest_result.periods)

    def test_benchmark_curve_present(self, backtest_result):
        assert len(backtest_result.benchmark_curve) > 0

    def test_ann_turnover_positive(self, backtest_result):
        assert backtest_result.ann_turnover >= 0

    def test_cost_drag_non_negative(self, backtest_result):
        assert backtest_result.cost_drag >= 0

    def test_max_drawdown_non_positive(self, backtest_result):
        assert backtest_result.max_drawdown <= 0

    def test_contamination_label_set(self, backtest_result):
        assert "OUT-OF-SAMPLE" in backtest_result.contamination_label

    def test_no_lookahead_empty_raises(self):
        with pytest.raises(ValueError):
            walk_forward_from_scores(
                scored_records=pd.DataFrame({"date": [date(2020, 1, 1)],
                                             "ticker": ["A"], "grade": ["Buy"],
                                             "period_return": [0.05]}),
                portfolio_name="x",
                rebalance_freq="quarterly",
                cost_model=CostModel(),
                config={},
            )


# ──────────────────────────────────────────────────────────────
# Calibrator tests
# ──────────────────────────────────────────────────────────────

class TestWeightGrid:
    def test_all_sum_to_one(self):
        grid = _weight_grid(step=0.10, min_w=0.10)
        for f, t, q in grid:
            assert abs(f + t + q - 1.0) < 1e-9

    def test_all_above_min(self):
        grid = _weight_grid(step=0.10, min_w=0.10)
        for f, t, q in grid:
            assert f >= 0.10 - 1e-9
            assert t >= 0.10 - 1e-9
            assert q >= 0.10 - 1e-9

    def test_grid_non_empty(self):
        assert len(_weight_grid()) > 0


class TestBlendComposite:
    def test_blend_sum_of_engines(self):
        df = pd.DataFrame({
            "fund_score": [60.0], "tech_score": [50.0], "quant_score": [40.0]
        })
        result = _blend_composite(df, 0.5, 0.3, 0.2)
        expected = 60 * 0.5 + 50 * 0.3 + 40 * 0.2
        assert abs(result["composite"].iloc[0] - expected) < 1e-9

    def test_blend_clips_to_100(self):
        df = pd.DataFrame({
            "fund_score": [100.0], "tech_score": [100.0], "quant_score": [100.0]
        })
        result = _blend_composite(df, 0.5, 0.3, 0.2)
        assert result["composite"].iloc[0] <= 100.0

    def test_blend_raises_without_engine_scores(self):
        df = pd.DataFrame({"composite": [60.0]})
        with pytest.raises(ValueError):
            _blend_composite(df, 0.5, 0.3, 0.2)


class TestCalibrateWeights:
    def test_returns_calibration_result(self, calibration_result):
        assert isinstance(calibration_result, CalibrationResult)

    def test_best_weights_sum_to_one(self, calibration_result):
        total = sum(calibration_result.best_weights.values())
        assert abs(total - 1.0) < 1e-9

    def test_all_weights_above_min(self, calibration_result):
        for w in calibration_result.best_weights.values():
            assert w >= 0.20 - 1e-9

    def test_val_ic_positive_for_synthetic(self, calibration_result):
        assert calibration_result.val_ic > 0

    def test_test_ic_present(self, calibration_result):
        assert calibration_result.test_ic is not None

    def test_grid_results_non_empty(self, calibration_result):
        assert len(calibration_result.grid_results) > 0

    def test_grid_results_sorted_by_val_ic(self, calibration_result):
        ics = [p.val_ic for p in calibration_result.grid_results]
        assert ics == sorted(ics, reverse=True)

    def test_empty_validation_raises(self, scores):
        with pytest.raises(ValueError):
            calibrate_weights(scores, train_end=date(2099, 1, 1), val_end=date(2099, 6, 1))


# ──────────────────────────────────────────────────────────────
# Report tests
# ──────────────────────────────────────────────────────────────

class TestRenderReports:
    def test_validation_report_contains_label(self, validation_report):
        md = render_validation_report(validation_report)
        assert "OUT-OF-SAMPLE" in md

    def test_validation_report_contains_ic_table(self, validation_report):
        md = render_validation_report(validation_report)
        assert "Information Coefficient" in md

    def test_validation_report_contains_monotonicity(self, validation_report):
        md = render_validation_report(validation_report)
        assert "Monotonicity" in md

    def test_backtest_report_contains_portfolio_name(self, backtest_result):
        md = render_backtest_report(backtest_result)
        assert backtest_result.portfolio_name in md

    def test_backtest_report_contains_sharpe(self, backtest_result):
        md = render_backtest_report(backtest_result)
        assert "Sharpe" in md

    def test_calibration_report_contains_best_weights(self, calibration_result):
        md = render_calibration_report(calibration_result)
        assert "Optimal Engine Weights" in md

    def test_calibration_report_contains_oos(self, calibration_result):
        md = render_calibration_report(calibration_result)
        assert "Test (OOS)" in md

    def test_to_json_validation_report(self, validation_report):
        j = to_json(validation_report)
        parsed = json.loads(j)
        assert "label" in parsed

    def test_to_json_backtest_result(self, backtest_result):
        j = to_json(backtest_result)
        parsed = json.loads(j)
        assert "portfolio_name" in parsed

    def test_to_json_calibration_result(self, calibration_result):
        j = to_json(calibration_result)
        parsed = json.loads(j)
        assert "best_weights" in parsed

    def test_save_report_creates_file(self, tmp_path, validation_report):
        md = render_validation_report(validation_report)
        path = tmp_path / "reports" / "test.md"
        save_report(md, path)
        assert path.exists()
        content = path.read_text(encoding="utf-8")
        assert "OUT-OF-SAMPLE" in content


# ──────────────────────────────────────────────────────────────
# Integration: full pipeline on synthetic data
# ──────────────────────────────────────────────────────────────

class TestEndToEndSynthetic:
    def test_grade_validation_positive_ic(self, validation_report):
        """Synthetic data with true IC ≈ 0.10 should produce positive measured IC."""
        ic12 = validation_report.ic_stats[12].mean
        assert ic12 > 0, f"Expected positive IC, got {ic12}"

    def test_walk_forward_outperforms_cash(self, backtest_result):
        """Portfolio should not lose everything (equity curve end > 0)."""
        assert float(backtest_result.equity_curve.iloc[-1]) > 0

    def test_calibrate_selects_reasonable_weights(self, calibration_result):
        """No single engine should dominate (weights should be <= 0.8)."""
        for w in calibration_result.best_weights.values():
            assert w <= 0.80, f"Extreme weight: {w}"

    def test_reports_all_render_without_error(
        self, validation_report, backtest_result, calibration_result
    ):
        md1 = render_validation_report(validation_report)
        md2 = render_backtest_report(backtest_result)
        md3 = render_calibration_report(calibration_result)
        assert all(len(md) > 100 for md in [md1, md2, md3])
