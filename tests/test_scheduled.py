"""
Tests for daily_run.py and weekly_run.py (Step 13).

All tests run via --dry-run or patching the analysis pipeline
so no live API calls are made.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml
from typer.testing import CliRunner

from daily_run import (
    app as daily_app,
    _load_state,
    _save_state,
    _grade_changed,
    _alert_price_hit,
    _circuit_breaker_fired,
    _price_crossed_boundary,
    _make_report,
)
from weekly_run import (
    app as weekly_app,
    _drift_report,
    _holdings_md,
    _mandate_md,
)

runner = CliRunner()


# ──────────────────────────────────────────────────────────────
# Daily alert detection helpers
# ──────────────────────────────────────────────────────────────

class TestGradeChanged:
    def test_detects_change(self):
        prev = {"grade": "Hold"}
        assert _grade_changed(prev, "Buy") is True

    def test_no_change(self):
        prev = {"grade": "Buy"}
        assert _grade_changed(prev, "Buy") is False

    def test_no_prior_returns_false(self):
        assert _grade_changed(None, "Buy") is False


class TestAlertPriceHit:
    def _mock_result(self, price: float):
        r = MagicMock()
        r.price = price
        return r

    def test_hit_at_exact_price(self):
        entry = {"alert_price": 140.0}
        assert _alert_price_hit(entry, self._mock_result(140.0)) is True

    def test_hit_below_alert(self):
        entry = {"alert_price": 140.0}
        assert _alert_price_hit(entry, self._mock_result(130.0)) is True

    def test_not_hit_above_alert(self):
        entry = {"alert_price": 140.0}
        assert _alert_price_hit(entry, self._mock_result(145.0)) is False

    def test_no_alert_price_returns_false(self):
        entry = {}
        assert _alert_price_hit(entry, self._mock_result(100.0)) is False


class TestCircuitBreakerFired:
    def _mock_result(self, active_cbs: list[str]):
        r = MagicMock()
        r.overall.circuit_breakers = {cb: True for cb in active_cbs}
        return r

    def test_new_cb_detected(self):
        prev = {"circuit_breakers": []}
        result = self._mock_result(["altman_z"])
        fired = _circuit_breaker_fired(prev, result)
        assert "altman_z" in fired

    def test_existing_cb_not_reported(self):
        prev = {"circuit_breakers": ["altman_z"]}
        result = self._mock_result(["altman_z"])
        fired = _circuit_breaker_fired(prev, result)
        assert fired == []

    def test_no_prev_returns_all(self):
        result = self._mock_result(["altman_z", "extreme_overvaluation"])
        fired = _circuit_breaker_fired(None, result)
        assert set(fired) == {"altman_z", "extreme_overvaluation"}


class TestMakeReport:
    def test_no_alerts_section(self):
        md = _make_report([], ["AAPL", "MSFT"], "2024-01-15")
        assert "No Actionable Alerts" in md

    def test_alerts_listed(self):
        alerts = [("AAPL", ["Grade changed: Hold → Buy ✅"])]
        md = _make_report(alerts, [], "2024-01-15")
        assert "AAPL" in md
        assert "Grade changed" in md

    def test_no_alerts_lists_tickers_with_no_change(self):
        md = _make_report([], ["AAPL", "MSFT"], "2024-01-15")
        assert "AAPL" in md
        assert "MSFT" in md


class TestStateIO:
    def test_roundtrip(self, tmp_path, monkeypatch):
        import daily_run as dr
        monkeypatch.setattr(dr, "_STATE_FILE", tmp_path / ".state.json")
        state = {"AAPL": {"grade": "Buy", "price": 150.0, "circuit_breakers": []}}
        _save_state(state)
        loaded = _load_state()
        assert loaded == state

    def test_missing_state_returns_empty(self, tmp_path, monkeypatch):
        import daily_run as dr
        monkeypatch.setattr(dr, "_STATE_FILE", tmp_path / ".nonexistent.json")
        assert _load_state() == {}


# ──────────────────────────────────────────────────────────────
# Daily CLI tests
# ──────────────────────────────────────────────────────────────

class TestDailyCLI:
    def test_dry_run_exits_2(self, tmp_path):
        result = runner.invoke(daily_app, ["--dry-run"])
        assert result.exit_code == 2

    def test_dry_run_prints_message(self, tmp_path):
        result = runner.invoke(daily_app, ["--dry-run"])
        assert "dry-run" in result.output.lower()

    def test_missing_watchlist_exits_1(self, tmp_path):
        result = runner.invoke(daily_app, ["--watchlist", str(tmp_path / "nope.yaml")])
        assert result.exit_code == 1

    def test_empty_watchlist_exits_0(self, tmp_path):
        wl = tmp_path / "wl.yaml"
        wl.write_text("watchlist: []\n", encoding="utf-8")
        result = runner.invoke(daily_app, ["--watchlist", str(wl)])
        assert result.exit_code == 0

    def test_mock_analysis_no_alerts_produces_no_change_message(self, tmp_path, monkeypatch):
        wl = tmp_path / "wl.yaml"
        wl.write_text(
            "watchlist:\n  - ticker: MOCK\n    notes: test\n",
            encoding="utf-8",
        )
        state_file = tmp_path / ".state.json"
        state_file.write_text(
            json.dumps({"MOCK": {"grade": "Buy", "price": 100.0, "circuit_breakers": []}}),
            encoding="utf-8",
        )

        mock_result = MagicMock()
        mock_result.overall.grade.value = "Buy"
        mock_result.overall.circuit_breakers = {}
        mock_result.price = 100.0
        mock_result.price_ladder = None

        import daily_run as dr
        monkeypatch.setattr(dr, "_STATE_FILE", state_file)
        monkeypatch.setattr(dr, "_RUNS_DIR", tmp_path / "daily")

        with patch("stockgrader.pipeline.run_analysis", return_value=mock_result), \
             patch("stockgrader.config.get_config", return_value={}):
            result = runner.invoke(daily_app, [
                "--watchlist", str(wl), "--no-save"
            ])
        assert result.exit_code == 0
        assert "No actionable alerts" in result.output

    def test_mock_analysis_grade_change_produces_alert(self, tmp_path, monkeypatch):
        wl = tmp_path / "wl.yaml"
        wl.write_text(
            "watchlist:\n  - ticker: MOCK\n    notes: test\n",
            encoding="utf-8",
        )
        state_file = tmp_path / ".state.json"
        # Prior state says Hold; current will be Buy
        state_file.write_text(
            json.dumps({"MOCK": {"grade": "Hold", "price": 100.0, "circuit_breakers": []}}),
            encoding="utf-8",
        )

        mock_result = MagicMock()
        mock_result.overall.grade.value = "Buy"
        mock_result.overall.circuit_breakers = {}
        mock_result.price = 100.0
        mock_result.price_ladder = None

        import daily_run as dr
        monkeypatch.setattr(dr, "_STATE_FILE", state_file)
        monkeypatch.setattr(dr, "_RUNS_DIR", tmp_path / "daily")

        with patch("stockgrader.pipeline.run_analysis", return_value=mock_result), \
             patch("stockgrader.config.get_config", return_value={}):
            result = runner.invoke(daily_app, [
                "--watchlist", str(wl), "--no-save"
            ])
        assert result.exit_code == 0
        assert "1 alert" in result.output


# ──────────────────────────────────────────────────────────────
# Weekly helpers
# ──────────────────────────────────────────────────────────────

class TestDriftReport:
    def test_shows_increase(self):
        prior   = {"AAPL": 0.05}
        current = {"AAPL": 0.10}
        md = _drift_report("balanced", prior, current)
        assert "↑" in md
        assert "AAPL" in md

    def test_shows_decrease(self):
        prior   = {"AAPL": 0.10}
        current = {"AAPL": 0.05}
        md = _drift_report("balanced", prior, current)
        assert "↓" in md

    def test_new_position_shown(self):
        prior   = {}
        current = {"MSFT": 0.08}
        md = _drift_report("balanced", prior, current)
        assert "MSFT" in md

    def test_removed_position_shown(self):
        prior   = {"TSLA": 0.06}
        current = {}
        md = _drift_report("balanced", prior, current)
        assert "TSLA" in md

    def test_tiny_change_omitted(self):
        # Changes < 0.1% are not shown
        prior   = {"AAPL": 0.0500}
        current = {"AAPL": 0.0501}
        md = _drift_report("balanced", prior, current)
        assert "AAPL" not in md


class TestHoldingsMd:
    def _mock_result(self, tickers: list[str]):
        holdings = []
        for i, t in enumerate(tickers):
            h = MagicMock()
            h.ticker = t
            h.weight = 1.0 / len(tickers)
            h.grade  = "Buy"
            h.composite = 65.0 + i
            holdings.append(h)
        result = MagicMock()
        result.holdings = holdings
        result.analytics.expected_return = 0.08
        result.analytics.volatility      = 0.15
        result.analytics.sharpe          = 0.52
        result.analytics.n_holdings      = len(tickers)
        return result

    def test_holdings_table_present(self):
        result = self._mock_result(["AAPL", "MSFT"])
        md = _holdings_md("balanced", result, "2024-01-15")
        assert "AAPL" in md
        assert "MSFT" in md

    def test_empty_holdings_message(self):
        result = MagicMock()
        result.holdings = []
        md = _holdings_md("balanced", result, "2024-01-15")
        assert "insufficient" in md.lower() or "No holdings" in md

    def test_none_result_message(self):
        md = _holdings_md("balanced", None, "2024-01-15")
        assert "No holdings" in md


class TestMandateMd:
    def test_pass_reported(self):
        result = MagicMock()
        result.mandate_check.passed    = True
        result.mandate_check.violations = []
        md = _mandate_md("balanced", result)
        assert "PASS" in md

    def test_fail_reported(self):
        result = MagicMock()
        result.mandate_check.passed     = False
        result.mandate_check.violations = ["Sector cap exceeded"]
        md = _mandate_md("balanced", result)
        assert "FAIL" in md
        assert "Sector cap exceeded" in md

    def test_none_result(self):
        md = _mandate_md("balanced", None)
        assert "unavailable" in md.lower()


# ──────────────────────────────────────────────────────────────
# Weekly CLI tests
# ──────────────────────────────────────────────────────────────

class TestWeeklyCLI:
    def test_dry_run_exits_2(self):
        result = runner.invoke(weekly_app, ["--dry-run"])
        assert result.exit_code == 2

    def test_dry_run_prints_config_info(self):
        result = runner.invoke(weekly_app, ["--dry-run"])
        assert "dry-run" in result.output.lower() or "Config" in result.output

    def test_mock_funnel_full_run(self, tmp_path, monkeypatch):
        import weekly_run as wr
        monkeypatch.setattr(wr, "_RUNS_DIR", tmp_path / "weekly")

        mock_holding = MagicMock()
        mock_holding.ticker  = "AAPL"
        mock_holding.weight  = 0.10
        mock_holding.grade   = "Buy"
        mock_holding.composite = 65.0

        mock_result = MagicMock()
        mock_result.holdings = [mock_holding]
        mock_result.analytics.expected_return = 0.08
        mock_result.analytics.volatility      = 0.15
        mock_result.analytics.sharpe          = 0.52
        mock_result.analytics.n_holdings      = 1
        mock_result.mandate_check.passed      = True
        mock_result.mandate_check.violations  = []

        with patch("stockgrader.portfolios.construction.run_full_funnel", return_value=mock_result), \
             patch("stockgrader.config.get_config", return_value={}):
            result = runner.invoke(weekly_app, ["--funds", "balanced"])

        assert result.exit_code == 0
        assert "balanced" in result.output.lower()

    def test_mock_funnel_writes_files(self, tmp_path, monkeypatch):
        import weekly_run as wr
        out_dir = tmp_path / "weekly"
        monkeypatch.setattr(wr, "_RUNS_DIR", out_dir)

        mock_holding = MagicMock()
        mock_holding.ticker  = "AAPL"
        mock_holding.weight  = 0.10
        mock_holding.grade   = "Buy"
        mock_holding.composite = 65.0

        mock_result = MagicMock()
        mock_result.holdings = [mock_holding]
        mock_result.analytics.expected_return = 0.08
        mock_result.analytics.volatility      = 0.15
        mock_result.analytics.sharpe          = 0.52
        mock_result.analytics.n_holdings      = 1
        mock_result.mandate_check.passed      = True
        mock_result.mandate_check.violations  = []

        with patch("stockgrader.portfolios.construction.run_full_funnel", return_value=mock_result), \
             patch("stockgrader.config.get_config", return_value={}):
            result = runner.invoke(weekly_app, ["--funds", "balanced"])

        assert result.exit_code == 0
        run_date = datetime.utcnow().strftime("%Y-%m-%d")
        assert (out_dir / run_date / "balanced_holdings.md").exists()
        assert (out_dir / run_date / "balanced_analytics.json").exists()
        assert (out_dir / run_date / "drift_report.md").exists()
