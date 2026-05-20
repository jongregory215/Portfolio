"""
Tests for daily_run.py and weekly_run.py (Step 13).

All tests use mocks / dry-run so no live API calls are made.
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
    _sub_grade_changes,
    _alert_price_hit,
    _circuit_breaker_fired,
    _price_crossed_boundary,
    _extract_state,
    _make_report,
    _send_notification,
)
from weekly_run import (
    app as weekly_app,
    _drift_report,
    _holdings_md,
    _mandate_md,
    _funnel_md,
    _load_checkpoint,
    _save_checkpoint,
)

runner = CliRunner()


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────

def _mock_result(grade: str = "Buy", price: float = 100.0,
                 cbs: list[str] | None = None,
                 sub_grades: dict[str, str] | None = None) -> MagicMock:
    """Build a minimal mock AnalysisResult."""
    r = MagicMock()
    r.overall.grade.value = grade
    r.overall.circuit_breakers = {cb: True for cb in (cbs or [])}
    r.price = price
    r.price_ladder = None

    # Portfolio sub-grades
    pg = MagicMock()
    for fund in ["very_conservative", "conservative", "balanced",
                 "aggressive", "very_aggressive"]:
        sleeve = MagicMock()
        sleeve.grade.value = (sub_grades or {}).get(fund, "Hold")
        setattr(pg, fund, sleeve)
    r.portfolio_grades = pg
    return r


# ──────────────────────────────────────────────────────────────
# Daily — alert detection helpers
# ──────────────────────────────────────────────────────────────

class TestGradeChanged:
    def test_detects_change(self):
        assert _grade_changed({"grade": "Hold"}, "Buy") is True

    def test_no_change(self):
        assert _grade_changed({"grade": "Buy"}, "Buy") is False

    def test_no_prior_returns_false(self):
        assert _grade_changed(None, "Buy") is False


class TestSubGradeChanges:
    def test_detects_sleeve_change(self):
        prev = {
            "sub_grades": {"aggressive": "Hold"}
        }
        result = _mock_result(sub_grades={"aggressive": "Buy"})
        msgs = _sub_grade_changes(prev, result)
        assert any("Aggressive" in m for m in msgs)
        assert any("Hold → Buy" in m for m in msgs)

    def test_no_change_returns_empty(self):
        prev = {"sub_grades": {"balanced": "Buy"}}
        result = _mock_result(sub_grades={"balanced": "Buy"})
        assert _sub_grade_changes(prev, result) == []

    def test_no_prior_returns_empty(self):
        result = _mock_result(sub_grades={"balanced": "Buy"})
        assert _sub_grade_changes(None, result) == []


class TestAlertPriceHit:
    def test_hit_at_exact_price(self):
        assert _alert_price_hit({"alert_price": 140.0}, _mock_result(price=140.0))

    def test_hit_below_alert(self):
        assert _alert_price_hit({"alert_price": 140.0}, _mock_result(price=130.0))

    def test_not_hit_above_alert(self):
        assert not _alert_price_hit({"alert_price": 140.0}, _mock_result(price=145.0))

    def test_no_alert_price_returns_false(self):
        assert not _alert_price_hit({}, _mock_result(price=100.0))


class TestCircuitBreakerFired:
    def test_new_cb_detected(self):
        prev = {"circuit_breakers": []}
        result = _mock_result(cbs=["altman_z"])
        assert "altman_z" in _circuit_breaker_fired(prev, result)

    def test_existing_cb_not_reported(self):
        prev = {"circuit_breakers": ["altman_z"]}
        result = _mock_result(cbs=["altman_z"])
        assert _circuit_breaker_fired(prev, result) == []

    def test_no_prev_reports_all(self):
        result = _mock_result(cbs=["altman_z", "extreme_overvaluation"])
        fired = _circuit_breaker_fired(None, result)
        assert set(fired) == {"altman_z", "extreme_overvaluation"}


class TestExtractState:
    def test_grade_captured(self):
        state = _extract_state(_mock_result(grade="Buy"))
        assert state["grade"] == "Buy"

    def test_price_captured(self):
        state = _extract_state(_mock_result(price=155.5))
        assert state["price"] == 155.5

    def test_circuit_breakers_captured(self):
        state = _extract_state(_mock_result(cbs=["altman_z"]))
        assert "altman_z" in state["circuit_breakers"]

    def test_sub_grades_captured(self):
        state = _extract_state(_mock_result(sub_grades={"balanced": "Buy"}))
        assert state["sub_grades"]["balanced"] == "Buy"


# ──────────────────────────────────────────────────────────────
# Daily — state I/O
# ──────────────────────────────────────────────────────────────

class TestStateIO:
    def test_save_and_load_rolling_state(self, tmp_path, monkeypatch):
        import daily_run as dr
        monkeypatch.setattr(dr, "_STATE_FILE", tmp_path / ".state.json")
        state = {"AAPL": {"grade": "Buy", "price": 150.0, "circuit_breakers": [],
                          "sub_grades": {}, "ladder_prices": {}}}
        _save_state(state, "2024-01-15", tmp_path / "daily")
        loaded = _load_state()
        assert loaded == state

    def test_save_writes_dated_snapshot(self, tmp_path, monkeypatch):
        import daily_run as dr
        monkeypatch.setattr(dr, "_STATE_FILE", tmp_path / ".state.json")
        runs_dir = tmp_path / "daily"
        state = {"MSFT": {"grade": "Hold", "price": 200.0, "circuit_breakers": [],
                           "sub_grades": {}, "ladder_prices": {}}}
        _save_state(state, "2024-01-15", runs_dir)
        snapshot = runs_dir / "2024-01-15.json"
        assert snapshot.exists()
        assert json.loads(snapshot.read_text(encoding="utf-8")) == state

    def test_missing_state_returns_empty(self, tmp_path, monkeypatch):
        import daily_run as dr
        monkeypatch.setattr(dr, "_STATE_FILE", tmp_path / ".nonexistent.json")
        assert _load_state() == {}


# ──────────────────────────────────────────────────────────────
# Daily — report builder
# ──────────────────────────────────────────────────────────────

class TestMakeReport:
    def test_no_alerts_section(self):
        md = _make_report([], ["AAPL", "MSFT"], [], "2024-01-15", first_run=False)
        assert "No Actionable Alerts" in md

    def test_no_alerts_count_line(self):
        md = _make_report([], ["AAPL"], [], "2024-01-15", first_run=False)
        assert "1 name" in md

    def test_alerts_listed(self):
        alerts = [("AAPL", ["Overall grade: Hold → Buy ✅"])]
        md = _make_report(alerts, [], [], "2024-01-15", first_run=False)
        assert "AAPL" in md
        assert "Hold → Buy" in md

    def test_first_run_message(self):
        md = _make_report([], ["AAPL"], [], "2024-01-15", first_run=True)
        assert "First run" in md

    def test_skipped_section(self):
        md = _make_report([], [], ["BAD"], "2024-01-15", first_run=False)
        assert "Skipped" in md
        assert "BAD" in md

    def test_no_alerts_with_tickers(self):
        md = _make_report([], ["AAPL", "MSFT"], [], "2024-01-15", first_run=False)
        assert "AAPL" in md
        assert "MSFT" in md


# ──────────────────────────────────────────────────────────────
# Daily — notification hook
# ──────────────────────────────────────────────────────────────

class TestNotificationHook:
    def test_off_by_default(self):
        # Should not raise — just a no-op
        alerts = [("AAPL", ["Grade changed"])]
        _send_notification(alerts, {})

    def test_enabled_flag(self, caplog):
        import logging
        alerts = [("AAPL", ["Grade changed"])]
        with caplog.at_level(logging.INFO, logger="daily_run"):
            _send_notification(alerts, {"notifications": {"enabled": True}})
        assert any("NOTIFICATION HOOK" in r.message for r in caplog.records)


# ──────────────────────────────────────────────────────────────
# Daily — CLI
# ──────────────────────────────────────────────────────────────

class TestDailyCLI:
    def test_dry_run_exits_2(self):
        result = runner.invoke(daily_app, ["--dry-run"])
        assert result.exit_code == 2

    def test_dry_run_prints_message(self):
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

    def test_first_run_records_baseline(self, tmp_path, monkeypatch):
        """First run with empty state: no alerts, 'baseline recorded' printed."""
        wl = tmp_path / "wl.yaml"
        wl.write_text("watchlist:\n  - ticker: MOCK\n    notes: test\n", encoding="utf-8")
        state_file = tmp_path / ".state.json"

        import daily_run as dr
        monkeypatch.setattr(dr, "_STATE_FILE", state_file)
        monkeypatch.setattr(dr, "_RUNS_DIR", tmp_path / "daily")

        mock_result = _mock_result(grade="Buy", price=100.0)
        with patch("stockgrader.pipeline.run_analysis", return_value=mock_result), \
             patch("stockgrader.config.get_config", return_value={}):
            result = runner.invoke(daily_app, ["--watchlist", str(wl)])

        assert result.exit_code == 0
        assert "Baseline" in result.output or "baseline" in result.output.lower()
        # State file should now exist
        assert state_file.exists()

    def test_second_run_no_change_no_alert(self, tmp_path, monkeypatch):
        """Second run with identical state → no alerts."""
        wl = tmp_path / "wl.yaml"
        wl.write_text("watchlist:\n  - ticker: MOCK\n    notes: test\n", encoding="utf-8")
        state_file = tmp_path / ".state.json"
        state_file.write_text(json.dumps({
            "MOCK": {"grade": "Buy", "price": 100.0, "circuit_breakers": [],
                     "sub_grades": {"balanced": "Buy"},
                     "ladder_prices": {}}
        }), encoding="utf-8")

        import daily_run as dr
        monkeypatch.setattr(dr, "_STATE_FILE", state_file)
        monkeypatch.setattr(dr, "_RUNS_DIR", tmp_path / "daily")

        mock_result = _mock_result(grade="Buy", price=100.0,
                                   sub_grades={"balanced": "Buy"})
        with patch("stockgrader.pipeline.run_analysis", return_value=mock_result), \
             patch("stockgrader.config.get_config", return_value={}):
            result = runner.invoke(daily_app, ["--watchlist", str(wl)])

        assert result.exit_code == 0
        assert "No alerts" in result.output

    def test_grade_change_produces_alert(self, tmp_path, monkeypatch):
        """Prior grade=Hold, current=Buy → alert fired."""
        wl = tmp_path / "wl.yaml"
        wl.write_text("watchlist:\n  - ticker: MOCK\n    notes: test\n", encoding="utf-8")
        state_file = tmp_path / ".state.json"
        state_file.write_text(json.dumps({
            "MOCK": {"grade": "Hold", "price": 100.0, "circuit_breakers": [],
                     "sub_grades": {}, "ladder_prices": {}}
        }), encoding="utf-8")

        import daily_run as dr
        monkeypatch.setattr(dr, "_STATE_FILE", state_file)
        monkeypatch.setattr(dr, "_RUNS_DIR", tmp_path / "daily")

        mock_result = _mock_result(grade="Buy", price=100.0)
        with patch("stockgrader.pipeline.run_analysis", return_value=mock_result), \
             patch("stockgrader.config.get_config", return_value={}):
            result = runner.invoke(daily_app, ["--watchlist", str(wl)])

        assert result.exit_code == 0
        assert "1 alert" in result.output

    def test_failed_ticker_skipped_gracefully(self, tmp_path, monkeypatch):
        """Data error for one ticker → it's skipped, run succeeds."""
        wl = tmp_path / "wl.yaml"
        wl.write_text("watchlist:\n  - ticker: BAD\n    notes: test\n", encoding="utf-8")
        state_file = tmp_path / ".state.json"

        import daily_run as dr
        monkeypatch.setattr(dr, "_STATE_FILE", state_file)
        monkeypatch.setattr(dr, "_RUNS_DIR", tmp_path / "daily")

        with patch("stockgrader.pipeline.run_analysis",
                   side_effect=RuntimeError("API timeout")), \
             patch("stockgrader.config.get_config", return_value={}):
            result = runner.invoke(daily_app, ["--watchlist", str(wl)])

        assert result.exit_code == 0
        assert "Skipped" in result.output or "skipped" in result.output.lower() or \
               "BAD" in result.output or "0" in result.output


# ──────────────────────────────────────────────────────────────
# Weekly — helpers
# ──────────────────────────────────────────────────────────────

class TestDriftReport:
    def test_shows_increase(self):
        md = _drift_report("balanced", {"AAPL": 0.05}, {"AAPL": 0.10})
        assert "↑" in md and "AAPL" in md

    def test_shows_decrease(self):
        md = _drift_report("balanced", {"AAPL": 0.10}, {"AAPL": 0.05})
        assert "↓" in md

    def test_entered_label(self):
        md = _drift_report("balanced", {}, {"MSFT": 0.08})
        assert "Entered" in md and "MSFT" in md

    def test_exited_label(self):
        md = _drift_report("balanced", {"TSLA": 0.06}, {})
        assert "Exited" in md and "TSLA" in md

    def test_tiny_change_omitted(self):
        md = _drift_report("balanced", {"AAPL": 0.0500}, {"AAPL": 0.0501})
        assert "AAPL" not in md or "No meaningful" in md


class TestHoldingsMd:
    def _mock_result(self, tickers: list[str]):
        holdings = []
        for i, t in enumerate(tickers):
            h = MagicMock()
            h.ticker = t
            h.weight = 1.0 / len(tickers)
            h.grade = "Buy"
            h.composite = 65.0 + i
            holdings.append(h)
        r = MagicMock()
        r.holdings = holdings
        r.analytics.expected_return = 0.08
        r.analytics.volatility = 0.15
        r.analytics.sharpe = 0.52
        r.analytics.n_holdings = len(tickers)
        return r

    def test_tickers_in_table(self):
        md = _holdings_md("balanced", self._mock_result(["AAPL", "MSFT"]), "2024-01-15")
        assert "AAPL" in md and "MSFT" in md

    def test_empty_holdings_message(self):
        r = MagicMock(); r.holdings = []
        md = _holdings_md("balanced", r, "2024-01-15")
        assert "No holdings" in md or "insufficient" in md.lower()

    def test_none_result_message(self):
        md = _holdings_md("balanced", None, "2024-01-15")
        assert "No holdings" in md


class TestMandateMd:
    def test_pass_reported(self):
        r = MagicMock()
        r.mandate_check.passed = True
        r.mandate_check.violations = []
        assert "PASS" in _mandate_md("balanced", r)

    def test_fail_with_violation(self):
        r = MagicMock()
        r.mandate_check.passed = False
        r.mandate_check.violations = ["Sector cap exceeded"]
        md = _mandate_md("balanced", r)
        assert "FAIL" in md and "Sector cap exceeded" in md

    def test_none_result(self):
        assert "unavailable" in _mandate_md("balanced", None).lower()


class TestFunnelMd:
    def test_contains_stage_headers(self):
        r = MagicMock()
        r.funnel_stats = None
        r.holdings = []
        md = _funnel_md("balanced", r, "2024-01-15")
        assert "Funnel" in md

    def test_none_result(self):
        md = _funnel_md("balanced", None, "2024-01-15")
        assert "not available" in md.lower()


# ──────────────────────────────────────────────────────────────
# Weekly — checkpointing
# ──────────────────────────────────────────────────────────────

class TestCheckpointing:
    def test_save_and_load(self, tmp_path):
        _save_checkpoint(tmp_path, {"balanced": True})
        loaded = _load_checkpoint(tmp_path)
        assert loaded == {"balanced": True}

    def test_missing_returns_empty(self, tmp_path):
        assert _load_checkpoint(tmp_path / "nonexistent") == {}

    def test_completed_fund_skipped(self, tmp_path, monkeypatch):
        import weekly_run as wr
        run_date = datetime.utcnow().strftime("%Y-%m-%d")
        out_dir = tmp_path / "weekly" / run_date
        out_dir.mkdir(parents=True)
        monkeypatch.setattr(wr, "_RUNS_DIR", tmp_path / "weekly")
        _save_checkpoint(out_dir, {"balanced": True})

        mock_holding = MagicMock()
        mock_holding.ticker = "AAPL"
        mock_holding.weight = 0.10
        mock_holding.grade = "Buy"
        mock_holding.composite = 65.0

        mock_result = MagicMock()
        mock_result.holdings = [mock_holding]
        mock_result.analytics.expected_return = 0.08
        mock_result.analytics.volatility = 0.15
        mock_result.analytics.sharpe = 0.52
        mock_result.analytics.n_holdings = 1
        mock_result.mandate_check.passed = True
        mock_result.mandate_check.violations = []
        mock_result.funnel_stats = None

        call_count = {"n": 0}
        def counted_funnel(fund, config):
            call_count["n"] += 1
            return mock_result

        with patch("stockgrader.portfolios.construction.run_full_funnel", side_effect=counted_funnel), \
             patch("stockgrader.config.get_config", return_value={}):
            result = runner.invoke(weekly_app, ["--funds", "balanced", "--resume"])

        # balanced was checkpointed → funnel should NOT be called
        assert call_count["n"] == 0


# ──────────────────────────────────────────────────────────────
# Weekly — CLI
# ──────────────────────────────────────────────────────────────

class TestWeeklyCLI:
    def test_dry_run_exits_2(self):
        result = runner.invoke(weekly_app, ["--dry-run"])
        assert result.exit_code == 2

    def test_dry_run_prints_config_info(self):
        result = runner.invoke(weekly_app, ["--dry-run"])
        assert "dry-run" in result.output.lower() or "Config" in result.output

    def test_full_run_writes_all_files(self, tmp_path, monkeypatch):
        import weekly_run as wr
        monkeypatch.setattr(wr, "_RUNS_DIR", tmp_path / "weekly")

        mock_holding = MagicMock()
        mock_holding.ticker = "AAPL"
        mock_holding.weight = 0.10
        mock_holding.grade = "Buy"
        mock_holding.composite = 65.0

        mock_result = MagicMock()
        mock_result.holdings = [mock_holding]
        mock_result.analytics.expected_return = 0.08
        mock_result.analytics.volatility = 0.15
        mock_result.analytics.sharpe = 0.52
        mock_result.analytics.n_holdings = 1
        mock_result.mandate_check.passed = True
        mock_result.mandate_check.violations = []
        mock_result.funnel_stats = None

        with patch("stockgrader.portfolios.construction.run_full_funnel",
                   return_value=mock_result), \
             patch("stockgrader.config.get_config", return_value={}):
            result = runner.invoke(weekly_app, ["--funds", "balanced"])

        assert result.exit_code == 0
        run_date = datetime.utcnow().strftime("%Y-%m-%d")
        out_dir = tmp_path / "weekly" / run_date
        assert (out_dir / "balanced_holdings.md").exists()
        assert (out_dir / "balanced_analytics.json").exists()
        assert (out_dir / "balanced_mandate.md").exists()
        assert (out_dir / "balanced_funnel.md").exists()
        assert (out_dir / "drift_report.md").exists()

    def test_funnel_error_exits_1(self, tmp_path, monkeypatch):
        import weekly_run as wr
        monkeypatch.setattr(wr, "_RUNS_DIR", tmp_path / "weekly")

        with patch("stockgrader.portfolios.construction.run_full_funnel",
                   side_effect=RuntimeError("API down")), \
             patch("stockgrader.config.get_config", return_value={}):
            result = runner.invoke(weekly_app, ["--funds", "balanced"])

        assert result.exit_code == 1

    def test_no_save_flag(self, tmp_path, monkeypatch):
        import weekly_run as wr
        monkeypatch.setattr(wr, "_RUNS_DIR", tmp_path / "weekly")

        mock_result = MagicMock()
        mock_result.holdings = []
        mock_result.analytics = None
        mock_result.mandate_check = None
        mock_result.funnel_stats = None

        with patch("stockgrader.portfolios.construction.run_full_funnel",
                   return_value=mock_result), \
             patch("stockgrader.config.get_config", return_value={}):
            result = runner.invoke(weekly_app, ["--funds", "balanced", "--no-save"])

        # No files should be written
        assert not (tmp_path / "weekly").exists() or \
               not any((tmp_path / "weekly").rglob("*.md"))
