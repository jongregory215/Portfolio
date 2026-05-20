"""
Markdown reporter — human-readable per-stock analysis report.

Sections (spec §13.2):
  1. Header — grade, composite, confidence, price
  2. Price Ladder — five-zone table + sensitivity grid
  3. Drivers — top-3 positive and negative
  4. Engine Breakdown — weighted composite table
  5. Fundamental pillars + key signals
  6. Technical regime + indicators
  7. Quantitative factors + risk metrics
  8. Portfolio Sub-Grades — 5-row table
  9. Data Quality — completeness, missing fields, sources
"""
from __future__ import annotations

import uuid
from datetime import datetime
from pathlib import Path

from stockgrader.models import AnalysisResult, Grade, PriceLadder
from stockgrader.reporting.base import (
    GRADE_EMOJI, GRADE_STARS, _fmt_float, _fmt_pct, _fmt_pct_abs, _fmt_price, _na,
)


class MarkdownReporter:
    """Produce the full per-stock markdown report."""

    def render(self, result: AnalysisResult) -> str:
        parts = [
            self._header(result),
            self._price_ladder_section(result),
            self._drivers_section(result),
            self._engine_breakdown(result),
            self._fundamental_detail(result),
            self._technical_detail(result),
            self._quantitative_detail(result),
            self._portfolio_table(result),
            self._data_quality(result),
            self._footer(result),
        ]
        return "\n\n".join(p for p in parts if p)

    def save(
        self,
        result:   AnalysisResult,
        runs_dir: str | Path = "runs",
        filename: str | None = None,
    ) -> Path:
        base = Path(runs_dir)
        base.mkdir(parents=True, exist_ok=True)
        if filename is None:
            date_str = result.as_of.strftime("%Y-%m-%d")
            run_id   = result.run_id or uuid.uuid4().hex[:8]
            filename = f"{result.ticker}_{date_str}_{run_id}.md"
        path = base / filename
        path.write_text(self.render(result), encoding="utf-8")
        return path

    # ── Sections ──────────────────────────────────────────────

    def _header(self, r: AnalysisResult) -> str:
        g    = r.overall.grade
        emoji = GRADE_EMOJI.get(g, "")
        stars = GRADE_STARS.get(g, "")
        conf  = f"{r.overall.confidence * 100:.0f}%"
        cb    = " | ".join(r.overall.circuit_breakers) if r.overall.circuit_breakers else "None"
        return (
            f"# {r.ticker} — {emoji} {g.value.upper()} {stars}\n\n"
            f"> **Composite:** {r.overall.composite:.1f}/100 &nbsp;|&nbsp; "
            f"**Confidence:** {conf} &nbsp;|&nbsp; "
            f"**Price:** {_fmt_price(r.price)} &nbsp;|&nbsp; "
            f"**As of:** {r.as_of.strftime('%Y-%m-%d %H:%M UTC')}\n>\n"
            f"> **Circuit Breakers:** {cb}"
        )

    def _price_ladder_section(self, r: AnalysisResult) -> str:
        pl = r.price_ladder
        if pl is None:
            return "## Price Ladder\n\n*Price ladder unavailable — insufficient valuation data.*"

        price = r.price

        def _zone(label: str, cond: bool, extra: str = "") -> str:
            cursor = " **← current price**" if cond else ""
            return f"| {label} | {extra}{cursor} |"

        rows = [
            "| Zone | |",
            "|------|---|",
            _zone(f"⭐ **Gotta Have** ≤ {_fmt_price(pl.gotta_have_at)}",
                  price <= pl.gotta_have_at),
            _zone(f"✅ **Buy** ≤ {_fmt_price(pl.buy_at)}",
                  pl.gotta_have_at < price <= pl.buy_at),
            _zone(f"⚖️ **Hold** {_fmt_price(pl.hold_low)} – {_fmt_price(pl.hold_high)}",
                  pl.hold_low < price <= pl.hold_high),
            _zone(f"⚠️ **Sell** > {_fmt_price(pl.sell_above)}",
                  pl.sell_above < price <= pl.stay_away_above),
            _zone(f"🚫 **Stay Away** > {_fmt_price(pl.stay_away_above)}",
                  price > pl.stay_away_above),
        ]

        fv_line = (
            f"**Fair Value:** {_fmt_price(pl.fair_value)} &nbsp;|&nbsp; "
            f"**Upside to FV:** {_fmt_pct(pl.upside_to_fv_pct)} &nbsp;|&nbsp; "
            f"**FV range:** {_fmt_price(pl.fair_value_sensitivity.low)} – "
            f"{_fmt_price(pl.fair_value_sensitivity.high)}"
        )

        if pl.implied_growth_rate is not None:
            fv_line += f" &nbsp;|&nbsp; **Implied growth:** {_fmt_pct_abs(pl.implied_growth_rate)}"

        grid_section = self._sensitivity_grid(pl)

        return "## Price Ladder\n\n" + "\n".join(rows) + "\n\n" + fv_line + "\n\n" + grid_section

    def _sensitivity_grid(self, pl: PriceLadder) -> str:
        sg = pl.sensitivity_grid
        if not sg or not sg.cells:
            return ""

        g_deltas  = sg.growth_deltas  or [-0.02, 0.00, 0.02]
        dr_deltas = sg.discount_rate_deltas or [-0.01, 0.00, 0.01]

        header = "| Growth ↓ / Rate → | " + " | ".join(
            f"DR {d:+.0%}" for d in dr_deltas
        ) + " |"
        sep = "|---|" + "---|" * len(dr_deltas)

        rows = [header, sep]
        for dg in reversed(g_deltas):    # higher growth at the top
            cells = []
            for dr in dr_deltas:
                key = f"{dg:+.2f}/{dr:+.2f}"
                fv  = sg.cells.get(key)
                cells.append(_fmt_price(fv, decimals=0) if fv else "—")
            rows.append(f"| G {dg:+.0%} | " + " | ".join(cells) + " |")

        return "### Fair Value Sensitivity Grid\n\n" + "\n".join(rows)

    def _drivers_section(self, r: AnalysisResult) -> str:
        pos = "\n".join(f"1. {d}" for d in r.overall.drivers_positive) or "*(none)*"
        neg = "\n".join(f"1. {d}" for d in r.overall.drivers_negative) or "*(none)*"
        return (
            "## Key Drivers\n\n"
            "**Positive signals:**\n" + pos + "\n\n"
            "**Negative signals:**\n" + neg
        )

    def _engine_breakdown(self, r: AnalysisResult) -> str:
        e  = r.engines
        ow = r.overall.weights_used
        fw = ow.get("fundamental",  0.50)
        tw = ow.get("technical",    0.30)
        qw = ow.get("quantitative", 0.20)

        f_contrib = fw * e.fundamental.score
        t_contrib = tw * e.technical.score
        q_contrib = qw * e.quantitative.score

        return (
            "## Engine Breakdown\n\n"
            "| Engine | Score | Weight | Contribution |\n"
            "|--------|-------|--------|--------------|\n"
            f"| Fundamental  | {e.fundamental.score:.1f} | {fw:.0%} | {f_contrib:.1f} |\n"
            f"| Technical    | {e.technical.score:.1f} | {tw:.0%} | {t_contrib:.1f} |\n"
            f"| Quantitative | {e.quantitative.score:.1f} | {qw:.0%} | {q_contrib:.1f} |\n"
            f"| **Composite** | **{r.overall.composite:.1f}** | | |"
        )

    def _fundamental_detail(self, r: AnalysisResult) -> str:
        f  = r.engines.fundamental
        p  = f.pillars
        lines = [
            "## Fundamental Analysis\n",
            "| Pillar | Score |",
            "|--------|-------|",
            f"| Valuation | {p.valuation:.0f}/100 |",
            f"| Profitability | {p.profitability:.0f}/100 |",
            f"| Growth | {p.growth:.0f}/100 |",
            f"| Financial Health | {p.financial_health:.0f}/100 |",
            f"| Capital Allocation | {p.capital_allocation:.0f}/100 |",
        ]

        signals = []
        if f.roic_vs_wacc is not None:
            dir_w = "above" if f.roic_vs_wacc >= 0 else "below"
            signals.append(f"ROIC {f.roic_vs_wacc*100:+.1f}ppt {dir_w} WACC")
        if f.altman_z is not None:
            zone = "safe" if f.altman_z >= 3 else ("grey zone" if f.altman_z >= 1.8 else "**distress**")
            signals.append(f"Altman Z {f.altman_z:.2f} ({zone})")
        if f.piotroski_f is not None:
            signals.append(f"Piotroski F {f.piotroski_f}/9")
        if f.quality_of_earnings_flag:
            signals.append("⚠️ Quality-of-earnings flag (accruals elevated)")
        if f.margin_trajectory_3yr is not None:
            direction = "expanding" if f.margin_trajectory_3yr > 0 else "contracting"
            signals.append(f"Operating margin {direction} ({f.margin_trajectory_3yr:+.2f} ppt/yr)")

        if signals:
            lines.append("\n*" + " &nbsp;|&nbsp; ".join(signals) + "*")

        if f.missing_fields:
            lines.append(f"\n> Missing: {', '.join(f.missing_fields[:6])}")

        return "\n".join(lines)

    def _technical_detail(self, r: AnalysisResult) -> str:
        t  = r.engines.technical
        p  = t.pillars
        lines = [
            "## Technical Analysis\n",
            f"**Regime:** `{t.regime}` &nbsp;|&nbsp; "
            f"Trend {p.trend:.0f}/100 &nbsp;|&nbsp; "
            f"Momentum {p.momentum:.0f}/100 &nbsp;|&nbsp; "
            f"Vol/Structure {p.volume_structure:.0f}/100",
        ]

        key_levels = []
        if t.nearest_support:
            key_levels.append(f"Support: {_fmt_price(t.nearest_support)}")
        if t.nearest_resistance:
            key_levels.append(f"Resistance: {_fmt_price(t.nearest_resistance)}")
        if t.week_52_high:
            key_levels.append(f"52w High: {_fmt_price(t.week_52_high)}")
        if t.week_52_low:
            key_levels.append(f"52w Low: {_fmt_price(t.week_52_low)}")
        if key_levels:
            lines.append(" &nbsp;|&nbsp; ".join(key_levels))

        # Technical details from pillars
        td = p.details
        trend_d = td.get("trend", {})
        mom_d   = td.get("momentum", {})

        detail_items = []
        for k, label in [("pct_above_sma50", "vs SMA50"), ("pct_above_sma200", "vs SMA200"),
                          ("adx", "ADX"), ("rsi", "RSI"), ("macd_hist", "MACD hist"),
                          ("roc_60_pct", "ROC(60d)")]:
            for d in [trend_d, mom_d]:
                v = d.get(k)
                if v is not None:
                    val = f"{v:+.1f}%" if "pct" in k else f"{v:.2f}"
                    detail_items.append(f"{label}: {val}")

        if detail_items:
            lines.append(" | ".join(detail_items[:6]))

        return "\n".join(lines)

    def _quantitative_detail(self, r: AnalysisResult) -> str:
        q  = r.engines.quantitative
        f  = q.factors
        rm = q.risk_metrics

        factor_rows = []
        for name, z_attr, p_attr in [
            ("Value",       "value_z",          "value_pct"),
            ("Quality",     "quality_z",        "quality_pct"),
            ("Momentum",    "momentum_z",       "momentum_pct"),
            ("Size",        "size_z",           "size_pct"),
            ("Low-Vol",     "low_volatility_z", "low_volatility_pct"),
        ]:
            z   = getattr(f, z_attr, None)
            pct = getattr(f, p_attr, None)
            if z is not None or pct is not None:
                factor_rows.append(
                    f"| {name} | {_fmt_float(z, 2)} | "
                    f"{'—' if pct is None else f'{pct:.0f}th'} |"
                )

        factor_table = ""
        if factor_rows:
            factor_table = (
                "| Factor | Z-Score | Percentile |\n"
                "|--------|---------|------------|\n"
                + "\n".join(factor_rows)
            )

        risk_items = []
        if rm.sharpe_1yr is not None:
            risk_items.append(f"Sharpe(1yr): {rm.sharpe_1yr:.2f}")
        if rm.sortino_1yr is not None:
            risk_items.append(f"Sortino(1yr): {rm.sortino_1yr:.2f}")
        if rm.max_drawdown_3yr is not None:
            risk_items.append(f"MaxDD(3yr): {rm.max_drawdown_3yr*100:.1f}%")
        if rm.beta_1yr is not None:
            risk_items.append(f"Beta: {rm.beta_1yr:.2f}")
        if rm.realized_vol_1yr is not None:
            risk_items.append(f"Vol(1yr): {rm.realized_vol_1yr*100:.1f}%")

        risk_line = " &nbsp;|&nbsp; ".join(risk_items) if risk_items else ""

        ff = q.ff_regression
        ff_line = ""
        if ff:
            ff_line = (
                f"\n**FF{ff.model[-1]}-factor regression** — "
                f"Alpha: {_fmt_pct(ff.alpha_annualized)} "
                f"(t={_fmt_float(ff.alpha_t_stat, 2)}) &nbsp;|&nbsp; "
                f"R²: {_fmt_float(ff.r_squared, 3)}"
            )

        return (
            "## Quantitative Analysis\n\n"
            + factor_table
            + ("\n\n" if risk_line else "")
            + risk_line
            + ff_line
        )

    def _portfolio_table(self, r: AnalysisResult) -> str:
        pg = r.portfolios
        rows = ["| Portfolio | Grade | Composite | Failed Gates | Rationale |",
                "|-----------|-------|-----------|--------------|-----------|"]

        for label, attr in [
            ("Very Conservative", "very_conservative"),
            ("Conservative",      "conservative"),
            ("Balanced",          "balanced"),
            ("Aggressive",        "aggressive"),
            ("Very Aggressive",   "very_aggressive"),
        ]:
            sub = getattr(pg, attr)
            emoji  = GRADE_EMOJI.get(sub.grade, "")
            comp   = f"{sub.composite:.1f}" if sub.composite > 0 else "—"
            failed = ", ".join(sub.failed_gates[:3]) if sub.failed_gates else "—"
            rat    = sub.rationale[:80] + ("…" if len(sub.rationale) > 80 else "")
            rows.append(f"| {label} | {emoji} {sub.grade.value} | {comp} | {failed} | {rat} |")

        return "## Portfolio Sub-Grades\n\n" + "\n".join(rows)

    def _data_quality(self, r: AnalysisResult) -> str:
        dq    = r.data_quality
        comp  = f"{dq.data_completeness * 100:.1f}%"
        miss  = ", ".join(dq.missing_fields[:8]) if dq.missing_fields else "None"
        warns = "\n".join(f"- {w}" for w in dq.warnings[:5]) if dq.warnings else "None"

        sources_str = ""
        if dq.sources:
            unique_sources = sorted(set(dq.sources.values()))
            sources_str = f"**Data sources:** {', '.join(unique_sources)}"

        return (
            "## Data Quality\n\n"
            f"**Completeness:** {comp} &nbsp;|&nbsp; "
            f"**Missing fields:** {miss}\n\n"
            + (sources_str + "\n\n" if sources_str else "")
            + ("**Warnings:**\n" + warns if dq.warnings else "")
        )

    def _footer(self, r: AnalysisResult) -> str:
        return (
            "---\n\n"
            f"*Run ID: `{r.run_id}` &nbsp;|&nbsp; "
            f"Config hash: `{r.config_hash}` &nbsp;|&nbsp; "
            f"Generated: {r.as_of.strftime('%Y-%m-%d %H:%M UTC')}*\n\n"
            "*Grades are decision-support tools, not financial advice. "
            "Past patterns do not guarantee future results.*"
        )
