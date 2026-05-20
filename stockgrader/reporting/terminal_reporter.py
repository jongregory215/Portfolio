"""
Terminal reporter — color-coded summary using the rich library (spec §13.3).

Outputs:
  1. Compact single-line alert (used by daily_run.py)
  2. Full summary panel with price ladder + portfolio grade row
"""
from __future__ import annotations

from typing import Any

from stockgrader.models import AnalysisResult, Grade
from stockgrader.reporting.base import (
    GRADE_EMOJI, GRADE_RICH_COLOR, GRADE_STARS,
    _fmt_float, _fmt_pct, _fmt_price,
    grade_label,
)

try:
    from rich.console import Console
    from rich.panel   import Panel
    from rich.table   import Table
    from rich.text    import Text
    _RICH = True
except ImportError:
    _RICH = False


def _get_console() -> Any:
    if not _RICH:
        raise ImportError("rich is required for terminal output: pip install rich")
    return Console()


class TerminalReporter:
    """Render a color-coded analysis summary to the terminal."""

    def __init__(self, console: Any | None = None):
        self._console = console   # injected; created lazily if None

    @property
    def console(self) -> Any:
        if self._console is None:
            self._console = _get_console()
        return self._console

    # ── Public API ─────────────────────────────────────────────

    def print_full(self, result: AnalysisResult) -> None:
        """Print the full formatted summary panel to the terminal."""
        if not _RICH:
            print(self.render_plain(result))
            return
        self.console.print(self._full_panel(result))

    def print_compact(self, result: AnalysisResult) -> None:
        """Print a single compact line (for watchlist alert lists)."""
        if not _RICH:
            print(self.compact_line(result))
            return
        self.console.print(self._compact_text(result))

    def render_plain(self, result: AnalysisResult) -> str:
        """Plain-text version (no rich markup) for piping / logging."""
        g = result.overall.grade
        portfolio_row = " | ".join(
            f"{lbl}: {grade_label(getattr(result.portfolios, attr).grade, short=True)}"
            for lbl, attr in [
                ("VC", "very_conservative"), ("C", "conservative"),
                ("B", "balanced"), ("A", "aggressive"), ("VA", "very_aggressive"),
            ]
        )
        ladder_pos = _ladder_position(result)
        return (
            f"{result.ticker:<6} {g.value:<12} {result.overall.composite:.1f}/100  "
            f"Price {_fmt_price(result.price)}  FV {_fmt_price(result.price_ladder.fair_value)}  "
            f"[{ladder_pos}]\n"
            f"  Engines: F={result.overall.fundamental_score:.0f} "
            f"T={result.overall.technical_score:.0f} "
            f"Q={result.overall.quantitative_score:.0f}\n"
            f"  {portfolio_row}"
        )

    def compact_line(self, result: AnalysisResult) -> str:
        """One-line plain text (no colors)."""
        g = result.overall.grade
        return (
            f"{result.ticker:<6} {grade_label(g, short=True):<5} "
            f"{result.overall.composite:.1f}/100  "
            f"${result.price:.2f}  "
            f"[{_ladder_position(result)}]"
        )

    # ── Rich rendering ─────────────────────────────────────────

    def _full_panel(self, result: AnalysisResult) -> Any:
        from rich.columns import Columns

        grade = result.overall.grade
        color = GRADE_RICH_COLOR[grade]
        emoji = GRADE_EMOJI[grade]
        stars = GRADE_STARS[grade]

        # ── Top line ──────────────────────────────────────────
        header = Text()
        header.append(f"{result.ticker}  ", style="bold white")
        header.append(f"{emoji} {grade.value} {stars}", style=f"bold {color}")
        header.append(f"  {result.overall.composite:.1f}/100", style="bold")
        header.append(f"  confidence {result.overall.confidence*100:.0f}%", style="dim")

        # ── Price ladder ──────────────────────────────────────
        ladder_tbl = _ladder_table(result)

        # ── Engine row ────────────────────────────────────────
        engine_line = Text(
            f"F={result.overall.fundamental_score:.0f}  "
            f"T={result.overall.technical_score:.0f}  "
            f"Q={result.overall.quantitative_score:.0f}",
            style="dim"
        )

        # ── Portfolio row ─────────────────────────────────────
        portfolio_tbl = _portfolio_row_table(result)

        # ── Circuit breakers ──────────────────────────────────
        cb_line: Any = None
        if result.overall.circuit_breakers:
            cb_line = Text("⚡ " + "  |  ".join(result.overall.circuit_breakers),
                           style="bold red")

        # ── Drivers ───────────────────────────────────────────
        pos_text = Text("\n".join(f"  + {d}" for d in result.overall.drivers_positive[:3]),
                        style="green")
        neg_text = Text("\n".join(f"  - {d}" for d in result.overall.drivers_negative[:3]),
                        style="red")

        from rich.console import Group
        content = Group(
            header, Text(""),
            ladder_tbl,
            Text(""),
            engine_line,
            Text(""),
            portfolio_tbl,
            *([Text(""), cb_line] if cb_line else []),
            Text(""),
            Text("Drivers:", style="bold"),
            pos_text,
            neg_text,
        )

        return Panel(
            content,
            title=f"[bold]{result.ticker} Analysis[/bold]",
            subtitle=f"[dim]{result.as_of.strftime('%Y-%m-%d %H:%M UTC')}[/dim]",
            border_style=color,
        )

    def _compact_text(self, result: AnalysisResult) -> Any:
        grade = result.overall.grade
        color = GRADE_RICH_COLOR[grade]
        emoji = GRADE_EMOJI[grade]
        t     = Text()
        t.append(f"{result.ticker:<6}", style="bold white")
        t.append(f" {emoji} {grade_label(grade, short=True):<5}", style=f"bold {color}")
        t.append(f" {result.overall.composite:.1f}/100", style="bold")
        t.append(f"  ${result.price:.2f}", style="white")
        t.append(f"  [{_ladder_position(result)}]", style="dim")
        return t


# ── Rich table helpers ────────────────────────────────────────

def _ladder_table(result: AnalysisResult) -> Any:
    if not _RICH:
        return None
    from rich.table import Table

    pl    = result.price_ladder
    price = result.price

    tbl = Table(show_header=False, box=None, padding=(0, 1))
    tbl.add_column("Zone",  style="dim")
    tbl.add_column("Level", justify="right")
    tbl.add_column("",      justify="left")

    def _row(emoji, label, price_str, active):
        arrow = "◀ current" if active else ""
        style = "bold" if active else "dim"
        tbl.add_row(
            Text(f"{emoji} {label}", style=style),
            Text(price_str, style=style),
            Text(arrow, style=f"bold {GRADE_RICH_COLOR[_zone_grade(label)]}" if active else "dim"),
        )

    _row("⭐", "Gotta Have ≤", _fmt_price(pl.gotta_have_at),
         price is not None and price <= pl.gotta_have_at)
    _row("✅", "Buy       ≤", _fmt_price(pl.buy_at),
         price is not None and pl.gotta_have_at < price <= pl.buy_at)
    _row("⚖️", "Hold       ", f"{_fmt_price(pl.hold_low)} – {_fmt_price(pl.hold_high)}",
         price is not None and pl.hold_low < price <= pl.hold_high)
    _row("⚠️", "Sell      >", _fmt_price(pl.sell_above),
         price is not None and pl.sell_above < price <= pl.stay_away_above)
    _row("🚫", "Stay Away >", _fmt_price(pl.stay_away_above),
         price is not None and price > pl.stay_away_above)

    tbl.add_row(
        Text("Fair Value", style="dim"),
        Text(_fmt_price(pl.fair_value), style="bold"),
        Text(f"upside {_fmt_pct(pl.upside_to_fv_pct)}", style="dim"),
    )
    return tbl


def _portfolio_row_table(result: AnalysisResult) -> Any:
    if not _RICH:
        return None
    from rich.table import Table, box as rich_box

    tbl = Table(show_header=True, box=rich_box.SIMPLE_HEAD, padding=(0, 1))
    for label in ["VC", "CON", "BAL", "AGG", "VA"]:
        tbl.add_column(label, justify="center")

    pg = result.portfolios
    cells = []
    for attr in ["very_conservative", "conservative", "balanced",
                 "aggressive", "very_aggressive"]:
        sub   = getattr(pg, attr)
        g     = sub.grade
        color = GRADE_RICH_COLOR[g]
        emoji = GRADE_EMOJI[g]
        comp  = f"\n{sub.composite:.0f}" if sub.composite > 0 else ""
        cells.append(Text(f"{emoji} {grade_label(g, short=True)}{comp}", style=color))

    tbl.add_row(*cells)
    return tbl


# ── Helpers ───────────────────────────────────────────────────

def _ladder_position(result: AnalysisResult) -> str:
    if result.price_ladder is None:
        return "no ladder"
    pl = result.price_ladder
    p  = result.price
    if p is None:
        return "no price"
    if p <= pl.gotta_have_at:    return "Gotta Have zone"
    if p <= pl.buy_at:           return "Buy zone"
    if p <= pl.hold_high:        return "Hold zone"
    if p <= pl.stay_away_above:  return "Sell zone"
    return "Stay Away zone"


def _zone_grade(label: str) -> Grade:
    mapping = {
        "Gotta Have": Grade.GOTTA_HAVE,
        "Buy":        Grade.BUY,
        "Hold":       Grade.HOLD,
        "Sell":       Grade.SELL,
        "Stay Away":  Grade.STAY_AWAY,
    }
    for k, v in mapping.items():
        if k in label:
            return v
    return Grade.HOLD
