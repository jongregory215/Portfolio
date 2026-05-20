"""Shared formatting utilities for all reporters."""
from __future__ import annotations

from stockgrader.models import Grade

# ── Grade presentation ────────────────────────────────────────

GRADE_EMOJI = {
    Grade.STAY_AWAY:  "🚫",
    Grade.SELL:       "⚠️",
    Grade.HOLD:       "⚖️",
    Grade.BUY:        "✅",
    Grade.GOTTA_HAVE: "⭐",
}

GRADE_RICH_COLOR = {
    Grade.STAY_AWAY:  "bright_red",
    Grade.SELL:       "red",
    Grade.HOLD:       "yellow",
    Grade.BUY:        "green",
    Grade.GOTTA_HAVE: "bright_green",
}

GRADE_STARS = {
    Grade.STAY_AWAY:  "★",
    Grade.SELL:       "★★",
    Grade.HOLD:       "★★★",
    Grade.BUY:        "★★★★",
    Grade.GOTTA_HAVE: "★★★★★",
}


def grade_label(grade: Grade, short: bool = False) -> str:
    if short:
        return {
            Grade.STAY_AWAY: "SA", Grade.SELL: "SELL",
            Grade.HOLD: "HOLD", Grade.BUY: "BUY",
            Grade.GOTTA_HAVE: "GH",
        }[grade]
    return grade.value


# ── Number formatting ─────────────────────────────────────────

def _fmt_price(v: float | None, decimals: int = 2) -> str:
    if v is None:
        return "—"
    return f"${v:,.{decimals}f}"


def _fmt_pct(v: float | None, decimals: int = 1) -> str:
    """Format as a percentage string (input is a decimal fraction)."""
    if v is None:
        return "—"
    return f"{v * 100:+.{decimals}f}%"


def _fmt_pct_abs(v: float | None, decimals: int = 1) -> str:
    """Format as percentage without sign."""
    if v is None:
        return "—"
    return f"{v * 100:.{decimals}f}%"


def _fmt_float(v: float | None, decimals: int = 2, suffix: str = "") -> str:
    if v is None:
        return "—"
    return f"{v:.{decimals}f}{suffix}"


def _na(v: object | None, fmt=str) -> str:
    return fmt(v) if v is not None else "—"
