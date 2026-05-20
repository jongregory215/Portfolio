"""
BaseEngine — shared interface for all three scoring engines.

Each engine receives a normalized data dict and returns a typed result.
Engines are stateless: same inputs → same output (required for backtesting).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseEngine(ABC):
    """Abstract scoring engine. Subclass for Fundamental, Technical, Quantitative."""

    def __init__(self, config: dict[str, Any]):
        self.config = config

    @abstractmethod
    def score(self, data: dict[str, Any]) -> Any:
        """
        Score the stock from normalized input data.

        Parameters
        ----------
        data:
            Normalized dict from the data layer. All engines receive the same
            top-level data dict; each engine consumes the keys it needs.

        Returns
        -------
        A typed result object (FundamentalResult, TechnicalResult, or
        QuantitativeResult as appropriate). Engines must never return None;
        use default/zero values and populate missing_fields instead.
        """

    # ── Shared utilities ──────────────────────────────────────

    @staticmethod
    def clamp(value: float, lo: float = 0.0, hi: float = 100.0) -> float:
        return max(lo, min(hi, value))

    @staticmethod
    def linear_score(
        value: float,
        lo: float,
        hi: float,
        invert: bool = False,
    ) -> float:
        """
        Map value linearly into [0, 100].

        value ≤ lo → 0  (or 100 if invert)
        value ≥ hi → 100 (or 0 if invert)
        """
        if hi == lo:
            return 50.0
        raw = (value - lo) / (hi - lo)
        raw = max(0.0, min(1.0, raw))
        return (1.0 - raw) * 100.0 if invert else raw * 100.0

    @staticmethod
    def percentile_score(value: float, peer_values: list[float]) -> float:
        """
        Score as the percentile rank of value within peer_values.
        Returns 0–100.
        """
        if not peer_values:
            return 50.0
        below = sum(1 for v in peer_values if v < value)
        return (below / len(peer_values)) * 100.0

    @staticmethod
    def weighted_average(scores: dict[str, float], weights: dict[str, float]) -> float:
        """
        Compute a weighted average of scores using the provided weights.

        Missing score keys are skipped and weights renormalized accordingly.
        """
        total_w = 0.0
        total_s = 0.0
        for key, w in weights.items():
            if key in scores and scores[key] is not None:
                total_w += w
                total_s += w * scores[key]
        if total_w == 0.0:
            return 50.0
        return total_s / total_w
