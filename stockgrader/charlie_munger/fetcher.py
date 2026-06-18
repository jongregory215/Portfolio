"""
Charlie Munger-specific data fetcher.

Thin wrapper around BuffettFetcher — all required fields are already fetched
there. The Munger criteria apply stricter thresholds and add a capex-intensity
check (capex ≈ operating_cf − free_cf, both already in the Buffett data dict).
"""
from __future__ import annotations

from stockgrader.warren_buffett.fetcher import BuffettFetcher


class MungerFetcher(BuffettFetcher):
    """Fetch raw data for a single ticker for Munger-style evaluation."""
    # No additional fields needed beyond BuffettFetcher.
