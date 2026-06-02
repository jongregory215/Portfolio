"""
SEC EDGAR EPS history provider via edgartools.

Provides up to 15+ years of annual diluted EPS from 10-K XBRL filings.
No API key required. Identity header (name + email) required by SEC.
"""
from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

_IDENTITY_SET = False


def _ensure_identity() -> None:
    global _IDENTITY_SET
    if _IDENTITY_SET:
        return
    try:
        from edgar import set_identity
        name  = os.environ.get("EDGAR_IDENTITY_NAME",  "Jonathan Gregory")
        email = os.environ.get("EDGAR_IDENTITY_EMAIL", "jmfgregory@gmail.com")
        set_identity(f"{name} {email}")
        _IDENTITY_SET = True
    except Exception as exc:
        logger.debug("edgar set_identity failed: %s", exc)


def fetch_annual_eps(ticker: str, years: int = 10) -> list[tuple[int, float]]:
    """
    Return up to `years` annual diluted EPS values from SEC EDGAR XBRL facts.

    Returns [(fiscal_year, eps), ...] newest-first, or [] on failure.
    Uses EarningsPerShareDiluted; falls back to EarningsPerShareBasic.
    """
    try:
        _ensure_identity()
        from edgar import Company
        company = Company(ticker)
        if company.not_found:
            logger.debug("EDGAR: company not found for %s", ticker)
            return []

        facts = company.get_facts()
        all_facts = facts.get_all_facts()

        result = _extract_annual_eps(all_facts, "EarningsPerShareDiluted")
        if not result:
            result = _extract_annual_eps(all_facts, "EarningsPerShareBasic")

        return result[:years]

    except ImportError:
        logger.debug("edgartools not installed — EDGAR EPS unavailable")
        return []
    except Exception as exc:
        logger.debug("EDGAR EPS fetch failed for %s: %s", ticker, exc)
        return []


def _extract_annual_eps(all_facts: list, concept_name: str) -> list[tuple[int, float]]:
    """
    Filter annual EPS facts, deduplicate by fiscal year (keep latest filing),
    and return as [(year, eps), ...] newest-first.
    """
    # Collect all annual (FY) facts for the concept
    candidates = [
        f for f in all_facts
        if concept_name in str(f.concept)
        and not f.is_dimensioned
        and f.fiscal_period == "FY"
        and f.period_end is not None
        and f.numeric_value is not None
    ]

    if not candidates:
        return []

    # Deduplicate by period_end year — keep the most recently filed version.
    # This ensures restated numbers from later 10-Ks take precedence.
    by_year: dict[int, Any] = {}
    for fact in candidates:
        year = fact.period_end.year
        existing = by_year.get(year)
        if existing is None:
            by_year[year] = fact
        else:
            existing_date = existing.filing_date or ""
            fact_date     = fact.filing_date or ""
            if fact_date > existing_date:
                by_year[year] = fact

    return [
        (year, float(by_year[year].numeric_value))
        for year in sorted(by_year.keys(), reverse=True)
    ]
