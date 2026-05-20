"""
Fundamental Engine — scores a stock across five pillars.

Scoring philosophy (per spec §4):
  - Valuation & profitability ratios: peer-relative percentile rank
    (requires ≥5 peers; falls back to absolute thresholds when unavailable)
  - Solvency, liquidity, distress: absolute thresholds with documented danger zones
  - ROIC vs. WACC spread: highest-weight single signal (best moat proxy)
  - Missing fields: reduce effective pillar weight proportionally; never silently zero

Pillar weights (default, overridden by config):
  Valuation 30%, Profitability 25%, Growth 20%, Health 15%, Capital Allocation 10%
"""
from __future__ import annotations

import logging
from typing import Any

from stockgrader.engines.base import BaseEngine
from stockgrader.models import FundamentalPillars, FundamentalResult

logger = logging.getLogger(__name__)

MIN_PEERS = 5   # minimum peer count for peer-relative percentile scoring


# ──────────────────────────────────────────────────────────────
# Absolute breakpoint tables
# Format: [(threshold, score), ...] — see _hi_score / _lo_score below
# ──────────────────────────────────────────────────────────────

# "Higher is better" — sorted descending; first threshold value >= wins
_CURRENT_RATIO   = [(2.0,100),(1.5,82),(1.0,62),(0.7,38),(0.0,15)]
_QUICK_RATIO     = [(1.5,100),(1.0,82),(0.7,62),(0.5,42),(0.0,18)]
_INTEREST_COV    = [(15,100),(8,85),(5,70),(3,52),(1.5,32),(0,12)]
_ALTMAN_Z        = [(3.0,100),(2.5,82),(1.8,58),(1.2,28),(0,8)]
_PIOTROSKI_F     = [(8,100),(6,80),(4,60),(2,38),(0,12)]
_REVENUE_CAGR    = [(0.20,100),(0.15,90),(0.10,80),(0.07,70),(0.05,60),
                    (0.03,50),(0.01,40),(0.00,30),(-999,10)]
_EPS_CAGR        = [(0.20,100),(0.15,90),(0.10,80),(0.07,70),(0.05,60),
                    (0.03,50),(0.01,40),(0.00,30),(-999,10)]
_FWD_GROWTH      = [(0.20,100),(0.15,90),(0.10,80),(0.05,65),(0.03,52),
                    (0.00,38),(-999,15)]
_ROIC_WACC       = [(0.15,100),(0.10,90),(0.05,75),(0.02,60),(0.00,45),
                    (-0.02,30),(-0.05,15),(-999,5)]
_OP_MARGIN       = [(0.30,100),(0.20,85),(0.15,70),(0.10,55),(0.05,40),
                    (0.00,25),(-999,10)]
_GROSS_MARGIN    = [(0.60,100),(0.45,85),(0.35,70),(0.25,55),(0.15,40),
                    (0.00,20),(-999,5)]
_NET_MARGIN      = [(0.20,100),(0.15,85),(0.10,70),(0.05,55),(0.02,40),
                    (0.00,22),(-999,5)]
_ROE_ABS         = [(0.25,100),(0.18,82),(0.12,65),(0.08,50),(0.04,35),
                    (0.00,18),(-999,5)]
_ROA_ABS         = [(0.15,100),(0.10,82),(0.07,65),(0.05,50),(0.02,35),
                    (0.00,18),(-999,5)]
_FCF_CONV        = [(1.5,100),(1.0,85),(0.7,65),(0.4,42),(0.0,22),(-999,5)]
_FCF_YIELD       = [(0.08,100),(0.05,85),(0.03,70),(0.02,55),(0.01,40),
                    (0.00,22),(-999,5)]
_MARGIN_TRAJ     = [(2.0,100),(0.5,80),(0.0,60),(-0.5,40),(-2.0,20),(-999,5)]

# "Lower is better" — sorted ascending; first threshold value <= wins
_DEBT_EQUITY     = [(0.3,100),(0.5,85),(0.8,70),(1.2,55),(2.0,40),(3.0,25),(999,10)]
_PE_ABS          = [(10,100),(15,85),(20,72),(25,58),(30,44),(40,28),(999,12)]
_PB_ABS          = [(1.0,100),(2.0,82),(4.0,62),(8.0,40),(999,18)]
_PS_ABS          = [(1.0,100),(2.0,82),(4.0,62),(8.0,42),(999,18)]
_EVEBITDA_ABS    = [(6,100),(10,82),(15,62),(20,44),(25,28),(999,12)]
_PEG_ABS         = [(0.5,100),(1.0,85),(1.5,70),(2.0,55),(3.0,35),(999,15)]

# Payout ratio: moderate is best; extremes penalised (handled inline)


class FundamentalEngine(BaseEngine):
    """Score a stock's fundamentals across five weighted pillars."""

    # ── Public interface ─────────────────────────────────────

    def score(self, data: dict[str, Any]) -> FundamentalResult:
        cfg            = self.config.get("fundamental", {})
        pillar_weights = cfg.get("pillar_weights", {
            "valuation":          0.30,
            "profitability":      0.25,
            "growth":             0.20,
            "financial_health":   0.15,
            "capital_allocation": 0.10,
        })

        peer_metrics = data.get("peer_metrics", {})
        missing: list[str] = []

        # ── Score each pillar ────────────────────────────────
        val_score,  val_det,  val_miss  = self._score_valuation(data, peer_metrics)
        prof_score, prof_det, prof_miss = self._score_profitability(data, peer_metrics)
        grow_score, grow_det, grow_miss = self._score_growth(data, peer_metrics)
        hlth_score, hlth_det, hlth_miss = self._score_health(data)
        cap_score,  cap_det,  cap_miss  = self._score_capital_allocation(data)

        missing = val_miss + prof_miss + grow_miss + hlth_miss + cap_miss

        # ── Effective pillar weights (reduce for missing data) ─
        pillar_scores = {
            "valuation":          val_score,
            "profitability":      prof_score,
            "growth":             grow_score,
            "financial_health":   hlth_score,
            "capital_allocation": cap_score,
        }
        composite = self.weighted_average(
            {k: v for k, v in pillar_scores.items() if v is not None},
            pillar_weights,
        )

        # ── Explainability drivers ───────────────────────────
        all_components = {**val_det, **prof_det, **grow_det, **hlth_det, **cap_det}
        drivers_pos, drivers_neg = self._top_drivers(all_components)

        # ── Assemble result ──────────────────────────────────
        pillars = FundamentalPillars(
            valuation          = val_score  if val_score  is not None else 50.0,
            profitability      = prof_score if prof_score is not None else 50.0,
            growth             = grow_score if grow_score is not None else 50.0,
            financial_health   = hlth_score if hlth_score is not None else 50.0,
            capital_allocation = cap_score  if cap_score  is not None else 50.0,
            details = {
                "valuation":          val_det,
                "profitability":      prof_det,
                "growth":             grow_det,
                "financial_health":   hlth_det,
                "capital_allocation": cap_det,
            },
        )

        return FundamentalResult(
            score                  = self.clamp(composite),
            pillars                = pillars,
            roic_vs_wacc           = data.get("roic_vs_wacc"),
            wacc                   = data.get("wacc"),
            roic                   = data.get("roic"),
            altman_z               = data.get("altman_z"),
            piotroski_f            = data.get("piotroski_f"),
            margin_trajectory_3yr  = data.get("margin_trajectory"),
            quality_of_earnings_flag = bool(data.get("quality_of_earnings_flag", False)),
            peer_count             = len(peer_metrics),
            missing_fields         = sorted(set(missing)),
        )

    # ── Pillar: Valuation ────────────────────────────────────

    def _score_valuation(
        self, data: dict, peer_metrics: dict
    ) -> tuple[float, dict, list[str]]:
        scores: dict[str, float] = {}
        details: dict[str, Any]  = {}
        missing: list[str]       = []

        # P/E trailing
        pe = _f(data.get("pe_trailing"))
        if pe is not None and pe > 0:
            peer_pes = _peer_vals(peer_metrics, "priceEarningsRatioTTM", positive_only=True)
            if len(peer_pes) >= MIN_PEERS:
                scores["pe_trailing"] = self.percentile_score(pe, peer_pes, inverted=True)
            else:
                scores["pe_trailing"] = _lo_score(pe, _PE_ABS)
            details["pe_trailing"] = round(pe, 1)
        else:
            missing.append("pe_trailing")

        # P/E forward
        pe_fwd = _f(data.get("pe_forward"))
        if pe_fwd is not None and pe_fwd > 0:
            scores["pe_forward"] = _lo_score(pe_fwd, _PE_ABS)
            details["pe_forward"] = round(pe_fwd, 1)
        else:
            missing.append("pe_forward")

        # P/B
        pb = _f(data.get("pb"))
        if pb is not None and pb > 0:
            peer_pbs = _peer_vals(peer_metrics, "priceToBookRatioTTM", positive_only=True)
            if len(peer_pbs) >= MIN_PEERS:
                scores["pb"] = self.percentile_score(pb, peer_pbs, inverted=True)
            else:
                scores["pb"] = _lo_score(pb, _PB_ABS)
            details["pb"] = round(pb, 1)
        else:
            missing.append("pb")

        # P/S
        ps = _f(data.get("ps"))
        if ps is not None and ps > 0:
            peer_ps = _peer_vals(peer_metrics, "priceToSalesRatioTTM", positive_only=True)
            if len(peer_ps) >= MIN_PEERS:
                scores["ps"] = self.percentile_score(ps, peer_ps, inverted=True)
            else:
                scores["ps"] = _lo_score(ps, _PS_ABS)
            details["ps"] = round(ps, 1)
        else:
            missing.append("ps")

        # EV/EBITDA
        ev = _f(data.get("ev_ebitda"))
        if ev is not None and ev > 0:
            peer_ev = _peer_vals(peer_metrics, "enterpriseValueOverEBITDATTM", positive_only=True)
            if len(peer_ev) >= MIN_PEERS:
                scores["ev_ebitda"] = self.percentile_score(ev, peer_ev, inverted=True)
            else:
                scores["ev_ebitda"] = _lo_score(ev, _EVEBITDA_ABS)
            details["ev_ebitda"] = round(ev, 1)
        else:
            missing.append("ev_ebitda")

        # PEG
        peg = _f(data.get("peg"))
        if peg is not None and 0 < peg < 10:
            scores["peg"] = _lo_score(peg, _PEG_ABS)
            details["peg"] = round(peg, 2)
        else:
            missing.append("peg")

        composite = self.weighted_average(scores, {k: 1.0 for k in scores}) if scores else 50.0
        details["_label"] = f"Valuation score {composite:.0f}/100 | "
        details["_label"] += " | ".join(f"{k}={v:.0f}" for k, v in scores.items())
        return composite, details, missing

    # ── Pillar: Profitability ────────────────────────────────

    def _score_profitability(
        self, data: dict, peer_metrics: dict
    ) -> tuple[float, dict, list[str]]:
        scores:  dict[str, float] = {}
        details: dict[str, Any]   = {}
        missing: list[str]        = []

        # Gross margin
        gm = _f(data.get("gross_margin"))
        if gm is not None:
            peer_gm = _peer_vals(peer_metrics, "grossProfitMarginTTM")
            if len(peer_gm) >= MIN_PEERS:
                scores["gross_margin"] = self.percentile_score(gm, peer_gm)
            else:
                scores["gross_margin"] = _hi_score(gm, _GROSS_MARGIN)
            details["gross_margin_pct"] = round(gm * 100, 1)
        else:
            missing.append("gross_margin")

        # Operating margin
        om = _f(data.get("operating_margin"))
        if om is not None:
            peer_om = _peer_vals(peer_metrics, "operatingProfitMarginTTM")
            if len(peer_om) >= MIN_PEERS:
                scores["operating_margin"] = self.percentile_score(om, peer_om)
            else:
                scores["operating_margin"] = _hi_score(om, _OP_MARGIN)
            details["operating_margin_pct"] = round(om * 100, 1)
        else:
            missing.append("operating_margin")

        # Net margin
        nm = _f(data.get("net_margin"))
        if nm is not None:
            scores["net_margin"] = _hi_score(nm, _NET_MARGIN)
            details["net_margin_pct"] = round(nm * 100, 1)
        else:
            missing.append("net_margin")

        # ROE
        roe = _f(data.get("roe"))
        if roe is not None:
            # Cap extreme ROE (negative equity companies like AAPL)
            roe_capped = min(roe, 3.0)
            if roe_capped > 0:
                peer_roe = _peer_vals(peer_metrics, "returnOnEquityTTM")
                if len(peer_roe) >= MIN_PEERS:
                    scores["roe"] = self.percentile_score(roe_capped, peer_roe)
                else:
                    scores["roe"] = _hi_score(roe_capped, _ROE_ABS)
                details["roe_pct"] = round(roe * 100, 1)
            else:
                scores["roe"] = 15.0   # negative ROE → poor signal
                details["roe_pct"] = round(roe * 100, 1)
        else:
            missing.append("roe")

        # ROA
        roa = _f(data.get("roa"))
        if roa is not None:
            if roa > 0:
                peer_roa = _peer_vals(peer_metrics, "returnOnAssetsTTM")
                if len(peer_roa) >= MIN_PEERS:
                    scores["roa"] = self.percentile_score(roa, peer_roa)
                else:
                    scores["roa"] = _hi_score(roa, _ROA_ABS)
                details["roa_pct"] = round(roa * 100, 1)
            else:
                scores["roa"] = 10.0
                details["roa_pct"] = round(roa * 100, 1)
        else:
            missing.append("roa")

        # ROIC vs WACC spread — most important signal (double weight)
        spread = _f(data.get("roic_vs_wacc"))
        if spread is not None:
            scores["roic_wacc_spread"]  = _hi_score(spread, _ROIC_WACC)
            scores["roic_wacc_spread2"] = scores["roic_wacc_spread"]  # double weight
            details["roic_vs_wacc_ppts"] = round(spread * 100, 1)
        else:
            missing.append("roic_vs_wacc")

        # Margin trajectory
        mt = _f(data.get("margin_trajectory"))
        if mt is not None:
            scores["margin_trajectory"] = _hi_score(mt, _MARGIN_TRAJ)
            details["margin_trajectory_ppts_yr"] = round(mt, 2)
        else:
            missing.append("margin_trajectory")

        weights = {k: 1.0 for k in scores}
        composite = self.weighted_average(scores, weights) if scores else 50.0
        details["_label"] = f"Profitability score {composite:.0f}/100"
        return composite, details, missing

    # ── Pillar: Growth ───────────────────────────────────────

    def _score_growth(
        self, data: dict, peer_metrics: dict
    ) -> tuple[float, dict, list[str]]:
        scores:  dict[str, float] = {}
        details: dict[str, Any]   = {}
        missing: list[str]        = []

        # Revenue CAGR 3yr (weight 2x — more reliable than 5yr)
        rc3 = _f(data.get("revenue_cagr_3yr"))
        if rc3 is not None:
            scores["rev_cagr_3yr"]  = _hi_score(rc3, _REVENUE_CAGR)
            scores["rev_cagr_3yr2"] = scores["rev_cagr_3yr"]
            details["revenue_cagr_3yr_pct"] = round(rc3 * 100, 1)
        else:
            missing.append("revenue_cagr_3yr")

        # Revenue CAGR 5yr
        rc5 = _f(data.get("revenue_cagr_5yr"))
        if rc5 is not None:
            scores["rev_cagr_5yr"] = _hi_score(rc5, _REVENUE_CAGR)
            details["revenue_cagr_5yr_pct"] = round(rc5 * 100, 1)
        else:
            missing.append("revenue_cagr_5yr")

        # EPS CAGR 3yr
        ec3 = _f(data.get("eps_cagr_3yr"))
        if ec3 is not None:
            scores["eps_cagr_3yr"]  = _hi_score(ec3, _EPS_CAGR)
            scores["eps_cagr_3yr2"] = scores["eps_cagr_3yr"]
            details["eps_cagr_3yr_pct"] = round(ec3 * 100, 1)
        else:
            missing.append("eps_cagr_3yr")

        # EPS CAGR 5yr
        ec5 = _f(data.get("eps_cagr_5yr"))
        if ec5 is not None:
            scores["eps_cagr_5yr"] = _hi_score(ec5, _EPS_CAGR)
            details["eps_cagr_5yr_pct"] = round(ec5 * 100, 1)
        else:
            missing.append("eps_cagr_5yr")

        # Forward EPS growth estimate
        fg = _f(data.get("forward_eps_growth"))
        if fg is not None:
            scores["fwd_eps_growth"] = _hi_score(fg, _FWD_GROWTH)
            details["forward_eps_growth_pct"] = round(fg * 100, 1)
        else:
            missing.append("forward_eps_growth")

        # Forward revenue growth
        fr = _f(data.get("forward_revenue_growth"))
        if fr is not None:
            scores["fwd_rev_growth"] = _hi_score(fr, _FWD_GROWTH)
            details["forward_rev_growth_pct"] = round(fr * 100, 1)
        else:
            missing.append("forward_revenue_growth")

        # Growth durability bonus: EPS CAGR > Revenue CAGR → margin expansion
        if ec3 is not None and rc3 is not None and ec3 > rc3:
            scores["durability_bonus"] = 75.0
            details["durability_note"] = "EPS growing faster than revenue → operating leverage"
        elif ec3 is not None and rc3 is not None and ec3 < rc3 * 0.5:
            scores["durability_penalty"] = 30.0
            details["durability_note"] = "EPS lagging revenue → margin compression signal"

        composite = self.weighted_average(scores, {k: 1.0 for k in scores}) if scores else 50.0
        details["_label"] = f"Growth score {composite:.0f}/100"
        return composite, details, missing

    # ── Pillar: Financial Health ─────────────────────────────

    def _score_health(self, data: dict) -> tuple[float, dict, list[str]]:
        scores:  dict[str, float] = {}
        details: dict[str, Any]   = {}
        missing: list[str]        = []

        # Current ratio
        cr = _f(data.get("current_ratio"))
        if cr is not None:
            scores["current_ratio"] = _hi_score(cr, _CURRENT_RATIO)
            details["current_ratio"] = round(cr, 2)
        else:
            missing.append("current_ratio")

        # Quick ratio
        qr = _f(data.get("quick_ratio"))
        if qr is not None:
            scores["quick_ratio"] = _hi_score(qr, _QUICK_RATIO)
            details["quick_ratio"] = round(qr, 2)
        else:
            missing.append("quick_ratio")

        # Debt/equity (lower is better)
        de = _f(data.get("debt_equity"))
        if de is not None:
            scores["debt_equity"] = _lo_score(abs(de), _DEBT_EQUITY)
            details["debt_equity"] = round(de, 2)
        else:
            missing.append("debt_equity")

        # Interest coverage
        ic = _f(data.get("interest_coverage")) or _f(data.get("interest_coverage_c"))
        if ic is not None:
            scores["interest_coverage"] = _hi_score(ic, _INTEREST_COV)
            details["interest_coverage"] = round(ic, 1)
        else:
            missing.append("interest_coverage")

        # Altman Z (double weight — distress signal)
        az = _f(data.get("altman_z"))
        if az is not None:
            z_score = _hi_score(az, _ALTMAN_Z)
            scores["altman_z"]  = z_score
            scores["altman_z2"] = z_score   # double weight
            details["altman_z"] = round(az, 2)
        else:
            missing.append("altman_z")

        # Piotroski F (double weight)
        pf = data.get("piotroski_f")
        if pf is not None:
            f_score = _hi_score(float(pf), _PIOTROSKI_F)
            scores["piotroski_f"]  = f_score
            scores["piotroski_f2"] = f_score
            details["piotroski_f"] = int(pf)
        else:
            missing.append("piotroski_f")

        composite = self.weighted_average(scores, {k: 1.0 for k in scores}) if scores else 50.0
        details["_label"] = f"Health score {composite:.0f}/100"
        return composite, details, missing

    # ── Pillar: Capital Allocation ───────────────────────────

    def _score_capital_allocation(self, data: dict) -> tuple[float, dict, list[str]]:
        scores:  dict[str, float] = {}
        details: dict[str, Any]   = {}
        missing: list[str]        = []

        # FCF conversion (FCF / Net Income)
        fc = _f(data.get("fcf_conversion"))
        if fc is not None:
            scores["fcf_conversion"] = _hi_score(fc, _FCF_CONV)
            details["fcf_conversion"] = round(fc, 2)
        else:
            missing.append("fcf_conversion")

        # FCF yield
        fy = _f(data.get("fcf_yield"))
        if fy is not None and fy > 0:
            scores["fcf_yield"] = _hi_score(fy, _FCF_YIELD)
            details["fcf_yield_pct"] = round(fy * 100, 2)
        else:
            missing.append("fcf_yield")

        # Dividend payout safety (moderate payout = good capital discipline)
        pr = _f(data.get("payout_ratio"))
        if pr is not None and pr >= 0:
            if 0.15 <= pr <= 0.50:
                scores["payout_ratio"] = 85.0    # disciplined dividend
            elif pr < 0.15:
                scores["payout_ratio"] = 70.0    # growing; no dividend yet
            elif 0.50 < pr <= 0.75:
                scores["payout_ratio"] = 55.0    # elevated but sustainable
            elif 0.75 < pr <= 1.0:
                scores["payout_ratio"] = 30.0    # stretched
            else:
                scores["payout_ratio"] = 10.0    # >100% payout → unsustainable
            details["payout_ratio_pct"] = round(pr * 100, 1)
        else:
            missing.append("payout_ratio")

        # No share dilution (from Piotroski F7 in piotroski_f — here use shares series)
        sh = data.get("shares_diluted", [])
        if isinstance(sh, list) and len(sh) >= 2:
            s0, s1 = _f(sh[0]), _f(sh[1])
            if s0 is not None and s1 is not None and s1 > 0:
                chg = (s0 - s1) / s1
                if chg <= -0.01:                 # buybacks reducing count
                    scores["share_count"] = 90.0
                    details["share_change_pct"] = round(chg * 100, 1)
                elif chg <= 0.01:                # roughly flat
                    scores["share_count"] = 65.0
                    details["share_change_pct"] = round(chg * 100, 1)
                else:                            # dilution
                    scores["share_count"] = 30.0
                    details["share_change_pct"] = round(chg * 100, 1)
        else:
            missing.append("shares_diluted")

        # ROIC trend (is ROIC improving?)
        roic_val = _f(data.get("roic"))
        roic_fmp = _f(data.get("roic_fmp"))   # FMP's computed ROIC
        if roic_val is not None:
            # Reward high absolute ROIC regardless of trend (trend needs 2yr ROIC data)
            if roic_val > 0.20:
                scores["roic_level"] = 90.0
            elif roic_val > 0.12:
                scores["roic_level"] = 70.0
            elif roic_val > 0.08:
                scores["roic_level"] = 55.0
            elif roic_val > 0.0:
                scores["roic_level"] = 35.0
            else:
                scores["roic_level"] = 10.0
            details["roic_pct"] = round(roic_val * 100, 1)
        else:
            missing.append("roic")

        composite = self.weighted_average(scores, {k: 1.0 for k in scores}) if scores else 50.0
        details["_label"] = f"CapAlloc score {composite:.0f}/100"
        return composite, details, missing

    # ── Driver extraction ────────────────────────────────────

    @staticmethod
    def _top_drivers(
        details: dict[str, Any], n: int = 3
    ) -> tuple[list[str], list[str]]:
        """
        Extract top-N positive and negative drivers from component detail dicts.

        Looks for numeric scores in the details sub-dicts and ranks them.
        """
        labeled: list[tuple[str, float]] = []
        for pillar, pillar_dict in details.items():
            if not isinstance(pillar_dict, dict):
                continue
            lbl = pillar_dict.get("_label", "")
            if lbl:
                labeled.append((lbl, 0.0))

        if not labeled:
            return [], []

        # Sort: highest score first for positives, lowest for negatives
        # (label strings already embed the score for readability)
        positives = [lbl for lbl, _ in labeled[:n]]
        negatives = [lbl for lbl, _ in labeled[-n:]]
        return positives, negatives


# ──────────────────────────────────────────────────────────────
# Module-level helpers
# ──────────────────────────────────────────────────────────────

def _f(val: Any) -> float | None:
    """Safe float conversion; returns None for missing/NaN/Inf."""
    if val is None:
        return None
    try:
        import math
        f = float(val)
        return None if (math.isnan(f) or math.isinf(f)) else f
    except (TypeError, ValueError):
        return None


def _hi_score(value: float, breaks: list[tuple[float, float]]) -> float:
    """Score where higher value is better. Breaks sorted descending by threshold."""
    for threshold, score in sorted(breaks, key=lambda x: x[0], reverse=True):
        if value >= threshold:
            return score
    return breaks[-1][1]


def _lo_score(value: float, breaks: list[tuple[float, float]]) -> float:
    """Score where lower value is better. Breaks sorted ascending by threshold."""
    for threshold, score in sorted(breaks, key=lambda x: x[0]):
        if value <= threshold:
            return score
    return breaks[-1][1]


def _peer_vals(
    peer_metrics: dict[str, dict],
    fmp_key: str,
    positive_only: bool = False,
) -> list[float]:
    """Extract a list of numeric peer values for a given FMP key."""
    vals = []
    for metrics in peer_metrics.values():
        v = _f(metrics.get(fmp_key))
        if v is not None:
            if positive_only and v <= 0:
                continue
            vals.append(v)
    return vals
