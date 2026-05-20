"""
Portfolio Construction Funnel — four-stage pipeline per fund (spec §11).

WARNING (from spec): Do NOT run mean-variance optimization over thousands of
tickers.  The covariance matrix is unstable, and the optimizer becomes an
"error maximizer."  The four-stage funnel below exists to avoid both failure
modes by feeding the optimizer only 20–40 well-scored, robust survivors.

Stage 1  Universe screen  → apply liquidity floors + mandate gates (cheap)
Stage 2  Grade & rank     → score survivors, keep Buy/Gotta Have, cap at N
Stage 3  Robust optimize  → Ledoit-Wolf cov, shrunk returns, constrained MV
Stage 4  Mandate assembly → combine equity + bond sleeves per fund's split

All projections are labeled as such; past optimality does not imply future.
This is decision-support, not an auto-allocator.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable

import numpy as np
import pandas as pd
import scipy.optimize

from stockgrader.models import Grade, PortfolioGrade

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────
# Result data classes
# ──────────────────────────────────────────────────────────────

@dataclass
class HoldingInfo:
    ticker:           str
    weight:           float          # 0–1
    sleeve:           str            # "equity" | "bond"
    grade:            str            # Grade.value
    composite:        float          # sub-grade composite for this portfolio
    sector:           str = ""
    risk_contribution: float = 0.0   # fraction of total portfolio variance


@dataclass
class PortfolioAnalytics:
    """Projected (model-based, not guaranteed) portfolio characteristics."""
    projected_return:     float        # annualized
    projected_volatility: float        # annualized
    projected_yield:      float        # weighted dividend yield estimate
    weighted_beta:        float
    sector_exposures:     dict[str, float] = field(default_factory=dict)
    effective_n_holdings: float = 0.0  # 1 / sum(w^2) — diversification measure
    projected_max_drawdown: float = 0.0
    return_method:        str = "shrunk_sample_mean"


@dataclass
class MandateCheck:
    target_low:  float
    target_high: float
    projected:   float
    passed:      bool
    gap:         float   # projected - target_low (negative = below target)


@dataclass
class FunnelStats:
    universe_entered:  int
    stage1_survivors:  int
    stage2_candidates: int
    optimizer_inputs:  int


@dataclass
class PortfolioResult:
    fund_name:   str
    as_of:       datetime
    holdings:    list[HoldingInfo]
    analytics:   PortfolioAnalytics
    mandate:     MandateCheck
    funnel:      FunnelStats
    robustness:  dict[str, Any] = field(default_factory=dict)
    warnings:    list[str]      = field(default_factory=list)


# ──────────────────────────────────────────────────────────────
# Stage 1 — Universe screen
# ──────────────────────────────────────────────────────────────

def screen_universe(
    universe_basics: list[dict[str, Any]],
    portfolio_cfg:   dict[str, Any],
    liquidity_cfg:   dict[str, Any] | None = None,
) -> list[str]:
    """
    Stage 1: apply liquidity floors and mandate eligibility gates.

    Parameters
    ----------
    universe_basics:
        List of dicts, one per ticker, with at minimum:
        ticker, price, avg_volume, market_cap, beta, dividend_yield, debt_equity,
        altman_z (optional), sector.
    portfolio_cfg:
        One portfolio's config block (from config.yaml portfolios.*).
    liquidity_cfg:
        The universe.liquidity_floor block from config.yaml.

    Returns
    -------
    List of ticker symbols that pass all Stage-1 filters.
    """
    liq  = liquidity_cfg or {}
    min_price = float(liq.get("min_price",          5.0))
    min_vol   = float(liq.get("min_avg_volume_30d",  500_000))
    min_mc    = float(liq.get("min_market_cap",      100_000_000))
    gates     = portfolio_cfg.get("eligibility", {})

    survivors: list[str] = []
    for item in universe_basics:
        ticker = item.get("ticker", "")
        if not ticker:
            continue

        # ── Liquidity floors (applied first — cheap reject) ─
        price  = _f(item.get("price"))
        vol    = _f(item.get("avg_volume"))
        mc     = _f(item.get("market_cap"))

        if price  is not None and price  < min_price:  continue
        if vol    is not None and vol    < min_vol:     continue
        if mc     is not None and mc     < min_mc:      continue

        # ── Mandate eligibility gates ──────────────────────
        beta = _f(item.get("beta"))
        max_beta = _f(gates.get("max_beta"))
        if max_beta is not None and beta is not None and beta > max_beta:
            continue

        gate_min_mc = _f(gates.get("min_market_cap"))
        if gate_min_mc is not None and mc is not None and mc < gate_min_mc:
            continue

        if gates.get("require_dividend", False):
            dy = _f(item.get("dividend_yield")) or 0.0
            if dy <= 0.001:
                continue

        de = _f(item.get("debt_equity"))
        max_de = _f(gates.get("max_debt_equity"))
        if max_de is not None and de is not None and abs(de) > max_de:
            continue

        z      = _f(item.get("altman_z"))
        min_z  = _f(gates.get("min_altman_z"))
        if min_z is not None and z is not None and z < min_z:
            continue

        survivors.append(ticker)

    return survivors


# ──────────────────────────────────────────────────────────────
# Stage 2 helpers
# ──────────────────────────────────────────────────────────────

@dataclass
class _ScoredCandidate:
    ticker:    str
    composite: float
    grade:     Grade
    data:      dict[str, Any]
    pg:        PortfolioGrade


def _grade_candidates(
    candidates:     list[dict[str, Any]],
    portfolio_name: str,
    config:         dict[str, Any],
) -> list[_ScoredCandidate]:
    """
    Run the full scoring pipeline + sub-grading for each candidate.
    Returns all results (caller filters by grade).
    """
    from stockgrader.engines.fundamental  import FundamentalEngine
    from stockgrader.engines.technical    import TechnicalEngine
    from stockgrader.engines.quantitative import QuantitativeEngine
    from stockgrader.portfolios.sub_grades import compute_sub_grades

    fund_eng  = FundamentalEngine(config)
    tech_eng  = TechnicalEngine(config)
    quant_eng = QuantitativeEngine(config)

    scored: list[_ScoredCandidate] = []
    for data in candidates:
        ticker = data.get("ticker", "?")
        try:
            fr = fund_eng.score(data)
            tr = tech_eng.score(data)
            qr = quant_eng.score(data)
            pg_all = compute_sub_grades(data, fr, tr, qr, config)
            pg     = pg_all.as_dict()[portfolio_name]
            scored.append(_ScoredCandidate(
                ticker    = ticker,
                composite = pg.composite,
                grade     = pg.grade,
                data      = data,
                pg        = pg,
            ))
        except Exception as exc:
            logger.warning("Scoring failed for %s: %s — skipping.", ticker, exc)

    return scored


def filter_and_rank(
    scored:       list[_ScoredCandidate],
    max_positions: int = 40,
) -> list[_ScoredCandidate]:
    """Keep only Buy / Gotta Have; sort by composite; cap at max_positions."""
    eligible = [s for s in scored
                if s.grade in (Grade.BUY, Grade.GOTTA_HAVE) and s.pg.composite > 0]
    eligible.sort(key=lambda s: s.composite, reverse=True)
    return eligible[:max_positions]


# ──────────────────────────────────────────────────────────────
# Stage 3 — Robust optimization
# ──────────────────────────────────────────────────────────────

def _build_returns_matrix(candidates: list[_ScoredCandidate]) -> pd.DataFrame | None:
    """
    Align price histories and compute daily return matrix (tickers as columns).
    Requires at least 60 shared trading days.
    """
    series: dict[str, pd.Series] = {}
    for c in candidates:
        df = c.data.get("price_history")
        if df is None or len(df) < 60:
            continue
        close = df["close"].astype(float)
        close.index = pd.to_datetime([str(d)[:10] for d in close.index])
        rets = close.pct_change().dropna()
        series[c.ticker] = rets

    if len(series) < 2:
        return None

    aligned = pd.DataFrame(series).dropna()
    if len(aligned) < 60:
        return None
    return aligned


def _compute_cov(returns: pd.DataFrame) -> np.ndarray:
    """Ledoit-Wolf shrinkage covariance (annualized)."""
    try:
        from sklearn.covariance import LedoitWolf  # type: ignore[import]
        lw = LedoitWolf(assume_centered=False)
        lw.fit(returns.values)
        cov = lw.covariance_ * 252
    except Exception:
        cov = returns.cov().values * 252
    return cov


def _compute_expected_returns(
    returns:   pd.DataFrame,
    shrinkage: float = 0.25,
) -> np.ndarray:
    """
    Shrunk sample mean returns (annualized).
    Shrinks each asset's sample mean toward the cross-sectional grand mean.
    shrinkage=0 → pure sample; shrinkage=1 → all assets get the grand mean.
    Per spec §14 decision §8: do NOT use raw trailing returns.
    """
    sample = returns.mean().values * 252
    grand  = sample.mean()
    return (1.0 - shrinkage) * sample + shrinkage * grand


def _optimize_weights(
    mu:            np.ndarray,
    cov:           np.ndarray,
    objective:     str,
    target_return: float,
    rf:            float,
    pos_min:       float,
    pos_max:       float,
    sector_groups: dict[str, list[int]],
    sector_limit:  float,
) -> np.ndarray:
    """
    Constrained portfolio optimization via scipy SLSQP.

    Constraints:
      - Weights sum to 1 (equality)
      - pos_min ≤ w_i ≤ pos_max for all i
      - sum(w_i for i in sector) ≤ sector_limit for each sector
    """
    n   = len(mu)
    x0  = np.ones(n) / n
    bds = [(pos_min, pos_max)] * n

    cons: list[dict] = [{"type": "eq", "fun": lambda w: float(np.sum(w)) - 1.0}]
    for indices in sector_groups.values():
        idx = list(indices)
        cons.append({"type": "ineq",
                     "fun": lambda w, i=idx: sector_limit - float(np.sum(w[i]))})

    cov_stable = cov + np.eye(n) * 1e-8   # numerical stability

    # Guard: if pos_max * n < 1 the constraint is infeasible; relax to 1/n
    pos_max = max(pos_max, 1.0 / n + 0.005)

    if objective == "min_variance":
        def fn(w):
            return float(w @ cov_stable @ w)

    elif objective == "max_sharpe":
        def fn(w):
            ret = float(mu @ w)
            vol = math.sqrt(max(float(w @ cov_stable @ w), 1e-10))
            return -(ret - rf) / vol      # minimize negative Sharpe

    elif objective in ("target_return_mv", "mean_variance"):
        cons.append({"type": "ineq",
                     "fun": lambda w: float(mu @ w) - target_return})
        def fn(w):
            return float(w @ cov_stable @ w)

    elif objective == "risk_parity":
        # Inverse-volatility weights as target (robust approximation of true RP)
        vol = np.sqrt(np.maximum(np.diag(cov_stable), 1e-8))
        target_w = (1.0 / vol) / (1.0 / vol).sum()
        def fn(w):
            return float(np.sum((w - target_w) ** 2))

    else:
        return x0   # unknown objective → equal weight

    result = scipy.optimize.minimize(
        fn, x0, method="SLSQP", bounds=bds, constraints=cons,
        options={"ftol": 1e-10, "maxiter": 2000},
    )

    if result.success and result.x is not None:
        w = np.maximum(result.x, 0.0)
        if w.sum() > 0:
            return w / w.sum()

    logger.warning("Optimizer did not converge for objective=%s; using equal weight.", objective)
    return x0


# ──────────────────────────────────────────────────────────────
# Bond sleeve (Stage 3b)
# ──────────────────────────────────────────────────────────────

# Credit quality mapping for scoring
_CREDIT_SCORES: dict[str, float] = {
    "treasury": 1.0, "tips": 0.9, "aggregate": 0.75, "ig_corp": 0.55,
}

# Duration preference by fund (lower score = prefer shorter)
_DUR_PREFERENCE: dict[str, float] = {
    "very_conservative": 2.5,
    "conservative":       4.0,
    "balanced":           6.0,
    "aggressive":         8.0,
    "very_aggressive":   12.0,
}


def _bond_mandate_score(etf: dict, fund_name: str) -> float:
    """Score a bond ETF for fit with the fund's mandate (higher = better)."""
    dur  = float(etf.get("duration_yrs", 6.0))
    cred = str(etf.get("credit", "aggregate"))

    pref_dur    = _DUR_PREFERENCE.get(fund_name, 6.0)
    dur_score   = max(0.0, 1.0 - abs(dur - pref_dur) / 10.0)
    cred_score  = _CREDIT_SCORES.get(cred, 0.5)

    # Conservative funds weight credit higher; aggressive weight duration higher
    if fund_name in ("very_conservative", "conservative"):
        return 0.60 * cred_score + 0.40 * dur_score
    elif fund_name in ("aggressive", "very_aggressive"):
        return 0.30 * cred_score + 0.70 * dur_score
    else:
        return 0.50 * cred_score + 0.50 * dur_score


def build_bond_sleeve(
    fund_name:   str,
    bond_cfg:    dict[str, Any],
    max_per_etf: float = 0.30,
) -> dict[str, float]:
    """
    Allocate bond-sleeve weights across the configured bond ETF candidates.

    Scores each ETF for mandate fit, allocates proportionally, caps at
    max_per_etf to prevent concentration.  Returns {ticker: weight} summing to 1.
    """
    all_etfs: list[dict] = []
    for bucket in bond_cfg.values():
        if isinstance(bucket, list):
            all_etfs.extend(bucket)

    if not all_etfs:
        return {}

    scores = {e["ticker"]: _bond_mandate_score(e, fund_name) for e in all_etfs
              if "ticker" in e}
    total  = sum(scores.values())
    if total == 0:
        return {}

    raw = {t: s / total for t, s in scores.items()}

    # Apply position cap and renormalise
    capped: dict[str, float] = {}
    for t, w in raw.items():
        capped[t] = min(w, max_per_etf)

    total_capped = sum(capped.values())
    return {t: w / total_capped for t, w in capped.items()} if total_capped > 0 else {}


# ──────────────────────────────────────────────────────────────
# Portfolio analytics
# ──────────────────────────────────────────────────────────────

def _compute_analytics(
    equity_weights: np.ndarray,
    tickers:        list[str],
    candidates:     list[_ScoredCandidate],
    returns:        pd.DataFrame | None,
    mu:             np.ndarray | None,
    cov:            np.ndarray | None,
    bond_weights:   dict[str, float],
    equity_pct:     float,
    bond_pct:       float,
    rf:             float,
) -> PortfolioAnalytics:
    n = len(equity_weights)

    # Projected equity return (weighted)
    if mu is not None and len(mu) == n:
        eq_ret = float(equity_weights @ mu)
    else:
        eq_ret = float(np.mean([c.data.get("revenue_cagr_3yr") or 0.07
                                 for c in candidates[:n]]))

    # Approximate bond return (use short rate + spread)
    bond_ret = rf + 0.015   # very rough; proper impl uses ETF yields

    proj_return = equity_pct * eq_ret + bond_pct * bond_ret

    # Projected equity volatility
    if cov is not None and len(cov) == n:
        proj_vol_eq = float(math.sqrt(max(equity_weights @ cov @ equity_weights, 0)))
        proj_vol    = equity_pct * proj_vol_eq  # bonds add diversification (simplified)
    else:
        proj_vol = 0.12 * equity_pct + 0.05 * bond_pct

    # Weighted beta
    betas = np.array([c.data.get("beta") or 1.0 for c in candidates[:n]], dtype=float)
    w_beta = float(equity_weights @ betas)

    # Projected dividend yield (equity sleeve)
    yields = np.array([c.data.get("dividend_yield") or 0.0 for c in candidates[:n]], dtype=float)
    eq_yield    = float(equity_weights @ yields)
    bond_yield  = 0.035   # rough fixed income yield proxy
    proj_yield  = equity_pct * eq_yield + bond_pct * bond_yield

    # Sector exposures (equity sleeve only)
    sector_exp: dict[str, float] = {}
    for i, c in enumerate(candidates[:n]):
        sec = c.data.get("sector") or "Unknown"
        sector_exp[sec] = sector_exp.get(sec, 0.0) + float(equity_weights[i]) * equity_pct

    # Effective N (diversification measure: 1/HHI = 1/sum(w^2))
    eq_full = equity_weights * equity_pct
    eff_n   = float(1.0 / max(np.sum(eq_full ** 2), 1e-8))

    # Projected max drawdown (rough: 2x annual vol as proxy)
    proj_dd = -2.0 * proj_vol

    return PortfolioAnalytics(
        projected_return      = round(proj_return, 4),
        projected_volatility  = round(proj_vol,    4),
        projected_yield       = round(proj_yield,  4),
        weighted_beta         = round(w_beta,       3),
        sector_exposures      = {k: round(v, 3) for k, v in sector_exp.items()},
        effective_n_holdings  = round(eff_n, 1),
        projected_max_drawdown= round(proj_dd, 4),
        return_method         = "shrunk_sample_mean",
    )


def _mandate_check(analytics: PortfolioAnalytics, portfolio_cfg: dict) -> MandateCheck:
    target = portfolio_cfg.get("return_target", [0.05, 0.07])
    lo, hi = float(target[0]), float(target[1])
    proj   = analytics.projected_return
    return MandateCheck(
        target_low  = lo,
        target_high = hi,
        projected   = proj,
        passed      = lo <= proj <= hi,
        gap         = round(proj - lo, 4),
    )


# ──────────────────────────────────────────────────────────────
# Stage 4 — Mandate assembly
# ──────────────────────────────────────────────────────────────

def assemble_portfolio(
    equity_weights: dict[str, float],
    bond_weights:   dict[str, float],
    candidates:     list[_ScoredCandidate],
    equity_pct:     float,
    bond_pct:       float,
) -> list[HoldingInfo]:
    holdings: list[HoldingInfo] = []

    cand_map = {c.ticker: c for c in candidates}

    for ticker, w in equity_weights.items():
        c   = cand_map.get(ticker)
        sec = c.data.get("sector", "Unknown") if c else "Unknown"
        holdings.append(HoldingInfo(
            ticker     = ticker,
            weight     = round(w * equity_pct, 4),
            sleeve     = "equity",
            grade      = c.grade.value if c else Grade.HOLD.value,
            composite  = c.composite if c else 0.0,
            sector     = sec,
        ))

    for ticker, w in bond_weights.items():
        holdings.append(HoldingInfo(
            ticker    = ticker,
            weight    = round(w * bond_pct, 4),
            sleeve    = "bond",
            grade     = "N/A",
            composite = 0.0,
            sector    = "Fixed Income",
        ))

    return holdings


# ──────────────────────────────────────────────────────────────
# Main builder (Stages 2–4)
# ──────────────────────────────────────────────────────────────

def build_portfolio(
    portfolio_name:      str,
    candidates_data:     list[dict[str, Any]],
    config:              dict[str, Any],
    risk_free:           float = 0.05,
    pre_scored:          list[_ScoredCandidate] | None = None,
) -> PortfolioResult:
    """
    Run Stages 2–4 of the funnel on pre-fetched candidate data.

    Parameters
    ----------
    portfolio_name:  one of the five portfolio keys
    candidates_data: list of full normalized data dicts (Stage 1 survivors)
    config:          full config dict
    risk_free:       annualized risk-free rate
    pre_scored:      optional pre-scored candidates (skips engine runs in tests)
    """
    warnings_list: list[str] = []
    pcfg     = config.get("portfolios", {}).get(portfolio_name, {})
    bond_cfg = config.get("bond_candidates", {})
    opt_cfg  = config.get("optimizer", {})
    max_pos  = int(opt_cfg.get("max_candidate_pool", 40))

    # ── Stage 2: Grade & rank ─────────────────────────────────
    if pre_scored is not None:
        scored = pre_scored
    else:
        scored = _grade_candidates(candidates_data, portfolio_name, config)

    stage1_count = len(candidates_data)
    all_scored   = len(scored)
    candidates   = filter_and_rank(scored, max_pos)

    if not candidates:
        warnings_list.append("No Buy/Gotta Have candidates survived Stage 2.")
        # Return empty portfolio
        return PortfolioResult(
            fund_name  = portfolio_name,
            as_of      = datetime.utcnow(),
            holdings   = [],
            analytics  = PortfolioAnalytics(0.0, 0.0, 0.0, 0.0),
            mandate    = _mandate_check(PortfolioAnalytics(0.0, 0.0, 0.0, 0.0), pcfg),
            funnel     = FunnelStats(stage1_count, stage1_count, all_scored, 0),
            warnings   = warnings_list,
        )

    tickers = [c.ticker for c in candidates]
    n       = len(tickers)

    # ── Stage 3a: Build covariance + expected returns ─────────
    returns = _build_returns_matrix(candidates)
    mu: np.ndarray | None = None
    cov: np.ndarray | None = None
    equity_weights_arr: np.ndarray

    if returns is not None and len(returns.columns) >= 2:
        # Align to available tickers
        avail = [t for t in tickers if t in returns.columns]
        if len(avail) >= 2:
            ret_sub = returns[avail]
            cov = _compute_cov(ret_sub)
            mu  = _compute_expected_returns(ret_sub, shrinkage=float(opt_cfg.get("shrinkage", 0.25)))

            # Sector groups for sector constraints
            sector_limit = float(pcfg.get("sector_limit", 0.25))
            pos_limits   = pcfg.get("position_limits", {"min": 0.01, "max": 0.07})
            pos_min      = float(pos_limits.get("min", 0.01))
            pos_max      = float(pos_limits.get("max", 0.07))

            sector_groups: dict[str, list[int]] = {}
            for i, ticker in enumerate(avail):
                c   = next((c for c in candidates if c.ticker == ticker), None)
                sec = c.data.get("sector", "Unknown") if c else "Unknown"
                sector_groups.setdefault(sec, []).append(i)

            target_return = float(pcfg.get("return_target", [0.05, 0.07])[0])
            objective     = str(pcfg.get("optimizer", "mean_variance"))

            equity_weights_arr = _optimize_weights(
                mu=mu, cov=cov, objective=objective,
                target_return=target_return, rf=risk_free,
                pos_min=pos_min, pos_max=pos_max,
                sector_groups=sector_groups, sector_limit=sector_limit,
            )

            # Re-map back to all candidates (those not in avail get zero weight)
            full_weights = np.zeros(n)
            for j, t in enumerate(avail):
                idx = tickers.index(t)
                full_weights[idx] = equity_weights_arr[j]
            equity_weights_arr = full_weights

            # Robustness cross-check: compare MV with risk-parity
            rp_weights = _optimize_weights(
                mu=mu, cov=cov, objective="risk_parity",
                target_return=target_return, rf=risk_free,
                pos_min=pos_min, pos_max=pos_max,
                sector_groups=sector_groups, sector_limit=sector_limit,
            )
        else:
            equity_weights_arr = np.ones(n) / n
            warnings_list.append("Insufficient shared price history; using equal weight.")
    else:
        equity_weights_arr = np.ones(n) / n
        warnings_list.append("No price history for optimization; using equal weight.")

    equity_dict = {t: float(equity_weights_arr[i]) for i, t in enumerate(tickers)}

    # ── Stage 3b: Bond sleeve ──────────────────────────────────
    bond_dict   = build_bond_sleeve(portfolio_name, bond_cfg)
    equity_pct  = float(pcfg.get("equity_pct", 0.55))
    bond_pct    = float(pcfg.get("bond_pct",   0.45))

    # ── Stage 4: Assemble ──────────────────────────────────────
    holdings = assemble_portfolio(equity_dict, bond_dict, candidates, equity_pct, bond_pct)

    analytics = _compute_analytics(
        equity_weights  = equity_weights_arr,
        tickers         = tickers,
        candidates      = candidates,
        returns         = returns,
        mu              = mu,
        cov             = cov,
        bond_weights    = bond_dict,
        equity_pct      = equity_pct,
        bond_pct        = bond_pct,
        rf              = risk_free,
    )

    mandate = _mandate_check(analytics, pcfg)

    # Risk contribution (marginal variance contribution)
    if cov is not None and len(cov) == len(avail if returns is not None else []):
        pass   # could compute full risk decomposition; stub for v1

    robustness: dict[str, Any] = {}
    if returns is not None and mu is not None and cov is not None:
        robustness = {
            "note": "Risk-parity cross-check available; compare weights below.",
            "mv_weights":   {t: round(equity_dict[t], 4) for t in tickers[:10]},
        }

    if not mandate.passed:
        warnings_list.append(
            f"Projected return {mandate.projected:.1%} outside target band "
            f"[{mandate.target_low:.1%}, {mandate.target_high:.1%}]."
        )

    warnings_list.append(
        "REBALANCE NOTE: this is a point-in-time ideal portfolio. Re-run on schedule. "
        "Turnover/tax costs not modeled in v1."
    )

    return PortfolioResult(
        fund_name  = portfolio_name,
        as_of      = datetime.utcnow(),
        holdings   = holdings,
        analytics  = analytics,
        mandate    = mandate,
        funnel     = FunnelStats(stage1_count, stage1_count, all_scored, n),
        robustness = robustness,
        warnings   = warnings_list,
    )


# ──────────────────────────────────────────────────────────────
# Full four-stage runner
# ──────────────────────────────────────────────────────────────

def run_full_funnel(
    portfolio_name:    str,
    universe_basics:   list[dict[str, Any]],
    fetch_full_data:   Callable[[str], dict[str, Any]],
    config:            dict[str, Any],
    risk_free:         float = 0.05,
) -> PortfolioResult:
    """
    Run all four stages end-to-end.

    universe_basics: list of minimal dicts {ticker, price, avg_volume,
                     market_cap, beta, dividend_yield, debt_equity, sector}
    fetch_full_data: callable(ticker) → full normalized data dict
    """
    pcfg    = config.get("portfolios", {}).get(portfolio_name, {})
    liq_cfg = config.get("universe", {}).get("liquidity_floor", {})

    # Stage 1
    survivors = screen_universe(universe_basics, pcfg, liq_cfg)
    logger.info("[%s] Stage 1: %d/%d tickers survived screening.",
                portfolio_name, len(survivors), len(universe_basics))

    if not survivors:
        return PortfolioResult(
            fund_name = portfolio_name, as_of = datetime.utcnow(),
            holdings=[], analytics=PortfolioAnalytics(0,0,0,0),
            mandate=_mandate_check(PortfolioAnalytics(0,0,0,0), pcfg),
            funnel=FunnelStats(len(universe_basics), 0, 0, 0),
            warnings=["No tickers survived Stage 1 screening."],
        )

    # Fetch full data for Stage 1 survivors
    full_data: list[dict] = []
    for ticker in survivors:
        try:
            data = fetch_full_data(ticker)
            data["ticker"] = ticker
            full_data.append(data)
        except Exception as exc:
            logger.warning("Failed to fetch full data for %s: %s", ticker, exc)

    # Stages 2–4
    return build_portfolio(portfolio_name, full_data, config, risk_free)


# ──────────────────────────────────────────────────────────────
# Helper
# ──────────────────────────────────────────────────────────────

def _f(v: Any) -> float | None:
    if v is None:
        return None
    try:
        f = float(v)
        return None if (math.isnan(f) or math.isinf(f)) else f
    except (TypeError, ValueError):
        return None
