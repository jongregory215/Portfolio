"""
Quantitative Engine — factor model + risk-adjusted return quality.

Factor exposures (§6.1):
  Value:         composite z of earnings yield, FCF yield, book-to-market
  Quality:       composite z of ROE, accruals (inverse), leverage (inverse)
  Momentum:      12-1 month return (skip most-recent month), risk-adjusted
  Size:          log market-cap decile (small = positive FF loading)
  Low-Volatility: inverse of realized vol and beta

Risk metrics (§6.3):
  Beta (1yr proxy), Sharpe/Sortino (1yr and 3yr), max drawdown (3yr),
  downside deviation, realized volatility.
  SPY / AGG correlations computed when benchmark data is present in
  data["benchmark_prices"]; otherwise recorded as None.

Fama-French regression (§6.2, optional):
  Runs if Ken French daily factor CSV is present at
  ~/.stockgrader/ff_factors_daily.csv; silently skipped otherwise.
  Download from: https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/
  Expected columns: Mkt-RF, SMB, HML, RMW, CMA, RF  (as decimals, not %)
"""
from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from stockgrader.engines.base import BaseEngine
from stockgrader.models import (
    FFRegression,
    FactorScores,
    QuantitativeResult,
    RiskMetrics,
)

logger = logging.getLogger(__name__)

# Empirical distribution parameters for absolute z-score fallback
# (S&P 500 cross-sectional estimates; update via backtesting calibration)
_ABS_PARAMS: dict[str, tuple[float, float]] = {
    "earnings_yield":   (0.050, 0.025),   # mean, std
    "fcf_yield":        (0.040, 0.025),
    "book_to_market":   (0.350, 0.250),
    "roe":              (0.160, 0.130),
    "quality_accruals": (1.100, 0.400),   # OCF/NI
    "leverage_inv":     (0.600, 0.200),   # 1/(1+D/E)
    "momentum_12_1":    (0.080, 0.300),
    "log_mktcap":       (25.0,  1.8),     # log of USD market cap
    "vol_inv":          (3.800, 1.500),   # 1/realized_vol (ann.)
    # beta_inv: low beta = positive low-vol z; distribution of 1/beta centred at 1.0
    "beta_inv":         (1.000, 0.400),   # 1/beta; mean ~1 for mkt, std ~0.4
}

# Ken French factor CSV path
_FF_PATH = Path.home() / ".stockgrader" / "ff_factors_daily.csv"


# ──────────────────────────────────────────────────────────────
# Pure helpers
# ──────────────────────────────────────────────────────────────

def _f(v: Any) -> float | None:
    if v is None:
        return None
    try:
        f = float(v)
        return None if (math.isnan(f) or math.isinf(f)) else f
    except (TypeError, ValueError):
        return None


def _peer_z(own: float, peers: list[float]) -> float | None:
    """Z-score of own relative to peer list (min 4 peers)."""
    if len(peers) < 4:
        return None
    mu  = float(np.mean(peers))
    sig = float(np.std(peers, ddof=1))
    if sig == 0:
        # All peers identical: return ±3 based on direction, 0 if equal
        return 3.0 if own > mu else (-3.0 if own < mu else 0.0)
    return (own - mu) / sig


def _abs_z(val: float, key: str) -> float:
    """Absolute z-score using pre-calibrated distribution parameters."""
    mu, sig = _ABS_PARAMS.get(key, (0.0, 1.0))
    return (val - mu) / sig if sig != 0 else 0.0


def _norm_cdf(z: float) -> float:
    """Standard-normal CDF → percentile (0–100)."""
    return float(0.5 * (1.0 + math.erf(z / math.sqrt(2)))) * 100.0


def _z_to_pct(z: float | None) -> float | None:
    if z is None:
        return None
    return _norm_cdf(max(-4.0, min(4.0, z)))


def _peer_vals_q(peer_metrics: dict, fmp_key: str) -> list[float]:
    vals = []
    for m in peer_metrics.values():
        v = _f(m.get(fmp_key))
        if v is not None:
            vals.append(v)
    return vals


def _ann_return(returns: pd.Series) -> float:
    return float(returns.mean() * 252)


def _ann_vol(returns: pd.Series) -> float:
    return float(returns.std() * math.sqrt(252))


# ──────────────────────────────────────────────────────────────
# Quantitative Engine
# ──────────────────────────────────────────────────────────────

class QuantitativeEngine(BaseEngine):

    def score(self, data: dict[str, Any]) -> QuantitativeResult:
        cfg_q = self.config.get("quantitative", {})
        rf    = _f(data.get("risk_free_rate_3mo")) or 0.05
        price_df     = data.get("price_history")
        peer_metrics = data.get("peer_metrics", {})
        missing: list[str] = []

        # ── 1. Risk metrics ──────────────────────────────────
        risk_raw  = self._compute_risk_metrics(price_df, rf)
        risk_obj  = self._build_risk_metrics(risk_raw, data)

        # ── 2. Factor exposures ──────────────────────────────
        fac_res = self._compute_factors(data, peer_metrics, risk_raw, cfg_q)
        fac_obj = FactorScores(
            value_z          = fac_res.get("value_z"),
            quality_z        = fac_res.get("quality_z"),
            momentum_z       = fac_res.get("momentum_z"),
            size_z           = fac_res.get("size_z"),
            low_volatility_z = fac_res.get("low_vol_z"),
            value_pct          = fac_res.get("value_pct"),
            quality_pct        = fac_res.get("quality_pct"),
            momentum_pct       = fac_res.get("momentum_pct"),
            size_pct           = fac_res.get("size_pct"),
            low_volatility_pct = fac_res.get("low_vol_pct"),
        )

        # ── 3. Factor composite score ─────────────────────────
        favorable = set(cfg_q.get("favorable_factors",
                                  ["value", "quality", "momentum", "low_volatility"]))
        fw = cfg_q.get("factor_weights_in_quant_score", {
            "value": 0.20, "quality": 0.30, "momentum": 0.25,
            "size":  0.10, "low_volatility": 0.15,
        })

        factor_scores_dict: dict[str, float] = {}
        for fname, weight in fw.items():
            pct_key = f"{fname}_pct".replace("low_volatility", "low_vol")
            pct = fac_res.get(pct_key)
            if pct is None:
                pct_key2 = f"{fname}_pct"
                pct = fac_res.get(pct_key2)
            if pct is not None:
                # For "favorable" factors, higher percentile = better
                # For size, percentile is just tracked (exposure, not directional)
                factor_scores_dict[fname] = pct if fname in favorable else 50.0
            else:
                missing.append(f"factor_{fname}")

        factor_composite = (
            self.weighted_average(factor_scores_dict, fw) if factor_scores_dict else 50.0
        )

        # ── 4. Risk quality score ─────────────────────────────
        risk_score = self._score_risk_quality(risk_raw, data)

        # ── 5. Combined quant score ───────────────────────────
        # 65% factor alignment, 35% risk-adjusted return quality
        if factor_scores_dict:
            composite = 0.65 * factor_composite + 0.35 * risk_score
        else:
            composite = risk_score   # fallback: risk only

        # ── 6. Optional Fama-French regression ───────────────
        ff_reg = self._fama_french_regression(price_df, rf)

        return QuantitativeResult(
            score        = self.clamp(composite),
            factors      = fac_obj,
            risk_metrics = risk_obj,
            ff_regression = ff_reg,
            missing_fields = sorted(set(missing)),
        )

    # ── Risk metric computation ───────────────────────────────

    def _compute_risk_metrics(
        self, price_df: pd.DataFrame | None, rf: float
    ) -> dict[str, Any]:
        if price_df is None or len(price_df) < 30:
            return {}

        close   = price_df["close"].astype(float)
        returns = close.pct_change().dropna()
        n       = len(returns)

        rf_daily = (1.0 + rf) ** (1.0 / 252) - 1.0

        # ── Realized volatility (annualized) ──────────────────
        vol_1yr = float(returns.iloc[-252:].std() * math.sqrt(252)) if n >= 252 \
                  else float(returns.std() * math.sqrt(252))

        # ── Max drawdown and current drawdown ─────────────────
        window = min(756, n)     # 3yr = 756 trading days
        prices_w = close.iloc[-window:]
        roll_max = prices_w.cummax()
        drawdowns = (prices_w - roll_max) / roll_max
        max_dd  = float(drawdowns.min())
        cur_dd  = float(drawdowns.iloc[-1])

        # ── Downside deviation (excess returns below zero) ────
        excess = returns - rf_daily
        neg    = excess[excess < 0]
        dd_dev = float(neg.std() * math.sqrt(252)) if len(neg) >= 10 else None

        # ── Sharpe and Sortino ─────────────────────────────────
        def _sharpe_sortino(rets: pd.Series):
            if len(rets) < 20:
                return None, None
            ann_r = _ann_return(rets)
            ann_v = _ann_vol(rets)
            sh    = (ann_r - rf) / ann_v if ann_v > 0 else None
            exc   = rets - rf_daily
            neg_v = exc[exc < 0].std() * math.sqrt(252) if len(exc[exc < 0]) >= 5 else None
            so    = (ann_r - rf) / neg_v if neg_v else None
            return sh, so

        sh1, so1 = _sharpe_sortino(returns.iloc[-252:] if n >= 252 else returns)
        sh3, so3 = _sharpe_sortino(returns.iloc[-756:]) if n >= 756 else (None, None)

        # ── 12-1 month momentum ────────────────────────────────
        # Return from t-252 to t-21 (skip most-recent 21 trading days)
        mom_12_1 = None
        if n >= 273:
            p_now  = float(close.iloc[-22])   # price 1 month ago
            p_past = float(close.iloc[-273])  # price 13 months ago
            mom_12_1 = (p_now / p_past - 1.0) if p_past != 0 else None

        # ── Benchmark correlations (if benchmark data present) ─
        bench  = price_df.attrs.get("benchmarks", {})  # optional dict
        corr_spy = _corr_with_bench(returns, bench.get("SPY"))
        corr_agg = _corr_with_bench(returns, bench.get("AGG"))

        return {
            "realized_vol_1yr":  vol_1yr,
            "max_drawdown_3yr":  max_dd,
            "current_drawdown":  cur_dd,
            "downside_dev_1yr":  dd_dev,
            "sharpe_1yr":        sh1,
            "sharpe_3yr":        sh3,
            "sortino_1yr":       so1,
            "sortino_3yr":       so3,
            "momentum_12_1":     mom_12_1,
            "corr_spy":          corr_spy,
            "corr_agg":          corr_agg,
        }

    def _build_risk_metrics(
        self, raw: dict, data: dict
    ) -> RiskMetrics:
        beta_val = _f(data.get("beta"))
        return RiskMetrics(
            beta_1yr         = beta_val,
            beta_3yr         = beta_val,   # same source; 3yr beta would need separate call
            sharpe_1yr       = raw.get("sharpe_1yr"),
            sharpe_3yr       = raw.get("sharpe_3yr"),
            sortino_1yr      = raw.get("sortino_1yr"),
            sortino_3yr      = raw.get("sortino_3yr"),
            max_drawdown_3yr = raw.get("max_drawdown_3yr"),
            current_drawdown = raw.get("current_drawdown"),
            downside_deviation_1yr = raw.get("downside_dev_1yr"),
            corr_spy         = raw.get("corr_spy"),
            corr_agg         = raw.get("corr_agg"),
            realized_vol_1yr = raw.get("realized_vol_1yr"),
        )

    # ── Factor computation ────────────────────────────────────

    def _compute_factors(
        self,
        data:         dict,
        peer_metrics: dict,
        risk_raw:     dict,
        cfg_q:        dict,
    ) -> dict[str, float | None]:
        result: dict[str, float | None] = {}

        # ── Value factor ─────────────────────────────────────
        val_subs: list[float] = []

        pe = _f(data.get("pe_trailing"))
        if pe and pe > 0:
            ey = 1.0 / pe
            peer_eys = [1.0 / v for v in _peer_vals_q(peer_metrics, "priceEarningsRatioTTM")
                        if _f(v) and _f(v) > 0]
            z = _peer_z(ey, peer_eys) if len(peer_eys) >= 4 else _abs_z(ey, "earnings_yield")
            val_subs.append(z)

        fy = _f(data.get("fcf_yield"))
        if fy is not None:
            peer_fy = _peer_vals_q(peer_metrics, "freeCashFlowYieldTTM")
            z = _peer_z(fy, peer_fy) if len(peer_fy) >= 4 else _abs_z(fy, "fcf_yield")
            val_subs.append(z)

        pb = _f(data.get("pb"))
        if pb and pb > 0:
            btm = 1.0 / pb
            peer_btm = [1.0 / v for v in _peer_vals_q(peer_metrics, "priceToBookRatioTTM")
                        if _f(v) and _f(v) > 0]
            z = _peer_z(btm, peer_btm) if len(peer_btm) >= 4 else _abs_z(btm, "book_to_market")
            val_subs.append(z)

        if val_subs:
            vz = float(np.mean(val_subs))
            result["value_z"]   = round(vz, 3)
            result["value_pct"] = _z_to_pct(vz)

        # ── Quality factor ────────────────────────────────────
        q_subs: list[float] = []

        roe = _f(data.get("roe"))
        if roe is not None:
            peer_roe = _peer_vals_q(peer_metrics, "returnOnEquityTTM")
            z = _peer_z(roe, peer_roe) if len(peer_roe) >= 4 else _abs_z(roe, "roe")
            q_subs.append(z)

        qoe = _f(data.get("quality_of_earnings"))
        if qoe is not None:
            z = _abs_z(qoe, "quality_accruals")
            q_subs.append(z)

        de = _f(data.get("debt_equity"))
        if de is not None:
            lev_inv = 1.0 / (1.0 + abs(de))
            peer_de = _peer_vals_q(peer_metrics, "debtToEquityTTM")
            peer_li = [1.0 / (1.0 + abs(v)) for v in peer_de if _f(v) is not None]
            z = _peer_z(lev_inv, peer_li) if len(peer_li) >= 4 else _abs_z(lev_inv, "leverage_inv")
            q_subs.append(z)

        if q_subs:
            qz = float(np.mean(q_subs))
            result["quality_z"]   = round(qz, 3)
            result["quality_pct"] = _z_to_pct(qz)

        # ── Momentum factor ───────────────────────────────────
        mom = risk_raw.get("momentum_12_1")
        if mom is not None:
            vol = risk_raw.get("realized_vol_1yr") or 0.20
            # Risk-adjusted: divide by realized vol (standard momentum factor construction)
            mom_adj = mom / vol if vol > 0 else mom
            mz      = _abs_z(mom_adj, "momentum_12_1")
            result["momentum_z"]   = round(mz, 3)
            result["momentum_pct"] = _z_to_pct(mz)

        # ── Size factor ───────────────────────────────────────
        mc = _f(data.get("market_cap"))
        if mc and mc > 0:
            log_mc = math.log(mc)
            sz     = _abs_z(log_mc, "log_mktcap")
            # Fama-French: small = positive SMB loading; negate for our scoring direction
            result["size_z"]   = round(-sz, 3)   # negative: large-cap has negative SMB loading
            result["size_pct"] = _z_to_pct(-sz)

        # ── Low-volatility factor ─────────────────────────────
        lv_subs: list[float] = []

        vol_1yr = risk_raw.get("realized_vol_1yr")
        if vol_1yr and vol_1yr > 0:
            lv_subs.append(_abs_z(1.0 / vol_1yr, "vol_inv"))

        beta = _f(data.get("beta"))
        if beta is not None and beta != 0:
            # Low beta = positive low-vol z; use beta_inv distribution (separate from vol_inv)
            lv_subs.append(_abs_z(1.0 / abs(beta), "beta_inv"))

        if lv_subs:
            lvz = float(np.mean(lv_subs))
            result["low_vol_z"]   = round(lvz, 3)
            result["low_vol_pct"] = _z_to_pct(lvz)

        return result

    # ── Risk quality scoring ──────────────────────────────────

    def _score_risk_quality(
        self, risk_raw: dict, data: dict
    ) -> float:
        scores: dict[str, float] = {}

        # Sharpe (1yr) — primary risk-adjusted return signal
        sh = _f(risk_raw.get("sharpe_1yr"))
        if sh is not None:
            if sh >= 2.0:    scores["sharpe"] = 98.0
            elif sh >= 1.5:  scores["sharpe"] = 90.0
            elif sh >= 1.0:  scores["sharpe"] = 78.0
            elif sh >= 0.5:  scores["sharpe"] = 63.0
            elif sh >= 0.0:  scores["sharpe"] = 48.0
            elif sh >= -0.5: scores["sharpe"] = 32.0
            else:            scores["sharpe"] = 15.0

        # Sortino (1yr) — penalises downside more
        so = _f(risk_raw.get("sortino_1yr"))
        if so is not None:
            if so >= 2.5:    scores["sortino"] = 95.0
            elif so >= 1.5:  scores["sortino"] = 82.0
            elif so >= 1.0:  scores["sortino"] = 70.0
            elif so >= 0.5:  scores["sortino"] = 55.0
            elif so >= 0.0:  scores["sortino"] = 40.0
            else:            scores["sortino"] = 18.0

        # Max drawdown (3yr) — lower magnitude (less negative) = better
        mdd = _f(risk_raw.get("max_drawdown_3yr"))
        if mdd is not None:
            if mdd > -0.10:   scores["max_dd"] = 92.0
            elif mdd > -0.20: scores["max_dd"] = 78.0
            elif mdd > -0.30: scores["max_dd"] = 62.0
            elif mdd > -0.40: scores["max_dd"] = 45.0
            elif mdd > -0.55: scores["max_dd"] = 28.0
            else:             scores["max_dd"] = 12.0

        # Current drawdown — penalise if stuck in a deep drawdown
        cur_dd = _f(risk_raw.get("current_drawdown"))
        if cur_dd is not None:
            if cur_dd > -0.05:   scores["cur_dd"] = 80.0
            elif cur_dd > -0.10: scores["cur_dd"] = 65.0
            elif cur_dd > -0.20: scores["cur_dd"] = 48.0
            elif cur_dd > -0.35: scores["cur_dd"] = 30.0
            else:                scores["cur_dd"] = 12.0

        # Realized volatility — lower is better (within reason)
        vol = _f(risk_raw.get("realized_vol_1yr"))
        if vol is not None:
            if vol < 0.12:    scores["vol"] = 88.0
            elif vol < 0.20:  scores["vol"] = 75.0
            elif vol < 0.30:  scores["vol"] = 60.0
            elif vol < 0.40:  scores["vol"] = 45.0
            elif vol < 0.55:  scores["vol"] = 30.0
            else:             scores["vol"] = 15.0

        # Beta: reward low-beta AND good Sharpe; penalise high-beta with bad Sharpe
        beta = _f(data.get("beta"))
        if beta is not None and sh is not None:
            beta_penalty = (abs(beta) - 1.0) * 20.0   # 0 at beta=1, -20 at beta=2
            sharpe_adj   = max(0.0, sh * 30.0)         # Sharpe contribution
            b_score      = 60.0 - beta_penalty + sharpe_adj
            scores["beta_sharpe"] = self.clamp(b_score)

        return self.weighted_average(scores, {k: 1.0 for k in scores}) if scores else 50.0

    # ── Fama-French regression ────────────────────────────────

    def _fama_french_regression(
        self, price_df: pd.DataFrame | None, rf: float
    ) -> FFRegression | None:
        if price_df is None or len(price_df) < 252:
            return None
        if not _FF_PATH.exists():
            logger.debug("Ken French daily factors not found at %s; skipping FF regression.", _FF_PATH)
            return None

        try:
            factors = pd.read_csv(_FF_PATH, index_col=0, parse_dates=True)
            # Ensure decimal (not percentage) form
            if factors.abs().max().max() > 1.0:
                factors = factors / 100.0

            close   = price_df["close"].astype(float)
            rets    = close.pct_change().dropna()
            rets.index = pd.to_datetime([str(d)[:10] for d in rets.index])

            merged = pd.merge(
                rets.rename("stock"),
                factors,
                left_index=True, right_index=True,
                how="inner",
            )

            if len(merged) < 60:
                return None

            from statsmodels.regression.linear_model import OLS  # type: ignore[import]
            from statsmodels.tools import add_constant             # type: ignore[import]

            rf_col  = "RF" if "RF" in merged.columns else None
            rf_series = merged[rf_col] if rf_col else pd.Series(
                (1 + rf) ** (1 / 252) - 1, index=merged.index
            )

            y = merged["stock"] - rf_series
            factor_cols = [c for c in ["Mkt-RF", "SMB", "HML", "RMW", "CMA"] if c in merged.columns]
            X = add_constant(merged[factor_cols])
            model = OLS(y, X).fit()

            alpha_d = float(model.params.get("const", 0.0))
            return FFRegression(
                model           = "FF5" if len(factor_cols) == 5 else "FF3",
                mkt_rf_beta     = _f(model.params.get("Mkt-RF")),
                smb_beta        = _f(model.params.get("SMB")),
                hml_beta        = _f(model.params.get("HML")),
                rmw_beta        = _f(model.params.get("RMW")),
                cma_beta        = _f(model.params.get("CMA")),
                alpha_annualized = float((1 + alpha_d) ** 252 - 1),
                alpha_t_stat    = _f(model.tvalues.get("const")),
                r_squared       = float(model.rsquared),
                period_years    = round(len(merged) / 252, 1),
            )

        except Exception as exc:
            logger.warning("Fama-French regression failed: %s", exc)
            return None


# ── Module-level helpers ──────────────────────────────────────

def _corr_with_bench(
    stock_rets: pd.Series, bench_df: pd.DataFrame | None
) -> float | None:
    if bench_df is None or bench_df.empty:
        return None
    try:
        bench_rets = bench_df["close"].astype(float).pct_change().dropna()
        bench_rets.index = pd.to_datetime([str(d)[:10] for d in bench_rets.index])
        stock_idx  = pd.to_datetime([str(d)[:10] for d in stock_rets.index])
        merged = pd.merge(
            stock_rets.set_axis(stock_idx).rename("s"),
            bench_rets.rename("b"),
            left_index=True, right_index=True,
            how="inner",
        )
        if len(merged) < 60:
            return None
        return float(merged["s"].corr(merged["b"]))
    except Exception:
        return None
