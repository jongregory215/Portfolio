#!/usr/bin/env python3
"""
Interactive Stock Analysis Dashboard — Streamlit web UI.

Run with: streamlit run stock_analyzer.py

Enter a ticker symbol, click Analyze, and view complete metrics including:
- Price ladder with all grade zones
- Fundamental metrics (valuation, profitability, growth, health, capital allocation)
- Technical indicators (trend, momentum, volume/structure)
- Quantitative factors and risk metrics
- Portfolio sub-grades for all 5 risk profiles
"""
import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env", override=False)
except ImportError:
    pass

import streamlit as st
from datetime import datetime
from stockgrader.pipeline import run_analysis
from stockgrader.data.fetcher import DataFetcher
from stockgrader.models import Grade, AnalysisResult


# ── Page config ────────────────────────────────────────────────
st.set_page_config(
    page_title="Stock Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("📊 Stock Analysis Dashboard")

# ── Sidebar: Input controls ────────────────────────────────────
with st.sidebar:
    st.header("Analysis Controls")
    ticker_input = st.text_input(
        "Enter ticker symbol",
        value="AAPL",
        placeholder="e.g., AAPL, MSFT, TSLA",
        help="Equity ticker symbol"
    ).upper().strip()

    col1, col2 = st.columns(2)
    with col1:
        use_deep = st.checkbox("Use FMP (deep)", value=False,
                               help="Requires FMP_API_KEY for richer data")
    with col2:
        no_cache = st.checkbox("Skip cache", value=False,
                               help="Fetch live data instead of cached")

    analyze_button = st.button("🔍 Analyze", use_container_width=True, type="primary")

# ── Session state management ───────────────────────────────────
if "result" not in st.session_state:
    st.session_state.result = None
if "error" not in st.session_state:
    st.session_state.error = None

# ── Run analysis on button click ────────────────────────────────
if analyze_button:
    if not ticker_input:
        st.error("Please enter a ticker symbol.")
    else:
        with st.spinner(f"Analyzing {ticker_input}…"):
            try:
                fetcher = DataFetcher(deep=use_deep)
                result = run_analysis(ticker_input, portfolio="all",
                                     no_cache=no_cache, fetcher=fetcher)
                st.session_state.result = result
                st.session_state.error = None
            except Exception as exc:
                st.session_state.result = None
                st.session_state.error = str(exc)

# ── Display error if present ───────────────────────────────────
if st.session_state.error:
    st.error(f"**Error:** {st.session_state.error}")
    st.stop()

# ── Display results if available ───────────────────────────────
if st.session_state.result is None:
    st.info("👈 Enter a ticker and click **Analyze** to get started.")
    st.stop()

result: AnalysisResult = st.session_state.result

# ── Header with grade and key metrics ──────────────────────────
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    grade_color = {
        Grade.STAY_AWAY: "🚫",
        Grade.SELL: "⚠️",
        Grade.HOLD: "⚖️",
        Grade.BUY: "✅",
        Grade.GOTTA_HAVE: "⭐",
    }
    st.metric("Grade", f"{grade_color[result.overall.grade]} {result.overall.grade.value}")
with col2:
    st.metric("Score", f"{result.overall.composite:.1f}/100")
with col3:
    st.metric("Confidence", f"{result.overall.confidence*100:.0f}%")
with col4:
    st.metric("Current Price", f"${result.price:.2f}")
with col5:
    st.metric("Fair Value", f"${result.price_ladder.fair_value:.2f}")

st.divider()

# ── Tabs for detailed sections ─────────────────────────────────
tab_ladder, tab_fund, tab_tech, tab_quant, tab_portfolio = st.tabs(
    ["Price Ladder", "Fundamental", "Technical", "Quantitative", "Portfolios"]
)

# ── Tab 1: Price Ladder ────────────────────────────────────────
with tab_ladder:
    st.subheader("Price Ladder & Zones")
    pl = result.price_ladder

    # Price zone indicator
    def get_zone(p):
        if p <= pl.gotta_have_at: return "Gotta Have", "🌟"
        if p <= pl.buy_at: return "Buy", "✅"
        if p <= pl.hold_high: return "Hold", "⚖️"
        if p <= pl.stay_away_above: return "Sell", "⚠️"
        return "Stay Away", "🚫"

    zone, emoji = get_zone(result.price)
    st.markdown(f"### Current: **${result.price:.2f}** — {emoji} **{zone} zone**")

    # Ladder table
    ladder_data = {
        "Zone": ["⭐ Gotta Have", "✅ Buy", "⚖️ Hold", "⚠️ Sell", "🚫 Stay Away", "📍 Fair Value"],
        "Price Level": [
            f"≤ ${pl.gotta_have_at:.2f}",
            f"≤ ${pl.buy_at:.2f}",
            f"${pl.hold_low:.2f} – ${pl.hold_high:.2f}",
            f"> ${pl.sell_above:.2f}",
            f"> ${pl.stay_away_above:.2f}",
            f"${pl.fair_value:.2f}",
        ]
    }
    st.dataframe(ladder_data, use_container_width=True, hide_index=True)

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Upside to Fair Value", f"{pl.upside_to_fv_pct*100:.1f}%")
    with col2:
        if pl.implied_growth_rate is not None:
            st.metric("Implied Growth Rate", f"{pl.implied_growth_rate*100:.2f}%")

# ── Tab 2: Fundamental Metrics ─────────────────────────────────
with tab_fund:
    st.subheader("Fundamental Analysis")
    fund = result.engines.fundamental

    col1, col2 = st.columns([1, 4])
    with col1:
        st.metric("Score", f"{fund.score:.0f}/100")
    with col2:
        pillars = fund.pillars
        pillar_cols = st.columns(5)
        pillar_names = ["Valuation", "Profitability", "Growth", "Financial Health", "Capital Allocation"]
        pillar_scores = [pillars.valuation, pillars.profitability, pillars.growth,
                         pillars.financial_health, pillars.capital_allocation]
        for col, name, score in zip(pillar_cols, pillar_names, pillar_scores):
            with col:
                st.metric(name, f"{score:.0f}")

    st.divider()

    # Detailed metrics from details dict
    details = fund.pillars.details

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Valuation**")
        val = details.get("valuation", {})
        val_data = {
            "Metric": ["P/E Trailing", "P/E Forward", "P/B", "P/S", "EV/EBITDA", "PEG"],
            "Value": [
                f"{val.get('pe_trailing', 'N/A'):.2f}x" if val.get('pe_trailing') else "N/A",
                f"{val.get('pe_forward', 'N/A'):.2f}x" if val.get('pe_forward') else "N/A",
                f"{val.get('pb', 'N/A'):.2f}x" if val.get('pb') else "N/A",
                f"{val.get('ps', 'N/A'):.2f}x" if val.get('ps') else "N/A",
                f"{val.get('ev_ebitda', 'N/A'):.2f}x" if val.get('ev_ebitda') else "N/A",
                f"{val.get('peg', 'N/A'):.2f}" if val.get('peg') else "N/A",
            ]
        }
        st.dataframe(val_data, use_container_width=True, hide_index=True)

        st.markdown("**Profitability**")
        prof = details.get("profitability", {})
        prof_data = {
            "Metric": ["Gross Margin", "Operating Margin", "Net Margin", "ROE", "ROA", "ROIC vs WACC"],
            "Value": [
                f"{prof.get('gross_margin_pct', 'N/A'):.1f}%" if prof.get('gross_margin_pct') is not None else "N/A",
                f"{prof.get('operating_margin_pct', 'N/A'):.1f}%" if prof.get('operating_margin_pct') is not None else "N/A",
                f"{prof.get('net_margin_pct', 'N/A'):.1f}%" if prof.get('net_margin_pct') is not None else "N/A",
                f"{prof.get('roe_pct', 'N/A'):.1f}%" if prof.get('roe_pct') is not None else "N/A",
                f"{prof.get('roa_pct', 'N/A'):.1f}%" if prof.get('roa_pct') is not None else "N/A",
                f"{prof.get('roic_vs_wacc_ppts', 'N/A'):.1f} ppts" if prof.get('roic_vs_wacc_ppts') is not None else "N/A",
            ]
        }
        st.dataframe(prof_data, use_container_width=True, hide_index=True)

    with col2:
        st.markdown("**Growth**")
        growth = details.get("growth", {})
        growth_data = {
            "Metric": ["Rev CAGR 3yr", "Rev CAGR 5yr", "EPS CAGR 3yr", "EPS CAGR 5yr", "Fwd EPS Growth", "Fwd Rev Growth"],
            "Value": [
                f"{growth.get('revenue_cagr_3yr_pct', 'N/A'):.1f}%" if growth.get('revenue_cagr_3yr_pct') is not None else "N/A",
                f"{growth.get('revenue_cagr_5yr_pct', 'N/A'):.1f}%" if growth.get('revenue_cagr_5yr_pct') is not None else "N/A",
                f"{growth.get('eps_cagr_3yr_pct', 'N/A'):.1f}%" if growth.get('eps_cagr_3yr_pct') is not None else "N/A",
                f"{growth.get('eps_cagr_5yr_pct', 'N/A'):.1f}%" if growth.get('eps_cagr_5yr_pct') is not None else "N/A",
                f"{growth.get('forward_eps_growth_pct', 'N/A'):.1f}%" if growth.get('forward_eps_growth_pct') is not None else "N/A",
                f"{growth.get('forward_rev_growth_pct', 'N/A'):.1f}%" if growth.get('forward_rev_growth_pct') is not None else "N/A",
            ]
        }
        st.dataframe(growth_data, use_container_width=True, hide_index=True)

        st.markdown("**Financial Health & Capital Allocation**")
        health = details.get("financial_health", {})
        capalloc = details.get("capital_allocation", {})
        combined_data = {
            "Metric": ["Current Ratio", "Quick Ratio", "Debt/Equity", "Interest Coverage",
                      "Altman Z-Score", "Piotroski F", "FCF Conversion", "FCF Yield"],
            "Value": [
                f"{health.get('current_ratio', 'N/A'):.2f}" if health.get('current_ratio') else "N/A",
                f"{health.get('quick_ratio', 'N/A'):.2f}" if health.get('quick_ratio') else "N/A",
                f"{health.get('debt_equity', 'N/A'):.2f}" if health.get('debt_equity') else "N/A",
                f"{health.get('interest_coverage', 'N/A'):.1f}x" if health.get('interest_coverage') else "N/A",
                f"{fund.altman_z:.2f}" if fund.altman_z else "N/A",
                f"{fund.piotroski_f}/9" if fund.piotroski_f is not None else "N/A",
                f"{capalloc.get('fcf_conversion', 'N/A'):.2f}" if capalloc.get('fcf_conversion') is not None else "N/A",
                f"{capalloc.get('fcf_yield_pct', 'N/A'):.1f}%" if capalloc.get('fcf_yield_pct') is not None else "N/A",
            ]
        }
        st.dataframe(combined_data, use_container_width=True, hide_index=True)

# ── Tab 3: Technical Metrics ───────────────────────────────────
with tab_tech:
    st.subheader("Technical Analysis")
    tech = result.engines.technical

    col1, col2 = st.columns([1, 4])
    with col1:
        st.metric("Score", f"{tech.score:.0f}/100")
    with col2:
        pillars = tech.pillars
        pillar_cols = st.columns(3)
        pillar_names = ["Trend", "Momentum", "Vol/Structure"]
        pillar_scores = [pillars.trend, pillars.momentum, pillars.volume_structure]
        for col, name, score in zip(pillar_cols, pillar_names, pillar_scores):
            with col:
                st.metric(name, f"{score:.0f}")

    st.markdown(f"**Regime:** `{tech.regime}`")

    st.divider()

    details = tech.pillars.details

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Trend Indicators**")
        trend = details.get("trend", {})
        trend_data = {
            "Indicator": ["SMA50 %", "SMA200 %", "SMA200 Slope", "ADX"],
            "Value": [
                f"{trend.get('pct_above_sma50', 'N/A'):.1f}%" if trend.get('pct_above_sma50') is not None else "N/A",
                f"{trend.get('pct_above_sma200', 'N/A'):.1f}%" if trend.get('pct_above_sma200') is not None else "N/A",
                f"{trend.get('sma200_slope_pct', 'N/A'):.1f}%" if trend.get('sma200_slope_pct') is not None else "N/A",
                f"{trend.get('adx', 'N/A'):.1f}" if trend.get('adx') is not None else "N/A",
            ]
        }
        st.dataframe(trend_data, use_container_width=True, hide_index=True)

        st.markdown("**Volume & Structure**")
        vol = details.get("volume_structure", {})
        vol_data = {
            "Metric": ["OBV Trend", "Vol Ratio", "52-Week Range"],
            "Value": [
                f"{vol.get('obv_trend_pct', 'N/A'):.1f}%" if vol.get('obv_trend_pct') is not None else "N/A",
                f"{vol.get('vol_ratio', 'N/A'):.2f}x" if vol.get('vol_ratio') is not None else "N/A",
                f"{vol.get('52w_range_pct', 'N/A'):.1f}%" if vol.get('52w_range_pct') is not None else "N/A",
            ]
        }
        st.dataframe(vol_data, use_container_width=True, hide_index=True)

    with col2:
        st.markdown("**Momentum Indicators**")
        mom = details.get("momentum", {})
        mom_data = {
            "Indicator": ["RSI", "MACD Histogram", "Bollinger %", "Stochastic K", "ROC 20-day", "ROC 60-day"],
            "Value": [
                f"{mom.get('rsi', 'N/A'):.1f}" if mom.get('rsi') is not None else "N/A",
                f"{mom.get('macd_hist', 'N/A'):.4f}" if mom.get('macd_hist') is not None else "N/A",
                f"{mom.get('bb_pct', 'N/A'):.1f}%" if mom.get('bb_pct') is not None else "N/A",
                f"{mom.get('stoch_k', 'N/A'):.1f}" if mom.get('stoch_k') is not None else "N/A",
                f"{mom.get('roc_20_pct', 'N/A'):.1f}%" if mom.get('roc_20_pct') is not None else "N/A",
                f"{mom.get('roc_60_pct', 'N/A'):.1f}%" if mom.get('roc_60_pct') is not None else "N/A",
            ]
        }
        st.dataframe(mom_data, use_container_width=True, hide_index=True)

        st.markdown("**Support & Resistance**")
        sr_data = {
            "Level": ["Support", "Resistance", "52-Week Low", "52-Week High"],
            "Price": [
                f"${tech.nearest_support:.2f}" if tech.nearest_support else "N/A",
                f"${tech.nearest_resistance:.2f}" if tech.nearest_resistance else "N/A",
                f"${tech.week_52_low:.2f}" if tech.week_52_low else "N/A",
                f"${tech.week_52_high:.2f}" if tech.week_52_high else "N/A",
            ]
        }
        st.dataframe(sr_data, use_container_width=True, hide_index=True)

# ── Tab 4: Quantitative Metrics ────────────────────────────────
with tab_quant:
    st.subheader("Quantitative Analysis")
    quant = result.engines.quantitative

    st.metric("Score", f"{quant.score:.0f}/100")

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Factor Scores (Z-Score & Percentile)**")
        factors = quant.factors
        factor_data = {
            "Factor": ["Value", "Quality", "Momentum", "Size", "Low Volatility"],
            "Z-Score": [
                f"{factors.value_z:.2f}" if factors.value_z is not None else "N/A",
                f"{factors.quality_z:.2f}" if factors.quality_z is not None else "N/A",
                f"{factors.momentum_z:.2f}" if factors.momentum_z is not None else "N/A",
                f"{factors.size_z:.2f}" if factors.size_z is not None else "N/A",
                f"{factors.low_volatility_z:.2f}" if factors.low_volatility_z is not None else "N/A",
            ],
            "Percentile": [
                f"{factors.value_pct:.0f}%" if factors.value_pct is not None else "N/A",
                f"{factors.quality_pct:.0f}%" if factors.quality_pct is not None else "N/A",
                f"{factors.momentum_pct:.0f}%" if factors.momentum_pct is not None else "N/A",
                f"{factors.size_pct:.0f}%" if factors.size_pct is not None else "N/A",
                f"{factors.low_volatility_pct:.0f}%" if factors.low_volatility_pct is not None else "N/A",
            ]
        }
        st.dataframe(factor_data, use_container_width=True, hide_index=True)

    with col2:
        st.markdown("**Risk Metrics**")
        risk = quant.risk_metrics
        risk_data = {
            "Metric": ["Beta (1yr)", "Sharpe Ratio (1yr)", "Sharpe Ratio (3yr)",
                      "Sortino (1yr)", "Max Drawdown (3yr)", "Realized Vol (1yr)", "Corr SPY"],
            "Value": [
                f"{risk.beta_1yr:.2f}" if risk.beta_1yr else "N/A",
                f"{risk.sharpe_1yr:.2f}" if risk.sharpe_1yr is not None else "N/A",
                f"{risk.sharpe_3yr:.2f}" if risk.sharpe_3yr is not None else "N/A",
                f"{risk.sortino_1yr:.2f}" if risk.sortino_1yr is not None else "N/A",
                f"{risk.max_drawdown_3yr*100:.1f}%" if risk.max_drawdown_3yr is not None else "N/A",
                f"{risk.realized_vol_1yr*100:.1f}%" if risk.realized_vol_1yr is not None else "N/A",
                f"{risk.corr_spy:.2f}" if risk.corr_spy is not None else "N/A",
            ]
        }
        st.dataframe(risk_data, use_container_width=True, hide_index=True)

    # Fama-French regression if available
    if quant.ff_regression:
        st.divider()
        st.markdown("**Fama-French Regression**")
        ff = quant.ff_regression
        ff_data = {
            "Factor": ["Market (Mkt-RF)", "Size (SMB)", "Value (HML)", "Profitability (RMW)", "Investment (CMA)", "Alpha", "R-Squared"],
            "Beta": [
                f"{ff.mkt_rf_beta:.4f}" if ff.mkt_rf_beta is not None else "N/A",
                f"{ff.smb_beta:.4f}" if ff.smb_beta is not None else "N/A",
                f"{ff.hml_beta:.4f}" if ff.hml_beta is not None else "N/A",
                f"{ff.rmw_beta:.4f}" if ff.rmw_beta is not None else "N/A",
                f"{ff.cma_beta:.4f}" if ff.cma_beta is not None else "N/A",
                f"{ff.alpha_annualized*100:.2f}%" if ff.alpha_annualized is not None else "N/A",
                f"{ff.r_squared:.3f}" if ff.r_squared is not None else "N/A",
            ]
        }
        st.dataframe(ff_data, use_container_width=True, hide_index=True)

# ── Tab 5: Portfolio Sub-Grades ────────────────────────────────
with tab_portfolio:
    st.subheader("Portfolio Sub-Grades")
    st.markdown("Grade for each portfolio risk profile:")

    portfolios = [
        ("Very Conservative", result.portfolios.very_conservative),
        ("Conservative", result.portfolios.conservative),
        ("Balanced", result.portfolios.balanced),
        ("Aggressive", result.portfolios.aggressive),
        ("Very Aggressive", result.portfolios.very_aggressive),
    ]

    for name, pg in portfolios:
        with st.expander(f"{name}: {pg.grade.value} ({pg.composite:.0f}/100)"):
            col1, col2 = st.columns([1, 3])
            with col1:
                grade_emoji = {
                    Grade.STAY_AWAY: "🚫",
                    Grade.SELL: "⚠️",
                    Grade.HOLD: "⚖️",
                    Grade.BUY: "✅",
                    Grade.GOTTA_HAVE: "⭐",
                }
                st.markdown(f"### {grade_emoji[pg.grade]} {pg.grade.value}")
                st.markdown(f"**Score:** {pg.composite:.1f}/100")

            with col2:
                if pg.gate_results:
                    st.markdown("**Eligibility Gates:**")
                    for gate in pg.gate_results:
                        status = "✅ PASS" if gate.passed else "❌ FAIL"
                        value_str = f" (value: {gate.value}, limit: {gate.limit})" if gate.value is not None else ""
                        st.markdown(f"- {status}: {gate.gate}{value_str}")

                if pg.rationale:
                    st.markdown(f"**Rationale:** {pg.rationale}")

# ── Footer with metadata ───────────────────────────────────────
st.divider()
col1, col2, col3 = st.columns(3)
with col1:
    st.caption(f"**Ticker:** {result.ticker}")
with col2:
    st.caption(f"**As of:** {result.as_of.strftime('%Y-%m-%d %H:%M UTC')}")
with col3:
    st.caption(f"**Run ID:** `{result.run_id[:8]}`")
