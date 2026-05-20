"""
Stock Grader Dashboard — Streamlit app.

Launch with:  streamlit run dashboard.py

Reads the latest runs/weekly/YYYY-MM-DD/universe_grades.json produced by
grade_universe.py and displays:
  - All Tickers tab: full ranked list, filterable by grade / sector / direction
  - One tab per portfolio: optimal holdings at top + all ranked tickers below
  - Click any ticker for a detailed analysis page with price ladder chart
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ──────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Stock Grader Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

_RUNS_DIR = Path("runs") / "weekly"
_ALL_FUNDS = ["very_conservative", "conservative", "balanced",
              "aggressive", "very_aggressive"]
_FUND_LABELS = {
    "very_conservative": "Very Conservative",
    "conservative":      "Conservative",
    "balanced":          "Balanced",
    "aggressive":        "Aggressive",
    "very_aggressive":   "Very Aggressive",
}
_GRADE_ORDER  = ["Gotta Have", "Buy", "Hold", "Sell", "Stay Away"]
_GRADE_COLORS = {
    "Gotta Have": "#1a7f37",
    "Buy":        "#2ea44f",
    "Hold":       "#d29922",
    "Sell":       "#cf6679",
    "Stay Away":  "#b62324",
}
_GRADE_EMOJI = {
    "Gotta Have": "⭐",
    "Buy":        "✅",
    "Hold":       "⚖️",
    "Sell":       "⚠️",
    "Stay Away":  "🚫",
}


# ──────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def load_latest_data() -> dict[str, Any] | None:
    dated_dirs = sorted(
        (d for d in _RUNS_DIR.glob("????-??-??") if d.is_dir()),
        reverse=True,
    )
    for ddir in dated_dirs:
        p = ddir / "universe_grades.json"
        if p.exists():
            try:
                return json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                pass
    return None


def build_df(tickers: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for ticker, rec in tickers.items():
        row = {"Ticker": ticker}
        row.update(rec)
        rows.append(row)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.sort_values("composite", ascending=False).reset_index(drop=True)
    df["Rank"] = df.index + 1
    return df


# ──────────────────────────────────────────────────────────────
# Grade badge helper
# ──────────────────────────────────────────────────────────────

def grade_badge(grade: str) -> str:
    emoji = _GRADE_EMOJI.get(grade, "")
    return f"{emoji} {grade}"


# ──────────────────────────────────────────────────────────────
# Sidebar filters
# ──────────────────────────────────────────────────────────────

def apply_sidebar_filters(df: pd.DataFrame, grade_changes: dict) -> pd.DataFrame:
    st.sidebar.header("Filters")

    # Grade filter
    grades = st.sidebar.multiselect(
        "Overall Grade",
        options=_GRADE_ORDER,
        default=_GRADE_ORDER,
    )
    if grades:
        df = df[df["grade"].isin(grades)]

    # Sector filter
    sectors = sorted(df["sector"].dropna().unique().tolist())
    sel_sectors = st.sidebar.multiselect("Sector", options=sectors, default=sectors)
    if sel_sectors:
        df = df[df["sector"].isin(sel_sectors)]

    # Grade direction filter
    direction = st.sidebar.radio(
        "Grade Direction",
        options=["All", "Upgraded", "Downgraded", "Unchanged"],
        index=0,
    )
    if direction == "Upgraded":
        upgraded = {t for t, c in grade_changes.items()
                    if _grade_rank(c["to"]) > _grade_rank(c["from"])}
        df = df[df["Ticker"].isin(upgraded)]
    elif direction == "Downgraded":
        downgraded = {t for t, c in grade_changes.items()
                      if _grade_rank(c["to"]) < _grade_rank(c["from"])}
        df = df[df["Ticker"].isin(downgraded)]
    elif direction == "Unchanged":
        changed = set(grade_changes.keys())
        df = df[~df["Ticker"].isin(changed)]

    # Search
    search = st.sidebar.text_input("Search ticker", "")
    if search:
        df = df[df["Ticker"].str.contains(search.upper())]

    return df


def _grade_rank(grade: str) -> int:
    return {"Stay Away": 0, "Sell": 1, "Hold": 2, "Buy": 3, "Gotta Have": 4}.get(grade, 2)


# ──────────────────────────────────────────────────────────────
# Price ladder chart
# ──────────────────────────────────────────────────────────────

def price_ladder_chart(ticker: str, rec: dict) -> go.Figure:
    ladder = rec.get("price_ladder", {})
    price  = rec.get("price")

    zones = [
        ("Gotta Have", ladder.get("gotta_have_at"), ladder.get("buy_at"),        "#1a7f37"),
        ("Buy",        ladder.get("buy_at"),         ladder.get("hold_low"),       "#2ea44f"),
        ("Hold",       ladder.get("hold_low"),        ladder.get("hold_high"),      "#d29922"),
        ("Sell",       ladder.get("hold_high"),       ladder.get("sell_above"),     "#cf6679"),
        ("Stay Away",  ladder.get("sell_above"),      ladder.get("stay_away_above"),"#b62324"),
    ]

    fig = go.Figure()

    # Draw zones as horizontal bands
    for label, lo, hi in [(z[0], z[1], z[2]) for z in zones]:
        color = _GRADE_COLORS.get(label, "#888")
        if lo is not None and hi is not None:
            fig.add_shape(
                type="rect",
                x0=0, x1=1, y0=lo, y1=hi,
                fillcolor=color, opacity=0.2,
                line=dict(width=0),
                layer="below",
            )
            fig.add_annotation(
                x=0.02, y=(lo + hi) / 2,
                text=f"{_GRADE_EMOJI.get(label, '')} {label}",
                showarrow=False, xref="paper",
                font=dict(size=11, color=color),
                xanchor="left",
            )

    # Fair value line
    fv = ladder.get("fair_value")
    if fv:
        fig.add_hline(y=fv, line_dash="dot", line_color="white",
                      annotation_text=f"Fair Value ${fv:,.2f}",
                      annotation_position="right")

    # Current price marker
    if price:
        fig.add_trace(go.Scatter(
            x=[0.5], y=[price],
            mode="markers+text",
            marker=dict(size=16, color="white", symbol="diamond",
                        line=dict(color="black", width=2)),
            text=[f"  ${price:,.2f}"],
            textposition="middle right",
            name="Current Price",
            showlegend=False,
        ))

    # Y axis bounds
    all_prices = [v for v in [
        ladder.get("gotta_have_at"), price, ladder.get("stay_away_above"), fv
    ] if v is not None]
    if all_prices:
        pad = (max(all_prices) - min(all_prices)) * 0.15 or 10
        fig.update_yaxes(range=[min(all_prices) - pad, max(all_prices) + pad],
                         tickprefix="$", tickformat=",.0f")

    fig.update_xaxes(visible=False)
    fig.update_layout(
        title=f"{ticker} — Price Ladder",
        height=400,
        margin=dict(l=10, r=120, t=40, b=10),
        paper_bgcolor="#0e1117",
        plot_bgcolor="#0e1117",
        font=dict(color="white"),
    )
    return fig


# ──────────────────────────────────────────────────────────────
# Engine score chart
# ──────────────────────────────────────────────────────────────

def engine_chart(rec: dict) -> go.Figure:
    labels  = ["Fundamental", "Technical", "Quantitative"]
    values  = [rec.get("fund_score"), rec.get("tech_score"), rec.get("quant_score")]
    colors  = ["#2ea44f" if (v or 0) >= 60 else "#d29922" if (v or 0) >= 40 else "#b62324"
               for v in values]
    values  = [v or 0 for v in values]

    fig = go.Figure(go.Bar(
        x=values, y=labels,
        orientation="h",
        marker_color=colors,
        text=[f"{v:.1f}" for v in values],
        textposition="outside",
    ))
    fig.add_vline(x=60, line_dash="dot", line_color="white", opacity=0.4,
                  annotation_text="Buy threshold", annotation_position="top right")
    fig.update_xaxes(range=[0, 105])
    fig.update_layout(
        title="Engine Scores (0–100)",
        height=220,
        margin=dict(l=10, r=60, t=40, b=10),
        paper_bgcolor="#0e1117",
        plot_bgcolor="#0e1117",
        font=dict(color="white"),
    )
    return fig


# ──────────────────────────────────────────────────────────────
# Ticker detail panel
# ──────────────────────────────────────────────────────────────

def show_ticker_detail(ticker: str, rec: dict, grade_changes: dict) -> None:
    grade = rec["grade"]
    color = _GRADE_COLORS.get(grade, "#888")

    st.markdown(f"## {_GRADE_EMOJI.get(grade,'')} {ticker} — {grade}")
    st.markdown(f"**Sector:** {rec.get('sector','—')}  |  "
                f"**Composite:** {rec.get('composite','—')}/100  |  "
                f"**Confidence:** {int((rec.get('confidence') or 0)*100)}%")

    if ticker in grade_changes:
        ch = grade_changes[ticker]
        st.info(f"Grade changed this week: {ch['from']} → {ch['to']}")

    # Metrics row
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Price",      f"${rec['price']:,.2f}"      if rec.get("price")      else "—")
    c2.metric("Fair Value", f"${rec['fair_value']:,.2f}" if rec.get("fair_value") else "—")
    c3.metric("Upside",
              f"{rec['upside_pct']:+.1f}%"               if rec.get("upside_pct") is not None else "—")
    c4.metric("Piotroski F", str(int(rec["piotroski_f"])) if rec.get("piotroski_f") is not None else "—")
    c5.metric("Beta",        f"{rec['beta']:.2f}"         if rec.get("beta")       else "—")

    # Charts
    col_l, col_r = st.columns([3, 2])
    with col_l:
        st.plotly_chart(price_ladder_chart(ticker, rec), use_container_width=True)
    with col_r:
        st.plotly_chart(engine_chart(rec), use_container_width=True)

        # Portfolio sub-grades
        st.markdown("**Portfolio Sub-Grades**")
        pg = rec.get("portfolio_grades", {})
        pc = rec.get("portfolio_composites", {})
        for fund in _ALL_FUNDS:
            g = pg.get(fund, "—")
            score = pc.get(fund)
            emoji = _GRADE_EMOJI.get(g, "")
            score_str = f"  ({score:.0f})" if score else ""
            st.markdown(f"{emoji} **{_FUND_LABELS[fund]}**: {g}{score_str}")

    # Drivers
    col_pos, col_neg = st.columns(2)
    with col_pos:
        st.markdown("**Positive drivers**")
        for d in rec.get("drivers_positive", []):
            st.markdown(f"+ {d}")
    with col_neg:
        st.markdown("**Negative drivers**")
        for d in rec.get("drivers_negative", []):
            st.markdown(f"- {d}")

    # Key stats
    st.markdown("---")
    s1, s2, s3 = st.columns(3)
    s1.metric("Altman Z",    f"{rec['altman_z']:.2f}"    if rec.get("altman_z")    is not None else "—")
    s2.metric("ROIC vs WACC",f"{rec['roic_vs_wacc']:+.1%}" if rec.get("roic_vs_wacc") is not None else "—")
    s3.metric("Sharpe (1yr)",f"{rec['sharpe']:.2f}"      if rec.get("sharpe")      is not None else "—")


# ──────────────────────────────────────────────────────────────
# Ticker table
# ──────────────────────────────────────────────────────────────

def show_ticker_table(df: pd.DataFrame, grade_changes: dict,
                      portfolio_col: str | None = None) -> str | None:
    """Render a compact ticker table and return the selected ticker (or None)."""
    if df.empty:
        st.info("No tickers match the current filters.")
        return None

    score_col = portfolio_col or "composite"

    display = pd.DataFrame({
        "Rank":      df["Rank"],
        "Ticker":    df["Ticker"],
        "Grade":     df["grade"].map(lambda g: grade_badge(g)),
        "Score":     df[score_col].map(lambda v: f"{v:.1f}" if pd.notna(v) else "—"),
        "Price":     df["price"].map(lambda v: f"${v:,.2f}" if pd.notna(v) else "—"),
        "Fair Value":df["fair_value"].map(lambda v: f"${v:,.2f}" if pd.notna(v) else "—"),
        "Upside":    df["upside_pct"].map(lambda v: f"{v:+.1f}%" if pd.notna(v) else "—"),
        "Sector":    df["sector"],
        "Regime":    df["regime"].map(lambda v: str(v).replace("_", " ") if pd.notna(v) else "—"),
        "Change":    df["Ticker"].map(
            lambda t: f"{grade_changes[t]['from']} → {grade_changes[t]['to']}"
            if t in grade_changes else ""
        ),
    })

    selected = st.dataframe(
        display,
        use_container_width=True,
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        height=min(600, 40 + len(display) * 35),
    )

    if selected and selected.selection and selected.selection.rows:
        row_idx = selected.selection.rows[0]
        return df.iloc[row_idx]["Ticker"]
    return None


# ──────────────────────────────────────────────────────────────
# Grade distribution chart
# ──────────────────────────────────────────────────────────────

def grade_dist_chart(df: pd.DataFrame) -> go.Figure:
    counts = df["grade"].value_counts().reindex(_GRADE_ORDER, fill_value=0)
    fig = go.Figure(go.Bar(
        x=counts.index,
        y=counts.values,
        marker_color=[_GRADE_COLORS[g] for g in counts.index],
        text=counts.values,
        textposition="outside",
    ))
    fig.update_layout(
        title="Grade Distribution",
        height=260,
        margin=dict(l=10, r=10, t=40, b=10),
        paper_bgcolor="#0e1117",
        plot_bgcolor="#0e1117",
        font=dict(color="white"),
        showlegend=False,
    )
    return fig


def sector_chart(df: pd.DataFrame) -> go.Figure:
    sector_grade = (
        df.groupby(["sector", "grade"])
          .size()
          .unstack(fill_value=0)
          .reindex(columns=_GRADE_ORDER, fill_value=0)
    )
    fig = go.Figure()
    for grade in _GRADE_ORDER:
        if grade in sector_grade.columns:
            fig.add_trace(go.Bar(
                name=grade,
                x=sector_grade.index,
                y=sector_grade[grade],
                marker_color=_GRADE_COLORS[grade],
            ))
    fig.update_layout(
        barmode="stack",
        title="Grades by Sector",
        height=300,
        margin=dict(l=10, r=10, t=40, b=80),
        paper_bgcolor="#0e1117",
        plot_bgcolor="#0e1117",
        font=dict(color="white"),
        xaxis_tickangle=-30,
        legend=dict(orientation="h", y=-0.3),
    )
    return fig


# ──────────────────────────────────────────────────────────────
# Portfolio tab
# ──────────────────────────────────────────────────────────────

def show_portfolio_tab(fund: str, data: dict, full_df: pd.DataFrame,
                       grade_changes: dict) -> str | None:
    portfolio = data.get("portfolios", {}).get(fund, {})
    holdings  = portfolio.get("holdings", [])
    analytics = portfolio.get("analytics", {})

    # ── Optimal portfolio ─────────────────────────────────────
    st.subheader(f"Optimal {_FUND_LABELS[fund]} Portfolio")

    if analytics:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Expected Return", f"{analytics.get('projected_return',0):.1%}")
        c2.metric("Volatility",      f"{analytics.get('projected_volatility',0):.1%}")
        c3.metric("Weighted Beta",   f"{analytics.get('weighted_beta',0):.2f}")
        c4.metric("Holdings",        analytics.get("n_holdings", 0))

    if holdings:
        eq = [h for h in holdings if h.get("sleeve") == "equity"]
        bd = [h for h in holdings if h.get("sleeve") == "bond"]
        if eq:
            st.markdown("**Equity holdings**")
            eq_df = pd.DataFrame(eq)[["ticker","weight","grade","composite","sector"]]
            eq_df["weight"] = eq_df["weight"].map(lambda v: f"{v:.1%}")
            eq_df["composite"] = eq_df["composite"].map(lambda v: f"{v:.1f}")
            eq_df["grade"] = eq_df["grade"].map(grade_badge)
            st.dataframe(eq_df, hide_index=True, use_container_width=True,
                         height=min(400, 50 + len(eq_df)*35))
        if bd:
            st.markdown("**Bond sleeve**")
            bd_df = pd.DataFrame(bd)[["ticker","weight","grade","composite"]]
            bd_df["weight"] = bd_df["weight"].map(lambda v: f"{v:.1%}")
            st.dataframe(bd_df, hide_index=True, use_container_width=True)
    else:
        st.info("No holdings — funnel returned empty for this portfolio. "
                "Run grade_universe.py --portfolios to populate.")

    st.divider()

    # ── Full ticker ranking for this portfolio ────────────────
    st.subheader(f"All Tickers — Ranked for {_FUND_LABELS[fund]}")

    score_key = f"portfolio_composites"
    fund_df = full_df.copy()
    fund_df["_port_score"] = fund_df["portfolio_composites"].map(
        lambda d: d.get(fund, 0) if isinstance(d, dict) else 0
    )
    fund_df["_port_grade"] = fund_df["portfolio_grades"].map(
        lambda d: d.get(fund, "Hold") if isinstance(d, dict) else "Hold"
    )
    fund_df = fund_df.sort_values("_port_score", ascending=False).reset_index(drop=True)
    fund_df["Rank"] = fund_df.index + 1

    # Show portfolio score instead of overall composite
    display = pd.DataFrame({
        "Rank":          fund_df["Rank"],
        "Ticker":        fund_df["Ticker"],
        "Portfolio Grade": fund_df["_port_grade"].map(grade_badge),
        "Port Score":    fund_df["_port_score"].map(lambda v: f"{v:.1f}"),
        "Overall Grade": fund_df["grade"].map(grade_badge),
        "Overall Score": fund_df["composite"].map(lambda v: f"{v:.1f}"),
        "Price":         fund_df["price"].map(lambda v: f"${v:,.2f}" if pd.notna(v) else "—"),
        "Upside":        fund_df["upside_pct"].map(lambda v: f"{v:+.1f}%" if pd.notna(v) else "—"),
        "Sector":        fund_df["sector"],
    })

    selected = st.dataframe(
        display,
        use_container_width=True,
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        height=500,
    )

    if selected and selected.selection and selected.selection.rows:
        row_idx = selected.selection.rows[0]
        return fund_df.iloc[row_idx]["Ticker"]
    return None


# ──────────────────────────────────────────────────────────────
# Main app
# ──────────────────────────────────────────────────────────────

def main() -> None:
    st.title("📊 Stock Grader Dashboard")

    data = load_latest_data()
    if data is None:
        st.error("No universe_grades.json found. Run `python grade_universe.py` first.")
        st.code("python grade_universe.py")
        return

    run_date     = data.get("run_date", "unknown")
    n_tickers    = data.get("n_tickers", 0)
    grade_changes = data.get("grade_changes", {})
    tickers_data  = data.get("tickers", {})

    st.caption(f"Last run: **{run_date}**  |  {n_tickers} tickers analysed  |  "
               f"{len(grade_changes)} grade changes vs prior week")

    full_df = build_df(tickers_data)
    if full_df.empty:
        st.warning("Data loaded but no tickers found.")
        return

    # Apply sidebar filters to All Tickers tab only
    filtered_df = apply_sidebar_filters(full_df.copy(), grade_changes)

    # ── Session state for selected ticker ─────────────────────
    if "selected_ticker" not in st.session_state:
        st.session_state["selected_ticker"] = None

    # ── Tabs ──────────────────────────────────────────────────
    tab_labels = ["🏆 All Tickers"] + [
        f"📁 {_FUND_LABELS[f]}" for f in _ALL_FUNDS
    ]
    tabs = st.tabs(tab_labels)

    # Tab 0: All Tickers
    with tabs[0]:
        col_dist, col_sect = st.columns([1, 2])
        with col_dist:
            st.plotly_chart(grade_dist_chart(filtered_df), use_container_width=True)
        with col_sect:
            st.plotly_chart(sector_chart(filtered_df), use_container_width=True)

        if grade_changes:
            with st.expander(f"Grade changes this week ({len(grade_changes)})"):
                upgrades   = [(t, c) for t, c in grade_changes.items()
                              if _grade_rank(c["to"]) > _grade_rank(c["from"])]
                downgrades = [(t, c) for t, c in grade_changes.items()
                              if _grade_rank(c["to"]) < _grade_rank(c["from"])]
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("**Upgrades**")
                    for t, c in sorted(upgrades):
                        st.markdown(f"✅ **{t}**: {c['from']} → {c['to']}")
                with c2:
                    st.markdown("**Downgrades**")
                    for t, c in sorted(downgrades):
                        st.markdown(f"⚠️ **{t}**: {c['from']} → {c['to']}")

        st.subheader("All Tickers — Ranked by Composite Score")
        selected = show_ticker_table(filtered_df, grade_changes)
        if selected:
            st.session_state["selected_ticker"] = selected

    # Tabs 1–5: Portfolio tabs
    for i, fund in enumerate(_ALL_FUNDS):
        with tabs[i + 1]:
            selected = show_portfolio_tab(fund, data, full_df, grade_changes)
            if selected:
                st.session_state["selected_ticker"] = selected

    # ── Ticker detail (shown below all tabs when a ticker is selected) ─
    sel = st.session_state.get("selected_ticker")
    if sel and sel in tickers_data:
        st.divider()
        col_title, col_close = st.columns([10, 1])
        with col_close:
            if st.button("✕ Close", key="close_detail"):
                st.session_state["selected_ticker"] = None
                st.rerun()
        show_ticker_detail(sel, tickers_data[sel], grade_changes)


if __name__ == "__main__":
    main()
