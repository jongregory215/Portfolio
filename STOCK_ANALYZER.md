# Stock Analyzer Dashboard

Interactive web dashboard for comprehensive single-stock analysis. Enter a ticker, click analyze, and view complete metrics across 5 detailed tabs.

## Quick Start

```bash
streamlit run stock_analyzer.py
```

This opens `http://localhost:8501` in your browser.

## Features

### Input & Controls
- **Ticker input:** Enter any equity ticker (AAPL, MSFT, TSLA, etc.)
- **Deep analysis:** Optional FMP API integration for richer fundamentals (requires `FMP_API_KEY` env var)
- **Cache control:** Skip disk cache to fetch live data

### Display Tabs

#### 1. **Price Ladder**
- Current price with zone indicator (Gotta Have → Buy → Hold → Sell → Stay Away)
- All 5 grade boundary prices
- Fair value estimate with upside/downside percentage
- Implied growth rate (reverse-DCF)

#### 2. **Fundamental Metrics** 
All metrics from the 5 fundamental pillars:
- **Valuation:** P/E (trailing/forward), P/B, P/S, EV/EBITDA, PEG
- **Profitability:** Gross/operating/net margin, ROE, ROA, ROIC vs WACC
- **Growth:** Revenue & EPS CAGR (3yr, 5yr), forward growth estimates
- **Financial Health:** Current/quick ratio, D/E, interest coverage, Altman Z, Piotroski F
- **Capital Allocation:** FCF conversion, yield, payout ratio, share dilution, ROIC

#### 3. **Technical Metrics**
All technical indicators across 3 pillars:
- **Trend:** SMA50/200 deviations, ADX, SMA200 slope, regime label
- **Momentum:** RSI, MACD histogram, Bollinger %, Stochastic K, ROC (20/60-day)
- **Volume/Structure:** OBV trend, volume ratio, support/resistance levels, 52-week range

#### 4. **Quantitative Metrics**
- **Factor Scores:** Value, Quality, Momentum, Size, Low-Volatility (z-scores + percentiles)
- **Risk Metrics:** Beta, Sharpe (1yr/3yr), Sortino, max drawdown, realized volatility, correlations
- **Fama-French (if available):** Factor betas, alpha, R-squared

#### 5. **Portfolio Sub-Grades**
Grade results for each of 5 risk profiles:
- **Very Conservative** — lowest beta/volatility
- **Conservative** — balanced risk
- **Balanced** — neutral weighting
- **Aggressive** — higher growth/momentum focus
- **Very Aggressive** — highest risk/return

For each: grade, composite score, eligibility gates (with pass/fail status), rationale.

## Key Metrics Explained

### Grade Scale
- 🚫 **Stay Away:** 0–19 (avoid)
- ⚠️ **Sell:** 20–39 (exit)
- ⚖️ **Hold:** 40–59 (maintain)
- ✅ **Buy:** 60–79 (accumulate)
- ⭐ **Gotta Have:** 80–100 (core position)

### Price Ladder Zones
Price targets computed from DCF (two-stage) + multiples-based fair value, adjusted for quality:
- **Gotta Have:** ≤ FV × (1 - 30%) — strong buy
- **Buy:** ≤ FV × (1 - 15%) — good entry
- **Hold:** FV × (1 - 15%) to FV × (1 + 15%) — fair value range
- **Sell:** > FV × (1 + 15%) — reduce position
- **Stay Away:** > FV × (1 + 40%) — extreme overvaluation

### Scoring Engines
- **Fundamental (50% weight):** DCF implied growth, financial health, profitability, valuation, capital efficiency
- **Technical (30% weight):** Price momentum, trend strength, regime, support/resistance
- **Quantitative (20% weight):** Factor exposures (value/quality/momentum), risk-adjusted returns

## Examples

### Analyze Apple (AAPL)
1. Type `AAPL` in the ticker input
2. Click "🔍 Analyze"
3. View results in tabs

### Deep analysis with FMP
1. Set `FMP_API_KEY` in `.env` or shell: `export FMP_API_KEY=your_key`
2. Check "Use FMP (deep)" box
3. Analyze

### Skip cache for live data
1. Check "Skip cache" box
2. Analyze (slower, but always fresh data)

## Data Sources

- **Prices & fundamentals:** yfinance (always available)
- **Rich fundamentals & peer metrics:** FMP (optional, requires API key)
- **Risk-free rate:** FRED (DGS3MO)
- **Factor data:** Ken French Data Library (optional)

## Session State

Results persist in the browser session (`st.session_state`). Changing ticker or clicking Analyze again updates the display without losing prior context.

## Troubleshooting

**"Invalid ticker"**
- Check ticker symbol (must be valid US equity)
- Some OTC/delisted tickers may not have full data

**"Missing data" / "N/A" in metrics**
- Small-cap or young companies may lack historical data
- Try a larger-cap ticker (AAPL, MSFT, etc.)

**FMP metrics not showing**
- `FMP_API_KEY` not set, or API limit exceeded (250 calls/day on free tier)
- Uncheck "Use FMP (deep)" to use yfinance only

**Slow analysis**
- First run caches data to disk (~30s)
- Subsequent runs use cache (< 1s)
- Uncheck "Skip cache" to use cached data

## Architecture

```
stock_analyzer.py
├── Streamlit UI (input form, tabs, display)
├── stockgrader.pipeline.run_analysis() — analysis engine
├── stockgrader.data.fetcher.DataFetcher — data fetch & normalize
├── stockgrader.engines — fundamental/technical/quantitative scoring
├── stockgrader.grading.price_ladder — DCF + multiples valuation
└── stockgrader.models — type-safe data models
```

## Compare with CLI

- **CLI** (`python analyze.py AAPL`) — terminal output, quick lookup, headless/scripting
- **Web Dashboard** (this app) — interactive, complete metrics, visual layout, browser-based

Use the dashboard for deep analysis; use CLI for quick checks or scripting.
