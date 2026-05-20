# Stock Analysis Engine — Build Specification

> **Handoff document for Claude Code.** This spec defines a stock analysis system that ingests live market data, scores a stock across fundamental, technical, and quantitative dimensions, and produces (1) an overall grade, (2) per-portfolio sub-grades aligned to five risk-tiered models, and (3) price targets that define the boundaries between grades.
>
> Read this whole document before writing code. Section 14 contains open implementation decisions — make a reasonable call on each, document the choice in the README, and flag anything you think the owner should revisit.

---

## 1. Purpose & Scope

Build a command-line + library tool (Python) that analyzes a single equity ticker on demand using **live data** and returns a structured, reproducible grade. The output must be both human-readable (terminal + markdown report) and machine-readable (JSON) so it can later feed a portfolio dashboard.

The system grades each stock in two layers:

1. **Overall Grade** — a universal, portfolio-agnostic assessment of the business and its current price.
2. **Portfolio Sub-Grades** — the same stock re-graded through the lens of each of five risk-tiered model portfolios. A stock can be a `Buy` overall and `Gotta Have` for the Very Aggressive sleeve while being `Stay Away` for the Very Conservative sleeve.

Both layers use the same five-category scale:

| Grade | Meaning | Numeric band (0–100 composite) |
|-------|---------|-------------------------------|
| **Stay Away** | Avoid entirely; structurally impaired, distressed, or grossly overvalued | 0–19 |
| **Sell** | Exit existing positions; risk/reward unfavorable | 20–39 |
| **Hold** | Maintain but don't add; fairly valued or mixed signal | 40–59 |
| **Buy** | Accumulate; favorable risk/reward at current price | 60–79 |
| **Gotta Have** | High-conviction; rare combination of quality + price + momentum | 80–100 |

> **Important grading principle:** The grade is a function of *both the business quality AND the current price*. A wonderful company can be a `Hold` or `Sell` if it's too expensive; a mediocre company can be a `Buy` if it's cheap enough. This is why price targets (Section 8) are first-class outputs.

---

## 2. High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                          CLI / API                            │
│   analyze TICKER [--portfolio all] [--format json|md|term]    │
└───────────────────────────────┬─────────────────────────────┘
                                │
                ┌───────────────▼───────────────┐
                │        Orchestrator            │
                │  (pipeline runner + caching)   │
                └───────────────┬───────────────┘
                                │
        ┌───────────────────────┼───────────────────────┐
        ▼                       ▼                       ▼
┌───────────────┐      ┌───────────────┐       ┌───────────────┐
│ Data Layer    │      │ Scoring Layer │       │ Grading Layer │
│ (live feeds)  │─────▶│ F / T / Q     │──────▶│ overall +     │
│ + normalize   │      │ sub-scores    │       │ portfolio +   │
│ + cache       │      │               │       │ price targets │
└───────────────┘      └───────────────┘       └───────────────┘
                                │
                                ▼
                       ┌───────────────┐
                       │  Reporters    │
                       │ json/md/term  │
                       └───────────────┘
```

Keep the three scoring engines (Fundamental, Technical, Quantitative) as independent, individually testable modules with a shared interface so weights and components can be tuned without touching the orchestrator.

---

## 3. Data Layer

### 3.1 Required data points

**Fundamental (from financial statements + market data):**
- Income statement: revenue, gross profit, operating income, net income, EPS (TTM + 5yr history)
- Balance sheet: total assets, total liabilities, total equity, cash & equivalents, total debt, current assets/liabilities
- Cash flow: operating cash flow, free cash flow, capex, dividends paid
- Per-share & valuation: P/E (trailing + forward), P/B, P/S, EV/EBITDA, PEG, dividend yield, payout ratio
- Growth: revenue CAGR (3/5yr), EPS CAGR (3/5yr), forward growth estimates
- Profitability: gross/operating/net margins, ROE, ROA, ROIC
- Health: current ratio, quick ratio, debt/equity, interest coverage, Altman Z-score, Piotroski F-score
- Sector & industry classification (for peer-relative scoring)

**Technical (from price/volume history, min 2 years daily):**
- OHLCV daily bars
- Moving averages: SMA/EMA 20, 50, 100, 200
- Momentum: RSI(14), MACD(12,26,9), Stochastic, rate-of-change
- Trend: ADX, directional movement
- Volatility: ATR, Bollinger Bands, historical volatility (realized)
- Volume: OBV, volume-weighted trends, accumulation/distribution
- Relative strength vs. S&P 500 and vs. sector ETF
- Key levels: 52-week high/low, support/resistance via swing points

**Quantitative (computed or sourced):**
- Beta (vs. SPY), 1yr and 3yr
- Factor exposures: value, quality, momentum, size, low-volatility (see Section 6)
- Sharpe & Sortino (trailing 1/3yr)
- Maximum drawdown (trailing 3yr)
- Correlation to SPY and to the bond proxy (AGG)
- Downside deviation, semi-variance
- Optional: Fama-French 3- or 5-factor regression loadings if data permits

### 3.2 Data sources (live)

Implement a **provider abstraction** (`DataProvider` interface) so the source can be swapped. Suggested primary/fallback chain:

| Layer | Primary | Fallback / notes |
|-------|---------|------------------|
| Price/volume | `yfinance` (free, no key) | Alpha Vantage, Tiingo, Polygon.io |
| Fundamentals | Financial Modeling Prep (FMP) API | Alpha Vantage `OVERVIEW`, Finnhub |
| Estimates/forward | FMP analyst estimates | Finnhub |
| Macro (risk-free rate) | FRED API (`DGS3MO`, `DGS10`) | hardcode fallback w/ warning |

- Read API keys from environment variables (`FMP_API_KEY`, etc.). **Never hardcode keys.**
- Build a **disk cache** (e.g., `~/.stockgrader/cache/`) keyed by ticker + date + endpoint, with a configurable TTL (default: intraday price 15 min, fundamentals 24 hr). This protects against rate limits and makes runs reproducible within a session.
- Every data point must carry a timestamp and source. If a required field is missing, the scoring module must degrade gracefully (impute or down-weight) and the final report must list what was missing and how it was handled.

---

## 4. Scoring Layer Overview

Each engine outputs a **0–100 sub-score** plus a structured breakdown of its components. The three sub-scores are combined into a composite via configurable weights. Default top-level weights:

```
composite = 0.50 * fundamental + 0.30 * technical + 0.20 * quantitative
```

These weights are **portfolio-dependent** — see Section 7. The 50/30/20 split is the baseline used for the *overall* grade. Make all weights live in a single `config.yaml` so they can be tuned without code changes.

Each component score should be computed on a normalized basis. Two normalization methods, selectable per component:

1. **Absolute thresholds** — e.g., current ratio > 2.0 = full marks. Good for health metrics with well-established danger zones.
2. **Peer-relative percentile** — rank vs. sector/industry peers, score = percentile. Good for valuation and margins, which are only meaningful relative to a peer set.

Document which metrics use which method. Prefer peer-relative for valuation/profitability, absolute for solvency/liquidity.

---

## 5. Fundamental Engine (default weight 50%)

Produce sub-scores for five pillars, then weight them:

| Pillar | Default weight | Example components |
|--------|---------------|-------------------|
| **Valuation** | 30% | P/E, P/FCF, EV/EBITDA, PEG, P/S — peer-relative percentile + vs. own 5yr history |
| **Profitability** | 25% | Gross/op/net margins, ROE, ROIC vs. WACC, margin trend |
| **Growth** | 20% | Revenue & EPS CAGR (3/5yr), forward estimates, growth durability |
| **Financial health** | 15% | Debt/equity, interest coverage, current ratio, Altman Z, Piotroski F |
| **Capital allocation** | 10% | FCF conversion, buyback discipline, dividend safety, ROIC trend, reinvestment runway |

**Key derived signals to implement explicitly:**
- **ROIC vs. WACC spread** — the single best proxy for whether the business creates value. Positive and widening = quality moat signal.
- **Quality of earnings** — net income vs. operating cash flow divergence; flag if accruals are inflating earnings.
- **Margin trajectory** — 3yr trend in operating margin, not just the level.
- **Altman Z-score** banding for distress risk (Z < 1.8 = distress zone → caps the overall grade at `Sell` or worse regardless of other signals).
- **Piotroski F-score** (0–9) as a financial-strength overlay.

Implement at least one absolute "**circuit breaker**" rule set (Section 9) so a fundamentally distressed company cannot receive a `Buy`/`Gotta Have` no matter how attractive the technicals look.

---

## 6. Quantitative Engine (default weight 20%)

This is where the owner's math background is leveraged. Implement:

### 6.1 Factor model
Compute the stock's loadings on standard equity factors. At minimum, build a cross-sectional z-score for each factor; ideally run a time-series regression where data allows.

- **Value**: composite z of earnings yield, FCF yield, book-to-market
- **Quality**: composite z of ROE, accruals (inverse), leverage (inverse), earnings stability
- **Momentum**: 12-1 month return (skip most recent month), risk-adjusted
- **Size**: market-cap decile (small = positive loading)
- **Low volatility**: inverse of trailing realized vol / beta

Output each factor as a z-score and a percentile vs. the chosen universe (default: S&P 500 constituents; make universe configurable).

### 6.2 Fama-French regression (optional but encouraged)
If you can source the FF factor returns (Ken French data library is freely downloadable), regress the stock's excess returns on FF3 (Mkt-RF, SMB, HML) and FF5 (add RMW, CMA). Report:
- Factor betas + t-stats
- Alpha (annualized) + significance
- R²

### 6.3 Risk metrics
- Beta (1yr, 3yr) vs. SPY
- Sharpe, Sortino (trailing 1yr, 3yr), using FRED risk-free rate
- Max drawdown (3yr) and current drawdown from peak
- Downside deviation
- Correlation to SPY and AGG (this feeds portfolio fit in Section 7)

### 6.4 Quant score construction
Combine factor alignment + risk-adjusted return quality into the 0–100 quant sub-score. Reward: positive risk-adjusted returns, favorable factor loadings (configurable which factors are "favorable"), low correlation that aids diversification. Penalize: high beta with no return premium, deep current drawdown with deteriorating momentum.

---

## 7. Technical Engine (default weight 30%)

Score three pillars:

| Pillar | Default weight | Components |
|--------|---------------|-----------|
| **Trend** | 40% | Price vs. SMA50/200, golden/death cross state, ADX strength, slope of 200-SMA |
| **Momentum** | 35% | RSI(14) regime, MACD histogram + signal cross, relative strength vs. SPY & sector |
| **Volume/structure** | 25% | OBV trend, accumulation/distribution, proximity to support/resistance, breakout/breakdown detection |

Technical scoring should be **regime-aware**: an RSI of 75 is a momentum positive in a strong uptrend but an overbought warning in a range. Document the regime logic.

The technical engine also feeds **entry-timing** for price targets — e.g., identifying the nearest meaningful support level becomes a candidate `Buy`-below price.

---

## 8. Price Targets (first-class output)

For every stock, compute the price levels at which the **overall grade** changes. The owner explicitly wants: *"Stock ABC might be a hold at $100 but a buy at $88."*

### 8.1 Method
1. Build a **fair-value estimate** using a blend of:
   - Multiple-based: apply a justified forward multiple (peer + own-history blended, quality-adjusted) to forward EPS/FCF.
   - Intrinsic: a transparent DCF (document all assumptions — growth, terminal rate, discount rate = WACC). Keep it simple, explicit, and sensitivity-tested.
   - Optional: reverse-DCF to show what growth the current price implies.
2. Treat fair value as the centerpoint. Derive grade-boundary prices by applying margin-of-safety bands around fair value, **modulated by the quality/quant scores** (higher-quality businesses earn tighter required margins of safety).

### 8.2 Required output: a price ladder
For each ticker, output the price thresholds:

```
Gotta Have  ≤  $X1     (deep discount to fair value, high MoS)
Buy         ≤  $X2
Hold         range $X2 – $X3   (around fair value)
Sell        ≥  $X3
Stay Away   ≥  $X4     (extreme overvaluation) — OR triggered by circuit breaker
```

- Show the current price's position on this ladder.
- Show implied upside/downside to fair value (%).
- Run a **sensitivity table**: fair value under low/base/high assumptions for growth and discount rate (a small grid). The owner's math background means a clean sensitivity matrix is more useful than a single point estimate.

> Price targets are for the *overall* grade. Portfolio sub-grades (Section 7→9) may shift the *category* a given price implies for a specific sleeve, but the price ladder itself is computed once per stock.

---

## 9. Grading Layer — Overall Grade

### 9.1 Composite → grade
1. Compute weighted composite (Section 4) on 0–100.
2. Apply **circuit breakers** (hard caps) before banding:
   - Altman Z < 1.8 → cap at `Sell`
   - Negative & worsening FCF with high leverage → cap at `Sell`
   - Going-concern / data integrity failure → `Stay Away`
   - Price > `Stay Away` valuation threshold → cap at `Sell` even if quality is elite (it's a great company at a terrible price)
3. Map the (possibly capped) composite to a grade band (Section 1 table).
4. Attach a **confidence score** reflecting data completeness and signal agreement across the three engines (high divergence = low confidence, surfaced in the report).

### 9.2 Explainability (required)
Every grade must ship with a human-readable rationale: the top 3 positive drivers, top 3 negative drivers, any circuit breakers triggered, and the price-ladder position. No black boxes — the owner needs to audit the logic.

---

## 10. Portfolio Sub-Grades

Re-grade the same stock for each of five risk-tiered model portfolios. Sub-grades reuse the engine sub-scores but apply **portfolio-specific weights, factor preferences, and eligibility filters**.

### 10.1 The five portfolios

| Portfolio | Stock/Bond context | What it wants from an *equity* holding |
|-----------|-------------------|----------------------------------------|
| **Very Conservative** (Risk 1–20) | 15/85 | Only the safest equities: low beta, high dividend safety, large-cap, low drawdown, fortress balance sheet. Most stocks should grade low here. |
| **Conservative** (Risk 21–40) | 30/70 | Quality dividend payers, low-vol, defensive sectors, modest valuation. |
| **Balanced** (Risk 41–60) | 55/45 | Blend of quality growth + value; moderate beta; the "default" lens closest to the overall grade. |
| **Aggressive** (Risk 61–80) | 75/25 | Growth, momentum, higher beta tolerated if risk-adjusted returns justify it. |
| **Very Aggressive** (Risk 81–99) | 90/10 | Maximum growth/momentum; tolerates high volatility and rich valuations for asymmetric upside. |

### 10.2 How sub-grades differ from the overall grade

Each portfolio defines a **weight profile** and **preference adjustments**:

```yaml
# Example: config weights per portfolio (illustrative — tune these)
very_conservative:
  weights:        { fundamental: 0.55, technical: 0.15, quantitative: 0.30 }
  factor_pref:    { low_vol: ++, quality: ++, value: +, momentum: -, size: large_only }
  eligibility:
    max_beta: 0.9
    min_market_cap: 10_000_000_000
    require_dividend: true
    max_drawdown_3yr: 0.35
    min_altman_z: 3.0
  valuation_penalty: high     # rich valuation hurts more here

balanced:
  weights:        { fundamental: 0.50, technical: 0.30, quantitative: 0.20 }
  factor_pref:    { quality: +, value: +, momentum: +, low_vol: neutral }
  eligibility:    { max_beta: 1.4, min_market_cap: 2_000_000_000 }
  valuation_penalty: medium

very_aggressive:
  weights:        { fundamental: 0.35, technical: 0.40, quantitative: 0.25 }
  factor_pref:    { momentum: ++, growth: ++, size: small_ok, low_vol: -- }
  eligibility:    { max_beta: null, min_market_cap: 300_000_000 }
  valuation_penalty: low      # willing to pay up for growth
```

**Eligibility filters** are gates: a stock that fails a portfolio's hard filter (e.g., beta 1.8 in Very Conservative) is automatically `Stay Away` for that sleeve, regardless of composite. This is the mechanism that makes "a Buy for aggressive ≠ a Buy for conservative" concrete.

### 10.3 Sub-grade output
For each portfolio: grade, composite score, which eligibility gates passed/failed, and a one-line rationale ("Excellent business but beta 1.6 and no dividend → fails Very Conservative mandate").

---

## 11. Portfolio Construction — Build the Optimal Portfolio per Fund

After grading, construct the **ideal portfolio for each of the five funds**. The owner's goal is to find the optimal holdings to achieve each fund's mandate *in the long run*, screening from the broadest possible universe ("every ticker that is trading"). This is impossible to do as a naive single-pass optimization — see the warning below — so it is built as a **funnel**: the entire tradable universe enters at Stage 1, but only robust survivors reach the optimizer.

> **Critical methodological warning — read before implementing.** You cannot run mean-variance optimization directly over thousands of tickers. The covariance matrix becomes unstable (estimating millions of pairwise covariances from limited return history), and the optimizer becomes an "error maximizer" — producing concentrated, fragile weights that look optimal in-sample and fail out-of-sample. Likewise, optimizing on trailing returns fits the past, not the future. The funnel architecture below exists specifically to avoid both failure modes. Do not shortcut it.

### 11.1 The four-stage funnel

Run this independently for each fund (the example narrative is for **Conservative: 30% equity / 70% bonds, 3–5% target return, near-retiree income focus**, but the same machinery runs for all five with fund-specific parameters).

**Stage 1 — Universe screen (full market in → ~50–250 names out).**
Apply the fund's hard eligibility gates (Section 10.2) to *every* tradable ticker. Pull the full listed universe (US equities + ETFs; make the universe configurable and document the source — e.g., the union of major exchange listings from the data provider). Apply liquidity floors first (min average dollar volume, min price to exclude penny stocks) to drop the long tail cheaply, then the fund's mandate gates. For Conservative this is approximately: dividend payer, sustainable payout ratio, beta ≤ ~0.9, large-cap, high Altman Z, bounded 3yr max drawdown. This is a cheap filter that runs before expensive scoring.

**Stage 2 — Grade & rank (~50–250 → ~30–60 names).**
Run survivors through the fund's sub-grade engine. Keep only `Buy` and `Gotta Have` for that fund. Rank by sub-grade composite. Cap the candidate pool at a configurable size (default ~40) to keep the optimizer well-conditioned.

**Stage 3 — Robust optimization (on the survivors only).**
Now the covariance matrix is estimable. Build the optimal weights for the **equity sleeve** and, separately, the **bond sleeve** (see 11.3). Requirements:
- Use a **shrinkage covariance estimator** (Ledoit-Wolf) rather than raw sample covariance.
- Use **robust/regularized expected-return inputs** — do *not* feed raw trailing returns. Prefer one of: capital-market-assumption-based returns, a Black-Litterman blend that uses the sub-grade conviction as the "views" overlay, or shrunk/winsorized long-run estimates. Document the choice.
- **Constrain the optimizer** so it cannot produce fragile corner solutions: per-position cap (e.g., ≤ 5–8%), per-sector cap (e.g., ≤ 25%), minimum position size (no dust), long-only by default.
- Offer multiple objective functions, selectable in config: mean-variance (target the fund's return band at min variance), **risk-parity** (often more robust for income mandates), and minimum-variance. Default for Conservative: target-return mean-variance with a risk-parity cross-check.

**Stage 4 — Honor the mandate (assemble the full fund).**
Combine sleeves to the fund's fixed split. Conservative = 30% equity sleeve (Stage 3 equity result) + 70% bond sleeve. The split is a hard constraint, not an optimizer output — the equity/bond ratio *is* the fund's risk identity. Then validate the assembled portfolio against the fund's target: does projected return land in the 3–5% band at acceptable projected volatility? If not, report the gap rather than forcing it.

### 11.2 Per-fund parameters

Each fund supplies its own funnel parameters in config:

| Fund | Equity / Bond | Stage 1 equity gates (illustrative) | Default objective | Return target |
|------|--------------|-------------------------------------|-------------------|---------------|
| Very Conservative | 15 / 85 | beta ≤ 0.8, large-cap, div required, Z ≥ 3.0, low drawdown | min-variance | 2–3% |
| Conservative | 30 / 70 | beta ≤ 0.9, div required, sustainable payout, Z ≥ 2.7 | target-return MV + risk-parity check | 3–5% |
| Balanced | 55 / 45 | beta ≤ 1.2, quality+value tilt | mean-variance | 5–7% |
| Aggressive | 75 / 25 | beta ≤ 1.5, growth/momentum tilt | mean-variance (max Sharpe) | 7–9% |
| Very Aggressive | 90 / 10 | no beta cap, momentum/growth | max-Sharpe / max-return at vol cap | 9–12% |

### 11.3 The bond sleeve

The bond sleeve is **optimized too**, but over a curated set of bond ETFs rather than individually graded stocks (individual bonds are out of scope for v1). Build a bond-ETF candidate set spanning the relevant axes — duration (short / intermediate / long), credit (Treasury / aggregate / investment-grade corporate / TIPS), and yield. Optimize the sleeve for the fund's needs: for Conservative/near-retiree income, weight toward shorter-to-intermediate duration and high credit quality, optimizing for yield per unit of duration/credit risk. Hold the sleeve's *size* fixed at the mandate (70% for Conservative); optimize only its internal composition. Make the bond candidate set configurable so the owner can pin it to the bond ETFs already used in the live RIA model portfolios if preferred.

### 11.4 Output per fund

For each fund, output:
- **Holdings table**: ticker, name, weight, sleeve (equity/bond), fund sub-grade, contribution to portfolio risk.
- **Portfolio analytics**: projected long-run return (with method stated), projected volatility, projected yield, weighted beta, sector exposures, effective number of holdings (diversification), and projected max drawdown.
- **Mandate check**: does projected return fall in the fund's target band? Pass/fail with the gap quantified.
- **Funnel transparency**: how many tickers entered Stage 1, survived each stage, and the final candidate pool — so the owner can audit the narrowing.
- **Rebalance note**: this is a point-in-time ideal; flag that it should be re-run on a schedule and that turnover/tax costs (not modeled in v1) matter in practice.

### 11.5 Honesty requirements (do not skip)

- **Label projections as projections.** Long-run return/volatility estimates are model outputs under stated assumptions, not promises. Surface the assumptions next to the numbers.
- **Report robustness, not just the point solution.** Include a brief sensitivity check (e.g., how weights shift under the risk-parity alternative vs. mean-variance) so the owner sees how stable the "optimal" portfolio actually is.
- **This is decision-support, not an auto-allocator.** The constructed portfolio informs the advisor's judgment and must still pass suitability/fiduciary review per client.

---

## 12. Backtesting & Validation

> **Why this section is the most important one in the spec.** Every grade band, engine weight, factor preference, and margin-of-safety threshold in this system is currently an *educated guess*. Until validated, the tool produces confident-sounding grades with unknown predictive value — precision without proven accuracy, which is the dangerous failure mode for anything touching client money. Backtesting is what converts the system from "plausible" to "trustworthy." It is a required capability, not a v2 nice-to-have. **No weight set should be trusted in production until it has survived out-of-sample validation here.**

### 12.1 The cardinal rule: point-in-time data, no look-ahead

The backtest is worthless — worse than worthless, actively misleading — if it uses data that wasn't knowable on the simulated date. This is the hardest part of the build and the part most likely to be silently wrong. Requirements:

- **As-reported (point-in-time) fundamentals.** Use the financials as they were *originally reported*, not later restatements. Today's databases show restated history; backtesting on restated numbers is look-ahead bias.
- **Reporting lag.** A fiscal-quarter's financials were not public on the quarter-end date. Apply a realistic lag (e.g., do not let the model "see" a 10-Q until its actual filing date). Make the lag explicit and configurable.
- **Survivorship-bias-free universe.** Delisted, acquired, and bankrupt tickers **must remain in the historical universe** on the dates they were live. A universe of only today's survivors systematically excludes the failures the system most needs to have graded `Stay Away` — and inflates every backtest result. This is non-negotiable.
- **Point-in-time index/sector membership** for peer-relative scoring (a stock's GICS peers and index membership change over time).
- **No future prices anywhere** in feature construction.

**Required data sources (proper point-in-time — recommend tiers in README):**
| Need | Recommended | Notes |
|------|-------------|-------|
| PIT fundamentals + survivorship-free equities | **Sharadar Core US Equities (SF1/SEP)** via Nasdaq Data Link | Affordable PIT fundamentals with delisted names retained; strong fit for an independent shop. ~low-hundreds/yr tier. |
| Factor returns | **Ken French Data Library** (free) | FF3/FF5 historical factor returns. |
| Alternative / higher-end | **CRSP**, **Compustat Point-in-Time**, **FactSet**, **Norgate Data** (futures/equities, survivorship-free) | More expensive; document if owner wants institutional-grade. |
| Risk-free / macro | **FRED** (free) | Treasury rates for Sharpe/discounting. |

The backtest engine **must detect and refuse to run silently on non-PIT data.** If handed a source that can't guarantee point-in-time/survivorship-free semantics, it either aborts with a clear error or runs only in an explicitly-labeled "INDICATIVE / LOOK-AHEAD-CONTAMINATED — DO NOT TRUST" mode. Never let a contaminated backtest produce clean-looking numbers.

### 12.2 Layer 1 — Grade validation (does a grade predict anything?)

The core question: **do forward returns improve monotonically across the grade scale?** If `Sell` names beat `Buy` names historically, the weights are wrong and must be recalibrated before use.

For a historical window, on each rebalance date, grade the full PIT universe, then track forward returns by grade bucket. Report:
- **Forward-return spread by grade** — mean/median forward return (1m, 3m, 6m, 12m) for each of Stay Away → Gotta Have. The headline test is monotonicity: Gotta Have > Buy > Hold > Sell > Stay Away.
- **Quintile/decile spread & long-short** — return of a Gotta-Have-minus-Stay-Away basket; is it positive and significant?
- **Information Coefficient (IC)** — rank correlation between composite score and forward return; report mean IC, IC volatility, and IC information ratio (IC mean / IC std).
- **Hit rate** — % of time each grade's directional call was right.
- **Risk-adjusted** — not just raw returns; Sharpe/Sortino by bucket, since higher grades may simply be taking more risk.
- **Per-engine attribution** — run the same validation using fundamental-only, technical-only, quant-only scores to see which engine actually carries the predictive load. This justifies (or refutes) the 50/30/20 weighting.

Also validate the **price ladder**: when price crossed below the computed `Buy` line, did forward returns actually improve as the model implies? This tests whether the fair-value/margin-of-safety machinery means anything.

### 12.3 Layer 2 — Walk-forward portfolio backtesting (with costs)

Validate the Section 11 construction funnel as it would have run in real time.

- **Walk-forward only.** On each historical rebalance date, construct each fund's portfolio using *only* data available then (PIT universe → screen → grade → optimize → assemble). Hold to the next rebalance. Never use the full-sample anything.
- **Per-fund realized performance** vs. its mandate: did Conservative actually deliver its 3–5% band at acceptable vol? Report realized return, volatility, max drawdown, Sharpe/Sortino, and worst 12-month period for each of the five funds.
- **Net of costs.** Model transaction costs (commissions where applicable + a configurable spread/slippage assumption) and **turnover**. A portfolio that rebalances into the "optimal" names weekly may be wrecked by costs and taxes; the backtest must show net, not gross. Report annual turnover per fund.
- **Tax awareness (flag, even if not fully modeled).** Note realized short- vs. long-term gain mix as a turnover proxy; full tax modeling can be v2 but surface the exposure.
- **Benchmark each fund** against a passive analog (e.g., Conservative vs. a 30/70 VTI/BND blend). Beating the mandate is necessary; beating the cheap passive alternative net of costs is the real bar.
- **Rebalance frequency** as a tunable parameter — show how performance and turnover trade off across monthly/quarterly/semi-annual.

### 12.4 Layer 3 — Weight calibration / optimization against history

Tune the system's free parameters (engine weights, pillar weights, grade-band cutoffs, per-fund factor preferences, margin-of-safety bands) against historical performance — **with discipline to avoid overfitting**, which is the cardinal sin here.

- **Strict train/validation/test split.** Calibrate on a training era, select on a validation era, and report final numbers **only** on a never-touched out-of-sample test era. Reserve the most recent N years as a final holdout.
- **Walk-forward / expanding-window calibration**, not single in-sample fit. Re-tune periodically using only prior data; measure performance on the subsequent unseen window.
- **Objective is configurable** — calibrate to maximize long-short IC, or fund-level risk-adjusted return net of costs, or grade monotonicity. Default: maximize out-of-sample IC information ratio subject to monotonicity holding.
- **Guard against overfitting explicitly:** penalize parameter complexity, prefer robust/flat optima over sharp peaks, cross-validate across non-overlapping periods, and **report in-sample vs. out-of-sample degradation** prominently. A strategy that's brilliant in-sample and mediocre out-of-sample is mediocre — say so loudly.
- **Regime checks.** Report performance separately across distinct market regimes (bull, bear, high-vol, rising-rate) so the owner sees where the system works and where it doesn't. A single blended number hides this.
- **Parameter stability.** Show how chosen weights move across calibration windows; wildly unstable weights are a red flag that the "edge" is noise.

### 12.5 Outputs & honesty requirements

- A **backtest report** (markdown + JSON) with: grade-bucket forward-return table, IC stats, long-short curve, per-fund walk-forward equity curves vs. benchmarks (net of costs), turnover, regime breakdown, and the in-sample-vs-out-of-sample comparison front and center.
- **Every headline number labeled** in-sample or out-of-sample. OOS is the only number that counts.
- **State assumptions next to results:** cost model, slippage, rebalance frequency, reporting lag, data source and its PIT guarantees.
- **No cherry-picking.** Report the full test period and all five funds, including the ones that underperform. Negative results are the point of the exercise.
- **Standing caveat in the report:** past performance does not guarantee future results; a validated backtest reduces — but does not eliminate — the risk that the edge is spurious or regime-dependent. This is decision-support evidence, not a guarantee.

### 12.6 CLI

```
backtest grades   --start 2010-01-01 --end 2024-12-31 --oos-from 2021-01-01
backtest portfolio --fund conservative --rebalance quarterly --costs default
backtest calibrate --train ... --validate ... --test ...   # weight optimization
```

All three must run from PIT data, refuse to run silently on contaminated data, and write dated reports to `runs/backtest/`.

---

## 13. Outputs

### 13.1 JSON (machine-readable, canonical)
```json
{
  "ticker": "ABC",
  "as_of": "2026-05-19T14:30:00Z",
  "price": 100.00,
  "overall": {
    "grade": "Hold",
    "composite": 54.2,
    "confidence": 0.78,
    "drivers_positive": ["ROIC 18% vs WACC 9%", "..."],
    "drivers_negative": ["P/E 32 vs peer median 19", "..."],
    "circuit_breakers": []
  },
  "price_ladder": {
    "gotta_have_at": 72.00,
    "buy_at": 88.00,
    "hold_range": [88.00, 112.00],
    "sell_above": 112.00,
    "stay_away_above": 140.00,
    "fair_value": 100.00,
    "fair_value_sensitivity": { "low": 84, "base": 100, "high": 121 },
    "upside_to_fv_pct": 0.0
  },
  "engines": {
    "fundamental": { "score": 58, "pillars": { "...": "..." } },
    "technical":   { "score": 49, "pillars": { "...": "..." } },
    "quantitative":{ "score": 51, "factors": { "value": 0.3, "momentum": -0.4, "...": "..." } }
  },
  "portfolios": {
    "very_conservative": { "grade": "Stay Away", "composite": 31, "failed_gates": ["require_dividend","max_beta"], "rationale": "..." },
    "conservative":      { "grade": "Hold",      "composite": 45, "failed_gates": [], "rationale": "..." },
    "balanced":          { "grade": "Hold",      "composite": 54, "failed_gates": [], "rationale": "..." },
    "aggressive":        { "grade": "Buy",       "composite": 66, "failed_gates": [], "rationale": "..." },
    "very_aggressive":   { "grade": "Buy",       "composite": 71, "failed_gates": [], "rationale": "..." }
  },
  "data_quality": { "missing_fields": [], "sources": { "...": "..." } }
}
```

### 13.2 Markdown report
A clean per-stock report: header with grade + price ladder, a section per engine with the component breakdown, the portfolio sub-grade table, the sensitivity matrix, and the explainability block. This is what the owner reads.

### 13.3 Terminal
Color-coded summary: ticker, overall grade, current price vs. ladder, and the five portfolio grades in a compact row.

---

## 14. Open Implementation Decisions (make a call, document it)

1. **Peer set definition** — how to pick the comparison universe for percentile scoring (sector via GICS? sub-industry? market-cap band?). Pick a default, make it configurable.
2. **DCF depth** — single-stage vs. two-stage. Recommend two-stage with explicit + terminal phases; keep assumptions in config.
3. **Factor universe** — S&P 500 as the cross-sectional universe is simplest. Document if you broaden it.
4. **Missing-data policy** — exact down-weighting / imputation rules per field.
5. **Rate-limit & cost** — FMP/Alpha Vantage free tiers are limited. Decide caching TTLs and document required API tiers.
6. **Backtesting** — now a full required capability (Section 12), not a deferred hook. The grading and construction functions must be pure (point-in-time inputs → grade/portfolio out) so they can be replayed historically. See Section 12 for the point-in-time data requirement and validation methodology.
7. **Full-universe source & refresh** — how to enumerate "every tradable ticker" for the Stage 1 screen (exchange listing files? provider universe endpoint?), and how often to refresh that list. Document the source and its coverage limits.
8. **Expected-return inputs for optimization** — the single most consequential choice in portfolio construction. Pick the approach (capital-market assumptions, Black-Litterman with sub-grade conviction as views, or shrunk long-run estimates) and document why. Do not use raw trailing returns.
9. **Optimizer objective defaults per fund** — confirm the default objective function for each of the five funds (Section 11.2) and the position/sector constraint values.

---

## 15. Tech Stack & Quality Bar

- **Language:** Python 3.11+
- **Core libs:** `pandas`, `numpy`, `scipy`, `statsmodels` (for regressions), `yfinance`, `requests`, `pydantic` (typed models for the output schema), `pyyaml`, `rich` (terminal), `click` or `typer` (CLI)
- **Optimization & backtesting:** `cvxpy` or `scipy.optimize` (constrained portfolio optimization), `scikit-learn` (Ledoit-Wolf shrinkage covariance, train/test discipline), and a PIT data adapter for the backtest (e.g., Sharadar via `nasdaqdatalink`). Keep the backtest engine dependency-isolated so the live tool doesn't require the PIT subscription to run.
- **Structure:** clean package layout (`data/`, `engines/`, `grading/`, `portfolios/`, `reporting/`, `config.yaml`), typed throughout.
- **Testing:** unit tests per engine with fixture data (don't hit live APIs in tests — use cached fixtures); golden-file tests for the report output; at least one end-to-end test on a cached ticker.
- **Reproducibility:** a run on cached data must be deterministic. Log the exact inputs and config hash in every report.
- **README:** setup, env vars, example commands, an explanation of the grading methodology, and a clear list of every default/assumption you chose in Section 14.

---

## 16. Suggested Build Order

1. Output schema (pydantic models) + `config.yaml` + project skeleton.
2. Data layer with one provider (yfinance for price, FMP for fundamentals) + caching.
3. Fundamental engine + unit tests on fixtures.
4. Technical engine + tests.
5. Quantitative engine (factors + risk) + tests.
6. Composite + circuit breakers + overall grade.
7. Price-ladder / fair-value module + sensitivity.
8. Portfolio sub-grade layer (config-driven weights + eligibility gates).
9. Portfolio construction funnel (Section 11): universe screen → grade/rank → robust optimization → mandate assembly, for all five funds + bond sleeve.
10. Reporters (JSON → markdown → terminal).
11. End-to-end CLI, README, and a worked example: a real ticker analysis *and* a constructed portfolio for at least one fund (Conservative).
12. **Backtesting & validation (Section 12)** — PIT data adapter (refuse-on-contamination), grade validation, walk-forward portfolio backtest with costs, then weight calibration with strict OOS holdout. Treat this as the gate before trusting any weights in production; expect to loop back and revise engine weights based on what it finds.
13. Scheduled-run scripts (Section 17): `daily_run.py` (watchlist, alerts-only) and `weekly_run.py` (full portfolio rebuild).

> **Sequencing note:** steps 1–11 build a *runnable* system on default (guessed) weights. Step 12 is what tells you whether those weights are any good — plan to iterate between 12 and the engine weights (steps 3–8) before relying on the daily alerts from step 13.

---

## 17. Scheduled Runs — Daily & Weekly Automation

The owner runs this as an unattended routine, not interactively. Build **two entry-point scripts** so it can be scheduled with cron (macOS/Linux) or Task Scheduler (Windows) and reviewed each morning over coffee. Both are thin wrappers over the existing library — no analysis logic lives here that isn't already in the engines.

### 16.1 `daily_run.py` — morning alerts (fast, watchlist-scoped)

**Purpose:** grade a small, owner-maintained watchlist every morning and surface **only actionable alerts**. This is intentionally *not* a full snapshot — silence means nothing actionable happened, which is itself the signal.

**Scope:** a config-driven watchlist (`watchlist.yaml` — tickers the owner actually cares about, plus any currently held positions). Default sized for tens of names, runs in a couple of minutes on standard data tiers.

**An "alert" fires when any of these is true for a watchlist name:**
- **Grade change** vs. the last run — overall grade or any fund sub-grade moved (e.g., `AAPL: Buy → Hold` overall; `MSFT: Hold → Buy` for Aggressive sleeve).
- **Price-ladder breach** — price crossed a grade-boundary on the ladder (e.g., "crossed *below* its $88.00 Buy line," "entered Gotta-Have zone ≤ $72.00," "exceeded Sell threshold $112.00").
- **Circuit breaker newly triggered** (e.g., Altman Z dropped into distress).
- **Target reached** — hit a price the owner flagged in the watchlist (optional per-ticker `alert_price`).

**State & change detection:** persist each run's grades and ladder positions to a dated store (e.g., `runs/daily/YYYY-MM-DD.json`). Each morning, diff against the most recent prior run to compute changes. First run has no baseline → record state, emit no change-alerts, say so plainly.

**Output (alerts-only):**
- Terminal: a tight, color-coded alert list. If nothing fired: print one line — "No alerts. N names checked, all grades unchanged, no ladder breaches." (Silence is a valid, good outcome.)
- A dated markdown alert file written to `runs/daily/` for the record.
- Optional notification hook (config flag, off by default): a function stub the owner can wire to email/SMS/Slack later. Don't build the integration; leave the seam.
- Exit code: nonzero only on *failure* (data outage, etc.), not on "alerts present" — so the scheduler can distinguish a broken run from a quiet market. Consider a separate signal if the owner wants to be pinged only when alerts exist; document the convention.

**Robustness:** must never hang or crash the schedule. Wrap data calls with timeouts + retries; if a ticker's data is unavailable, skip it, note it in the run log, and continue. A partial run that flags 18 of 20 names is far better than a failed run that flags none.

### 16.2 `weekly_run.py` — full portfolio rebuild (heavy)

**Purpose:** run the full Section 11 construction funnel across the broad universe and produce the ideal portfolio for each of the five funds. This is the expensive job that cannot run daily on standard data tiers.

**Scope:** full tradable universe → funnel → five fund portfolios + bond sleeves. Expect a long runtime; design for it (see below).

**Scheduling:** weekly cron (e.g., Sunday pre-market) so fresh target portfolios are ready for the week.

**Output:**
- Per-fund portfolio files (holdings, analytics, mandate check, funnel transparency) per Section 11.4, written to a dated folder `runs/weekly/YYYY-MM-DD/`.
- A **drift report** vs. the prior weekly run: which names entered/left each fund portfolio and how weights shifted — the weekly analog of the daily change-alert.

**Runtime & rate-limit handling (critical):** the full-universe screen is thousands of tickers. Implement: aggressive caching (Stage-1 screening data has a longer TTL than intraday prices), batched/paginated provider calls, checkpointing so an interrupted run resumes rather than restarts, and a configurable concurrency cap that respects the provider's rate limit. Document the expected runtime and the API tier actually required — if the free tier can't complete a weekly full-universe run in a reasonable window, say so explicitly in the README and recommend the specific paid tier.

### 16.3 Shared requirements
- Both scripts read the same `config.yaml` + the typed output schema; they import the library, never re-implement scoring.
- Both write dated, append-only history so the owner accumulates a time series (useful later for the backtesting work in Section 12).
- Setup docs in the README: exact cron lines *and* Windows Task Scheduler steps, where output lands, how to edit the watchlist, and how to read an alert file.
- A `--dry-run` flag on both (use cached data, write nothing) so the owner can test the schedule safely.

---

## 18. Definition of Done

- `analyze TICKER` runs end-to-end on live data and emits JSON + markdown + terminal output.
- Overall grade, price ladder with sensitivity, and all five portfolio sub-grades are present and internally consistent (e.g., failed eligibility gate ⇒ `Stay Away` for that sleeve).
- Every grade has an explainability block; no unexplained numbers.
- Circuit breakers demonstrably prevent distressed names from grading `Buy`+.
- Portfolio construction produces an ideal portfolio per fund via the Section 11 funnel: every tradable ticker enters Stage 1, the optimizer runs only on robust survivors, the equity/bond split is honored as a hard constraint, and the bond sleeve is composed from the bond-ETF set. Each fund portfolio ships with its analytics, mandate check, and funnel transparency.
- Projections are labeled as projections with assumptions surfaced; robustness/sensitivity is reported alongside the point solution.
- **Backtesting (Section 12) runs on point-in-time, survivorship-bias-free data and refuses to run silently on contaminated data.** Grade validation reports forward-return monotonicity + IC; portfolio backtest is walk-forward and net of costs/turnover; weight calibration uses a strict out-of-sample holdout with in-sample-vs-OOS degradation reported. Negative results are reported, not hidden.
- No production weight set is blessed until it has survived OOS validation; the README states which weights are validated vs. still default guesses.
- `daily_run.py` grades the watchlist, emits **alerts only** (grade changes, price-ladder breaches, new circuit breakers, target hits), diffs against the prior run, writes dated history, and prints a clean "no alerts" line when the market is quiet. It degrades gracefully on missing data and never crashes the schedule.
- `weekly_run.py` runs the full Section 11 funnel for all five funds, writes dated per-fund portfolios + a drift report vs. the prior week, and handles rate limits via caching/checkpointing/concurrency caps.
- Both scheduled scripts support `--dry-run`, share `config.yaml`, and have cron + Windows Task Scheduler setup documented in the README.
- Weights, thresholds, peer logic, and portfolio profiles all live in config, not code.
- Tests pass on cached fixtures; a documented example run is in the README.
- Section 14 decisions are all made and written down.

---

### Notes from the owner
- Mathematical depth is welcome — favor transparent, well-documented quantitative methods over hidden heuristics.
- The five portfolio tiers correspond to the owner's RIA model portfolios (Very Conservative → Very Aggressive, mapped to risk scores 1–99).
- This is decision-support tooling, not an automated trading system. Grades inform human judgment.
- The owner runs this as a scheduled routine: a fast daily watchlist alert check (alerts-only — silence means nothing actionable) and a heavy weekly full-universe portfolio rebuild. Optimize the daily path for speed and signal-to-noise; optimize the weekly path for completeness and robustness.
