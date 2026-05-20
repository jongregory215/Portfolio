# Stock Analysis Engine

A command-line + library tool that grades a single equity on demand using live market data. Produces an overall grade, five portfolio sub-grades aligned to risk-tiered model portfolios, and a price ladder that defines the boundaries between grades.

## Grade scale

| Grade | Composite (0–100) | Meaning |
|-------|-------------------|---------|
| **Stay Away** | 0–19 | Avoid; distressed or grossly overvalued |
| **Sell** | 20–39 | Exit; unfavorable risk/reward |
| **Hold** | 40–59 | Maintain but don't add |
| **Buy** | 60–79 | Accumulate; favorable risk/reward |
| **Gotta Have** | 80–100 | High-conviction; quality + price + momentum |

The grade reflects **both business quality and current price**. A great company can be a Hold or Sell if it is too expensive.

---

## Setup

### 1. Prerequisites
- Python 3.11+
- Git

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Set environment variables
```bash
# Required for fundamental data
export FMP_API_KEY=your_fmp_key

# Required for risk-free rate (Sharpe, DCF discount rate)
export FRED_API_KEY=your_fred_key

# Optional — override config location
export STOCKGRADER_CONFIG=/path/to/config.yaml
```

On Windows (PowerShell):
```powershell
$env:FMP_API_KEY = "your_fmp_key"
$env:FRED_API_KEY = "your_fred_key"
```

### 4. API tiers required
| Provider | Tier | Cost | Used for |
|----------|------|------|----------|
| yfinance | Free | $0 | Price history, basic info |
| Financial Modeling Prep | Free (250 req/day) or Starter ($14/mo) | [fmp](https://financialmodelingprep.com/developer/docs) | Fundamentals, estimates, universe list |
| FRED | Free | $0 | Risk-free rates |
| Sharadar (Nasdaq Data Link) | Core US Equities (~$50/mo) | Optional | **Backtesting only** — point-in-time fundamentals |

The free FMP tier is sufficient for single-ticker analysis. The weekly full-universe portfolio rebuild requires FMP Starter or higher.

---

## Example commands

```bash
# Analyze a ticker (terminal output)
python analyze.py AAPL

# JSON output
python analyze.py MSFT --format json

# Markdown report
python analyze.py JPM --format md

# Only show Conservative portfolio sub-grade
python analyze.py JNJ --portfolio conservative

# Daily alert check (watchlist)
python daily_run.py

# Weekly portfolio rebuild (all five funds)
python weekly_run.py

# Dry run (no live data, no file writes)
python daily_run.py --dry-run
python weekly_run.py --dry-run
```

---

## Project structure

```
stockgrader/
├── models.py          # Pydantic output schema (AnalysisResult, Grade, PriceLadder, …)
├── config.py          # Config loader (finds and hashes config.yaml)
├── data/
│   ├── base.py        # DataProvider abstract interface
│   ├── cache.py       # Disk cache (TTL-keyed, ~/.stockgrader/cache/)
│   ├── yfinance_provider.py   # Price history (Step 2)
│   └── fmp_provider.py        # Fundamentals + universe (Step 2)
├── engines/
│   ├── base.py        # BaseEngine interface + shared utilities
│   ├── fundamental.py # Valuation, profitability, growth, health, capalloc (Step 3)
│   ├── technical.py   # Trend, momentum, volume/structure (Step 4)
│   └── quantitative.py# Factors, risk metrics, FF regression (Step 5)
├── grading/
│   ├── composite.py   # Weighted composite + circuit breakers (Step 6)
│   └── price_ladder.py# DCF fair value + grade-boundary prices (Step 7)
├── portfolios/
│   ├── sub_grades.py  # Config-driven re-grading per portfolio (Step 8)
│   └── construction.py# Four-stage funnel + optimizer (Step 9)
├── reporting/
│   ├── json_reporter.py
│   ├── markdown_reporter.py
│   └── terminal_reporter.py   # (Step 10)
└── backtest/          # Point-in-time validation engine (Step 12)

analyze.py             # CLI entry point
daily_run.py           # Scheduled: morning watchlist alerts
weekly_run.py          # Scheduled: full portfolio rebuild
config.yaml            # All weights, thresholds, assumptions
watchlist.yaml         # Tickers checked daily
```

---

## Configuration

All tunable parameters live in `config.yaml`. Key sections:

- **`weights.overall`** — fundamental/technical/quantitative split (default 50/30/20)
- **`fundamental.pillar_weights`** — valuation/profitability/growth/health/capalloc weights
- **`portfolios.*`** — per-fund weights, factor preferences, and eligibility gates
- **`fair_value.dcf`** — DCF assumptions (stage years, terminal growth, ERP)
- **`fair_value.margin_of_safety`** — grade-boundary discount/premium bands
- **`circuit_breakers`** — hard caps (Altman Z, FCF, going concern)
- **`bond_candidates`** — bond ETF universe for the bond sleeve
- **`optimizer`** — covariance estimator, expected-return method, constraints

No production weight set should be trusted until it has survived out-of-sample backtesting (Section 12 of the spec). The weights shipped in `config.yaml` are **educated-guess defaults** pending validation.

---

## Implementation decisions (Spec §14)

### 1. Peer set definition
Default: GICS sub-industry peers sourced from FMP. Falls back to GICS industry if fewer than 5 sub-industry peers are available. Configurable via `fundamental.peer_set.method`. Valuation and profitability metrics are scored as percentile ranks within the peer set; solvency/liquidity metrics use absolute thresholds.

### 2. DCF depth
Two-stage DCF: 5-year explicit forecast period + 5-year fade-to-terminal period + Gordon Growth terminal value. All assumptions (ERP, terminal growth, debt spread, tax rate) are in `config.yaml` under `fair_value.dcf`. A reverse-DCF is computed to show what growth rate the current price implies. Sensitivity grid: 3 × 3 (growth delta × discount-rate delta).

### 3. Factor universe
S&P 500 constituents as the cross-sectional universe for factor z-scores and percentile ranking. Configurable to Russell 1000 or Russell 3000 via `quantitative.factor_universe`. A smaller universe keeps the computation tractable on free data tiers; the S&P 500 is a reasonable approximation for large/mid-cap coverage.

### 4. Missing-data policy
Default: **downweight**. Each missing field reduces the effective weight of its parent pillar proportionally to the fraction of total expected fields that are unavailable. Missing fields are listed in `data_quality.missing_fields`; imputed fields (filled with sector median) are in `data_quality.imputed_fields`. The report always shows what was missing.

### 5. Rate limits and API cost
FMP free tier: ~250 requests/day. Cache TTLs: intraday price 15 min, EOD price/fundamentals 24 hr, universe 7 days. Single-ticker analysis typically uses 4–6 FMP calls; the cache means subsequent same-day runs hit the disk. The weekly full-universe screen (thousands of tickers) requires FMP Starter (~$14/mo). The free tier is sufficient for daily watchlist runs on tens of names.

### 6. Backtesting data (PIT)
The backtest engine requires point-in-time, survivorship-bias-free data. **Recommended: Sharadar Core US Equities (SF1/SEP)** via Nasdaq Data Link. yfinance data is NOT point-in-time and the backtest engine will abort (or label output as `INDICATIVE / LOOK-AHEAD-CONTAMINATED`) when handed a non-PIT source. The live scoring engine (Steps 1–11) does not depend on the PIT subscription.

### 7. Full-universe source
FMP `available-traded/list` endpoint filtered by `[NYSE, NASDAQ, AMEX]` exchanges and the liquidity floors in `universe.liquidity_floor`. Refreshed every 7 days (TTL configurable). The universe list is cached; only Stage-1 screening data is fetched per-ticker. yfinance does not expose a clean universe endpoint.

### 8. Expected-return inputs for optimization
Default: **shrunk long-run estimates** — Ledoit-Wolf shrinkage of the sample mean return toward the cross-sectional grand mean. This is more robust than raw trailing returns (which over-fit recent winners) and simpler than full Black-Litterman (which requires market-cap weights and a prior). Black-Litterman mode is available in config; it uses the sub-grade composite (0–100) as the view signal and maps it to a view uncertainty via `optimizer.black_litterman.confidence_scale`. Raw trailing returns are never used as expected-return inputs.

### 9. Optimizer objective defaults
| Fund | Default objective | Rationale |
|------|------------------|-----------|
| Very Conservative | min_variance | Income mandate; minimize volatility |
| Conservative | target_return_mv + risk-parity cross-check | Balance return target with stability |
| Balanced | mean_variance | Standard MV at moderate constraints |
| Aggressive | max_sharpe | Maximize risk-adjusted return |
| Very Aggressive | max_sharpe | Maximize Sharpe; tolerate high vol |

Position cap: 5–10% (fund-specific, see `portfolios.*.position_limits`). Sector cap: 20–30%. Ledoit-Wolf covariance estimator always. Long-only constraint. No more than 40 names fed to the optimizer (Stage 2 cap).

---

## Scheduled runs

### Daily (`daily_run.py`)
Grades the watchlist tickers each morning and emits **only actionable alerts**:
- Overall or portfolio sub-grade changed vs. prior run
- Price crossed a grade-boundary on the ladder
- Circuit breaker newly triggered
- `alert_price` hit for a watchlist entry

Output: color-coded terminal list + dated markdown file in `runs/daily/`. Exit code is nonzero only on failure (not on "alerts present"). Runs in ~1–2 minutes on the free FMP tier.

Add `--dry-run` to test without writing files or hitting live APIs.

### Weekly (`weekly_run.py`)
Runs the full Section 11 portfolio-construction funnel for all five funds: universe screen → sub-grade and rank → constrained optimization → mandate assembly. Outputs per-fund holdings tables, analytics, mandate checks, and a drift report vs. the prior week, written to `runs/weekly/YYYY-MM-DD/`.

Expects FMP Starter+ for the universe endpoint. Implements batched calls, TTL caching, and checkpointing so an interrupted run resumes.

Add `--dry-run` to validate without writing files.

### Cron (macOS/Linux)
```cron
# Daily at 7:00 AM
0 7 * * 1-5  cd /path/to/Portfolio && python daily_run.py >> logs/daily.log 2>&1

# Weekly Sunday at 6:00 AM
0 6 * * 0    cd /path/to/Portfolio && python weekly_run.py >> logs/weekly.log 2>&1
```

### Windows Task Scheduler
1. Open Task Scheduler → Create Basic Task
2. **Daily run**: Trigger = Daily, 7:00 AM (weekdays only → Advanced → repeat Mon–Fri); Action = `python.exe` with arguments `C:\Users\Jonat\OneDrive\Desktop\Portfolio\daily_run.py`; Start in = `C:\Users\Jonat\OneDrive\Desktop\Portfolio`
3. **Weekly run**: Trigger = Weekly, Sunday 6:00 AM; same action with `weekly_run.py`

---

## Backtesting

```bash
# Grade validation — does Gotta Have beat Buy beat Hold beat Sell beat Stay Away?
python -m stockgrader.backtest grades --start 2010-01-01 --end 2024-12-31 --oos-from 2021-01-01

# Walk-forward portfolio backtest (net of costs)
python -m stockgrader.backtest portfolio --fund conservative --rebalance quarterly --costs default

# Weight calibration with strict OOS holdout
python -m stockgrader.backtest calibrate --train 2010-2018 --validate 2019-2020 --test 2021-2024
```

All backtest runs write dated reports to `runs/backtest/`. OOS results are clearly separated from in-sample results in every report. Every headline number is labeled in-sample or out-of-sample. Negative results are reported, not hidden.

**Standing caveat:** past performance does not guarantee future results. A validated backtest reduces — but does not eliminate — the risk that the observed edge is spurious or regime-dependent.

---

## Build status

| Step | Description | Status |
|------|-------------|--------|
| 1 | Schema, config, skeleton | ✅ Complete |
| 2 | Data layer (yfinance + FMP + cache) | ⬜ Pending |
| 3 | Fundamental engine + tests | ⬜ Pending |
| 4 | Technical engine + tests | ⬜ Pending |
| 5 | Quantitative engine + tests | ⬜ Pending |
| 6 | Composite + circuit breakers | ⬜ Pending |
| 7 | Price ladder / fair value | ⬜ Pending |
| 8 | Portfolio sub-grades | ⬜ Pending |
| 9 | Portfolio construction funnel | ⬜ Pending |
| 10 | Reporters (JSON / MD / terminal) | ⬜ Pending |
| 11 | End-to-end CLI + worked example | ⬜ Pending |
| 12 | Backtesting & validation | ⬜ Pending |
| 13 | Scheduled scripts (daily / weekly) | ⬜ Pending |
