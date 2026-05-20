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

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Set API keys (required for full fundamental data)
export FMP_API_KEY=your_key    # or $env:FMP_API_KEY = "..." on Windows
export FRED_API_KEY=your_key

# Analyze a ticker (color-coded terminal output + save JSON + MD)
python analyze.py AAPL --save

# JSON output to stdout
python analyze.py MSFT --format json

# Markdown report to stdout
python analyze.py JPM --format md

# Grade through a specific portfolio lens
python analyze.py TSLA --portfolio aggressive

# Bypass cache for fresh data
python analyze.py NVDA --no-cache

# All formats + save files to a custom directory
python analyze.py AAPL --format all --save --runs-dir /path/to/reports

# Daily alert check (watchlist in watchlist.yaml)
python daily_run.py

# Weekly portfolio rebuild (all five funds)
python weekly_run.py

# Dry-run (no live data, no file writes)
python daily_run.py --dry-run
```

## Worked example

With API keys set, `python analyze.py AAPL` produces output like:

```
╭──────────────────── AAPL Analysis ─────────────────────╮
│ AAPL  ✅ BUY ★★★★  67.2/100  confidence 81%           │
│                                                          │
│ ⭐ Gotta Have ≤ $128          |                         │
│ ✅ Buy       ≤ $155           |                         │
│ ⚖️ Hold       $155 – $210     | ◀ current ($182.74)    │
│ ⚠️ Sell      > $210           |                         │
│ 🚫 Stay Away > $256           |                         │
│ Fair Value  $183.00  upside +0.1%                       │
│                                                          │
│ F=70  T=60  Q=58                                        │
│                                                          │
│  VC          CON      BAL      AGG      VA               │
│  🚫 SA       ⚖️ HOLD  ✅ BUY  ✅ BUY  ✅ BUY            │
│                                                          │
│ Drivers:                                                 │
│  + ROIC +9.0ppt above WACC (strong moat)                │
│  + Profitability 78/100                                  │
│  + Quality factor 72nd percentile                        │
│  - Momentum 4th percentile (recent pullback)             │
│  - Valuation 52/100 — P/E near peer median              │
│  - Max drawdown -18.2% (3yr)                            │
╰─────────────────── 2026-05-20 14:30 UTC ───────────────╯
```

The JSON output (`--format json`) matches the canonical schema in spec §13.1.
The Markdown report (`--format md`) contains all nine sections: price ladder,
sensitivity grid, engine breakdown, factor table, risk metrics, and portfolio
sub-grade table.

## Example commands

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
- Overall or any fund sub-grade changed vs. prior run (e.g., `AAPL: Hold → Buy` overall; `MSFT: Hold → Buy` in the Aggressive sleeve)
- Price crossed a grade-boundary on the price ladder (e.g., "dropped below Buy line $88.00")
- Circuit breaker newly triggered (e.g., Altman Z dropped into distress)
- `alert_price` hit for a watchlist entry

**Silence is the signal.** If nothing fires, a single line is printed: "No alerts. N names checked, all grades unchanged, no ladder breaches." A quiet morning is a good morning.

**First run:** no prior baseline exists → state is recorded, no change-alerts fire. A plain "Baseline recorded" message is printed. Run again the next morning to start diffing.

**Robustness:** if a ticker's data is unavailable it is skipped, noted in the run log, and the rest of the watchlist continues. A partial run is far better than a failed run.

**Notification hook:** implement `_send_notification()` in `daily_run.py` and set `notifications.enabled: true` in `config.yaml` to wire alerts to email/SMS/Slack. The seam is there; the integration is left for you.

```bash
python daily_run.py                        # live run, results to terminal + file
python daily_run.py --dry-run              # no data, no writes (exit 2)
python daily_run.py --format md            # Markdown to stdout instead of terminal
python daily_run.py --watchlist custom.yaml
```

**Where output lands:**
```
runs/
  .daily_state.json          ← rolling latest state (diff baseline for next run)
  daily/
    YYYY-MM-DD.json          ← dated state snapshot (append-only history)
    YYYY-MM-DD.md            ← dated alert report
```

**How to read an alert file** (`runs/daily/YYYY-MM-DD.md`):
- `## Actionable Alerts` — tickers that need attention today with bullet-point reasons
- `## Tickers — No Change` — names checked but nothing fired
- `## Skipped` — tickers where data was unavailable (investigate separately)

**How to edit the watchlist** (`watchlist.yaml`):
```yaml
watchlist:
  - ticker: AAPL
    notes: "Core holding candidate"
  - ticker: JNJ
    alert_price: 140.00    # fires when price ≤ this value
```

Runs in ~1–2 minutes on the free FMP tier for a 10–20 name watchlist.

---

### Weekly (`weekly_run.py`)

Runs the full Section 11 portfolio-construction funnel for all five funds:
universe screen → sub-grade / rank → constrained optimization → assemble.

**Output per run** (written to `runs/weekly/YYYY-MM-DD/`):
| File | Contents |
|------|----------|
| `{fund}_holdings.md` | Holdings table: ticker, weight, grade, composite score |
| `{fund}_analytics.json` | Expected return, vol, Sharpe, weights dict |
| `{fund}_mandate.md` | Mandate check: PASS/FAIL + violations |
| `{fund}_funnel.md` | Funnel transparency: how many names passed each of 4 stages |
| `drift_report.md` | Weight drift vs. prior week (enters, exits, Δ weight) |

**Checkpointing:** completed funds are saved to `.checkpoint.json` in the dated folder. An interrupted run resumes from where it left off — delete the checkpoint to force a full rebuild.

**API tier required:** FMP Starter ($14/mo) or higher. The free tier (250 calls/day) cannot complete a full-universe screen in one day. Expected runtime: ~20–60 min for a 3,000-ticker universe on FMP Starter.

```bash
python weekly_run.py                        # rebuild all five funds
python weekly_run.py --funds balanced,aggressive   # subset of funds
python weekly_run.py --dry-run              # validate config only (exit 2)
python weekly_run.py --no-save             # run but don't write files
python weekly_run.py --no-resume           # ignore checkpoint, full rebuild
```

---

### Cron (macOS/Linux)
```cron
# Daily at 7:00 AM weekdays
0 7 * * 1-5  cd /path/to/Portfolio && python daily_run.py >> logs/daily.log 2>&1

# Weekly Sunday at 6:00 AM
0 6 * * 0    cd /path/to/Portfolio && python weekly_run.py >> logs/weekly.log 2>&1
```

### Windows Task Scheduler
1. Open Task Scheduler → **Create Basic Task**
2. **Daily run**: Trigger = Daily, 7:00 AM; in Advanced Settings → check "Repeat Mon–Fri"; Action = `python.exe`, Arguments = `C:\Users\Jonat\OneDrive\Desktop\Portfolio\daily_run.py`; Start in = `C:\Users\Jonat\OneDrive\Desktop\Portfolio`
3. **Weekly run**: Trigger = Weekly, Sunday 6:00 AM; same pattern with `weekly_run.py`
4. In both tasks: set "Run whether user is logged on or not" + "Run with highest privileges" so the task fires unattended

Exit code conventions:
- `0` — success (alerts may or may not be present; check the output)
- `1` — runtime error (data outage, bad config) — scheduler should alert the owner
- `2` — `--dry-run` (intentional, not an error)

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
| 2 | Data layer (yfinance + FMP + cache) | ✅ Complete |
| 3 | Fundamental engine + tests | ✅ Complete |
| 4 | Technical engine + tests | ✅ Complete |
| 5 | Quantitative engine + tests | ✅ Complete |
| 6 | Composite + circuit breakers | ✅ Complete |
| 7 | Price ladder / fair value | ✅ Complete |
| 8 | Portfolio sub-grades | ✅ Complete |
| 9 | Portfolio construction funnel | ✅ Complete |
| 10 | Reporters (JSON / MD / terminal) | ✅ Complete |
| 11 | End-to-end CLI + worked example | ✅ Complete |
| 12 | Backtesting & validation | ✅ Complete |
| 13 | Scheduled scripts (daily / weekly) | ✅ Complete |
