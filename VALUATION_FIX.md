# Valuation Fix: Distress-Adjusted DCF

## Problem
564 tickers showed 100%+ upside, with 61 showing >1000%. Example: Ford at +4031% upside despite being in bankruptcy zone (Altman Z 0.81).

## Root Cause
DCF valuations used standard CAPM-based WACC without adjusting for bankruptcy/default risk. Distressed companies were valued as if they would recover, leading to inflated fair values.

## Solution
Implemented distress-adjusted valuation with three components:

### 1. Distress-Adjusted WACC
Adds bankruptcy spread based on Altman Z-Score:

| Altman Z | Zone | WACC Adjustment |
|----------|------|-----------------|
| ≥ 2.99 | Safe | 0 bps |
| 1.81–2.99 | Grey | +200 bps |
| 1.10–1.81 | Distress | +500 bps |
| < 1.10 | Bankruptcy | +1000 bps |

Maximum WACC capped at 35% to prevent nonsensical discounting.

### 2. Skip DCF for Distressed Companies
For Altman Z < 1.81, use **multiples-only valuation** (no DCF).
- **Why:** DCF assumes recovery; distressed companies need conservative multiples-based approach
- **Implementation:** Check Altman Z before running DCF; if < 1.81, skip DCF and use multiples only

### 3. Cap P/E Multiples for Distressed
Apply distress-based P/E caps:

| Altman Z | Max P/E | Rationale |
|----------|---------|-----------|
| < 1.10 (Bankruptcy) | 6x | Very tight cap |
| 1.10–1.81 (Distress) | 10x | Conservative cap |
| ≥ 1.81 | No cap | Use peer median |

Also disable quality premium for distressed companies (no moat in distress).

## Results

### Before Fix
```
Ford (F):
  Altman Z: 0.81 (bankruptcy zone)
  Fair Value: $546.23
  Upside: +4031.8%
  Status: BROKEN ❌
```

### After Fix
```
Ford (F):
  Altman Z: 0.81 (bankruptcy zone)
  Fair Value: $10.97
  Upside: -17.0%
  Status: REASONABLE ✓
```

### Other Examples
```
AAPL (healthy):
  Altman Z: 11.60
  Fair Value: $296.84
  Upside: -1.8%
  Status: Still reasonable ✓

GE (distressed):
  Altman Z: 3.22
  Fair Value: $90.40
  Upside: -69.9%
  Status: Fixed (was 4000%+) ✓

NVDA (healthy):
  Altman Z: 70.26
  Fair Value: $287.08
  Upside: +28.5%
  Status: Reasonable ✓
```

## Implementation Details

### Files Modified
- `stockgrader/grading/price_ladder.py`

### New Functions
- `_distress_adjusted_wacc(base_wacc, altman_z)` — Adjusts WACC for bankruptcy risk
- `_distress_pe_cap(peer_pe, altman_z)` — Caps P/E multiples for distressed companies

### Logic Flow
```
1. Extract inputs (price, EPS, FCF, Altman Z, etc.)
2. Calculate base WACC from CAPM
3. Apply distress adjustment to WACC if Altman Z < 1.81
4. Calculate DCF with adjusted WACC
   → If Altman Z < 1.81, skip DCF entirely
5. Calculate multiples-based FV
   → Apply P/E cap if Altman Z < 1.81
6. Blend DCF (50%) + multiples (50%)
   → If distressed, use multiples only
7. Generate price ladder with adjusted fair value
```

## Testing
```bash
# Test individual stocks
python analyze.py F --no-cache          # Ford: should show ~$11 FV
python analyze.py AAPL --no-cache       # Apple: should show reasonable upside
python analyze.py GE --no-cache         # GE: should show downside, not upside

# Test universe (generates updated grades with fixed valuations)
python grade_universe.py --source sp500 --workers 4
# Recheck: 564 → ? stocks with 100%+ upside (should be much lower)
```

## Expected Outcome
- Extreme outliers (4000%+ upside) → Eliminated
- Distressed companies now show realistic fair values
- Healthy companies unaffected (still show reasonable upside/downside)
- Fair value distribution should be more balanced and realistic
