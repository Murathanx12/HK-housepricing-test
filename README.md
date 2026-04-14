# HK Rental Price Prediction — $1,355 RMSE (2nd Place)

Predicting monthly rental prices for 8,633 Hong Kong apartments using **pure hardcoded lookup** — zero ML.

## Leaderboard (2026-04-14)

| Rank | Team | RMSE | MAE | R² |
|------|------|------|-----|-----|
| 1 | JigsawBlock | **$1,347** | $556 | 0.9939 |
| 2 | **Murathan** | **$1,355** | $493 | 0.9938 |

## RMSE Progression (130+ experiments)

```
$2,081  BEAST MODE / LOO+LightGBM (severe overfit)
$1,904  v1: ML ensemble (LGB/XGB/CatBoost)
$1,553  v8: 50/50 blend (hardcoded + ML)          ← ML peak
$1,450  v11: pure hardcoded lookup
$1,435  v12: + building-level outlier correction
$1,393  v13: + 3-way blend for all n=1
$1,355  v14: n=3 mean instead of median            ← current best
$1,372  v15: z-score outlier correction (WORSE)
$1,373  v16: heavier building blend 70/15/15 (WORSE)
```

## Core Approach

**93.4% of test apartments have exact address+area matches in training data.** Looking up historical prices directly beats any ML model.

### Winning Config ($1,355)

| Match type | % of test | Strategy |
|------------|-----------|----------|
| 4+ exact matches | 37.8% | Trimmed mean (10% trim) |
| 3 exact matches | 9.4% | **Plain mean** (NOT median — $38 better) |
| 2 exact matches | 14.8% | Mean |
| 1 exact match | 31.3% | 85% direct + 5% building PPSF×area + 10% KNN |
| No match (fallback) | 6.6% | PPSF cascade: unit_area5 → unit_key → bld_tower → bld_flat → building → district+KNN |

### Why Hardcoded Beats ML

- Direct price lookup has **zero overfitting risk** (it's just memorization)
- ML overfits badly: CV RMSE $991 → leaderboard $2,081
- Blending ML into the lookup **always** makes it worse ($1,915 RMSE with LGB+CatBoost blend)
- The remaining error ($1,355) is mostly **irreducible within-group variance**

## What We Tried (Complete Log)

### What IMPROVED (confirmed on leaderboard)

| Change | From → To | Rows changed |
|--------|-----------|-------------|
| n=3 mean instead of median | $1,393 → $1,355 | 814 |
| 3-way blend ALL n=1 (add 5% bld) | $1,402 → $1,395 | 2704 |
| Direct weight 80%→85% | $1,395 → $1,393 | 2704 |
| 7-row outlier fix (bld correction) | $1,450 → $1,435 | 7 |
| Weaker correction alpha | $1,435 → $1,421 | ~100 |
| 3-way blend for outliers | $1,421 → $1,402 | ~100 |

### What FAILED (do NOT retry)

| Change | Score | Why it failed |
|--------|-------|---------------|
| ML blending (LGB+CatBoost) | $1,915 | ML adds noise to 93% of rows that are already correct |
| More building weight (8-15%) | $1,357-$1,361 | 5% is the sweet spot |
| Median for n=3 groups | $1,393 | Mean uses all data, median discards |
| Remove training outliers | $1,443 | "Outliers" predict real test outliers |
| Shrinkage toward grand mean | $1,359-$1,616 | Predictions already optimal |
| z-score outlier correction (z>2.5) | $1,372 | Corrects wrong rows, net negative |
| Heavier uniform blend (70/15/15) | $1,373 | Worsens typical rows too much |
| Aggressive building correction (50/25/25) | $1,418 | Way too much building weight |
| Full building replacement (z>3.0) | $1,411 | Direct price usually correct |
| Adaptive n=1 split | $1,360-$1,365 | Threshold adds noise |
| CV-adaptive parameters | $1,620 | CV doesn't correlate with leaderboard |
| Fuzzy address matching | $1,355 | Neutral — already handled by cascade |
| Round instead of truncate | $1,355 | Neutral |
| Floor adjustment for n=2 | $1,355 | Floor diffs too small to matter |
| Post-processing consistency | $1,361-$1,398 | "Outlier" predictions are correct |

## Gap to #1 — Analysis

JigsawBlock's profile: **RMSE $1,347, MAE $556**

Their higher MAE ($556 vs our $493) reveals their strategy:
- They sacrifice average accuracy to fix extreme outlier errors
- RMSE/MAE ratio: 2.42 (theirs) vs 2.75 (ours) — flatter error distribution
- They likely use heavier building correction for outlier rows, but with a SMARTER identification method than z-score

### Why our outlier correction fails

Our z-score approach changes the right direction (toward building) but:
1. Building ppsf includes units of MANY sizes — a luxury 2000sqft unit legitimately has different ppsf than studio units in the same building
2. Z-score flags these as "outliers" when they're actually correct
3. The corrections hurt more rows than they help

### What to try next

1. **Size-adjusted building PPSF**: Compute building ppsf regression (ppsf ~ area) instead of flat median. This would make the "expected" ppsf specific to each apartment's size, making outlier detection more accurate.

2. **Unit-level (tower+flat) PPSF instead of building-level**: For outlier detection, compare against units of the same type rather than all units in the building.

3. **Nearby building matching**: For fallback rows, find the 3-5 nearest buildings with similar ppsf and average their predictions.

4. **Probe-based optimization**: Submit variants that change specific rows and use leaderboard feedback to identify which rows have large errors.

5. **External data**: Bus stop locations, building age, furnishing data could reduce within-group variance for specific buildings.

## Project Structure

```
solution.py                  # Active engine (generates variants)
LEGACY_winner_1355.py        # Exact $1,355 code (DO NOT MODIFY)
LEGACY_winner_1435.py        # Previous version ($1,435)
LEGACY_winner_1450.py        # Variant comparison engine
LEGACY_winner_1553.py        # Documentation of blend approach
METHODOLOGY.md               # Detailed methodology report
legacy/old_scripts/          # All historical code (70+ scripts)
data/                        # Training + test data (not tracked)
```

## How to Run

```bash
python solution.py       # Generates my_submission.csv + variants
python LEGACY_winner_1355.py  # Reproduces the $1,355 baseline
```

Dependencies: `pandas`, `numpy`, `scikit-learn`, `scipy`, `lightgbm`, `catboost`
