# HK Rental Price Prediction — Instructions for Claude

## Current State (2026-04-14)
- **#1 on leaderboard: $1,293 RMSE** (MAE $482, R² 0.9944)
- Improved from $1,355 → $1,293 in one session ($62 drop)
- $1,293 is SATURATED — 6 different configs all hit it

## Winning Architecture

### n≥2 matched rows (62.1%): Gaussian floor-weighted mean
```python
weight = exp(-|floor_diff|² / (2 × 0.7²))
prediction = weighted_mean(group_prices, weights)
```

### n=1 matched rows (31.3%): 85/5/10 blend
```python
prediction = 0.85 * direct + 0.05 * building_ppsf * area + 0.10 * KNN(k=10)
```

### Fallback rows (6.6%): cascade + 10% KNN(k=5) nudge
```python
prediction = 0.90 * cascade_lookup + 0.10 * KNN(k=5)
```

## Proven Rules
1. NEVER blend ML — catastrophic ($1,915)
2. Direct n=1 price is ALWAYS correct — never override
3. Luxury predictions correct — never clip/shrink
4. Smaller KNN k (3-5) better for fallback, k=10 fine for n=1
5. Floor proximity weighting crucial for n≥2 groups
6. Only leaderboard gives reliable scores
7. Building median PPSF too coarse for outlier detection

## $1,293 is the Hard Floor
Saturated across k=3-5, nudge 8-12%, n=1 k=3-10.
Remaining error = irreducible within-group variance from
furnished/unfurnished, lease terms, parking, view, renovation.
