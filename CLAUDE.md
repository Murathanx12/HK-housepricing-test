# HK Rental Price Prediction — Instructions for Claude

## Current State (2026-04-14)
- **#1 on leaderboard: $1,241 RMSE** (MAE $473, R² 0.9948)
- Session: $1,355 → $1,300 → $1,293 → $1,280 → $1,247 → $1,241
- Total improvement: $114 in one session

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

### Fallback rows (6.6%): cascade + 45% enriched KNN(k=5) nudge
```python
prediction = 0.55 * cascade_lookup + 0.45 * enriched_KNN(k=5, 17 features)
```

## 17 Enriched KNN Features (the social/demand layer)
1. wgs_lat, wgs_lon (location)
2. area_sqft, floor (unit specs)
3. dist_mtr (MTR station proximity)
4. dist_cbd (Central Business District distance)
5. bld_ppsf (building median price per sqft — price tier)
6. bld_cls (building name classification: 0=estate, 1=old, 2=modern, 3=premium)
7. log_area (log-transformed area)
8. mall_1km (shopping mall density)
9. mtr_1km (MTR station density)
10. dist_lkf (Lan Kwai Fong — nightlife proximity)
11. dist_harbour (Victoria Harbour — sea view proxy)
12. dist_intl_sch (international school — expat area indicator)
13. intl_sch_2km (international school density)
14. region (HK Island=4, Kowloon=2-3, NT=0-1)
15. bld_age (from HK government building permits data)

## Proven Rules
1. NEVER blend ML into matched predictions — catastrophic ($1,915)
2. Direct n=1 price is ALWAYS correct — never override
3. Luxury predictions correct — never clip/shrink
4. Smaller KNN k (3-5) better for fallback, k=10 fine for n=1
5. Floor proximity weighting crucial for n≥2 groups
6. Only leaderboard gives reliable scores
7. Social/demand features in KNN = key to breaking plateaus
8. Higher fallback nudge keeps improving (10% → 45%)
