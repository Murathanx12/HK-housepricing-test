# HK Rental Price Prediction — 1st Place Solution ($1,450 RMSE)

Predicting monthly rental prices for 8,633 Hong Kong apartments. **1st place** on the class leaderboard with **$1,450 RMSE** using pure hardcoded lookup — zero ML.

## Results

| Version | RMSE | Approach |
|---------|------|----------|
| v1 | $1,904 | ML ensemble (LGB/XGB/CatBoost) + building stats |
| v2 | $1,800 | + KNN neighbors + unit matching |
| v3 | $1,746 | + unit+floorcat + Huber loss (best pure ML) |
| v4 | $1,836 | Full address stats — overfit |
| v5 | $2,081 | BEAST MODE stacking — severe overfit |
| v6 | $2,081 | LOO + LightGBM hybrid — overfit |
| v7 | — | First hardcoded lookup |
| v8 | $1,553 | 50/50 blend (hardcoded + ML) |
| v9-v10 | — | Hardcoded iterations |
| **v11** | **$1,450** | **Trimmed mean + KNN blend — #1 PLACE** |

## Key Insight

**93.4% of test apartments have exact address+area matches in training data.** For these, looking up the historical price directly beats any ML model. The winning solution is ~100 lines of pure Python lookup logic.

### Why Hardcoded Beats ML

- ML overfits even with LOO target encoding (CV RMSE 991 → leaderboard $2,081)
- Direct price lookup has zero overfitting risk
- Different aggregation strategies for different confidence levels:
  - **4+ matches**: Trimmed mean (removes outlier prices)
  - **2-3 matches**: Median
  - **1 match**: 80% direct + 20% KNN (smooths single outliers)
  - **No match**: Hierarchical fallback (unit → building → district → KNN)

## Project Structure

```
├── LEGACY_winner_1450.py       # Current winner — hardcoded v3 variant4
├── LEGACY_winner_1553.py       # Previous winner — 50/50 blend documentation
├── METHODOLOGY.md              # Detailed methodology report
├── project_introduction.md     # Contest instructions
├── starter_notebook.ipynb      # Provided starter code
├── legacy/                     # All code versions with RMSE in filename
│   ├── v1_RMSE_1904_building_stats.py
│   ├── v2_RMSE_1800_knn_unit_matching.py
│   ├── v3_RMSE_1746_unit_floorcat.py
│   ├── v4_RMSE_1836_full_address_overfit.py
│   ├── v5_RMSE_2081_beast_mode_overfit.py
│   ├── v6_hybrid_RMSE_2081_loo_lightgbm.py
│   ├── v7_hardcode_best.py
│   ├── v8_blend_RMSE_1553_equal_blend.py
│   ├── v9_hardcode_v2.py
│   ├── v10_hardcode_ultra.py
│   ├── v11_hardcode_v3_RMSE_1450_winner.py
│   ├── analyze_v1.py
│   └── analyze_v2.py
└── data/                       # Not tracked (too large)
```

## How to Run the Winner

```bash
# Requires: data/HK_house_transactions.csv and data/test_features.csv
python LEGACY_winner_1450.py
# Outputs: sub_hc3_v4_trimmed_knn.csv (submit this)
```

Dependencies: `pandas`, `numpy`, `scikit-learn`, `scipy`

## RMSE Progression

```
$2,081  ── BEAST MODE / LOO+LightGBM (severe overfit)
$1,904  ── v1: ML ensemble
$1,836  ── v4: full address stats (overfit)
$1,800  ── v2: + KNN + unit matching
$1,746  ── v3: + unit+floorcat (best pure ML)
$1,553  ── 50/50 blend (hardcoded + ML)
$1,450  ── Pure hardcoded lookup ★ #1 PLACE
```

The paradox: **simpler is better**. Each ML improvement brought diminishing returns, while a simple lookup table crushed them all.
