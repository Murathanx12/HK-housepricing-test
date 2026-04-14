"""
Hong Kong Rental Price Prediction — Improvement over $1,450 RMSE
================================================================
Vectorized LOO cross-validation to find optimal parameters.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import trim_mean
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
import time

DATA_DIR = Path("./data")

print("Loading...")
train = pd.read_csv(DATA_DIR / "HK_house_transactions.csv")
test = pd.read_csv(DATA_DIR / "test_features.csv")

def get_building(addr):
    if pd.isna(addr): return "UNKNOWN"
    return addr.split(",")[0].strip()

for df in [train, test]:
    df["building"] = df["address"].apply(get_building)
    df["floor"] = pd.to_numeric(df["floor"], errors="coerce").fillna(10)
    df["unit_key"] = df["building"] + "|" + df["Tower"].fillna("X").astype(str) + "|" + df["Flat"].fillna("X")
    df["bld_tower"] = df["building"] + "|T" + df["Tower"].fillna("X").astype(str)
    df["bld_flat"] = df["building"] + "|F" + df["Flat"].fillna("X")
    df["full_addr"] = df["address"].fillna("") + "|" + df["area_sqft"].astype(str)
    df["area_bin5"] = (df["area_sqft"] / 5).round() * 5
    df["unit_area5"] = df["unit_key"] + "|" + df["area_bin5"].astype(str)

train["ppsf"] = train["price"] / train["area_sqft"]

# ── VECTORIZED LOO STATS ──
# For each full_addr group, compute LOO statistics efficiently
print("Computing LOO stats per full_addr group...")
t0 = time.time()

fa_grp = train.groupby("full_addr")
# Group-level aggregates
fa_sum = fa_grp["price"].transform("sum")
fa_count = fa_grp["price"].transform("count")
fa_logsum = fa_grp["price"].transform(lambda x: np.log(x).sum())

# LOO stats: remove self from group
train["loo_count"] = fa_count - 1
train["loo_sum"] = fa_sum - train["price"]
train["loo_mean"] = np.where(train["loo_count"] > 0, train["loo_sum"] / train["loo_count"], np.nan)
train["loo_logsum"] = fa_logsum - np.log(train["price"])
train["loo_geomean"] = np.where(train["loo_count"] > 0, np.exp(train["loo_logsum"] / train["loo_count"]), np.nan)

# LOO median: need per-group computation
# For groups with n>=3 (LOO n>=2), median barely changes when removing 1 point
# We'll use group median as approximation for LOO median (accurate for n>=4)
fa_median = fa_grp["price"].transform("median")
train["loo_median_approx"] = fa_median  # close enough for n>=3

# For exact LOO median on small groups, compute directly
print("Computing exact LOO medians for small groups...")
exact_loo_median = np.full(len(train), np.nan)
for fa, group in fa_grp:
    if len(group) <= 5:  # Only compute exact LOO for small groups
        idxs = group.index.values
        prices = group["price"].values
        for i, idx in enumerate(idxs):
            others = np.delete(prices, i)
            if len(others) > 0:
                exact_loo_median[idx] = np.median(others)

# Use exact where available, approx otherwise
has_exact = ~np.isnan(exact_loo_median)
train["loo_median"] = np.where(has_exact, exact_loo_median, train["loo_median_approx"])

# LOO trimmed mean for groups with n>=5 (LOO n>=4)
print("Computing LOO trimmed means...")
exact_loo_trimmed = np.full(len(train), np.nan)
for fa, group in fa_grp:
    if len(group) >= 5:  # LOO count >= 4
        idxs = group.index.values
        prices = group["price"].values
        for i, idx in enumerate(idxs):
            others = np.delete(prices, i)
            exact_loo_trimmed[idx] = trim_mean(others, 0.1)

train["loo_trimmed"] = exact_loo_trimmed

# Floor slopes per building
def floor_slope(g):
    if len(g) < 5 or g["floor"].std() < 1: return 0.0
    return np.polyfit(g["floor"], g["ppsf"], 1)[0]

bld_slopes_dict = train.groupby("building").apply(floor_slope, include_groups=False).to_dict()
train["bld_slope"] = train["building"].map(bld_slopes_dict).fillna(0)

# KNN predictions
print("Building KNN...")
scaler = StandardScaler()
X_knn = scaler.fit_transform(train[["wgs_lat", "wgs_lon", "area_sqft", "floor"]].values)
knn = KNeighborsRegressor(n_neighbors=11, weights="distance", n_jobs=-1)  # k=11 so LOO-ish (self is nearest)
knn.fit(X_knn, train["price"].values)
train["knn_pred"] = knn.predict(X_knn)

# Also build KNN on PPSF
knn_ppsf = KNeighborsRegressor(n_neighbors=11, weights="distance", n_jobs=-1)
knn_ppsf.fit(X_knn, train["ppsf"].values)
train["knn_ppsf_pred"] = knn_ppsf.predict(X_knn) * train["area_sqft"]

# Unit-level median price (for blending with single matches)
unit_med = train.groupby("unit_key")["price"].median()
unit_cnt = train.groupby("unit_key")["price"].count()
train["unit_med_price"] = train["unit_key"].map(unit_med)
train["unit_count"] = train["unit_key"].map(unit_cnt)

# Building-level median ppsf
bld_med_ppsf = train.groupby("building")["ppsf"].median()
train["bld_med_ppsf"] = train["building"].map(bld_med_ppsf)

# District-level median ppsf
dist_med_ppsf = train.groupby("district")["ppsf"].median()
train["dist_med_ppsf"] = train["district"].map(dist_med_ppsf)

# Floor group stats
fa_floor_mean = fa_grp["floor"].transform("mean")
fa_floor_std = fa_grp["floor"].transform("std").fillna(0)
train["fa_floor_mean"] = fa_floor_mean
train["fa_floor_std"] = fa_floor_std

print(f"Stats computed in {time.time()-t0:.1f}s")

# ── FALLBACK PREDICTION ──
# For rows where loo_count == 0 (single observation in full_addr group)
# We need a fallback from broader groups
print("\nComputing fallback predictions...")

# Cascade: unit_area5 -> unit_key -> bld_tower -> bld_flat -> building -> district -> KNN
ua_med_ppsf = train.groupby("unit_area5")["ppsf"].median()
ua_cnt = train.groupby("unit_area5")["price"].count()
bt_med_ppsf = train.groupby("bld_tower")["ppsf"].median()
bt_cnt = train.groupby("bld_tower")["price"].count()
bf_med_ppsf = train.groupby("bld_flat")["ppsf"].median()
bf_cnt = train.groupby("bld_flat")["price"].count()

fallback = np.full(len(train), np.nan)
for i in range(len(train)):
    row = train.iloc[i]
    area = row["area_sqft"]

    uak = row["unit_area5"]
    if uak in ua_med_ppsf.index and ua_cnt[uak] >= 3:
        fallback[i] = area * ua_med_ppsf[uak]
        continue

    uk = row["unit_key"]
    if uk in unit_med.index and unit_cnt[uk] >= 2:
        fallback[i] = unit_med[uk]
        continue

    btk = row["bld_tower"]
    if btk in bt_med_ppsf.index and bt_cnt[btk] >= 3:
        fallback[i] = area * bt_med_ppsf[btk]
        continue

    bk = row["building"]
    if bk in bld_med_ppsf.index:
        fallback[i] = area * bld_med_ppsf[bk]
        continue

    dk = row["district"]
    if dk in dist_med_ppsf.index:
        fallback[i] = area * dist_med_ppsf[dk]
        continue

    fallback[i] = row["knn_pred"]

train["fallback_pred"] = fallback

# ── NOW TEST STRATEGIES ──
print("\n" + "="*60)
print("TESTING STRATEGIES (LOO RMSE)")
print("="*60)

actual = train["price"].values

def compute_rmse(preds):
    p = np.clip(preds, 2000, 500000)
    return np.sqrt(((actual - p) ** 2).mean())

# BASELINE: Reproduce variant4
def strategy_baseline():
    """Exact reproduction of the $1,450 winner."""
    preds = np.full(len(train), np.nan)

    # loo_count >= 4: trimmed mean
    m4 = train["loo_count"] >= 4
    preds[m4] = train.loc[m4, "loo_trimmed"]

    # loo_count 2-3: median
    m23 = (train["loo_count"] >= 2) & (train["loo_count"] <= 3)
    preds[m23] = train.loc[m23, "loo_median"]

    # loo_count 1: 80% direct + 20% KNN
    m1 = train["loo_count"] == 1
    preds[m1] = 0.8 * train.loc[m1, "loo_mean"] + 0.2 * train.loc[m1, "knn_pred"]

    # loo_count 0: fallback
    m0 = train["loo_count"] == 0
    preds[m0] = train.loc[m0, "fallback_pred"]

    return preds

baseline = strategy_baseline()
rmse_base = compute_rmse(baseline)
print(f"\nBASELINE (variant4 replica): LOO RMSE = ${rmse_base:,.0f}")

# Error breakdown
for name, mask in [("4+ matches", train["loo_count"] >= 4),
                   ("2-3 matches", (train["loo_count"] >= 2) & (train["loo_count"] <= 3)),
                   ("1 match", train["loo_count"] == 1),
                   ("0 (fallback)", train["loo_count"] == 0)]:
    subset_err = (actual[mask] - np.clip(baseline[mask], 2000, 500000)) ** 2
    rmse = np.sqrt(subset_err.mean())
    contrib = subset_err.sum() / ((actual - np.clip(baseline, 2000, 500000)) ** 2).sum() * 100
    print(f"  {name:15s}: n={mask.sum():6d} ({100*mask.sum()/len(train):.1f}%), RMSE=${rmse:,.0f}, contrib={contrib:.1f}%")

# TEST: Different alpha for single matches
print("\n--- Single-match blend ratio ---")
for alpha in [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]:
    preds = baseline.copy()
    m1 = train["loo_count"] == 1
    preds[m1] = alpha * train.loc[m1, "loo_mean"].values + (1 - alpha) * train.loc[m1, "knn_pred"].values
    print(f"  alpha={alpha:.2f}: RMSE ${compute_rmse(preds):,.0f}")

# TEST: Single-match blend with unit median instead of KNN
print("\n--- Single-match: unit median vs KNN ---")
for alpha in [0.7, 0.8, 0.9]:
    preds = baseline.copy()
    m1 = train["loo_count"] == 1
    # Use unit median where available (count>=3), else KNN
    blend_partner = np.where(
        train["unit_count"] >= 3,
        train["unit_med_price"],
        train["knn_pred"]
    )
    preds[m1] = alpha * train.loc[m1, "loo_mean"].values + (1 - alpha) * blend_partner[m1]
    print(f"  alpha={alpha:.1f} (unit>KNN): RMSE ${compute_rmse(preds):,.0f}")

# TEST: Floor adjustment on multi-match rows
print("\n--- Floor adjustment on matched rows ---")
for scale in [0.5, 0.75, 1.0, 1.5]:
    preds = baseline.copy()
    multi = train["loo_count"] >= 2
    floor_diff = train["floor"] - train["fa_floor_mean"]
    adj = scale * train["bld_slope"] * floor_diff * train["area_sqft"]
    # Only apply where floor variance exists
    apply_mask = multi & (train["fa_floor_std"] > 0.5)
    preds[apply_mask] = preds[apply_mask] + adj[apply_mask].values
    print(f"  scale={scale:.2f}: RMSE ${compute_rmse(preds):,.0f}")

# TEST: Geometric mean vs arithmetic for 2-3 matches
print("\n--- Geometric mean for 2-3 matches ---")
preds = baseline.copy()
m23 = (train["loo_count"] >= 2) & (train["loo_count"] <= 3)
preds[m23] = train.loc[m23, "loo_geomean"]
print(f"  Geo mean 2-3: RMSE ${compute_rmse(preds):,.0f}")

# TEST: Geometric mean for ALL multi-match
preds = baseline.copy()
m2plus = train["loo_count"] >= 2
preds[m2plus] = train.loc[m2plus, "loo_geomean"]
print(f"  Geo mean all: RMSE ${compute_rmse(preds):,.0f}")

# TEST: Mean instead of median for 2-3
preds = baseline.copy()
m23 = (train["loo_count"] >= 2) & (train["loo_count"] <= 3)
preds[m23] = train.loc[m23, "loo_mean"]
print(f"  Arith mean 2-3: RMSE ${compute_rmse(preds):,.0f}")

# TEST: Trimmed mean for 3+ instead of 4+
print("\n--- Trim threshold ---")
for min_trim in [3, 4, 5, 6]:
    preds = np.full(len(train), np.nan)
    m_trim = train["loo_count"] >= min_trim
    m_med = (train["loo_count"] >= 2) & (train["loo_count"] < min_trim)
    m1 = train["loo_count"] == 1
    m0 = train["loo_count"] == 0

    # For trimmed: use exact where available, else median
    preds[m_trim] = np.where(
        ~np.isnan(train.loc[m_trim, "loo_trimmed"]),
        train.loc[m_trim, "loo_trimmed"],
        train.loc[m_trim, "loo_median"]
    )
    preds[m_med] = train.loc[m_med, "loo_median"]
    preds[m1] = 0.8 * train.loc[m1, "loo_mean"].values + 0.2 * train.loc[m1, "knn_pred"].values
    preds[m0] = train.loc[m0, "fallback_pred"]
    print(f"  trim_threshold={min_trim}: RMSE ${compute_rmse(preds):,.0f}")

# TEST: Blend trimmed + median for 4+ (hedging)
print("\n--- Blend trimmed + median for 4+ ---")
for w in [0.3, 0.5, 0.7, 0.8, 0.9, 1.0]:
    preds = baseline.copy()
    m4 = train["loo_count"] >= 4
    blended = w * train.loc[m4, "loo_trimmed"].values + (1 - w) * train.loc[m4, "loo_median"].values
    preds[m4] = blended
    print(f"  w_trim={w:.1f}: RMSE ${compute_rmse(preds):,.0f}")

# TEST: Cap extreme predictions (outlier clipping relative to broader stats)
print("\n--- Capping outlier predictions ---")
for cap_mult in [2.0, 2.5, 3.0, 4.0]:
    preds = baseline.copy()
    # If prediction is > cap_mult * building median ppsf * area, clip it
    bld_pred = train["bld_med_ppsf"] * train["area_sqft"]
    too_high = preds > cap_mult * bld_pred
    too_low = preds < bld_pred / cap_mult
    preds_capped = preds.copy()
    preds_capped[too_high] = cap_mult * bld_pred[too_high]
    preds_capped[too_low] = bld_pred[too_low] / cap_mult
    changed = (too_high | too_low).sum()
    print(f"  cap={cap_mult:.1f}x: RMSE ${compute_rmse(preds_capped):,.0f} ({changed} rows capped)")

# TEST: KNN-based ppsf for single matches
print("\n--- KNN ppsf for single matches ---")
for alpha in [0.7, 0.8, 0.9]:
    preds = baseline.copy()
    m1 = train["loo_count"] == 1
    preds[m1] = alpha * train.loc[m1, "loo_mean"].values + (1 - alpha) * train.loc[m1, "knn_ppsf_pred"].values
    print(f"  alpha={alpha:.1f} (knn_ppsf): RMSE ${compute_rmse(preds):,.0f}")

# TEST: Bayesian shrinkage for ALL groups
print("\n--- Bayesian shrinkage (all groups) ---")
for k in [0.5, 1.0, 1.5, 2.0, 3.0, 5.0]:
    preds = np.full(len(train), np.nan)
    has_match = train["loo_count"] >= 1
    n = train["loo_count"].values
    w = np.where(n > 0, n / (n + k), 0)

    # FA prediction: use loo_mean for simplicity
    fa_pred = train["loo_mean"].values
    broader = train["fallback_pred"].values

    preds[has_match] = w[has_match] * fa_pred[has_match] + (1 - w[has_match]) * broader[has_match]

    m0 = train["loo_count"] == 0
    preds[m0] = train.loc[m0, "fallback_pred"]
    print(f"  k={k:.1f}: RMSE ${compute_rmse(preds):,.0f}")

# TEST: Bayesian with median instead of mean as FA pred
print("\n--- Bayesian shrinkage (median FA pred) ---")
for k in [0.5, 1.0, 1.5, 2.0, 3.0]:
    preds = np.full(len(train), np.nan)

    m4 = train["loo_count"] >= 4
    m23 = (train["loo_count"] >= 2) & (train["loo_count"] < 4)
    m1 = train["loo_count"] == 1
    m0 = train["loo_count"] == 0

    n = train["loo_count"].values.astype(float)
    w = n / (n + k)
    broader = train["fallback_pred"].values

    fa_pred_arr = np.where(m4, train["loo_trimmed"].values,
                   np.where(m23, train["loo_median"].values,
                   train["loo_mean"].values))

    has_match = train["loo_count"] >= 1
    preds[has_match] = w[has_match] * fa_pred_arr[has_match] + (1 - w[has_match]) * broader[has_match]
    preds[m0] = broader[m0]
    print(f"  k={k:.1f}: RMSE ${compute_rmse(preds):,.0f}")

print("\n" + "="*60)
print("DONE — Use the best parameters to build final submission")
print("="*60)
