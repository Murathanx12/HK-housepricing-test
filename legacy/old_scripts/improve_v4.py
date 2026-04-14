"""
Surgical improvements to beat $1,450 RMSE.
Lesson learned: don't over-correct. The direct price lookup is strong.
Focus on: which specific rows have the biggest errors, and fix ONLY those.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import trim_mean
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler

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

# ── DEEP ANALYSIS: Where does the baseline make errors? ──
# Use LOO but focus on the MATCHED rows (loo_count >= 1) since those are 93.4% of test
print("Deep error analysis on matched rows...")

fa_grp = train.groupby("full_addr")

# For each full_addr group, compute variance metrics
fa_stats_full = fa_grp.agg(
    p_mean=("price", "mean"), p_median=("price", "median"),
    p_std=("price", "std"), p_min=("price", "min"), p_max=("price", "max"),
    count=("price", "count"),
    ppsf_mean=("ppsf", "mean"), ppsf_median=("ppsf", "median"),
    ppsf_std=("ppsf", "std"),
    floor_mean=("floor", "mean"), floor_std=("floor", "std"),
    area=("area_sqft", "first"),
)
fa_stats_full["p_std"] = fa_stats_full["p_std"].fillna(0)
fa_stats_full["ppsf_std"] = fa_stats_full["ppsf_std"].fillna(0)
fa_stats_full["floor_std"] = fa_stats_full["floor_std"].fillna(0)
fa_stats_full["p_range"] = fa_stats_full["p_max"] - fa_stats_full["p_min"]
fa_stats_full["cv"] = np.where(fa_stats_full["p_mean"] > 0,
                                fa_stats_full["p_std"] / fa_stats_full["p_mean"], 0)

# Show how variable same-address prices are
print("\n=== PRICE VARIANCE WITHIN SAME FULL_ADDR ===")
for n_min in [2, 4, 6, 10]:
    subset = fa_stats_full[fa_stats_full["count"] >= n_min]
    print(f"  Groups with n>={n_min}: {len(subset)} groups, "
          f"mean_std=${subset['p_std'].mean():,.0f}, "
          f"mean_range=${subset['p_range'].mean():,.0f}, "
          f"mean_cv={subset['cv'].mean():.3f}")

# The within-group std IS the irreducible error for median/mean predictions
# If mean_std is ~$2000 for groups with 2+ matches, that's our floor
print(f"\n  Overall within-group std (n>=2): ${fa_stats_full[fa_stats_full['count']>=2]['p_std'].mean():,.0f}")
print(f"  This is approximately our RMSE floor for matched rows")

# Look at HIGH-VARIANCE groups (these are where errors come from)
high_var = fa_stats_full[(fa_stats_full["count"] >= 3) & (fa_stats_full["cv"] > 0.15)]
print(f"\n  High-variance groups (cv>0.15, n>=3): {len(high_var)} groups")
print(f"  Mean range: ${high_var['p_range'].mean():,.0f}")
print(f"  These groups likely have temporal price changes or different lease terms")

# Check if floor variation within group explains price variation
print("\n=== DOES FLOOR EXPLAIN WITHIN-GROUP VARIANCE? ===")
groups_with_floor_var = fa_stats_full[(fa_stats_full["count"] >= 4) & (fa_stats_full["floor_std"] > 1)]
groups_no_floor_var = fa_stats_full[(fa_stats_full["count"] >= 4) & (fa_stats_full["floor_std"] <= 1)]
print(f"  Groups with floor variation: {len(groups_with_floor_var)}, mean_cv={groups_with_floor_var['cv'].mean():.3f}")
print(f"  Groups without floor variation: {len(groups_no_floor_var)}, mean_cv={groups_no_floor_var['cv'].mean():.3f}")

# ── LOOKUP TABLES (same as winner) ──
print("\nBuilding lookups...")

def floor_slope(g):
    if len(g) < 5 or g["floor"].std() < 1: return 0.0
    return np.polyfit(g["floor"], g["ppsf"], 1)[0]

bld_slopes = train.groupby("building").apply(floor_slope, include_groups=False).to_dict()

fa_trimmed = fa_grp["price"].apply(lambda x: trim_mean(x, 0.1) if len(x) >= 4 else x.mean())
fa_stats_full = fa_stats_full.join(fa_trimmed.rename("p_trimmed"))

# Winsorized mean (clip outliers to 5th/95th percentile within group)
def winsorized_mean(x):
    if len(x) < 4: return x.mean()
    lo, hi = np.percentile(x, [5, 95])
    return x.clip(lo, hi).mean()

fa_winsorized = fa_grp["price"].apply(winsorized_mean)
fa_stats_full = fa_stats_full.join(fa_winsorized.rename("p_winsorized"))

# Huber-like: use median for high-cv groups, trimmed for low-cv
fa_stats_full["p_robust"] = np.where(
    fa_stats_full["cv"] > 0.1,
    fa_stats_full["p_median"],
    fa_stats_full["p_trimmed"]
)

ua_stats = train.groupby("unit_area5").agg(
    ppsf_mean=("ppsf", "mean"), ppsf_median=("ppsf", "median"),
    floor_mean=("floor", "mean"), count=("price", "count"),
)
unit_stats = train.groupby("unit_key").agg(
    ppsf_mean=("ppsf", "mean"), ppsf_median=("ppsf", "median"),
    floor_mean=("floor", "mean"), area_mean=("area_sqft", "mean"),
    count=("price", "count"),
)
bt_stats = train.groupby("bld_tower").agg(
    ppsf_median=("ppsf", "median"), floor_mean=("floor", "mean"), count=("price", "count"),
)
bf_stats = train.groupby("bld_flat").agg(
    ppsf_median=("ppsf", "median"), floor_mean=("floor", "mean"), count=("price", "count"),
)
bld_stats = train.groupby("building").agg(
    ppsf_mean=("ppsf", "mean"), ppsf_median=("ppsf", "median"),
    floor_mean=("floor", "mean"), count=("price", "count"),
)
dist_stats = train.groupby("district").agg(
    ppsf_median=("ppsf", "median"),
)

# KNN
print("Building KNN...")
scaler = StandardScaler()
X_knn_tr = scaler.fit_transform(train[["wgs_lat", "wgs_lon", "area_sqft", "floor"]].values)
X_knn_te = scaler.transform(test[["wgs_lat", "wgs_lon", "area_sqft", "floor"]].values)

knn10 = KNeighborsRegressor(n_neighbors=10, weights="distance", n_jobs=-1)
knn10.fit(X_knn_tr, train["price"].values)
test["knn10"] = knn10.predict(X_knn_te)


def get_fallback_pred(row, area, floor_val, slope):
    uak = row["unit_area5"]
    uk = row["unit_key"]
    btk = row["bld_tower"]
    bfk = row["bld_flat"]
    bk = row["building"]
    dk = row["district"]

    if uak in ua_stats.index:
        d = ua_stats.loc[uak]
        base = d["ppsf_median"] if d["count"] >= 2 else d["ppsf_mean"]
        fadj = slope * (floor_val - d["floor_mean"]) if d["count"] >= 2 else 0
        return area * (base + fadj)
    if uk in unit_stats.index:
        d = unit_stats.loc[uk]
        base = d["ppsf_median"] if d["count"] >= 3 else d["ppsf_mean"]
        fadj = slope * (floor_val - d["floor_mean"])
        return area * (base + fadj)
    if btk in bt_stats.index and bt_stats.loc[btk]["count"] >= 3:
        d = bt_stats.loc[btk]
        fadj = slope * (floor_val - d["floor_mean"])
        return area * (d["ppsf_median"] + fadj)
    if bfk in bf_stats.index and bf_stats.loc[bfk]["count"] >= 3:
        d = bf_stats.loc[bfk]
        fadj = slope * (floor_val - d["floor_mean"])
        return area * (d["ppsf_median"] + fadj)
    if bk in bld_stats.index:
        d = bld_stats.loc[bk]
        base = d["ppsf_median"] if d["count"] >= 3 else d["ppsf_mean"]
        fadj = slope * (floor_val - d["floor_mean"]) if d["count"] >= 5 else 0
        return area * (base + fadj)
    knn_p = row["knn10"]
    if dk in dist_stats.index:
        dp = area * dist_stats.loc[dk]["ppsf_median"]
        return 0.4 * knn_p + 0.6 * dp
    return knn_p


# ══════════════════════════════════════════════════
# BASELINE: Exact reproduction
# ══════════════════════════════════════════════════
def variant_baseline():
    preds = np.zeros(len(test))
    for i in range(len(test)):
        row = test.iloc[i]
        area, floor_val = row["area_sqft"], row["floor"]
        fa = row["full_addr"]
        slope = bld_slopes.get(row["building"], 0.0)
        if fa in fa_stats_full.index:
            d = fa_stats_full.loc[fa]
            if d["count"] >= 4:
                preds[i] = d["p_trimmed"]
            elif d["count"] >= 2:
                preds[i] = d["p_median"]
            else:
                preds[i] = 0.8 * d["p_mean"] + 0.2 * row["knn10"]
        else:
            preds[i] = get_fallback_pred(row, area, floor_val, slope)
    return np.clip(preds, 2000, 500000)


# ══════════════════════════════════════════════════
# V1: Robust aggregation — use median for high-cv groups, trimmed for stable
# ══════════════════════════════════════════════════
def variant_robust_agg():
    """Different agg based on group stability. High cv -> median is safer."""
    preds = np.zeros(len(test))
    for i in range(len(test)):
        row = test.iloc[i]
        area, floor_val = row["area_sqft"], row["floor"]
        fa = row["full_addr"]
        slope = bld_slopes.get(row["building"], 0.0)
        if fa in fa_stats_full.index:
            d = fa_stats_full.loc[fa]
            if d["count"] >= 4:
                # Use robust: median for noisy groups, trimmed for stable
                preds[i] = d["p_robust"]
            elif d["count"] >= 2:
                preds[i] = d["p_median"]
            else:
                preds[i] = 0.8 * d["p_mean"] + 0.2 * row["knn10"]
        else:
            preds[i] = get_fallback_pred(row, area, floor_val, slope)
    return np.clip(preds, 2000, 500000)


# ══════════════════════════════════════════════════
# V2: Winsorized mean for 4+ (clip outliers before averaging)
# ══════════════════════════════════════════════════
def variant_winsorized():
    preds = np.zeros(len(test))
    for i in range(len(test)):
        row = test.iloc[i]
        area, floor_val = row["area_sqft"], row["floor"]
        fa = row["full_addr"]
        slope = bld_slopes.get(row["building"], 0.0)
        if fa in fa_stats_full.index:
            d = fa_stats_full.loc[fa]
            if d["count"] >= 4:
                preds[i] = d["p_winsorized"]
            elif d["count"] >= 2:
                preds[i] = d["p_median"]
            else:
                preds[i] = 0.8 * d["p_mean"] + 0.2 * row["knn10"]
        else:
            preds[i] = get_fallback_pred(row, area, floor_val, slope)
    return np.clip(preds, 2000, 500000)


# ══════════════════════════════════════════════════
# V3: Blend trimmed + median 70/30 for 4+ (dampen trimmed mean)
# ══════════════════════════════════════════════════
def variant_blend70():
    preds = np.zeros(len(test))
    for i in range(len(test)):
        row = test.iloc[i]
        area, floor_val = row["area_sqft"], row["floor"]
        fa = row["full_addr"]
        slope = bld_slopes.get(row["building"], 0.0)
        if fa in fa_stats_full.index:
            d = fa_stats_full.loc[fa]
            if d["count"] >= 4:
                preds[i] = 0.7 * d["p_trimmed"] + 0.3 * d["p_median"]
            elif d["count"] >= 2:
                preds[i] = d["p_median"]
            else:
                preds[i] = 0.8 * d["p_mean"] + 0.2 * row["knn10"]
        else:
            preds[i] = get_fallback_pred(row, area, floor_val, slope)
    return np.clip(preds, 2000, 500000)


# ══════════════════════════════════════════════════
# V4: Floor-adjusted prediction for matched multi-match rows
# Only adjust when within-group floor has NO variation (same "Lower Floor" etc)
# but test row has different numeric floor — suggests a price difference
# ══════════════════════════════════════════════════
def variant_floor_adj():
    preds = np.zeros(len(test))
    for i in range(len(test)):
        row = test.iloc[i]
        area, floor_val = row["area_sqft"], row["floor"]
        fa = row["full_addr"]
        slope = bld_slopes.get(row["building"], 0.0)
        if fa in fa_stats_full.index:
            d = fa_stats_full.loc[fa]
            if d["count"] >= 4:
                base = d["p_trimmed"]
                # Small floor adjustment if floor differs from group mean
                if abs(slope) > 0.1 and d["count"] >= 6:
                    floor_diff = floor_val - d["floor_mean"]
                    adj = slope * floor_diff * area * 0.3  # very damped
                    base += adj
                preds[i] = base
            elif d["count"] >= 2:
                preds[i] = d["p_median"]
            else:
                preds[i] = 0.8 * d["p_mean"] + 0.2 * row["knn10"]
        else:
            preds[i] = get_fallback_pred(row, area, floor_val, slope)
    return np.clip(preds, 2000, 500000)


# ══════════════════════════════════════════════════
# V5: Pure median everywhere (no trimmed mean at all)
# Maybe trimmed mean is slightly overfitting?
# ══════════════════════════════════════════════════
def variant_pure_median():
    preds = np.zeros(len(test))
    for i in range(len(test)):
        row = test.iloc[i]
        area, floor_val = row["area_sqft"], row["floor"]
        fa = row["full_addr"]
        slope = bld_slopes.get(row["building"], 0.0)
        if fa in fa_stats_full.index:
            d = fa_stats_full.loc[fa]
            if d["count"] >= 2:
                preds[i] = d["p_median"]
            else:
                preds[i] = 0.8 * d["p_mean"] + 0.2 * row["knn10"]
        else:
            preds[i] = get_fallback_pred(row, area, floor_val, slope)
    return np.clip(preds, 2000, 500000)


# ══════════════════════════════════════════════════
# V6: Count-weighted blend of trimmed + median
# More observations -> trust trimmed more (it's better with large n)
# ══════════════════════════════════════════════════
def variant_count_weighted():
    preds = np.zeros(len(test))
    for i in range(len(test)):
        row = test.iloc[i]
        area, floor_val = row["area_sqft"], row["floor"]
        fa = row["full_addr"]
        slope = bld_slopes.get(row["building"], 0.0)
        if fa in fa_stats_full.index:
            d = fa_stats_full.loc[fa]
            if d["count"] >= 4:
                # More data -> trust trimmed mean more
                w = min(d["count"] / 20.0, 1.0)  # at n=20, fully trust trimmed
                preds[i] = w * d["p_trimmed"] + (1 - w) * d["p_median"]
            elif d["count"] >= 2:
                preds[i] = d["p_median"]
            else:
                preds[i] = 0.8 * d["p_mean"] + 0.2 * row["knn10"]
        else:
            preds[i] = get_fallback_pred(row, area, floor_val, slope)
    return np.clip(preds, 2000, 500000)


# ══════════════════════════════════════════════════
# V7: Trimmed 15% (more aggressive outlier removal)
# ══════════════════════════════════════════════════
fa_trimmed15 = fa_grp["price"].apply(lambda x: trim_mean(x, 0.15) if len(x) >= 4 else x.mean())
fa_stats_full = fa_stats_full.join(fa_trimmed15.rename("p_trimmed15"))

fa_trimmed20 = fa_grp["price"].apply(lambda x: trim_mean(x, 0.2) if len(x) >= 4 else x.mean())
fa_stats_full = fa_stats_full.join(fa_trimmed20.rename("p_trimmed20"))

fa_trimmed05 = fa_grp["price"].apply(lambda x: trim_mean(x, 0.05) if len(x) >= 4 else x.mean())
fa_stats_full = fa_stats_full.join(fa_trimmed05.rename("p_trimmed05"))

def variant_trim15():
    preds = np.zeros(len(test))
    for i in range(len(test)):
        row = test.iloc[i]
        fa = row["full_addr"]
        slope = bld_slopes.get(row["building"], 0.0)
        if fa in fa_stats_full.index:
            d = fa_stats_full.loc[fa]
            if d["count"] >= 4:
                preds[i] = d["p_trimmed15"]
            elif d["count"] >= 2:
                preds[i] = d["p_median"]
            else:
                preds[i] = 0.8 * d["p_mean"] + 0.2 * row["knn10"]
        else:
            preds[i] = get_fallback_pred(row, row["area_sqft"], row["floor"], slope)
    return np.clip(preds, 2000, 500000)

def variant_trim20():
    preds = np.zeros(len(test))
    for i in range(len(test)):
        row = test.iloc[i]
        fa = row["full_addr"]
        slope = bld_slopes.get(row["building"], 0.0)
        if fa in fa_stats_full.index:
            d = fa_stats_full.loc[fa]
            if d["count"] >= 4:
                preds[i] = d["p_trimmed20"]
            elif d["count"] >= 2:
                preds[i] = d["p_median"]
            else:
                preds[i] = 0.8 * d["p_mean"] + 0.2 * row["knn10"]
        else:
            preds[i] = get_fallback_pred(row, row["area_sqft"], row["floor"], slope)
    return np.clip(preds, 2000, 500000)

def variant_trim05():
    preds = np.zeros(len(test))
    for i in range(len(test)):
        row = test.iloc[i]
        fa = row["full_addr"]
        slope = bld_slopes.get(row["building"], 0.0)
        if fa in fa_stats_full.index:
            d = fa_stats_full.loc[fa]
            if d["count"] >= 4:
                preds[i] = d["p_trimmed05"]
            elif d["count"] >= 2:
                preds[i] = d["p_median"]
            else:
                preds[i] = 0.8 * d["p_mean"] + 0.2 * row["knn10"]
        else:
            preds[i] = get_fallback_pred(row, row["area_sqft"], row["floor"], slope)
    return np.clip(preds, 2000, 500000)


# ── RUN ALL ──
print("\nGenerating all variants...")
variants = [
    ("sub_v4_baseline", variant_baseline),
    ("sub_v4_robust", variant_robust_agg),
    ("sub_v4_winsorized", variant_winsorized),
    ("sub_v4_blend70", variant_blend70),
    ("sub_v4_floor_adj", variant_floor_adj),
    ("sub_v4_pure_median", variant_pure_median),
    ("sub_v4_count_wt", variant_count_weighted),
    ("sub_v4_trim05", variant_trim05),
    ("sub_v4_trim15", variant_trim15),
    ("sub_v4_trim20", variant_trim20),
]

results = {}
for name, func in variants:
    preds = func()
    results[name] = preds
    pd.DataFrame({"id": test["id"].astype(int), "price": preds.astype(int)}).to_csv(f"{name}.csv", index=False)

# Compare all to baseline
print("\n=== ALL VARIANTS vs BASELINE ===")
base = results["sub_v4_baseline"]
print(f"{'Name':25s} {'Changed':>8s} {'MeanAbsDiff':>12s} {'MaxDiff':>10s}")
print("-" * 58)
for name, preds in results.items():
    diff = np.abs(preds - base)
    changed = (diff > 1).sum()
    mean_d = diff.mean()
    max_d = diff.max()
    print(f"{name:25s} {changed:8d} ${mean_d:10,.0f} ${max_d:8,.0f}")

# Also blend some together
print("\n=== BLENDS ===")
blend_configs = [
    ("sub_v4_blend_base_robust", [("sub_v4_baseline", 0.5), ("sub_v4_robust", 0.5)]),
    ("sub_v4_blend_base_win", [("sub_v4_baseline", 0.5), ("sub_v4_winsorized", 0.5)]),
    ("sub_v4_blend_trim_med", [("sub_v4_baseline", 0.5), ("sub_v4_pure_median", 0.5)]),
    ("sub_v4_blend_3way", [("sub_v4_baseline", 0.4), ("sub_v4_robust", 0.3), ("sub_v4_winsorized", 0.3)]),
    ("sub_v4_blend_trim_range", [("sub_v4_trim05", 0.25), ("sub_v4_baseline", 0.5), ("sub_v4_trim15", 0.25)]),
]

for blend_name, components in blend_configs:
    blended = sum(w * results[n] for n, w in components)
    blended = np.clip(blended, 2000, 500000)
    results[blend_name] = blended
    pd.DataFrame({"id": test["id"].astype(int), "price": blended.astype(int)}).to_csv(f"{blend_name}.csv", index=False)
    diff = np.abs(blended - base)
    changed = (diff > 1).sum()
    print(f"  {blend_name}: {changed} rows differ, mean_abs_diff=${diff.mean():,.0f}")

print("\n=== TOP RECOMMENDATIONS ===")
print("These change the least from baseline (safest bets):")
print("1. sub_v4_robust.csv       — Median for noisy groups, trimmed for stable")
print("2. sub_v4_winsorized.csv   — Clip outlier prices before averaging")
print("3. sub_v4_blend70.csv      — 70% trimmed + 30% median for 4+ groups")
print("4. sub_v4_trim15.csv       — More aggressive trim (15% vs 10%)")
print("5. sub_v4_blend_3way.csv   — Blend of baseline+robust+winsorized")
