"""
Hong Kong Rental Price Prediction — Hardcoded Ultra
=====================================================
5 pure-lookup variants. No ML. ~30 seconds total.

What we learned from leaderboard:
  - Pure hardcoded lookup is the best approach
  - ML adds noise / overfitting even with LOO
  - 93.4% of test rows have exact full_addr+area matches
  - Median price std within matched addresses is only $212

Variants try different:
  1. Aggregation methods (mean vs median vs trimmed mean)
  2. Floor adjustment strengths
  3. Fallback strategies
  4. Area-matching granularity
  5. KNN blending for low-confidence rows
"""

import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler

DATA_DIR = Path("./data")

# ─────────────────────────────────────────────
# LOAD & PARSE
# ─────────────────────────────────────────────
print("Loading data...")
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
    # Multiple area bin sizes
    df["area_bin5"] = (df["area_sqft"] / 5).round() * 5
    df["area_bin10"] = (df["area_sqft"] / 10).round() * 10
    df["area_bin20"] = (df["area_sqft"] / 20).round() * 20
    df["unit_area5"] = df["unit_key"] + "|" + df["area_bin5"].astype(str)
    df["unit_area10"] = df["unit_key"] + "|" + df["area_bin10"].astype(str)
    df["unit_area20"] = df["unit_key"] + "|" + df["area_bin20"].astype(str)

train["ppsf"] = train["price"] / train["area_sqft"]

# ─────────────────────────────────────────────
# PRECOMPUTE ALL LOOKUP TABLES
# ─────────────────────────────────────────────
print("Building lookup tables...")

def trimmed_mean(x, pct=0.1):
    """Mean after removing top/bottom pct."""
    if len(x) < 4: return x.mean()
    n = max(1, int(len(x) * pct))
    s = x.sort_values().iloc[n:-n]
    return s.mean() if len(s) > 0 else x.mean()

# Building floor slopes
def floor_slope(g):
    if len(g) < 5 or g["floor"].std() < 1: return 0.0
    return np.polyfit(g["floor"], g["ppsf"], 1)[0]

bld_slopes = train.groupby("building").apply(floor_slope, include_groups=False).to_dict()

# Unit-level floor slopes (more precise)
def unit_floor_slope(g):
    if len(g) < 3 or g["floor"].std() < 1: return np.nan
    return np.polyfit(g["floor"], g["ppsf"], 1)[0]

unit_slopes = train.groupby("unit_key").apply(unit_floor_slope, include_groups=False).to_dict()

# Full address stats
fa_stats = train.groupby("full_addr").agg(
    p_mean=("price", "mean"), p_median=("price", "median"),
    p_std=("price", "std"), count=("price", "count"),
    ppsf_mean=("ppsf", "mean"), ppsf_median=("ppsf", "median"),
    floor_mean=("floor", "mean"),
)
fa_stats["p_std"] = fa_stats["p_std"].fillna(0)

# Unit+area (multiple granularities)
for area_key in ["unit_area5", "unit_area10", "unit_area20"]:
    globals()[f"{area_key}_stats"] = train.groupby(area_key).agg(
        ppsf_mean=("ppsf", "mean"), ppsf_median=("ppsf", "median"),
        p_mean=("price", "mean"), p_median=("price", "median"),
        floor_mean=("floor", "mean"), count=("price", "count"),
    )

# Unit stats
unit_stats = train.groupby("unit_key").agg(
    ppsf_mean=("ppsf", "mean"), ppsf_median=("ppsf", "median"),
    p_mean=("price", "mean"), p_median=("price", "median"),
    floor_mean=("floor", "mean"), area_mean=("area_sqft", "mean"),
    count=("price", "count"),
)

# Building+tower
bt_stats = train.groupby("bld_tower").agg(
    ppsf_mean=("ppsf", "mean"), ppsf_median=("ppsf", "median"),
    floor_mean=("floor", "mean"), count=("price", "count"),
)

# Building+flat
bf_stats = train.groupby("bld_flat").agg(
    ppsf_mean=("ppsf", "mean"), ppsf_median=("ppsf", "median"),
    floor_mean=("floor", "mean"), count=("price", "count"),
)

# Building
bld_stats = train.groupby("building").agg(
    ppsf_mean=("ppsf", "mean"), ppsf_median=("ppsf", "median"),
    floor_mean=("floor", "mean"), count=("price", "count"),
)

# District
dist_stats = train.groupby("district").agg(
    ppsf_mean=("ppsf", "mean"), ppsf_median=("ppsf", "median"),
    floor_mean=("floor", "mean"), count=("price", "count"),
)

# KNN for fallback
print("Building KNN...")
scaler = StandardScaler()
X_knn_tr = scaler.fit_transform(train[["wgs_lat", "wgs_lon", "area_sqft", "floor"]].values)
X_knn_te = scaler.transform(test[["wgs_lat", "wgs_lon", "area_sqft", "floor"]].values)

knn5 = KNeighborsRegressor(n_neighbors=5, weights="distance", n_jobs=-1)
knn5.fit(X_knn_tr, train["price"].values)
knn5_prices = knn5.predict(X_knn_te)

knn10 = KNeighborsRegressor(n_neighbors=10, weights="distance", n_jobs=-1)
knn10.fit(X_knn_tr, train["price"].values)
knn10_prices = knn10.predict(X_knn_te)

knn20 = KNeighborsRegressor(n_neighbors=20, weights="distance", n_jobs=-1)
knn20.fit(X_knn_tr, train["price"].values)
knn20_prices = knn20.predict(X_knn_te)

knn_ppsf = KNeighborsRegressor(n_neighbors=10, weights="distance", n_jobs=-1)
knn_ppsf.fit(X_knn_tr, train["ppsf"].values)
knn_ppsf_vals = knn_ppsf.predict(X_knn_te)

# ─────────────────────────────────────────────
# PREDICTION FUNCTION
# ─────────────────────────────────────────────

def predict_all(variant_name, config):
    """Generate predictions for one variant."""
    use_mean = config.get("use_mean", False)  # median vs mean
    floor_strength = config.get("floor_strength", 1.0)  # floor adj multiplier
    area_bin = config.get("area_bin", "unit_area5")
    knn_blend = config.get("knn_blend", 0.0)  # blend with KNN for all
    fa_use_ppsf = config.get("fa_use_ppsf", False)  # use ppsf*area instead of direct price for full_addr

    ua_stats_table = globals()[f"{area_bin}_stats"]

    predictions = np.zeros(len(test))
    methods = []

    for i in range(len(test)):
        row = test.iloc[i]
        area = row["area_sqft"]
        floor_val = row["floor"]
        fa = row["full_addr"]
        uak = row[area_bin]
        uk = row["unit_key"]
        btk = row["bld_tower"]
        bfk = row["bld_flat"]
        bk = row["building"]
        dk = row["district"]

        # Get slope (prefer unit-level, fallback to building)
        u_slope = unit_slopes.get(uk, np.nan)
        b_slope = bld_slopes.get(bk, 0.0)
        slope = (u_slope if not np.isnan(u_slope) else b_slope) * floor_strength

        pred = None

        # Level 0: full address + area match
        if fa in fa_stats.index:
            d = fa_stats.loc[fa]
            if d["count"] >= 2:
                if fa_use_ppsf:
                    ppsf_val = d["ppsf_median"] if not use_mean else d["ppsf_mean"]
                    floor_adj = slope * (floor_val - d["floor_mean"])
                    pred = area * (ppsf_val + floor_adj)
                else:
                    pred = d["p_median"] if not use_mean else d["p_mean"]
                methods.append("fa>=2")
            else:
                if fa_use_ppsf:
                    pred = area * d["ppsf_mean"]
                else:
                    pred = d["p_mean"]
                methods.append("fa=1")

        # Level 1: unit + area bin
        if pred is None and uak in ua_stats_table.index:
            d = ua_stats_table.loc[uak]
            base = d["ppsf_median"] if (not use_mean and d["count"] >= 2) else d["ppsf_mean"]
            floor_adj = slope * (floor_val - d["floor_mean"]) if d["count"] >= 2 else 0
            pred = area * (base + floor_adj)
            methods.append(f"ua>={'2' if d['count']>=2 else '1'}")

        # Level 2: unit
        if pred is None and uk in unit_stats.index:
            d = unit_stats.loc[uk]
            base = d["ppsf_median"] if (not use_mean and d["count"] >= 3) else d["ppsf_mean"]
            floor_adj = slope * (floor_val - d["floor_mean"])
            pred = area * (base + floor_adj)
            methods.append("unit")

        # Level 3: building+tower
        if pred is None and btk in bt_stats.index and bt_stats.loc[btk]["count"] >= 3:
            d = bt_stats.loc[btk]
            base = d["ppsf_median"] if not use_mean else d["ppsf_mean"]
            floor_adj = b_slope * floor_strength * (floor_val - d["floor_mean"])
            pred = area * (base + floor_adj)
            methods.append("bt")

        # Level 4: building+flat
        if pred is None and bfk in bf_stats.index and bf_stats.loc[bfk]["count"] >= 3:
            d = bf_stats.loc[bfk]
            base = d["ppsf_median"] if not use_mean else d["ppsf_mean"]
            floor_adj = b_slope * floor_strength * (floor_val - d["floor_mean"])
            pred = area * (base + floor_adj)
            methods.append("bf")

        # Level 5: building
        if pred is None and bk in bld_stats.index:
            d = bld_stats.loc[bk]
            base = d["ppsf_median"] if (not use_mean and d["count"] >= 3) else d["ppsf_mean"]
            floor_adj = b_slope * floor_strength * (floor_val - d["floor_mean"]) if d["count"] >= 5 else 0
            pred = area * (base + floor_adj)
            methods.append("bld")

        # Level 6: district + KNN
        if pred is None:
            if dk in dist_stats.index:
                d = dist_stats.loc[dk]
                dist_pred = area * d["ppsf_median"]
                pred = 0.4 * knn10_prices[i] + 0.6 * dist_pred
            else:
                pred = knn10_prices[i]
            methods.append("fallback")

        # Optional KNN blending for all predictions
        if knn_blend > 0:
            pred = (1 - knn_blend) * pred + knn_blend * knn10_prices[i]

        predictions[i] = pred

    predictions = np.clip(predictions, 2000, 500000)

    mc = Counter(methods)
    print(f"\n{variant_name}:")
    for m, c in sorted(mc.items(), key=lambda x: -x[1]):
        print(f"  {m:12s}: {c:5d} ({c/len(test)*100:.1f}%)")
    print(f"  Price: ${predictions.min():,.0f} - ${predictions.max():,.0f}, mean ${predictions.mean():,.0f}")

    return predictions

# ─────────────────────────────────────────────
# GENERATE 5 VARIANTS
# ─────────────────────────────────────────────
print("\nGenerating 5 variants...")

variants = {
    # V1: Our previous best approach (baseline for comparison)
    "v1_baseline": {
        "use_mean": False, "floor_strength": 1.0,
        "area_bin": "unit_area5", "knn_blend": 0.0, "fa_use_ppsf": False,
    },
    # V2: Use ppsf*area for full_addr matches too (allows floor adjustment)
    "v2_ppsf_everywhere": {
        "use_mean": False, "floor_strength": 1.0,
        "area_bin": "unit_area5", "knn_blend": 0.0, "fa_use_ppsf": True,
    },
    # V3: Stronger floor adjustments
    "v3_strong_floor": {
        "use_mean": False, "floor_strength": 1.5,
        "area_bin": "unit_area5", "knn_blend": 0.0, "fa_use_ppsf": True,
    },
    # V4: Wider area bins (more data per bin, less noise)
    "v4_wider_bins": {
        "use_mean": False, "floor_strength": 1.0,
        "area_bin": "unit_area10", "knn_blend": 0.0, "fa_use_ppsf": False,
    },
    # V5: Light KNN blend (5% KNN for smoothing)
    "v5_knn_smooth": {
        "use_mean": False, "floor_strength": 1.0,
        "area_bin": "unit_area5", "knn_blend": 0.05, "fa_use_ppsf": False,
    },
}

all_preds = {}
for name, config in variants.items():
    preds = predict_all(name, config)
    all_preds[name] = preds
    sub = pd.DataFrame({"id": test["id"].astype(int), "price": preds.astype(int)})
    sub.to_csv(f"sub_{name}.csv", index=False)

# Also create blends between variants
print("\n\nGenerating blends...")

# Blend of all 5 (equal weight)
avg_all = np.mean(list(all_preds.values()), axis=0)
avg_all = np.clip(avg_all, 2000, 500000)
pd.DataFrame({"id": test["id"].astype(int), "price": avg_all.astype(int)}).to_csv("sub_blend_all5.csv", index=False)
print(f"blend_all5: mean ${avg_all.mean():,.0f}")

# Blend v1 + v2 (direct price vs ppsf for full_addr)
b12 = 0.5 * all_preds["v1_baseline"] + 0.5 * all_preds["v2_ppsf_everywhere"]
b12 = np.clip(b12, 2000, 500000)
pd.DataFrame({"id": test["id"].astype(int), "price": b12.astype(int)}).to_csv("sub_blend_v1v2.csv", index=False)
print(f"blend_v1v2: mean ${b12.mean():,.0f}")

# Blend v1 + v4 (tight vs wide bins)
b14 = 0.5 * all_preds["v1_baseline"] + 0.5 * all_preds["v4_wider_bins"]
b14 = np.clip(b14, 2000, 500000)
pd.DataFrame({"id": test["id"].astype(int), "price": b14.astype(int)}).to_csv("sub_blend_v1v4.csv", index=False)
print(f"blend_v1v4: mean ${b14.mean():,.0f}")

print("\n=== FILES GENERATED ===")
print("Individual variants:")
for name in variants:
    print(f"  sub_{name}.csv")
print("Blends:")
print("  sub_blend_all5.csv  (average of all 5)")
print("  sub_blend_v1v2.csv  (baseline + ppsf_everywhere)")
print("  sub_blend_v1v4.csv  (tight bins + wide bins)")
