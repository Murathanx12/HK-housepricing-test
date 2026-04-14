"""
Hong Kong Rental Price Prediction — Hardcoded v3
==================================================
Focused on the error sources:
  - 31.3% single-match rows: blend with broader stats to reduce noise
  - 6.6% without full_addr match: smarter fallback
  - 62.1% multi-match: try different aggregations

5 submissions that are ACTUALLY different from each other.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from scipy.stats import trim_mean

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

# ── LOOKUP TABLES ──
print("Building lookups...")

def floor_slope(g):
    if len(g) < 5 or g["floor"].std() < 1: return 0.0
    return np.polyfit(g["floor"], g["ppsf"], 1)[0]

bld_slopes = train.groupby("building").apply(floor_slope, include_groups=False).to_dict()

# Full address: precompute multiple agg types
fa_grp = train.groupby("full_addr")
fa_stats = fa_grp.agg(
    p_mean=("price", "mean"), p_median=("price", "median"),
    p_std=("price", "std"), count=("price", "count"),
    ppsf_mean=("ppsf", "mean"), ppsf_median=("ppsf", "median"),
    floor_mean=("floor", "mean"),
)
fa_stats["p_std"] = fa_stats["p_std"].fillna(0)

# Trimmed mean for full_addr (robust to outliers)
fa_trimmed = fa_grp["price"].apply(lambda x: trim_mean(x, 0.1) if len(x) >= 4 else x.mean())
fa_trimmed.name = "p_trimmed"
fa_stats = fa_stats.join(fa_trimmed)

# Unit+area
ua_stats = train.groupby("unit_area5").agg(
    ppsf_mean=("ppsf", "mean"), ppsf_median=("ppsf", "median"),
    p_mean=("price", "mean"), p_median=("price", "median"),
    floor_mean=("floor", "mean"), count=("price", "count"),
)

# Unit
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
    floor_mean=("floor", "mean"),
)

# KNN
print("Building KNN...")
scaler = StandardScaler()
X_knn_tr = scaler.fit_transform(train[["wgs_lat", "wgs_lon", "area_sqft", "floor"]].values)
X_knn_te = scaler.transform(test[["wgs_lat", "wgs_lon", "area_sqft", "floor"]].values)

for k in [5, 10, 20]:
    knn = KNeighborsRegressor(n_neighbors=k, weights="distance", n_jobs=-1)
    knn.fit(X_knn_tr, train["price"].values)
    test[f"knn{k}"] = knn.predict(X_knn_te)

knn_ppsf = KNeighborsRegressor(n_neighbors=10, weights="distance", n_jobs=-1)
knn_ppsf.fit(X_knn_tr, train["ppsf"].values)
test["knn_ppsf10"] = knn_ppsf.predict(X_knn_te)


def get_fallback_pred(row, area, floor_val, slope):
    """Common fallback logic for rows without full_addr match."""
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
        return area * (base + fadj), "ua"

    if uk in unit_stats.index:
        d = unit_stats.loc[uk]
        base = d["ppsf_median"] if d["count"] >= 3 else d["ppsf_mean"]
        fadj = slope * (floor_val - d["floor_mean"])
        return area * (base + fadj), "unit"

    if btk in bt_stats.index and bt_stats.loc[btk]["count"] >= 3:
        d = bt_stats.loc[btk]
        fadj = slope * (floor_val - d["floor_mean"])
        return area * (d["ppsf_median"] + fadj), "bt"

    if bfk in bf_stats.index and bf_stats.loc[bfk]["count"] >= 3:
        d = bf_stats.loc[bfk]
        fadj = slope * (floor_val - d["floor_mean"])
        return area * (d["ppsf_median"] + fadj), "bf"

    if bk in bld_stats.index:
        d = bld_stats.loc[bk]
        base = d["ppsf_median"] if d["count"] >= 3 else d["ppsf_mean"]
        fadj = slope * (floor_val - d["floor_mean"]) if d["count"] >= 5 else 0
        return area * (base + fadj), "bld"

    # KNN + district blend
    knn_p = row["knn10"]
    if dk in dist_stats.index:
        dp = area * dist_stats.loc[dk]["ppsf_median"]
        return 0.4 * knn_p + 0.6 * dp, "fallback"
    return knn_p, "fallback"


# ── VARIANT 1: Pure median (our baseline) ──
def variant1():
    preds = np.zeros(len(test))
    for i in range(len(test)):
        row = test.iloc[i]
        area, floor_val = row["area_sqft"], row["floor"]
        fa = row["full_addr"]
        slope = bld_slopes.get(row["building"], 0.0)

        if fa in fa_stats.index:
            d = fa_stats.loc[fa]
            preds[i] = d["p_median"] if d["count"] >= 2 else d["p_mean"]
        else:
            preds[i], _ = get_fallback_pred(row, area, floor_val, slope)
    return np.clip(preds, 2000, 500000)


# ── VARIANT 2: Single-match rows blended with unit stats ──
def variant2():
    """For count=1 full_addr matches, blend 70% direct + 30% unit-level prediction."""
    preds = np.zeros(len(test))
    for i in range(len(test)):
        row = test.iloc[i]
        area, floor_val = row["area_sqft"], row["floor"]
        fa = row["full_addr"]
        uk = row["unit_key"]
        slope = bld_slopes.get(row["building"], 0.0)

        if fa in fa_stats.index:
            d = fa_stats.loc[fa]
            if d["count"] >= 2:
                preds[i] = d["p_median"]
            else:
                # Single match — blend with unit-level
                direct = d["p_mean"]
                if uk in unit_stats.index:
                    ud = unit_stats.loc[uk]
                    unit_pred = area * (ud["ppsf_median"] + slope * (floor_val - ud["floor_mean"]))
                    preds[i] = 0.7 * direct + 0.3 * unit_pred
                else:
                    preds[i] = direct
        else:
            preds[i], _ = get_fallback_pred(row, area, floor_val, slope)
    return np.clip(preds, 2000, 500000)


# ── VARIANT 3: Use ppsf*area + floor adj for EVERYTHING ──
def variant3():
    """Never use direct price. Always use ppsf * area + floor adjustment."""
    preds = np.zeros(len(test))
    for i in range(len(test)):
        row = test.iloc[i]
        area, floor_val = row["area_sqft"], row["floor"]
        fa = row["full_addr"]
        slope = bld_slopes.get(row["building"], 0.0)

        if fa in fa_stats.index:
            d = fa_stats.loc[fa]
            base = d["ppsf_median"] if d["count"] >= 2 else d["ppsf_mean"]
            fadj = slope * (floor_val - d["floor_mean"]) if d["count"] >= 2 else 0
            preds[i] = area * (base + fadj)
        else:
            preds[i], _ = get_fallback_pred(row, area, floor_val, slope)
    return np.clip(preds, 2000, 500000)


# ── VARIANT 4: Trimmed mean for multi-match, KNN-blended for single ──
def variant4():
    """Trimmed mean for >= 4 matches. Blend with KNN for single matches."""
    preds = np.zeros(len(test))
    for i in range(len(test)):
        row = test.iloc[i]
        area, floor_val = row["area_sqft"], row["floor"]
        fa = row["full_addr"]
        slope = bld_slopes.get(row["building"], 0.0)

        if fa in fa_stats.index:
            d = fa_stats.loc[fa]
            if d["count"] >= 4:
                preds[i] = d["p_trimmed"]
            elif d["count"] >= 2:
                preds[i] = d["p_median"]
            else:
                # Single match: 80% direct + 20% KNN
                preds[i] = 0.8 * d["p_mean"] + 0.2 * row["knn10"]
        else:
            preds[i], _ = get_fallback_pred(row, area, floor_val, slope)
    return np.clip(preds, 2000, 500000)


# ── VARIANT 5: Blend of v1 + v3 (direct price + ppsf-based) ──
def variant5():
    """50% direct price lookup + 50% ppsf*area. Best of both worlds."""
    v1 = variant1()
    v3 = variant3()
    return np.clip(0.5 * v1 + 0.5 * v3, 2000, 500000)


# ── RUN ALL ──
print("\nRunning variants...")
results = {}
for name, func in [
    ("sub_hc3_v1_median", variant1),
    ("sub_hc3_v2_blend_single", variant2),
    ("sub_hc3_v3_ppsf_only", variant3),
    ("sub_hc3_v4_trimmed_knn", variant4),
    ("sub_hc3_v5_half_half", variant5),
]:
    preds = func()
    results[name] = preds
    pd.DataFrame({"id": test["id"].astype(int), "price": preds.astype(int)}).to_csv(f"{name}.csv", index=False)
    print(f"  {name}: mean ${preds.mean():,.0f}, median ${np.median(preds):,.0f}")

# Show how different the variants actually are
print("\n=== DIFFERENCES FROM v1 BASELINE ===")
base = results["sub_hc3_v1_median"]
for name, preds in results.items():
    if name == "sub_hc3_v1_median": continue
    diff = np.abs(preds - base)
    changed = (diff > 1).sum()
    print(f"  {name}: {changed} rows changed, mean diff ${diff.mean():,.0f}, max diff ${diff.max():,.0f}")

print("\n=== FILES ===")
print("sub_hc3_v1_median.csv       — Pure median (baseline)")
print("sub_hc3_v2_blend_single.csv — Single-match blended with unit stats")
print("sub_hc3_v3_ppsf_only.csv    — ppsf*area + floor adj everywhere")
print("sub_hc3_v4_trimmed_knn.csv  — Trimmed mean + KNN blend for singles")
print("sub_hc3_v5_half_half.csv    — 50% v1 + 50% v3")
