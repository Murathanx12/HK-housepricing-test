"""
Final push: spatially-enhanced KNN for fallback + extreme outlier correction.
Add distance-to-MTR, distance-to-CBD, nearby malls/schools as KNN features.
Also: blend 7 extreme outlier single-match corrections.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import trim_mean
from sklearn.neighbors import KNeighborsRegressor, BallTree
from sklearn.preprocessing import StandardScaler

DATA_DIR = Path("./data")

print("Loading...")
train = pd.read_csv(DATA_DIR / "HK_house_transactions.csv")
test = pd.read_csv(DATA_DIR / "test_features.csv")

# Spatial data
mtr = pd.read_csv(DATA_DIR / "HK_mtr_station.csv")
cbd = pd.read_csv(DATA_DIR / "HK_city_center.csv")
malls = pd.read_csv(DATA_DIR / "HK_mall.csv")
schools = pd.read_csv(DATA_DIR / "HK_school.csv")
parks = pd.read_csv(DATA_DIR / "HK_park.csv")
hospitals = pd.read_csv(DATA_DIR / "HK_hospital.csv")

cbd_lat, cbd_lon = cbd["lat"].values[0], cbd["lon"].values[0]

def haversine(lat1, lon1, lat2, lon2):
    """Distance in km."""
    R = 6371
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon/2)**2
    return R * 2 * np.arcsin(np.sqrt(a))

def nearest_distance(lat, lon, ref_lats, ref_lons):
    """Distance to nearest point in reference set."""
    dists = haversine(lat, lon, ref_lats, ref_lons)
    return np.min(dists)

def count_within(lat, lon, ref_lats, ref_lons, radius_km):
    """Count reference points within radius."""
    dists = haversine(lat, lon, ref_lats, ref_lons)
    return np.sum(dists <= radius_km)

# Precompute spatial features
print("Computing spatial features...")
mtr_lats, mtr_lons = mtr["lat"].values, mtr["lon"].values
mall_lats, mall_lons = malls["lat"].values, malls["lon"].values
school_lats, school_lons = schools["lat"].values, schools["lon"].values

for df in [train, test]:
    df["dist_mtr"] = [nearest_distance(lat, lon, mtr_lats, mtr_lons)
                      for lat, lon in zip(df["wgs_lat"], df["wgs_lon"])]
    df["dist_cbd"] = haversine(df["wgs_lat"].values, df["wgs_lon"].values, cbd_lat, cbd_lon)
    df["n_malls_1km"] = [count_within(lat, lon, mall_lats, mall_lons, 1.0)
                         for lat, lon in zip(df["wgs_lat"], df["wgs_lon"])]
    df["n_schools_1km"] = [count_within(lat, lon, school_lats, school_lons, 1.0)
                           for lat, lon in zip(df["wgs_lat"], df["wgs_lon"])]

print("Spatial features done.")

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

fa_grp = train.groupby("full_addr")
fa_stats = fa_grp.agg(
    p_mean=("price", "mean"), p_median=("price", "median"),
    count=("price", "count"),
    ppsf_median=("ppsf", "median"),
    floor_mean=("floor", "mean"),
)
fa_trimmed = fa_grp["price"].apply(lambda x: trim_mean(x, 0.1) if len(x) >= 4 else x.mean())
fa_stats = fa_stats.join(fa_trimmed.rename("p_trimmed"))

ua_stats = train.groupby("unit_area5").agg(
    ppsf_mean=("ppsf", "mean"), ppsf_median=("ppsf", "median"),
    floor_mean=("floor", "mean"), count=("price", "count"),
)
unit_stats = train.groupby("unit_key").agg(
    ppsf_mean=("ppsf", "mean"), ppsf_median=("ppsf", "median"),
    floor_mean=("floor", "mean"), count=("price", "count"),
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

# ── KNN models ──
print("Building KNN models...")

# Standard KNN (baseline)
feat_basic = ["wgs_lat", "wgs_lon", "area_sqft", "floor"]
scaler_basic = StandardScaler()
X_basic_tr = scaler_basic.fit_transform(train[feat_basic].values)
X_basic_te = scaler_basic.transform(test[feat_basic].values)
knn_basic = KNeighborsRegressor(n_neighbors=10, weights="distance", n_jobs=-1)
knn_basic.fit(X_basic_tr, train["price"].values)
test["knn10"] = knn_basic.predict(X_basic_te)

# Enhanced KNN (with spatial features)
feat_enhanced = ["wgs_lat", "wgs_lon", "area_sqft", "floor", "dist_mtr", "dist_cbd", "n_malls_1km", "n_schools_1km"]
scaler_enh = StandardScaler()
X_enh_tr = scaler_enh.fit_transform(train[feat_enhanced].values)
X_enh_te = scaler_enh.transform(test[feat_enhanced].values)

for k in [5, 7, 10, 15]:
    knn_enh = KNeighborsRegressor(n_neighbors=k, weights="distance", n_jobs=-1)
    knn_enh.fit(X_enh_tr, train["price"].values)
    test[f"knn_enh{k}"] = knn_enh.predict(X_enh_te)

# PPSF-based enhanced KNN
knn_ppsf_enh = KNeighborsRegressor(n_neighbors=10, weights="distance", n_jobs=-1)
knn_ppsf_enh.fit(X_enh_tr, train["ppsf"].values)
test["knn_ppsf_enh10"] = knn_ppsf_enh.predict(X_enh_te) * test["area_sqft"]

print("KNN models done.")

# ── Compare KNN predictions for fallback rows ──
fallback_mask = ~test["full_addr"].isin(fa_stats.index)
print(f"\n=== KNN COMPARISON ON FALLBACK ROWS ({fallback_mask.sum()}) ===")
fb = test[fallback_mask]
for col in ["knn10", "knn_enh5", "knn_enh7", "knn_enh10", "knn_enh15", "knn_ppsf_enh10"]:
    diff = np.abs(fb[col] - fb["knn10"])
    print(f"  {col}: mean_diff_from_basic=${diff.mean():,.0f}, max=${diff.max():,.0f}")


def get_fallback(row, area, floor_val, slope, knn_col="knn10"):
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
    knn_p = row[knn_col]
    if dk in dist_stats.index:
        dp = area * dist_stats.loc[dk]["ppsf_median"]
        return 0.4 * knn_p + 0.6 * dp
    return knn_p


# ══════════════════════════════════════════════════
# GENERATE SUBMISSIONS
# ══════════════════════════════════════════════════
print("\n=== GENERATING SUBMISSIONS ===")

# Baseline
def gen_baseline():
    preds = np.zeros(len(test))
    for i in range(len(test)):
        row = test.iloc[i]
        area, floor_val = row["area_sqft"], row["floor"]
        fa = row["full_addr"]
        slope = bld_slopes.get(row["building"], 0.0)
        if fa in fa_stats.index:
            d = fa_stats.loc[fa]
            if d["count"] >= 4: preds[i] = d["p_trimmed"]
            elif d["count"] >= 2: preds[i] = d["p_median"]
            else: preds[i] = 0.8 * d["p_mean"] + 0.2 * row["knn10"]
        else:
            preds[i] = get_fallback(row, area, floor_val, slope, "knn10")
    return np.clip(preds, 2000, 500000)

# V1: Enhanced KNN in fallback chain
def gen_enhanced_knn():
    preds = np.zeros(len(test))
    for i in range(len(test)):
        row = test.iloc[i]
        area, floor_val = row["area_sqft"], row["floor"]
        fa = row["full_addr"]
        slope = bld_slopes.get(row["building"], 0.0)
        if fa in fa_stats.index:
            d = fa_stats.loc[fa]
            if d["count"] >= 4: preds[i] = d["p_trimmed"]
            elif d["count"] >= 2: preds[i] = d["p_median"]
            else: preds[i] = 0.8 * d["p_mean"] + 0.2 * row["knn_enh10"]
        else:
            preds[i] = get_fallback(row, area, floor_val, slope, "knn_enh10")
    return np.clip(preds, 2000, 500000)

# V2: Enhanced KNN + extreme outlier correction
def gen_enh_extreme():
    preds = np.zeros(len(test))
    for i in range(len(test)):
        row = test.iloc[i]
        area, floor_val = row["area_sqft"], row["floor"]
        fa = row["full_addr"]
        slope = bld_slopes.get(row["building"], 0.0)
        bk = row["building"]
        if fa in fa_stats.index:
            d = fa_stats.loc[fa]
            if d["count"] >= 4:
                preds[i] = d["p_trimmed"]
            elif d["count"] >= 2:
                preds[i] = d["p_median"]
            else:
                direct = d["p_mean"]
                # Check for extreme outlier
                if bk in bld_stats.index:
                    bld_pred = area * bld_stats.loc[bk]["ppsf_median"]
                    ratio = direct / bld_pred if bld_pred > 0 else 1.0
                    if ratio > 2.0 or ratio < 0.5:
                        preds[i] = 0.4 * direct + 0.6 * bld_pred
                    elif ratio > 1.5 or ratio < 0.67:
                        preds[i] = 0.65 * direct + 0.35 * bld_pred
                    else:
                        preds[i] = 0.8 * direct + 0.2 * row["knn_enh10"]
                else:
                    preds[i] = 0.8 * direct + 0.2 * row["knn_enh10"]
        else:
            preds[i] = get_fallback(row, area, floor_val, slope, "knn_enh10")
    return np.clip(preds, 2000, 500000)

# V3: PPSF-based enhanced KNN for singles and fallback
def gen_ppsf_enh():
    preds = np.zeros(len(test))
    for i in range(len(test)):
        row = test.iloc[i]
        area, floor_val = row["area_sqft"], row["floor"]
        fa = row["full_addr"]
        slope = bld_slopes.get(row["building"], 0.0)
        if fa in fa_stats.index:
            d = fa_stats.loc[fa]
            if d["count"] >= 4: preds[i] = d["p_trimmed"]
            elif d["count"] >= 2: preds[i] = d["p_median"]
            else:
                # Blend direct with PPSF-based enhanced KNN
                preds[i] = 0.8 * d["p_mean"] + 0.2 * row["knn_ppsf_enh10"]
        else:
            preds[i] = get_fallback(row, area, floor_val, slope, "knn_enh10")
    return np.clip(preds, 2000, 500000)

# V4: Multi-KNN blend for fallback (blend basic + enhanced + ppsf)
def gen_multi_knn():
    preds = np.zeros(len(test))
    for i in range(len(test)):
        row = test.iloc[i]
        area, floor_val = row["area_sqft"], row["floor"]
        fa = row["full_addr"]
        slope = bld_slopes.get(row["building"], 0.0)
        if fa in fa_stats.index:
            d = fa_stats.loc[fa]
            if d["count"] >= 4: preds[i] = d["p_trimmed"]
            elif d["count"] >= 2: preds[i] = d["p_median"]
            else:
                # Blend of 3 KNN predictions for more stability
                knn_blend = (row["knn10"] + row["knn_enh10"] + row["knn_ppsf_enh10"]) / 3
                preds[i] = 0.8 * d["p_mean"] + 0.2 * knn_blend
        else:
            # Multi-KNN blend fallback
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
                preds[i] = area * (base + fadj)
            elif uk in unit_stats.index:
                d = unit_stats.loc[uk]
                base = d["ppsf_median"] if d["count"] >= 3 else d["ppsf_mean"]
                fadj = slope * (floor_val - d["floor_mean"])
                preds[i] = area * (base + fadj)
            elif btk in bt_stats.index and bt_stats.loc[btk]["count"] >= 3:
                d = bt_stats.loc[btk]
                fadj = slope * (floor_val - d["floor_mean"])
                preds[i] = area * (d["ppsf_median"] + fadj)
            elif bfk in bf_stats.index and bf_stats.loc[bfk]["count"] >= 3:
                d = bf_stats.loc[bfk]
                fadj = slope * (floor_val - d["floor_mean"])
                preds[i] = area * (d["ppsf_median"] + fadj)
            elif bk in bld_stats.index:
                d = bld_stats.loc[bk]
                base = d["ppsf_median"] if d["count"] >= 3 else d["ppsf_mean"]
                fadj = slope * (floor_val - d["floor_mean"]) if d["count"] >= 5 else 0
                preds[i] = area * (base + fadj)
            else:
                # Multi-KNN blend
                knn_blend = (row["knn10"] + row["knn_enh10"] + row["knn_ppsf_enh10"]) / 3
                if dk in dist_stats.index:
                    dp = area * dist_stats.loc[dk]["ppsf_median"]
                    preds[i] = 0.3 * knn_blend + 0.7 * dp
                else:
                    preds[i] = knn_blend
    return np.clip(preds, 2000, 500000)

# Generate all
base = gen_baseline()
variants = {
    "sub_v8_baseline": base,
    "sub_v8_enh_knn": gen_enhanced_knn(),
    "sub_v8_enh_extreme": gen_enh_extreme(),
    "sub_v8_ppsf_enh": gen_ppsf_enh(),
    "sub_v8_multi_knn": gen_multi_knn(),
}

# Also generate blends of baseline with improved
variants["sub_v8_blend_base_enh"] = np.clip(0.5 * base + 0.5 * variants["sub_v8_enh_knn"], 2000, 500000)
variants["sub_v8_blend_base_extreme"] = np.clip(0.5 * base + 0.5 * variants["sub_v8_enh_extreme"], 2000, 500000)

print(f"\n{'Name':30s} {'Changed':>8s} {'MeanDiff':>10s} {'MaxDiff':>10s}")
print("-" * 62)
for name, preds in variants.items():
    pd.DataFrame({"id": test["id"].astype(int), "price": preds.astype(int)}).to_csv(f"{name}.csv", index=False)
    diff = np.abs(preds - base)
    changed = (diff > 1).sum()
    print(f"{name:30s} {changed:8d} ${diff.mean():8,.0f} ${diff.max():8,.0f}")

print("\n=== SUBMIT PRIORITY ===")
print("1. sub_v8_enh_extreme.csv      — Enhanced KNN + outlier correction (best combo)")
print("2. sub_v8_enh_knn.csv          — Just enhanced KNN (spatial features)")
print("3. sub_v8_multi_knn.csv        — Multi-KNN blend for stability")
print("4. sub_v8_blend_base_extreme   — 50/50 hedge baseline+extreme")
print("5. sub_v8_ppsf_enh.csv         — PPSF-based enhanced KNN")
