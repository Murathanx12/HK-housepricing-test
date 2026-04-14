"""
FINAL optimized submissions to beat $1,450 RMSE.
Each variant changes only what the analysis shows could help.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import trim_mean
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler

DATA_DIR = Path("./data")

train = pd.read_csv(DATA_DIR / "HK_house_transactions.csv")
test = pd.read_csv(DATA_DIR / "test_features.csv")

# Spatial
mtr = pd.read_csv(DATA_DIR / "HK_mtr_station.csv")
cbd = pd.read_csv(DATA_DIR / "HK_city_center.csv")
cbd_lat, cbd_lon = cbd["lat"].values[0], cbd["lon"].values[0]

def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1))*np.cos(np.radians(lat2))*np.sin(dlon/2)**2
    return R * 2 * np.arcsin(np.sqrt(a))

mtr_lats, mtr_lons = mtr["lat"].values, mtr["lon"].values
for df in [train, test]:
    df["dist_mtr"] = [np.min(haversine(lat, lon, mtr_lats, mtr_lons))
                      for lat, lon in zip(df["wgs_lat"], df["wgs_lon"])]
    df["dist_cbd"] = haversine(df["wgs_lat"].values, df["wgs_lon"].values, cbd_lat, cbd_lon)

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
    df["addr_only"] = df["address"].fillna("")

train["ppsf"] = train["price"] / train["area_sqft"]

# ── LOOKUPS ──
def floor_slope(g):
    if len(g) < 5 or g["floor"].std() < 1: return 0.0
    return np.polyfit(g["floor"], g["ppsf"], 1)[0]

bld_slopes = train.groupby("building").apply(floor_slope, include_groups=False).to_dict()

fa_grp = train.groupby("full_addr")
fa_stats = fa_grp.agg(
    p_mean=("price", "mean"), p_median=("price", "median"),
    count=("price", "count"), ppsf_median=("ppsf", "median"),
    floor_mean=("floor", "mean"),
)
fa_trimmed = fa_grp["price"].apply(lambda x: trim_mean(x, 0.1) if len(x) >= 4 else x.mean())
fa_stats = fa_stats.join(fa_trimmed.rename("p_trimmed"))

ao_stats = train.groupby("addr_only").agg(
    ppsf_median=("ppsf", "median"), count=("price", "count"), floor_mean=("floor", "mean"),
)
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
dist_stats = train.groupby("district").agg(ppsf_median=("ppsf", "median"))

# KNN (basic + enhanced)
scaler_b = StandardScaler()
X_tr_b = scaler_b.fit_transform(train[["wgs_lat", "wgs_lon", "area_sqft", "floor"]].values)
X_te_b = scaler_b.transform(test[["wgs_lat", "wgs_lon", "area_sqft", "floor"]].values)
knn_b = KNeighborsRegressor(n_neighbors=10, weights="distance", n_jobs=-1)
knn_b.fit(X_tr_b, train["price"].values)
test["knn10"] = knn_b.predict(X_te_b)

feat_enh = ["wgs_lat", "wgs_lon", "area_sqft", "floor", "dist_mtr", "dist_cbd"]
scaler_e = StandardScaler()
X_tr_e = scaler_e.fit_transform(train[feat_enh].values)
X_te_e = scaler_e.transform(test[feat_enh].values)
knn_e = KNeighborsRegressor(n_neighbors=10, weights="distance", n_jobs=-1)
knn_e.fit(X_tr_e, train["price"].values)
test["knn_enh"] = knn_e.predict(X_te_e)

# ── FALLBACK FUNCTIONS ──
def fallback_original(row, area, floor_val, slope):
    uak, uk, btk, bfk, bk, dk = row["unit_area5"], row["unit_key"], row["bld_tower"], row["bld_flat"], row["building"], row["district"]
    if uak in ua_stats.index:
        d = ua_stats.loc[uak]
        base = d["ppsf_median"] if d["count"] >= 2 else d["ppsf_mean"]
        fadj = slope * (floor_val - d["floor_mean"]) if d["count"] >= 2 else 0
        return area * (base + fadj)
    if uk in unit_stats.index:
        d = unit_stats.loc[uk]
        base = d["ppsf_median"] if d["count"] >= 3 else d["ppsf_mean"]
        return area * (base + slope * (floor_val - d["floor_mean"]))
    if btk in bt_stats.index and bt_stats.loc[btk]["count"] >= 3:
        d = bt_stats.loc[btk]
        return area * (d["ppsf_median"] + slope * (floor_val - d["floor_mean"]))
    if bfk in bf_stats.index and bf_stats.loc[bfk]["count"] >= 3:
        d = bf_stats.loc[bfk]
        return area * (d["ppsf_median"] + slope * (floor_val - d["floor_mean"]))
    if bk in bld_stats.index:
        d = bld_stats.loc[bk]
        base = d["ppsf_median"] if d["count"] >= 3 else d["ppsf_mean"]
        fadj = slope * (floor_val - d["floor_mean"]) if d["count"] >= 5 else 0
        return area * (base + fadj)
    knn_p = row["knn10"]
    if dk in dist_stats.index:
        return 0.4 * knn_p + 0.6 * area * dist_stats.loc[dk]["ppsf_median"]
    return knn_p

def fallback_improved(row, area, floor_val, slope):
    """addr_only first, then enhanced KNN at end."""
    ao = row["addr_only"]
    if ao in ao_stats.index and ao_stats.loc[ao]["count"] >= 3:
        d = ao_stats.loc[ao]
        fadj = slope * (floor_val - d["floor_mean"]) if d["count"] >= 3 else 0
        return area * (d["ppsf_median"] + fadj)
    uak, uk, btk, bfk, bk, dk = row["unit_area5"], row["unit_key"], row["bld_tower"], row["bld_flat"], row["building"], row["district"]
    if uak in ua_stats.index:
        d = ua_stats.loc[uak]
        base = d["ppsf_median"] if d["count"] >= 2 else d["ppsf_mean"]
        fadj = slope * (floor_val - d["floor_mean"]) if d["count"] >= 2 else 0
        return area * (base + fadj)
    if uk in unit_stats.index:
        d = unit_stats.loc[uk]
        base = d["ppsf_median"] if d["count"] >= 3 else d["ppsf_mean"]
        return area * (base + slope * (floor_val - d["floor_mean"]))
    if btk in bt_stats.index and bt_stats.loc[btk]["count"] >= 3:
        d = bt_stats.loc[btk]
        return area * (d["ppsf_median"] + slope * (floor_val - d["floor_mean"]))
    if bfk in bf_stats.index and bf_stats.loc[bfk]["count"] >= 3:
        d = bf_stats.loc[bfk]
        return area * (d["ppsf_median"] + slope * (floor_val - d["floor_mean"]))
    if bk in bld_stats.index:
        d = bld_stats.loc[bk]
        base = d["ppsf_median"] if d["count"] >= 3 else d["ppsf_mean"]
        fadj = slope * (floor_val - d["floor_mean"]) if d["count"] >= 5 else 0
        return area * (base + fadj)
    knn_p = row["knn_enh"]  # enhanced KNN
    if dk in dist_stats.index:
        return 0.4 * knn_p + 0.6 * area * dist_stats.loc[dk]["ppsf_median"]
    return knn_p


# ══════════════════════════════════════════════════
# SUBMISSION A: Exact baseline ($1,450 reproduction)
# ══════════════════════════════════════════════════
def sub_baseline():
    preds = np.zeros(len(test))
    for i in range(len(test)):
        row = test.iloc[i]
        area, fv = row["area_sqft"], row["floor"]
        fa = row["full_addr"]
        slope = bld_slopes.get(row["building"], 0.0)
        if fa in fa_stats.index:
            d = fa_stats.loc[fa]
            if d["count"] >= 4: preds[i] = d["p_trimmed"]
            elif d["count"] >= 2: preds[i] = d["p_median"]
            else: preds[i] = 0.8 * d["p_mean"] + 0.2 * row["knn10"]
        else:
            preds[i] = fallback_original(row, area, fv, slope)
    return np.clip(preds, 2000, 500000)


# ══════════════════════════════════════════════════
# SUBMISSION B: Fix 7 extreme single-match outliers
# Only change: shrink 7 outlier predictions toward building median
# ══════════════════════════════════════════════════
def sub_extreme7():
    preds = np.zeros(len(test))
    fixes = 0
    for i in range(len(test)):
        row = test.iloc[i]
        area, fv = row["area_sqft"], row["floor"]
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
                if bk in bld_stats.index:
                    bld_pred = area * bld_stats.loc[bk]["ppsf_median"]
                    ratio = direct / bld_pred if bld_pred > 0 else 1.0
                    if ratio > 1.5 or ratio < 0.67:
                        # Shrink toward building: 50/50 blend
                        preds[i] = 0.5 * direct + 0.5 * bld_pred
                        fixes += 1
                    else:
                        preds[i] = 0.8 * direct + 0.2 * row["knn10"]
                else:
                    preds[i] = 0.8 * direct + 0.2 * row["knn10"]
        else:
            preds[i] = fallback_original(row, area, fv, slope)
    print(f"  sub_extreme7: {fixes} rows corrected")
    return np.clip(preds, 2000, 500000)


# ══════════════════════════════════════════════════
# SUBMISSION C: Improved fallback only (keep matched identical)
# ══════════════════════════════════════════════════
def sub_better_fallback():
    preds = np.zeros(len(test))
    for i in range(len(test)):
        row = test.iloc[i]
        area, fv = row["area_sqft"], row["floor"]
        fa = row["full_addr"]
        slope = bld_slopes.get(row["building"], 0.0)
        if fa in fa_stats.index:
            d = fa_stats.loc[fa]
            if d["count"] >= 4: preds[i] = d["p_trimmed"]
            elif d["count"] >= 2: preds[i] = d["p_median"]
            else: preds[i] = 0.8 * d["p_mean"] + 0.2 * row["knn10"]
        else:
            preds[i] = fallback_improved(row, area, fv, slope)
    return np.clip(preds, 2000, 500000)


# ══════════════════════════════════════════════════
# SUBMISSION D: Extreme fix + improved fallback (both changes)
# ══════════════════════════════════════════════════
def sub_combo():
    preds = np.zeros(len(test))
    for i in range(len(test)):
        row = test.iloc[i]
        area, fv = row["area_sqft"], row["floor"]
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
                if bk in bld_stats.index:
                    bld_pred = area * bld_stats.loc[bk]["ppsf_median"]
                    ratio = direct / bld_pred if bld_pred > 0 else 1.0
                    if ratio > 1.5 or ratio < 0.67:
                        preds[i] = 0.5 * direct + 0.5 * bld_pred
                    else:
                        preds[i] = 0.8 * direct + 0.2 * row["knn10"]
                else:
                    preds[i] = 0.8 * direct + 0.2 * row["knn10"]
        else:
            preds[i] = fallback_improved(row, area, fv, slope)
    return np.clip(preds, 2000, 500000)


# ══════════════════════════════════════════════════
# SUBMISSION E: Blend of baseline + extreme7 (50/50 hedge)
# ══════════════════════════════════════════════════

# ── RUN ──
print("Generating final submissions...\n")
base = sub_baseline()
v_extreme = sub_extreme7()
v_fallback = sub_better_fallback()
v_combo = sub_combo()
v_blend = np.clip(0.5 * base + 0.5 * v_extreme, 2000, 500000)

variants = {
    "my_submission_baseline": base,
    "my_submission_extreme7": v_extreme,
    "my_submission_fallback": v_fallback,
    "my_submission_combo": v_combo,
    "my_submission_blend": v_blend,
}

print(f"\n{'Name':30s} {'Changed':>8s} {'MeanDiff':>10s} {'MaxDiff':>10s}")
print("-" * 62)
for name, preds in variants.items():
    pd.DataFrame({"id": test["id"].astype(int), "price": preds.astype(int)}).to_csv(f"{name}.csv", index=False)
    diff = np.abs(preds - base)
    changed = (diff > 1).sum()
    print(f"{name:30s} {changed:8d} ${diff.mean():8,.0f} ${diff.max():8,.0f}")

# Show exactly what changed for extreme7
print("\n=== EXTREME7 CORRECTIONS ===")
for i in range(len(test)):
    d = abs(v_extreme[i] - base[i])
    if d > 1:
        row = test.iloc[i]
        fa = row["full_addr"]
        direct = fa_stats.loc[fa]["p_mean"]
        bld_pred = row["area_sqft"] * bld_stats.loc[row["building"]]["ppsf_median"]
        ratio = direct / bld_pred
        print(f"  id={int(row['id']):5d}: ${base[i]:>8,.0f} -> ${v_extreme[i]:>8,.0f} "
              f"(direct=${direct:,.0f} bld=${bld_pred:,.0f} ratio={ratio:.2f}) "
              f"{row['building'][:35]}")

print("\n=== SUBMIT ORDER ===")
print("1. my_submission_extreme7.csv  — Fix 7 outlier single-match rows (HIGH PRIORITY)")
print("2. my_submission_combo.csv     — Extreme fix + improved fallback")
print("3. my_submission_fallback.csv  — Only improve fallback chain")
print("4. my_submission_blend.csv     — 50/50 hedge of baseline + extreme7")
