"""
Combined approach: every small improvement at once.
Also: investigate systematic bias and try post-hoc corrections.
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

# Spatial features
mtr = pd.read_csv(DATA_DIR / "HK_mtr_station.csv")
cbd = pd.read_csv(DATA_DIR / "HK_city_center.csv")
cbd_lat, cbd_lon = cbd["lat"].values[0], cbd["lon"].values[0]

def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon/2)**2
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

# ── SYSTEMATIC BIAS ANALYSIS ──
print("\n=== SYSTEMATIC BIAS CHECK ===")

# Check: is there a floor-based systematic bias?
# Training prices: do higher floors command more in ALL buildings?
high_floor = train[train["floor"] >= 20]["ppsf"].median()
low_floor = train[train["floor"] <= 8]["ppsf"].median()
print(f"High floor (>=20) median ppsf: ${high_floor:.1f}")
print(f"Low floor (<=8) median ppsf: ${low_floor:.1f}")
print(f"Premium: {100*(high_floor/low_floor - 1):.1f}%")

# Check: ppsf by area size brackets
for lo, hi, label in [(0, 300, "Nano"), (300, 500, "Small"), (500, 800, "Medium"),
                       (800, 1200, "Large"), (1200, 9999, "Luxury")]:
    subset = train[(train["area_sqft"] >= lo) & (train["area_sqft"] < hi)]
    print(f"  {label:8s} ({lo}-{hi}sqft): n={len(subset):5d}, median_ppsf=${subset['ppsf'].median():.1f}, "
          f"mean_price=${subset['price'].mean():,.0f}")

# Check: ppsf by district
print("\nPPSF by district:")
for dist, grp in sorted(train.groupby("district"), key=lambda x: -x[1]["ppsf"].median()):
    print(f"  {dist:45s}: n={len(grp):5d}, med_ppsf=${grp['ppsf'].median():.1f}")

# ── LOOKUP TABLES ──
print("\nBuilding lookups...")

def floor_slope(g):
    if len(g) < 5 or g["floor"].std() < 1: return 0.0
    return np.polyfit(g["floor"], g["ppsf"], 1)[0]

bld_slopes = train.groupby("building").apply(floor_slope, include_groups=False).to_dict()

fa_grp = train.groupby("full_addr")
fa_stats = fa_grp.agg(
    p_mean=("price", "mean"), p_median=("price", "median"),
    count=("price", "count"),
    ppsf_mean=("ppsf", "mean"), ppsf_median=("ppsf", "median"),
    floor_mean=("floor", "mean"),
)
fa_trimmed = fa_grp["price"].apply(lambda x: trim_mean(x, 0.1) if len(x) >= 4 else x.mean())
fa_stats = fa_stats.join(fa_trimmed.rename("p_trimmed"))

# Addr only (no area restriction)
ao_stats = train.groupby("addr_only").agg(
    ppsf_mean=("ppsf", "mean"), ppsf_median=("ppsf", "median"),
    count=("price", "count"), floor_mean=("floor", "mean"),
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
dist_stats = train.groupby("district").agg(
    ppsf_median=("ppsf", "median"),
)

# KNN
print("Building KNN...")
feat = ["wgs_lat", "wgs_lon", "area_sqft", "floor", "dist_mtr", "dist_cbd"]
scaler = StandardScaler()
X_tr = scaler.fit_transform(train[feat].values)
X_te = scaler.transform(test[feat].values)
knn10 = KNeighborsRegressor(n_neighbors=10, weights="distance", n_jobs=-1)
knn10.fit(X_tr, train["price"].values)
test["knn10"] = knn10.predict(X_te)

# Also basic KNN for comparison
scaler2 = StandardScaler()
X_tr2 = scaler2.fit_transform(train[["wgs_lat", "wgs_lon", "area_sqft", "floor"]].values)
X_te2 = scaler2.transform(test[["wgs_lat", "wgs_lon", "area_sqft", "floor"]].values)
knn_basic = KNeighborsRegressor(n_neighbors=10, weights="distance", n_jobs=-1)
knn_basic.fit(X_tr2, train["price"].values)
test["knn_basic"] = knn_basic.predict(X_te2)


def get_fallback(row, area, floor_val, slope):
    uak = row["unit_area5"]
    uk = row["unit_key"]
    btk = row["bld_tower"]
    bfk = row["bld_flat"]
    bk = row["building"]
    dk = row["district"]
    # Try addr_only for PPSF (same apartment, different area)
    ao = row["addr_only"]
    if ao in ao_stats.index and ao_stats.loc[ao]["count"] >= 3:
        d = ao_stats.loc[ao]
        fadj = slope * (floor_val - d["floor_mean"]) if d["count"] >= 3 else 0
        return area * (d["ppsf_median"] + fadj)
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
# BASELINE
# ══════════════════════════════════════════════════
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
            else: preds[i] = 0.8 * d["p_mean"] + 0.2 * row["knn_basic"]
        else:
            # Use original fallback chain (no addr_only)
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
                knn_p = row["knn_basic"]
                if dk in dist_stats.index:
                    dp = area * dist_stats.loc[dk]["ppsf_median"]
                    preds[i] = 0.4 * knn_p + 0.6 * dp
                else:
                    preds[i] = knn_p
    return np.clip(preds, 2000, 500000)


# ══════════════════════════════════════════════════
# COMBO V1: All small improvements together
# - Enhanced KNN with spatial features
# - addr_only in fallback chain
# - Extreme outlier correction (only 7 rows)
# - Keep matched rows EXACTLY as baseline
# ══════════════════════════════════════════════════
def gen_combo1():
    """Only change fallback and extreme outliers. Keep matched rows intact."""
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
                # Extreme outlier check
                if bk in bld_stats.index:
                    bld_pred = area * bld_stats.loc[bk]["ppsf_median"]
                    ratio = direct / bld_pred if bld_pred > 0 else 1.0
                    if ratio > 2.0 or ratio < 0.5:
                        preds[i] = 0.4 * direct + 0.6 * bld_pred
                    else:
                        preds[i] = 0.8 * direct + 0.2 * row["knn10"]
                else:
                    preds[i] = 0.8 * direct + 0.2 * row["knn10"]
        else:
            preds[i] = get_fallback(row, area, floor_val, slope)
    return np.clip(preds, 2000, 500000)


# ══════════════════════════════════════════════════
# COMBO V2: Keep matched exactly, improved fallback only
# ══════════════════════════════════════════════════
def gen_combo2():
    """Same as baseline for matched rows. Only improve fallback chain."""
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
            else: preds[i] = 0.8 * d["p_mean"] + 0.2 * row["knn_basic"]
        else:
            # Improved fallback with addr_only and enhanced KNN
            preds[i] = get_fallback(row, area, floor_val, slope)
    return np.clip(preds, 2000, 500000)


# ══════════════════════════════════════════════════
# COMBO V3: Only extreme outlier correction (minimal change)
# ══════════════════════════════════════════════════
def gen_extreme_only():
    """Baseline except: correct the 7 extreme outlier single-match rows."""
    preds = np.zeros(len(test))
    corrections = []
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
                if bk in bld_stats.index:
                    bld_pred = area * bld_stats.loc[bk]["ppsf_median"]
                    ratio = direct / bld_pred if bld_pred > 0 else 1.0
                    if ratio > 2.0 or ratio < 0.5:
                        old = 0.8 * direct + 0.2 * row["knn_basic"]
                        new = 0.4 * direct + 0.6 * bld_pred
                        preds[i] = new
                        corrections.append((int(row["id"]), old, new, ratio, bk[:30]))
                    else:
                        preds[i] = 0.8 * direct + 0.2 * row["knn_basic"]
                else:
                    preds[i] = 0.8 * direct + 0.2 * row["knn_basic"]
        else:
            # Use original fallback (same as baseline)
            uak = row["unit_area5"]
            uk = row["unit_key"]
            btk = row["bld_tower"]
            bfk = row["bld_flat"]
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
                knn_p = row["knn_basic"]
                if dk in dist_stats.index:
                    dp = area * dist_stats.loc[dk]["ppsf_median"]
                    preds[i] = 0.4 * knn_p + 0.6 * dp
                else:
                    preds[i] = knn_p
    print(f"  Extreme corrections made: {len(corrections)}")
    for c in corrections:
        print(f"    id={c[0]}: ${c[1]:,.0f} -> ${c[2]:,.0f} (ratio={c[3]:.2f}, {c[4]})")
    return np.clip(preds, 2000, 500000)


# ══════════════════════════════════════════════════
# COMBO V4: Only correct the TOP 2 most extreme outliers
# ══════════════════════════════════════════════════
def gen_top2_extreme():
    """Only correct rows with ratio > 2.0 or < 0.5 (most extreme only)."""
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
                if bk in bld_stats.index:
                    bld_pred = area * bld_stats.loc[bk]["ppsf_median"]
                    ratio = direct / bld_pred if bld_pred > 0 else 1.0
                    # Only the most extreme cases
                    if ratio > 2.5 or ratio < 0.4:
                        preds[i] = 0.3 * direct + 0.7 * bld_pred
                    else:
                        preds[i] = 0.8 * direct + 0.2 * row["knn_basic"]
                else:
                    preds[i] = 0.8 * direct + 0.2 * row["knn_basic"]
        else:
            uak = row["unit_area5"]
            uk = row["unit_key"]
            btk = row["bld_tower"]
            bfk = row["bld_flat"]
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
                knn_p = row["knn_basic"]
                if dk in dist_stats.index:
                    dp = area * dist_stats.loc[dk]["ppsf_median"]
                    preds[i] = 0.4 * knn_p + 0.6 * dp
                else:
                    preds[i] = knn_p
    return np.clip(preds, 2000, 500000)


# ── RUN ──
print("\nGenerating...")
base = gen_baseline()

variants = {
    "sub_v9_baseline": base,
    "sub_v9_combo1": gen_combo1(),
    "sub_v9_combo2": gen_combo2(),
    "sub_v9_extreme_only": gen_extreme_only(),
    "sub_v9_top2_extreme": gen_top2_extreme(),
}

print(f"\n{'Name':25s} {'Changed':>8s} {'MeanDiff':>10s} {'MaxDiff':>10s}")
print("-" * 56)
for name, preds in variants.items():
    pd.DataFrame({"id": test["id"].astype(int), "price": preds.astype(int)}).to_csv(f"{name}.csv", index=False)
    diff = np.abs(preds - base)
    changed = (diff > 1).sum()
    print(f"{name:25s} {changed:8d} ${diff.mean():8,.0f} ${diff.max():8,.0f}")

print("\n=== SUBMIT THESE ===")
print("1. sub_v9_extreme_only.csv — Only fix 2-3 most extreme outlier singles")
print("2. sub_v9_combo1.csv       — Extreme fix + improved fallback + enhanced KNN")
print("3. sub_v9_combo2.csv       — Only improved fallback (addr_only chain)")
print("4. sub_v9_top2_extreme.csv — Ultra-conservative: only fix ratio>2.5 outliers")
