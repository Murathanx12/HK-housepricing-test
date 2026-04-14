"""
Focus on the 570 FALLBACK rows (no full_addr match).
These likely contribute most of the RMSE error.
If fallback RMSE is $5000, they contribute $5000^2 * 570 / 8633 = ~$1650 to overall RMSE.
Fixing these is the only way to get under $1,400.
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
    # Also create a relaxed full_addr: address only (no area restriction)
    df["addr_only"] = df["address"].fillna("")
    # Building + area bin
    df["bld_area5"] = df["building"] + "|" + df["area_bin5"].astype(str)
    # Building + floor band
    df["floor_band"] = pd.cut(df["floor"], bins=[0, 8, 15, 25, 100], labels=["L", "M", "H", "VH"])
    df["bld_floor"] = df["building"] + "|" + df["floor_band"].astype(str)

train["ppsf"] = train["price"] / train["area_sqft"]

# ── Identify fallback test rows ──
fa_grp = train.groupby("full_addr")
fa_count = fa_grp["price"].count()

fallback_mask = ~test["full_addr"].isin(fa_count.index)
fallback_test = test[fallback_mask].copy()
matched_test = test[~fallback_mask].copy()
print(f"Matched test rows: {len(matched_test)} ({100*len(matched_test)/len(test):.1f}%)")
print(f"Fallback test rows: {len(fallback_test)} ({100*len(fallback_test)/len(test):.1f}%)")

# ── ANALYZE FALLBACK ROWS ──
print("\n=== FALLBACK ROW ANALYSIS ===")

# What matches DO they have?
addr_only_counts = train.groupby("addr_only")["price"].count()
bld_counts = train.groupby("building")["price"].count()
unit_counts = train.groupby("unit_key")["price"].count()
ua5_counts = train.groupby("unit_area5")["price"].count()
bt_counts = train.groupby("bld_tower")["price"].count()
ba5_counts = train.groupby("bld_area5")["price"].count()

for name, key, counts in [
    ("addr_only (same address, diff area)", "addr_only", addr_only_counts),
    ("unit_key (bld+tower+flat)", "unit_key", unit_counts),
    ("unit_area5 (unit+area)", "unit_area5", ua5_counts),
    ("bld_tower", "bld_tower", bt_counts),
    ("bld_area5 (building+area)", "bld_area5", ba5_counts),
    ("building", "building", bld_counts),
]:
    matched = fallback_test[key].isin(counts.index).sum()
    print(f"  {name:40s}: {matched}/{len(fallback_test)} ({100*matched/len(fallback_test):.1f}%)")

# The fallback rows that DON'T match full_addr but DO match addr_only
# = same apartment, different area. This is a GREAT match!
addr_only_match = fallback_test[fallback_test["addr_only"].isin(addr_only_counts.index)]
print(f"\n  Fallback rows with addr_only match: {len(addr_only_match)}")

# Check: what's the typical area difference?
addr_only_stats = train.groupby("addr_only").agg(
    ppsf_mean=("ppsf", "mean"), ppsf_median=("ppsf", "median"),
    p_mean=("price", "mean"), p_median=("price", "median"),
    area_mean=("area_sqft", "mean"), area_std=("area_sqft", "std"),
    floor_mean=("floor", "mean"),
    count=("price", "count"),
)
addr_only_stats["area_std"] = addr_only_stats["area_std"].fillna(0)

n_same_area = 0
n_diff_area = 0
for _, row in addr_only_match.iterrows():
    ao = row["addr_only"]
    if ao in addr_only_stats.index:
        ao_data = addr_only_stats.loc[ao]
        area_diff = abs(row["area_sqft"] - ao_data["area_mean"])
        if area_diff < 10:
            n_same_area += 1
        else:
            n_diff_area += 1

print(f"    Same area (diff < 10): {n_same_area}")
print(f"    Different area: {n_diff_area}")
print(f"    These are likely the SAME APARTMENT with slightly different reported area")
print(f"    -> Use addr_only PPSF * test_area for these rows!")

# ── BUILD LOOKUP TABLES ──
print("\nBuilding all lookup tables...")

def floor_slope(g):
    if len(g) < 5 or g["floor"].std() < 1: return 0.0
    return np.polyfit(g["floor"], g["ppsf"], 1)[0]

bld_slopes = train.groupby("building").apply(floor_slope, include_groups=False).to_dict()

fa_stats = fa_grp.agg(
    p_mean=("price", "mean"), p_median=("price", "median"),
    count=("price", "count"),
    ppsf_mean=("ppsf", "mean"), ppsf_median=("ppsf", "median"),
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

# Building+area5 stats (NEW — more specific than building alone)
ba5_stats = train.groupby("bld_area5").agg(
    ppsf_mean=("ppsf", "mean"), ppsf_median=("ppsf", "median"),
    floor_mean=("floor", "mean"), count=("price", "count"),
)

# Building+floor band stats (NEW)
bf_band_stats = train.groupby("bld_floor").agg(
    ppsf_mean=("ppsf", "mean"), ppsf_median=("ppsf", "median"),
    count=("price", "count"),
)

# KNN with multiple k values
print("Building KNN models...")
scaler = StandardScaler()
X_knn_tr = scaler.fit_transform(train[["wgs_lat", "wgs_lon", "area_sqft", "floor"]].values)
X_knn_te = scaler.transform(test[["wgs_lat", "wgs_lon", "area_sqft", "floor"]].values)

for k in [5, 7, 10, 15, 20, 30]:
    knn = KNeighborsRegressor(n_neighbors=k, weights="distance", n_jobs=-1)
    knn.fit(X_knn_tr, train["price"].values)
    test[f"knn{k}"] = knn.predict(X_knn_te)

# KNN on PPSF
knn_ppsf = KNeighborsRegressor(n_neighbors=10, weights="distance", n_jobs=-1)
knn_ppsf.fit(X_knn_tr, train["ppsf"].values)
test["knn_ppsf10"] = knn_ppsf.predict(X_knn_te) * test["area_sqft"]

# Geographically weighted KNN: only use neighbors from same district
# This is more specific than regular KNN
print("Building district-specific KNN...")
test["knn_district"] = 0.0
for district in test["district"].unique():
    d_train = train[train["district"] == district]
    d_test_mask = test["district"] == district
    if len(d_train) < 10:
        test.loc[d_test_mask, "knn_district"] = test.loc[d_test_mask, "knn10"]
        continue
    k = min(10, len(d_train))
    d_scaler = StandardScaler()
    X_d_tr = d_scaler.fit_transform(d_train[["wgs_lat", "wgs_lon", "area_sqft", "floor"]].values)
    X_d_te = d_scaler.transform(test.loc[d_test_mask, ["wgs_lat", "wgs_lon", "area_sqft", "floor"]].values)
    d_knn = KNeighborsRegressor(n_neighbors=k, weights="distance", n_jobs=-1)
    d_knn.fit(X_d_tr, d_train["price"].values)
    test.loc[d_test_mask, "knn_district"] = d_knn.predict(X_d_te)


# ══════════════════════════════════════════════════
# IMPROVED FALLBACK CHAIN
# ══════════════════════════════════════════════════
def get_improved_fallback(row, area, floor_val, slope):
    """Enhanced fallback with addr_only and building+area matches."""
    ao = row["addr_only"]
    uak = row["unit_area5"]
    uk = row["unit_key"]
    btk = row["bld_tower"]
    bfk = row["bld_flat"]
    bk = row["building"]
    ba5k = row["bld_area5"]
    bfloork = row["bld_floor"]
    dk = row["district"]

    # NEW: addr_only match = same apartment, slightly different area
    if ao in addr_only_stats.index:
        d = addr_only_stats.loc[ao]
        if d["count"] >= 2:
            fadj = slope * (floor_val - d["floor_mean"]) if d["count"] >= 3 else 0
            return area * (d["ppsf_median"] + fadj), "addr_only"
        elif d["count"] == 1:
            fadj = 0
            return area * (d["ppsf_mean"] + fadj), "addr_only_1"

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

    # NEW: building + area bin (more specific than building alone)
    if ba5k in ba5_stats.index and ba5_stats.loc[ba5k]["count"] >= 3:
        d = ba5_stats.loc[ba5k]
        return area * d["ppsf_median"], "bld_area"

    if bk in bld_stats.index:
        d = bld_stats.loc[bk]
        base = d["ppsf_median"] if d["count"] >= 3 else d["ppsf_mean"]
        fadj = slope * (floor_val - d["floor_mean"]) if d["count"] >= 5 else 0
        return area * (base + fadj), "bld"

    # Last resort: blend KNN + district
    knn_p = row["knn10"]
    dk_knn = row["knn_district"]
    if dk in dist_stats.index:
        dp = area * dist_stats.loc[dk]["ppsf_median"]
        return 0.3 * knn_p + 0.3 * dk_knn + 0.4 * dp, "fallback_blend"
    return 0.5 * knn_p + 0.5 * dk_knn, "fallback_knn"


# ══════════════════════════════════════════════════
# ORIGINAL FALLBACK (for comparison)
# ══════════════════════════════════════════════════
def get_original_fallback(row, area, floor_val, slope):
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
# GENERATE SUBMISSIONS
# ══════════════════════════════════════════════════
print("\nGenerating submissions...")

def make_preds(fallback_func, use_improved=False):
    preds = np.zeros(len(test))
    match_info = []
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
                preds[i] = 0.8 * d["p_mean"] + 0.2 * row["knn10"]
            match_info.append("matched")
        else:
            if use_improved:
                preds[i], mt = fallback_func(row, area, floor_val, slope)
                match_info.append(mt)
            else:
                preds[i] = fallback_func(row, area, floor_val, slope)
                match_info.append("fallback")
    return np.clip(preds, 2000, 500000), match_info


# Baseline
preds_base, _ = make_preds(get_original_fallback)

# Improved fallback
preds_improved, match_types = make_preds(get_improved_fallback, use_improved=True)

# Show fallback match types
from collections import Counter
fb_types = [mt for mt in match_types if mt != "matched"]
print(f"\n=== IMPROVED FALLBACK MATCH TYPES ({len(fb_types)} rows) ===")
for mt, cnt in Counter(fb_types).most_common():
    print(f"  {mt:20s}: {cnt:4d} ({100*cnt/len(fb_types):.1f}%)")

# Diff analysis
diff = np.abs(preds_improved - preds_base)
fb_mask = np.array([mt != "matched" for mt in match_types])
print(f"\nFallback rows changed: {(diff[fb_mask] > 1).sum()}/{fb_mask.sum()}")
print(f"Mean abs diff on fallback rows: ${diff[fb_mask].mean():,.0f}")
print(f"Max abs diff: ${diff[fb_mask].max():,.0f}")

# Save
pd.DataFrame({"id": test["id"].astype(int), "price": preds_base.astype(int)}).to_csv("sub_v5_baseline.csv", index=False)
pd.DataFrame({"id": test["id"].astype(int), "price": preds_improved.astype(int)}).to_csv("sub_v5_improved_fb.csv", index=False)

# Also try: improved fallback + slightly different matched handling
# Blend baseline matched with winsorized/robust for 4+ groups
print("\n--- Additional variants ---")

# V5B: Improved fallback + trim15 for 4+ matches
preds_v5b = np.zeros(len(test))
fa_trimmed15 = fa_grp["price"].apply(lambda x: trim_mean(x, 0.15) if len(x) >= 4 else x.mean())
fa_t15 = fa_trimmed15.to_dict()

for i in range(len(test)):
    row = test.iloc[i]
    area, floor_val = row["area_sqft"], row["floor"]
    fa = row["full_addr"]
    slope = bld_slopes.get(row["building"], 0.0)

    if fa in fa_stats.index:
        d = fa_stats.loc[fa]
        if d["count"] >= 4:
            preds_v5b[i] = fa_t15.get(fa, d["p_trimmed"])
        elif d["count"] >= 2:
            preds_v5b[i] = d["p_median"]
        else:
            preds_v5b[i] = 0.8 * d["p_mean"] + 0.2 * row["knn10"]
    else:
        preds_v5b[i], _ = get_improved_fallback(row, area, floor_val, slope)

preds_v5b = np.clip(preds_v5b, 2000, 500000)
pd.DataFrame({"id": test["id"].astype(int), "price": preds_v5b.astype(int)}).to_csv("sub_v5b_improved_trim15.csv", index=False)

# V5C: Improved fallback + blend baseline+improved 50/50
preds_v5c = 0.5 * preds_base + 0.5 * preds_improved
preds_v5c = np.clip(preds_v5c, 2000, 500000)
pd.DataFrame({"id": test["id"].astype(int), "price": preds_v5c.astype(int)}).to_csv("sub_v5c_blend.csv", index=False)

# V5D: Improved fallback + knn_district blend for single matches
preds_v5d = np.zeros(len(test))
for i in range(len(test)):
    row = test.iloc[i]
    area, floor_val = row["area_sqft"], row["floor"]
    fa = row["full_addr"]
    slope = bld_slopes.get(row["building"], 0.0)

    if fa in fa_stats.index:
        d = fa_stats.loc[fa]
        if d["count"] >= 4:
            preds_v5d[i] = d["p_trimmed"]
        elif d["count"] >= 2:
            preds_v5d[i] = d["p_median"]
        else:
            # Use district KNN instead of regular KNN for single matches
            preds_v5d[i] = 0.8 * d["p_mean"] + 0.2 * row["knn_district"]
    else:
        preds_v5d[i], _ = get_improved_fallback(row, area, floor_val, slope)

preds_v5d = np.clip(preds_v5d, 2000, 500000)
pd.DataFrame({"id": test["id"].astype(int), "price": preds_v5d.astype(int)}).to_csv("sub_v5d_dknn_single.csv", index=False)

print("\n=== SUMMARY ===")
for name, p in [("baseline", preds_base), ("improved_fb", preds_improved),
                ("v5b_trim15", preds_v5b), ("v5c_blend", preds_v5c), ("v5d_dknn", preds_v5d)]:
    d = np.abs(p - preds_base)
    changed = (d > 1).sum()
    print(f"  {name:15s}: mean=${p.mean():,.0f}, {changed} rows differ from base, mean_diff=${d.mean():,.0f}")

print("\n=== SUBMIT THESE (priority order) ===")
print("1. sub_v5_improved_fb.csv  — addr_only + bld_area fallback (HIGHEST PRIORITY)")
print("2. sub_v5d_dknn_single.csv — improved fb + district KNN for singles")
print("3. sub_v5b_improved_trim15.csv — improved fb + trim15")
print("4. sub_v5c_blend.csv       — 50/50 hedge")
