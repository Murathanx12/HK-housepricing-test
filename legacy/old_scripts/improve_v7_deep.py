"""
Deep analysis: find the actual error sources.
Check unused features: Public_Housing, Phase, Block.
Find the specific rows where predictions diverge most.
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
    df["full_addr"] = df["address"].fillna("") + "|" + df["area_sqft"].astype(str)
    df["area_bin5"] = (df["area_sqft"] / 5).round() * 5
    df["unit_area5"] = df["unit_key"] + "|" + df["area_bin5"].astype(str)
    df["bld_tower"] = df["building"] + "|T" + df["Tower"].fillna("X").astype(str)
    df["bld_flat"] = df["building"] + "|F" + df["Flat"].fillna("X")

train["ppsf"] = train["price"] / train["area_sqft"]

# ── PUBLIC HOUSING ANALYSIS ──
print("\n=== PUBLIC HOUSING ===")
ph_train = train["Public_Housing"].value_counts()
ph_test = test["Public_Housing"].value_counts()
print(f"Training: {ph_train.to_dict()}")
print(f"Test: {ph_test.to_dict()}")

# PPSF by public housing status
for ph in [True, False]:
    subset = train[train["Public_Housing"] == ph]
    print(f"  Public_Housing={ph}: n={len(subset)}, "
          f"mean_ppsf=${subset['ppsf'].mean():.1f}, median_ppsf=${subset['ppsf'].median():.1f}, "
          f"mean_price=${subset['price'].mean():,.0f}")

# Are there test rows where Public_Housing status differs from training match?
print("\n=== PHASE AND BLOCK ANALYSIS ===")
print(f"Train phases: {train['Phase'].nunique()} unique, {train['Phase'].isna().sum()} null")
print(f"Train blocks: {train['Block'].nunique()} unique, {train['Block'].isna().sum()} null")
print(f"Test phases: {test['Phase'].nunique()} unique, {test['Phase'].isna().sum()} null")
print(f"Test blocks: {test['Block'].nunique()} unique, {test['Block'].isna().sum()} null")

# Does Phase+Block add info beyond building+tower?
# Check: same building+tower but different phase -> different ppsf?
train["btp"] = train["building"] + "|T" + train["Tower"].fillna("X").astype(str) + "|P" + train["Phase"].fillna("X").astype(str)
test["btp"] = test["building"] + "|T" + test["Tower"].fillna("X").astype(str) + "|P" + test["Phase"].fillna("X").astype(str)

# ── WHERE ARE RMSE CONTRIBUTIONS COMING FROM? ──
# Simulate by computing PPSF-based prediction for every test row
# and seeing how much it differs from direct price prediction
print("\n=== PREDICTION VARIANCE ANALYSIS ===")

fa_grp = train.groupby("full_addr")
fa_stats = fa_grp.agg(
    p_mean=("price", "mean"), p_median=("price", "median"),
    count=("price", "count"),
    ppsf_mean=("ppsf", "mean"), ppsf_median=("ppsf", "median"),
    p_std=("price", "std"),
    floor_mean=("floor", "mean"),
)
fa_stats["p_std"] = fa_stats["p_std"].fillna(0)
fa_trimmed = fa_grp["price"].apply(lambda x: trim_mean(x, 0.1) if len(x) >= 4 else x.mean())
fa_stats = fa_stats.join(fa_trimmed.rename("p_trimmed"))

# For each test row, compute multiple prediction methods
print("Computing multi-method predictions...")
preds_methods = {}

# Method 1: baseline (trimmed mean / median / 0.8+0.2 KNN)
def floor_slope(g):
    if len(g) < 5 or g["floor"].std() < 1: return 0.0
    return np.polyfit(g["floor"], g["ppsf"], 1)[0]

bld_slopes = train.groupby("building").apply(floor_slope, include_groups=False).to_dict()

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

# Also with Phase+Block
btp_stats = train.groupby("btp").agg(
    ppsf_mean=("ppsf", "mean"), ppsf_median=("ppsf", "median"),
    floor_mean=("floor", "mean"), count=("price", "count"),
)

scaler = StandardScaler()
X_knn_tr = scaler.fit_transform(train[["wgs_lat", "wgs_lon", "area_sqft", "floor"]].values)
X_knn_te = scaler.transform(test[["wgs_lat", "wgs_lon", "area_sqft", "floor"]].values)
knn10 = KNeighborsRegressor(n_neighbors=10, weights="distance", n_jobs=-1)
knn10.fit(X_knn_tr, train["price"].values)
test["knn10"] = knn10.predict(X_knn_te)

# Separate KNN for public/private housing
print("Building public housing specific KNN...")
for ph in [True, False]:
    ph_train = train[train["Public_Housing"] == ph]
    if len(ph_train) < 20: continue
    ph_scaler = StandardScaler()
    X_ph_tr = ph_scaler.fit_transform(ph_train[["wgs_lat", "wgs_lon", "area_sqft", "floor"]].values)
    ph_test_mask = test["Public_Housing"] == ph
    X_ph_te = ph_scaler.transform(test.loc[ph_test_mask, ["wgs_lat", "wgs_lon", "area_sqft", "floor"]].values)
    ph_knn = KNeighborsRegressor(n_neighbors=10, weights="distance", n_jobs=-1)
    ph_knn.fit(X_ph_tr, ph_train["price"].values)
    test.loc[ph_test_mask, f"knn_ph"] = ph_knn.predict(X_ph_te)

# Fill missing
test["knn_ph"] = test["knn_ph"].fillna(test["knn10"])

# ── Look at LARGEST test rows (high price = high error potential) ──
print("\n=== TEST ROW ANALYSIS ===")
# Rows with very high or very low areas
print(f"Test area: min={test['area_sqft'].min()}, max={test['area_sqft'].max()}, "
      f"mean={test['area_sqft'].mean():.0f}")
print(f"Test floor: min={test['floor'].min()}, max={test['floor'].max()}")

# Show how baseline prediction spread looks
def get_baseline_pred(row, area, floor_val, slope):
    fa = row["full_addr"]
    if fa in fa_stats.index:
        d = fa_stats.loc[fa]
        if d["count"] >= 4: return d["p_trimmed"], "trimmed"
        elif d["count"] >= 2: return d["p_median"], "median"
        else: return 0.8 * d["p_mean"] + 0.2 * row["knn10"], "single"
    # Fallback
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
        return area * (base + fadj), "fb_ua"
    if uk in unit_stats.index:
        d = unit_stats.loc[uk]
        base = d["ppsf_median"] if d["count"] >= 3 else d["ppsf_mean"]
        fadj = slope * (floor_val - d["floor_mean"])
        return area * (base + fadj), "fb_unit"
    if btk in bt_stats.index and bt_stats.loc[btk]["count"] >= 3:
        d = bt_stats.loc[btk]
        fadj = slope * (floor_val - d["floor_mean"])
        return area * (d["ppsf_median"] + fadj), "fb_bt"
    if bfk in bf_stats.index and bf_stats.loc[bfk]["count"] >= 3:
        d = bf_stats.loc[bfk]
        fadj = slope * (floor_val - d["floor_mean"])
        return area * (d["ppsf_median"] + fadj), "fb_bf"
    if bk in bld_stats.index:
        d = bld_stats.loc[bk]
        base = d["ppsf_median"] if d["count"] >= 3 else d["ppsf_mean"]
        fadj = slope * (floor_val - d["floor_mean"]) if d["count"] >= 5 else 0
        return area * (base + fadj), "fb_bld"
    knn_p = row["knn10"]
    if dk in dist_stats.index:
        dp = area * dist_stats.loc[dk]["ppsf_median"]
        return 0.4 * knn_p + 0.6 * dp, "fb_dist"
    return knn_p, "fb_knn"

# Compute baseline and identify suspicious predictions
test_preds = []
test_types = []
for i in range(len(test)):
    row = test.iloc[i]
    area = row["area_sqft"]
    floor_val = row["floor"]
    slope = bld_slopes.get(row["building"], 0.0)
    p, t = get_baseline_pred(row, area, floor_val, slope)
    test_preds.append(p)
    test_types.append(t)

test["baseline_pred"] = np.clip(test_preds, 2000, 500000)
test["pred_type"] = test_types

# Identify potentially problematic rows
# 1. Single-match rows where single price differs greatly from building median
print("\n=== SUSPICIOUS SINGLE-MATCH ROWS ===")
single_mask = test["pred_type"] == "single"
single_test = test[single_mask].copy()
print(f"Single-match test rows: {len(single_test)}")

# For each, compare single price with building ppsf * area
suspicious = 0
for i, (_, row) in enumerate(single_test.iterrows()):
    fa = row["full_addr"]
    bk = row["building"]
    area = row["area_sqft"]

    if fa in fa_stats.index and bk in bld_stats.index:
        single_price = fa_stats.loc[fa]["p_mean"]
        bld_ppsf = bld_stats.loc[bk]["ppsf_median"]
        bld_pred = area * bld_ppsf
        ratio = single_price / bld_pred if bld_pred > 0 else 1.0
        if ratio > 1.5 or ratio < 0.67:
            suspicious += 1
            if suspicious <= 10:
                print(f"  id={int(row['id'])}: single=${single_price:,.0f} vs bld=${bld_pred:,.0f} "
                      f"ratio={ratio:.2f} bld={bk[:30]} area={area}")

print(f"\nTotal suspicious (ratio > 1.5 or < 0.67): {suspicious}/{len(single_test)}")

# 2. Fallback rows: compare different prediction methods
print("\n=== FALLBACK ROW PREDICTIONS ===")
fb_mask = test["pred_type"].str.startswith("fb_")
fb_test = test[fb_mask]
print(f"Fallback rows: {len(fb_test)}")
print(f"  Public housing: {fb_test['Public_Housing'].sum()}")
print(f"  Mean baseline pred: ${fb_test['baseline_pred'].mean():,.0f}")

# Check if public housing KNN differs significantly
ph_diff = np.abs(test.loc[fb_mask, "knn_ph"] - test.loc[fb_mask, "knn10"])
print(f"  KNN vs PH-KNN mean diff: ${ph_diff.mean():,.0f}")

# ── Try Phase/Block enhanced matching ──
print("\n=== PHASE/BLOCK ENHANCED MATCHING ===")
# For fallback rows that have Phase/Block, can we match better?
fb_with_btp = fb_test[fb_test["btp"].isin(btp_stats.index)]
print(f"Fallback rows matching btp (building+tower+phase): {len(fb_with_btp)}/{len(fb_test)}")

# ── GENERATE FINAL OPTIMIZED SUBMISSIONS ──
print("\n=== GENERATING FINAL SUBMISSIONS ===")

# V1: Use Public Housing specific KNN for fallback
def variant_ph_knn():
    preds = np.zeros(len(test))
    for i in range(len(test)):
        row = test.iloc[i]
        area = row["area_sqft"]
        floor_val = row["floor"]
        fa = row["full_addr"]
        slope = bld_slopes.get(row["building"], 0.0)
        if fa in fa_stats.index:
            d = fa_stats.loc[fa]
            if d["count"] >= 4:
                preds[i] = d["p_trimmed"]
            elif d["count"] >= 2:
                preds[i] = d["p_median"]
            else:
                # Use PH-specific KNN instead of general KNN
                preds[i] = 0.8 * d["p_mean"] + 0.2 * row["knn_ph"]
        else:
            # Enhanced fallback: use PH-KNN
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
                # Use PH-KNN + district
                knn_p = row["knn_ph"]
                if dk in dist_stats.index:
                    dp = area * dist_stats.loc[dk]["ppsf_median"]
                    preds[i] = 0.4 * knn_p + 0.6 * dp
                else:
                    preds[i] = knn_p
    return np.clip(preds, 2000, 500000)


# V2: Extreme outlier correction for single matches only
def variant_extreme_correct():
    """Only correct single-match prices that are extreme outliers (>2x building pred)."""
    preds = np.zeros(len(test))
    corrections = 0
    for i in range(len(test)):
        row = test.iloc[i]
        area = row["area_sqft"]
        floor_val = row["floor"]
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
                # Check if extreme outlier
                if bk in bld_stats.index:
                    bld_pred = area * bld_stats.loc[bk]["ppsf_median"]
                    ratio = direct / bld_pred if bld_pred > 0 else 1.0
                    if ratio > 2.0 or ratio < 0.5:
                        # Extreme outlier — shrink heavily toward building pred
                        preds[i] = 0.3 * direct + 0.7 * bld_pred
                        corrections += 1
                    elif ratio > 1.5 or ratio < 0.67:
                        # Moderate outlier
                        preds[i] = 0.6 * direct + 0.4 * bld_pred
                        corrections += 1
                    else:
                        preds[i] = 0.8 * direct + 0.2 * row["knn10"]
                else:
                    preds[i] = 0.8 * direct + 0.2 * row["knn10"]
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
                knn_p = row["knn10"]
                if dk in dist_stats.index:
                    dp = area * dist_stats.loc[dk]["ppsf_median"]
                    preds[i] = 0.4 * knn_p + 0.6 * dp
                else:
                    preds[i] = knn_p
    print(f"  Extreme corrections: {corrections}")
    return np.clip(preds, 2000, 500000)


# V3: Use unit_key PPSF (not KNN) as blend partner for single matches
def variant_unit_blend():
    """For single matches, blend with unit_key ppsf * area instead of KNN."""
    preds = np.zeros(len(test))
    for i in range(len(test)):
        row = test.iloc[i]
        area = row["area_sqft"]
        floor_val = row["floor"]
        fa = row["full_addr"]
        slope = bld_slopes.get(row["building"], 0.0)
        uk = row["unit_key"]

        if fa in fa_stats.index:
            d = fa_stats.loc[fa]
            if d["count"] >= 4:
                preds[i] = d["p_trimmed"]
            elif d["count"] >= 2:
                preds[i] = d["p_median"]
            else:
                direct = d["p_mean"]
                # Use unit_key ppsf as partner (more specific than KNN)
                if uk in unit_stats.index and unit_stats.loc[uk]["count"] >= 5:
                    ud = unit_stats.loc[uk]
                    unit_pred = area * (ud["ppsf_median"] + slope * (floor_val - ud["floor_mean"]))
                    preds[i] = 0.8 * direct + 0.2 * unit_pred
                else:
                    preds[i] = 0.8 * direct + 0.2 * row["knn10"]
        else:
            uak = row["unit_area5"]
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
                knn_p = row["knn10"]
                if dk in dist_stats.index:
                    dp = area * dist_stats.loc[dk]["ppsf_median"]
                    preds[i] = 0.4 * knn_p + 0.6 * dp
                else:
                    preds[i] = knn_p
    return np.clip(preds, 2000, 500000)


# V4: Blend baseline with extreme_correct (hedge)
def variant_blend_extreme():
    v_base = variant_ph_knn()  # use PH as base
    v_extreme = variant_extreme_correct()
    return np.clip(0.5 * v_base + 0.5 * v_extreme, 2000, 500000)


# Run
print("\nGenerating...")
base_preds = np.clip(test["baseline_pred"].values, 2000, 500000)

for name, func in [("sub_v7_ph_knn", variant_ph_knn),
                    ("sub_v7_extreme", variant_extreme_correct),
                    ("sub_v7_unit_blend", variant_unit_blend),
                    ("sub_v7_blend_ext", variant_blend_extreme)]:
    preds = func()
    pd.DataFrame({"id": test["id"].astype(int), "price": preds.astype(int)}).to_csv(f"{name}.csv", index=False)
    diff = np.abs(preds - base_preds)
    changed = (diff > 1).sum()
    print(f"  {name}: {changed} rows differ, mean_diff=${diff.mean():,.0f}, max=${diff.max():,.0f}")

print("\n=== SUBMIT PRIORITY ===")
print("1. sub_v7_extreme.csv    — Only fix extreme single-match outliers")
print("2. sub_v7_ph_knn.csv     — Public housing specific KNN")
print("3. sub_v7_unit_blend.csv — Unit PPSF blend for singles")
