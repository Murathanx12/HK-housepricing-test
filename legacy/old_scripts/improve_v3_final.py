"""
Final improved submissions — focused on the 31.3% single-match rows.
Key insight: for single matches, validate against unit-level PPSF.
If the single price is an outlier relative to its unit, shrink toward unit prediction.
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
    ppsf_mean=("ppsf", "mean"), ppsf_median=("ppsf", "median"),
    floor_mean=("floor", "mean"), floor_std=("floor", "std"),
)
fa_stats["floor_std"] = fa_stats["floor_std"].fillna(0)

fa_trimmed = fa_grp["price"].apply(lambda x: trim_mean(x, 0.1) if len(x) >= 4 else x.mean())
fa_trimmed.name = "p_trimmed"
fa_stats = fa_stats.join(fa_trimmed)

# Unit+area5 stats
ua_stats = train.groupby("unit_area5").agg(
    ppsf_mean=("ppsf", "mean"), ppsf_median=("ppsf", "median"),
    p_mean=("price", "mean"), p_median=("price", "median"),
    floor_mean=("floor", "mean"), count=("price", "count"),
)

# Unit stats (for validating single matches)
unit_stats = train.groupby("unit_key").agg(
    ppsf_mean=("ppsf", "mean"), ppsf_median=("ppsf", "median"),
    ppsf_std=("ppsf", "std"),
    p_mean=("price", "mean"), p_median=("price", "median"),
    floor_mean=("floor", "mean"), area_mean=("area_sqft", "mean"),
    count=("price", "count"),
)
unit_stats["ppsf_std"] = unit_stats["ppsf_std"].fillna(0)

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
    ppsf_std=("ppsf", "std"),
    floor_mean=("floor", "mean"), count=("price", "count"),
)
bld_stats["ppsf_std"] = bld_stats["ppsf_std"].fillna(0)

# District
dist_stats = train.groupby("district").agg(
    ppsf_mean=("ppsf", "mean"), ppsf_median=("ppsf", "median"),
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


def get_unit_pred(row, area, floor_val, slope):
    """Get the best unit-level prediction for this row."""
    uak = row["unit_area5"]
    uk = row["unit_key"]
    btk = row["bld_tower"]
    bk = row["building"]

    if uak in ua_stats.index and ua_stats.loc[uak]["count"] >= 2:
        d = ua_stats.loc[uak]
        fadj = slope * (floor_val - d["floor_mean"])
        return area * (d["ppsf_median"] + fadj), "ua"

    if uk in unit_stats.index and unit_stats.loc[uk]["count"] >= 3:
        d = unit_stats.loc[uk]
        fadj = slope * (floor_val - d["floor_mean"])
        return area * (d["ppsf_median"] + fadj), "unit"

    if btk in bt_stats.index and bt_stats.loc[btk]["count"] >= 5:
        d = bt_stats.loc[btk]
        fadj = slope * (floor_val - d["floor_mean"])
        return area * (d["ppsf_median"] + fadj), "bt"

    if bk in bld_stats.index and bld_stats.loc[bk]["count"] >= 5:
        d = bld_stats.loc[bk]
        fadj = slope * (floor_val - d["floor_mean"])
        return area * (d["ppsf_median"] + fadj), "bld"

    return row["knn10"], "knn"


# ══════════════════════════════════════════════════
# VARIANT 1: Smart single-match handling
# For count=1, check if the single price is consistent with unit-level stats.
# If it's an outlier, shrink more toward unit prediction.
# ══════════════════════════════════════════════════
def variant_smart_single():
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
                # Single match — smart blending
                single_price = d["p_mean"]  # = the one observation
                unit_pred, src = get_unit_pred(row, area, floor_val, slope)

                # Check if single price is "reasonable" vs unit prediction
                ratio = single_price / unit_pred if unit_pred > 0 else 1.0
                if 0.8 <= ratio <= 1.2:
                    # Consistent — trust the direct observation more
                    preds[i] = 0.85 * single_price + 0.15 * unit_pred
                elif 0.6 <= ratio <= 1.5:
                    # Somewhat off — blend more
                    preds[i] = 0.6 * single_price + 0.4 * unit_pred
                else:
                    # Outlier — trust unit prediction more
                    preds[i] = 0.3 * single_price + 0.7 * unit_pred
        else:
            preds[i] = get_fallback_pred(row, area, floor_val, slope)
    return np.clip(preds, 2000, 500000)


# ══════════════════════════════════════════════════
# VARIANT 2: Unit-validated everywhere
# For ALL matches, compare FA prediction with unit prediction.
# If they diverge wildly, blend toward unit.
# ══════════════════════════════════════════════════
def variant_validated():
    preds = np.zeros(len(test))
    for i in range(len(test)):
        row = test.iloc[i]
        area, floor_val = row["area_sqft"], row["floor"]
        fa = row["full_addr"]
        slope = bld_slopes.get(row["building"], 0.0)

        if fa in fa_stats.index:
            d = fa_stats.loc[fa]

            # Base prediction from FA
            if d["count"] >= 4:
                fa_pred = d["p_trimmed"]
            elif d["count"] >= 2:
                fa_pred = d["p_median"]
            else:
                fa_pred = d["p_mean"]

            # Unit-level validation
            unit_pred, src = get_unit_pred(row, area, floor_val, slope)
            ratio = fa_pred / unit_pred if unit_pred > 0 else 1.0

            if d["count"] >= 3:
                # High confidence in FA — only correct extreme outliers
                if ratio > 1.5 or ratio < 0.67:
                    preds[i] = 0.7 * fa_pred + 0.3 * unit_pred
                else:
                    preds[i] = fa_pred
            elif d["count"] == 2:
                if ratio > 1.3 or ratio < 0.77:
                    preds[i] = 0.6 * fa_pred + 0.4 * unit_pred
                else:
                    preds[i] = fa_pred
            else:
                # Single match — more aggressive blending
                if 0.8 <= ratio <= 1.2:
                    preds[i] = 0.85 * fa_pred + 0.15 * unit_pred
                elif 0.6 <= ratio <= 1.5:
                    preds[i] = 0.5 * fa_pred + 0.5 * unit_pred
                else:
                    preds[i] = 0.3 * fa_pred + 0.7 * unit_pred
        else:
            preds[i] = get_fallback_pred(row, area, floor_val, slope)
    return np.clip(preds, 2000, 500000)


# ══════════════════════════════════════════════════
# VARIANT 3: Original baseline (for comparison)
# ══════════════════════════════════════════════════
def variant_baseline():
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
                preds[i] = 0.8 * d["p_mean"] + 0.2 * row["knn10"]
        else:
            preds[i] = get_fallback_pred(row, area, floor_val, slope)
    return np.clip(preds, 2000, 500000)


# ══════════════════════════════════════════════════
# VARIANT 4: Best simple change — alpha=0.7 for singles
# ══════════════════════════════════════════════════
def variant_alpha07():
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
                preds[i] = 0.7 * d["p_mean"] + 0.3 * row["knn10"]
        else:
            preds[i] = get_fallback_pred(row, area, floor_val, slope)
    return np.clip(preds, 2000, 500000)


# ══════════════════════════════════════════════════
# VARIANT 5: Blend baseline + smart_single (hedge)
# ══════════════════════════════════════════════════
def variant_blend():
    v_base = variant_baseline()
    v_smart = variant_smart_single()
    return np.clip(0.5 * v_base + 0.5 * v_smart, 2000, 500000)


# ── RUN ──
print("\nGenerating...")
variants = [
    ("sub_baseline_check", variant_baseline),
    ("sub_smart_single", variant_smart_single),
    ("sub_validated", variant_validated),
    ("sub_alpha07", variant_alpha07),
    ("sub_blend_smart", variant_blend),
]

results = {}
for name, func in variants:
    preds = func()
    results[name] = preds
    pd.DataFrame({"id": test["id"].astype(int), "price": preds.astype(int)}).to_csv(f"{name}.csv", index=False)
    print(f"  {name}: mean=${preds.mean():,.0f}")

# Differences
print("\n=== DIFFERENCES FROM BASELINE ===")
base = results["sub_baseline_check"]
for name, preds in results.items():
    if name == "sub_baseline_check": continue
    diff = np.abs(preds - base)
    changed = (diff > 10).sum()
    mean_diff = diff[diff > 10].mean() if changed > 0 else 0
    print(f"  {name}: {changed} rows changed by >${10}, avg change=${mean_diff:,.0f}")

print("\n=== RECOMMENDED SUBMISSION ORDER (5/day limit) ===")
print("1. sub_smart_single.csv    — Outlier-aware single-match blending")
print("2. sub_validated.csv       — Unit-validated ALL predictions")
print("3. sub_alpha07.csv         — Simple: 0.7/0.3 blend for singles")
print("4. sub_blend_smart.csv     — 50/50 hedge of baseline + smart_single")
print("5. sub_baseline_check.csv  — Verify baseline reproduces $1,450")
