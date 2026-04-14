"""
Generate improved submissions based on LOO analysis.
Multiple variants targeting different error sources.
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

# Full address stats
fa_grp = train.groupby("full_addr")
fa_stats = fa_grp.agg(
    p_mean=("price", "mean"), p_median=("price", "median"),
    p_std=("price", "std"), count=("price", "count"),
    ppsf_mean=("ppsf", "mean"), ppsf_median=("ppsf", "median"),
    floor_mean=("floor", "mean"), floor_std=("floor", "std"),
)
fa_stats["p_std"] = fa_stats["p_std"].fillna(0)
fa_stats["floor_std"] = fa_stats["floor_std"].fillna(0)

# Trimmed mean
fa_trimmed = fa_grp["price"].apply(lambda x: trim_mean(x, 0.1) if len(x) >= 4 else x.mean())
fa_trimmed.name = "p_trimmed"
fa_stats = fa_stats.join(fa_trimmed)

# Geometric mean
fa_geomean = fa_grp["price"].apply(lambda x: np.exp(np.log(x).mean()))
fa_geomean.name = "p_geomean"
fa_stats = fa_stats.join(fa_geomean)

# Unit+area5
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

# ── TEST SET MATCH DISTRIBUTION ──
print("\n=== TEST SET MATCH DISTRIBUTION ===")
match_counts = []
for i in range(len(test)):
    fa = test.iloc[i]["full_addr"]
    if fa in fa_stats.index:
        match_counts.append(int(fa_stats.loc[fa]["count"]))
    else:
        match_counts.append(0)

test["fa_match_count"] = match_counts
for label, lo, hi in [("0 (fallback)", 0, 0), ("1", 1, 1), ("2-3", 2, 3), ("4+", 4, 9999)]:
    n = ((test["fa_match_count"] >= lo) & (test["fa_match_count"] <= hi)).sum()
    print(f"  {label:15s}: {n:5d} ({100*n/len(test):.1f}%)")


def get_fallback_pred(row, area, floor_val, slope):
    """Hierarchical fallback for unmatched rows."""
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


# ── VARIANT A: Original winner (baseline) ──
def variant_baseline():
    """Exact reproduction of $1,450 winner (variant4 from hardcode_v3)."""
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


# ── VARIANT B: More KNN for single matches (alpha=0.5) ──
def variant_more_knn():
    """Alpha=0.5 for single matches — LOO showed this is best."""
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
                preds[i] = 0.5 * d["p_mean"] + 0.5 * row["knn10"]
        else:
            preds[i] = get_fallback_pred(row, area, floor_val, slope)
    return np.clip(preds, 2000, 500000)


# ── VARIANT C: Blend trimmed+median for 4+, alpha=0.6 for singles ──
def variant_blend_trimmed():
    """Blend trimmed+median 50/50 for 4+, alpha=0.6 for singles."""
    preds = np.zeros(len(test))
    for i in range(len(test)):
        row = test.iloc[i]
        area, floor_val = row["area_sqft"], row["floor"]
        fa = row["full_addr"]
        slope = bld_slopes.get(row["building"], 0.0)

        if fa in fa_stats.index:
            d = fa_stats.loc[fa]
            if d["count"] >= 4:
                preds[i] = 0.5 * d["p_trimmed"] + 0.5 * d["p_median"]
            elif d["count"] >= 2:
                preds[i] = d["p_median"]
            else:
                preds[i] = 0.6 * d["p_mean"] + 0.4 * row["knn10"]
        else:
            preds[i] = get_fallback_pred(row, area, floor_val, slope)
    return np.clip(preds, 2000, 500000)


# ── VARIANT D: Geo mean everywhere + floor adj ──
def variant_geo_floor():
    """Geo mean for all multi-match, floor adj, alpha=0.5 singles."""
    preds = np.zeros(len(test))
    for i in range(len(test)):
        row = test.iloc[i]
        area, floor_val = row["area_sqft"], row["floor"]
        fa = row["full_addr"]
        slope = bld_slopes.get(row["building"], 0.0)

        if fa in fa_stats.index:
            d = fa_stats.loc[fa]
            if d["count"] >= 2:
                base = d["p_geomean"]
                # Floor adjustment
                if d["floor_std"] > 0.5 and d["count"] >= 3:
                    floor_diff = floor_val - d["floor_mean"]
                    base += slope * floor_diff * area * 0.5  # damped
                preds[i] = base
            else:
                preds[i] = 0.5 * d["p_mean"] + 0.5 * row["knn10"]
        else:
            preds[i] = get_fallback_pred(row, area, floor_val, slope)
    return np.clip(preds, 2000, 500000)


# ── VARIANT E: Combined best — outlier capping on fallback ──
def variant_combined():
    """Best of everything: trimmed+median blend, alpha=0.5, capped fallback."""
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
                preds[i] = 0.5 * d["p_trimmed"] + 0.5 * d["p_median"]
            elif d["count"] >= 2:
                preds[i] = d["p_median"]
            else:
                preds[i] = 0.5 * d["p_mean"] + 0.5 * row["knn10"]
        else:
            preds[i] = get_fallback_pred(row, area, floor_val, slope)

        # Cap extreme predictions relative to building stats
        if bk in bld_stats.index:
            bld_pred = area * bld_stats.loc[bk]["ppsf_median"]
            if preds[i] > 2.5 * bld_pred:
                preds[i] = 2.5 * bld_pred
            elif preds[i] < bld_pred / 2.5:
                preds[i] = bld_pred / 2.5

    return np.clip(preds, 2000, 500000)


# ── VARIANT F: Blend of baseline + variant_more_knn ──
def variant_blend_old_new():
    """50/50 blend of baseline ($1,450) and more_knn variant."""
    v_base = variant_baseline()
    v_knn = variant_more_knn()
    return np.clip(0.5 * v_base + 0.5 * v_knn, 2000, 500000)


# ── VARIANT G: Adaptive alpha based on building variance ──
def variant_adaptive_alpha():
    """For single matches, alpha depends on how noisy the building is."""
    # Precompute building price std
    bld_cv = train.groupby("building")["ppsf"].agg(lambda x: x.std()/x.mean() if len(x) >= 3 else 0.5)

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
                # High-variance building -> trust KNN more
                cv = bld_cv.get(bk, 0.5)
                alpha = max(0.3, min(0.95, 1.0 - cv))  # high cv -> low alpha
                preds[i] = alpha * d["p_mean"] + (1 - alpha) * row["knn10"]
        else:
            preds[i] = get_fallback_pred(row, area, floor_val, slope)
    return np.clip(preds, 2000, 500000)


# ── RUN ALL ──
print("\nGenerating submissions...")
variants = [
    ("sub_baseline", variant_baseline),
    ("sub_more_knn", variant_more_knn),
    ("sub_blend_trim", variant_blend_trimmed),
    ("sub_geo_floor", variant_geo_floor),
    ("sub_combined", variant_combined),
    ("sub_blend_old_new", variant_blend_old_new),
    ("sub_adaptive", variant_adaptive_alpha),
]

results = {}
for name, func in variants:
    preds = func()
    results[name] = preds
    pd.DataFrame({"id": test["id"].astype(int), "price": preds.astype(int)}).to_csv(f"{name}.csv", index=False)
    print(f"  {name}: mean=${preds.mean():,.0f}, median=${np.median(preds):,.0f}, min=${preds.min():,.0f}, max=${preds.max():,.0f}")

# Show differences from baseline
print("\n=== DIFFERENCES FROM BASELINE ===")
base = results["sub_baseline"]
for name, preds in results.items():
    if name == "sub_baseline": continue
    diff = preds - base
    abs_diff = np.abs(diff)
    changed = (abs_diff > 1).sum()
    print(f"  {name}: {changed} rows changed, mean_abs_diff=${abs_diff.mean():,.0f}, max=${abs_diff.max():,.0f}")
    # Show distribution of changes by match count
    for label, lo, hi in [("1-match", 1, 1), ("2-3", 2, 3), ("4+", 4, 9999), ("fallback", 0, 0)]:
        mask = (test["fa_match_count"] >= lo) & (test["fa_match_count"] <= hi)
        if mask.sum() > 0:
            md = abs_diff[mask].mean()
            if md > 0.5:
                print(f"    {label}: mean_abs_diff=${md:,.0f}")

print("\n=== FILES ===")
for name, _ in variants:
    print(f"  {name}.csv")
print("\nSubmit these and compare scores!")
