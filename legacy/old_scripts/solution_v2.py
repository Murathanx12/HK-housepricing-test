"""
Solution v2: Build on $1,435 success.
Test additional corrections beyond single-match outliers:
  1. Optimized threshold/alpha (from LOO)
  2. Unit-level correction target (more specific than building)
  3. Also correct high-variance 2-3 match groups
  4. Multiple submissions with LOO ranking
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

def get_building(addr):
    if pd.isna(addr): return "UNKNOWN"
    return addr.split(",")[0].strip()

for df in [train, test]:
    df["building"] = df["address"].apply(get_building)
    df["floor"] = pd.to_numeric(df["floor"], errors="coerce").fillna(10)
    df["unit_key"] = (df["building"] + "|" +
                      df["Tower"].fillna("X").astype(str) + "|" +
                      df["Flat"].fillna("X"))
    df["bld_tower"] = df["building"] + "|T" + df["Tower"].fillna("X").astype(str)
    df["bld_flat"] = df["building"] + "|F" + df["Flat"].fillna("X")
    df["full_addr"] = df["address"].fillna("") + "|" + df["area_sqft"].astype(str)
    df["area_bin5"] = (df["area_sqft"] / 5).round() * 5
    df["unit_area5"] = df["unit_key"] + "|" + df["area_bin5"].astype(str)

train["ppsf"] = train["price"] / train["area_sqft"]

# ── LOOKUPS ──
def floor_slope(g):
    if len(g) < 5 or g["floor"].std() < 1: return 0.0
    return np.polyfit(g["floor"], g["ppsf"], 1)[0]

bld_slopes = train.groupby("building").apply(floor_slope, include_groups=False).to_dict()

fa_grp = train.groupby("full_addr")
fa_stats = fa_grp.agg(
    p_mean=("price", "mean"), p_median=("price", "median"),
    p_std=("price", "std"), count=("price", "count"),
    ppsf_median=("ppsf", "median"), floor_mean=("floor", "mean"),
)
fa_stats["p_std"] = fa_stats["p_std"].fillna(0)
fa_stats["cv"] = np.where(fa_stats["p_mean"] > 0, fa_stats["p_std"] / fa_stats["p_mean"], 0)
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
dist_stats = train.groupby("district").agg(ppsf_median=("ppsf", "median"))

scaler = StandardScaler()
X_tr = scaler.fit_transform(train[["wgs_lat", "wgs_lon", "area_sqft", "floor"]].values)
X_te = scaler.transform(test[["wgs_lat", "wgs_lon", "area_sqft", "floor"]].values)
knn = KNeighborsRegressor(n_neighbors=10, weights="distance", n_jobs=-1)
knn.fit(X_tr, train["price"].values)
test["knn10"] = knn.predict(X_te)


def fallback_pred(row, area, floor_val, slope):
    uak, uk, btk, bfk, bk, dk = (row["unit_area5"], row["unit_key"],
        row["bld_tower"], row["bld_flat"], row["building"], row["district"])
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


def get_unit_pred(row, area, floor_val, slope):
    """Most specific unit-level prediction available."""
    uak = row["unit_area5"]
    if uak in ua_stats.index and ua_stats.loc[uak]["count"] >= 2:
        d = ua_stats.loc[uak]
        return area * (d["ppsf_median"] + slope * (floor_val - d["floor_mean"]))
    uk = row["unit_key"]
    if uk in unit_stats.index and unit_stats.loc[uk]["count"] >= 2:
        d = unit_stats.loc[uk]
        return area * (d["ppsf_median"] + slope * (floor_val - d["floor_mean"]))
    btk = row["bld_tower"]
    if btk in bt_stats.index and bt_stats.loc[btk]["count"] >= 3:
        d = bt_stats.loc[btk]
        return area * (d["ppsf_median"] + slope * (floor_val - d["floor_mean"]))
    bk = row["building"]
    if bk in bld_stats.index:
        d = bld_stats.loc[bk]
        fadj = slope * (floor_val - d["floor_mean"]) if d["count"] >= 5 else 0
        return area * (d["ppsf_median"] + fadj)
    return None


# ══════════════════════════════════════════════════
# PREDICTION VARIANTS
# ══════════════════════════════════════════════════

def predict_v1_winner(test_df):
    """$1,435 winner: t=1.5, a=0.5, building-level correction."""
    return _predict(test_df, threshold=1.5, blend_alpha=0.5, use_unit=False)

def predict_v2_optimized(test_df):
    """LOO-optimized: t=1.35, a=0.55, building-level."""
    return _predict(test_df, threshold=1.35, blend_alpha=0.55, use_unit=False)

def predict_v3_unit(test_df):
    """Unit-level correction target instead of building."""
    return _predict(test_df, threshold=1.35, blend_alpha=0.55, use_unit=True)

def predict_v4_aggressive(test_df):
    """More aggressive: lower threshold, correct more rows."""
    return _predict(test_df, threshold=1.25, blend_alpha=0.5, use_unit=False)

def predict_v5_conservative(test_df):
    """Conservative: high threshold, strong correction."""
    return _predict(test_df, threshold=1.5, blend_alpha=0.4, use_unit=False)


def _predict(test_df, threshold, blend_alpha, use_unit):
    preds = np.zeros(len(test_df))
    for i in range(len(test_df)):
        row = test_df.iloc[i]
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
                # Get correction target
                if use_unit:
                    correction_target = get_unit_pred(row, area, floor_val, slope)
                    if correction_target is None and bk in bld_stats.index:
                        correction_target = area * bld_stats.loc[bk]["ppsf_median"]
                else:
                    correction_target = (area * bld_stats.loc[bk]["ppsf_median"]
                                         if bk in bld_stats.index else None)

                if correction_target is not None and correction_target > 0:
                    ratio = direct / correction_target
                    if ratio > threshold or ratio < 1.0 / threshold:
                        preds[i] = blend_alpha * direct + (1 - blend_alpha) * correction_target
                    else:
                        preds[i] = 0.8 * direct + 0.2 * row["knn10"]
                else:
                    preds[i] = 0.8 * direct + 0.2 * row["knn10"]
        else:
            preds[i] = fallback_pred(row, area, floor_val, slope)

    return np.clip(preds, 2000, 500000)


# ══════════════════════════════════════════════════
# LOO EVALUATION
# ══════════════════════════════════════════════════
print("Running LOO evaluation on all variants...\n")

# Precompute LOO stats
fa_sum = fa_grp["price"].transform("sum")
fa_cnt = fa_grp["price"].transform("count")
loo_count = fa_cnt - 1
loo_sum = fa_sum - train["price"]

# Cache LOO median/trimmed for each row
loo_median = {}
loo_trimmed = {}
for fa, group in fa_grp:
    prices = group["price"].values
    idxs = group.index.values
    for i_local, idx in enumerate(idxs):
        others = np.delete(prices, i_local)
        if len(others) >= 1:
            loo_median[idx] = np.median(others)
        if len(others) >= 4:
            loo_trimmed[idx] = trim_mean(others, 0.1)

train["_loo_count"] = loo_count
train["_loo_sum"] = loo_sum

eval_mask = train["_loo_count"] >= 1
eval_train = train[eval_mask]
actual = eval_train["price"].values


def loo_predict(threshold, blend_alpha, use_unit):
    """LOO prediction for evaluation."""
    preds = np.zeros(len(eval_train))
    for j, (idx, row) in enumerate(eval_train.iterrows()):
        area = row["area_sqft"]
        floor_val = row["floor"]
        slope = bld_slopes.get(row["building"], 0.0)
        bk = row["building"]
        n_loo = int(row["_loo_count"])

        if n_loo >= 4 and idx in loo_trimmed:
            preds[j] = loo_trimmed[idx]
        elif n_loo >= 2:
            preds[j] = loo_median.get(idx, row["_loo_sum"] / n_loo)
        else:
            direct = row["_loo_sum"]  # single other price

            if use_unit:
                correction_target = get_unit_pred(row, area, floor_val, slope)
                if correction_target is None and bk in bld_stats.index:
                    correction_target = area * bld_stats.loc[bk]["ppsf_median"]
            else:
                correction_target = (area * bld_stats.loc[bk]["ppsf_median"]
                                     if bk in bld_stats.index else None)

            if correction_target is not None and correction_target > 0:
                ratio = direct / correction_target
                if ratio > threshold or ratio < 1.0 / threshold:
                    preds[j] = blend_alpha * direct + (1 - blend_alpha) * correction_target
                else:
                    preds[j] = 0.8 * direct + 0.2 * correction_target
            else:
                preds[j] = direct

    return np.clip(preds, 2000, 500000)


variants = [
    ("v1_winner_t1.5_a0.5",   1.5,  0.5,  False),
    ("v2_optimized_t1.35",     1.35, 0.55, False),
    ("v3_unit_target",         1.35, 0.55, True),
    ("v4_aggressive_t1.25",    1.25, 0.5,  False),
    ("v5_conservative_t1.5",   1.5,  0.4,  False),
    ("v6_tight_t1.3_a0.45",   1.3,  0.45, False),
    ("v7_unit_t1.4_a0.5",     1.4,  0.5,  True),
    ("v8_unit_t1.5_a0.5",     1.5,  0.5,  True),
]

print(f"{'Variant':30s} {'LOO RMSE':>10s} {'Corrected':>10s}")
print("-" * 55)

results = []
for name, t, a, use_u in variants:
    loo_preds = loo_predict(t, a, use_u)
    rmse = np.sqrt(((actual - loo_preds) ** 2).mean())

    # Count corrected rows
    corrected = 0
    for j, (idx, row) in enumerate(eval_train.iterrows()):
        if int(row["_loo_count"]) == 1:
            direct = row["_loo_sum"]
            bk = row["building"]
            area = row["area_sqft"]
            if use_u:
                ct = get_unit_pred(row, area, row["floor"], bld_slopes.get(bk, 0))
                if ct is None and bk in bld_stats.index:
                    ct = area * bld_stats.loc[bk]["ppsf_median"]
            else:
                ct = area * bld_stats.loc[bk]["ppsf_median"] if bk in bld_stats.index else None
            if ct and ct > 0:
                ratio = direct / ct
                if ratio > t or ratio < 1.0/t:
                    corrected += 1

    print(f"{name:30s} ${rmse:>8,.0f} {corrected:>10d}")
    results.append((name, t, a, use_u, rmse))

# Sort by LOO RMSE
results.sort(key=lambda x: x[4])

print(f"\n{'='*55}")
print("TOP 3 BY LOO RMSE:")
for i, (name, t, a, use_u, rmse) in enumerate(results[:3]):
    print(f"  {i+1}. {name}: RMSE=${rmse:,.0f}")

# ══════════════════════════════════════════════════
# GENERATE TOP SUBMISSIONS
# ══════════════════════════════════════════════════
print("\nGenerating submissions...\n")

# Generate top 3 + the $1,435 winner for reference
gen_list = results[:3] + [r for r in results if "winner" in r[0]]
gen_list = list({r[0]: r for r in gen_list}.values())  # dedupe

base_preds = predict_v1_winner(test)

for name, t, a, use_u, rmse in gen_list:
    preds = _predict(test, threshold=t, blend_alpha=a, use_unit=use_u)
    fname = f"my_submission.csv" if name == results[0][0] else f"sub_{name}.csv"
    pd.DataFrame({"id": test["id"].astype(int), "price": preds.astype(int)}).to_csv(fname, index=False)

    diff = np.abs(preds - base_preds)
    changed = (diff > 1).sum()
    print(f"  {fname:35s} (LOO=${rmse:,.0f}) {changed} rows differ from $1,435 winner")

print(f"\nmy_submission.csv = best LOO variant: {results[0][0]}")
print("Submit my_submission.csv first!")
