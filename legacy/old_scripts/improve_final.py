"""
Deep data-driven hardcoding: adapt aggregation to actual data patterns.
Key insight from LOO: within-group std of $666 means ~$666 is our floor for
matched rows. The only lever is improving single-match (31.3%) and fallback (6.6%).

For single matches: the best blend partner isn't KNN but the UNIT-LEVEL stats
(same building+tower+flat, any area). Use ppsf * area from unit level.

For fallback: try every possible matching key and pick the most specific one
that has enough data points.

Also test: using the training data MORE granularly by computing per-building
optimal strategies.
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
    df["unit_key"] = df["building"] + "|" + df["Tower"].fillna("X").astype(str) + "|" + df["Flat"].fillna("X")
    df["bld_tower"] = df["building"] + "|T" + df["Tower"].fillna("X").astype(str)
    df["bld_flat"] = df["building"] + "|F" + df["Flat"].fillna("X")
    df["full_addr"] = df["address"].fillna("") + "|" + df["area_sqft"].astype(str)
    df["area_bin5"] = (df["area_sqft"] / 5).round() * 5
    df["area_bin10"] = (df["area_sqft"] / 10).round() * 10
    df["unit_area5"] = df["unit_key"] + "|" + df["area_bin5"].astype(str)
    df["unit_area10"] = df["unit_key"] + "|" + df["area_bin10"].astype(str)
    df["addr_only"] = df["address"].fillna("")

train["ppsf"] = train["price"] / train["area_sqft"]
train["log_price"] = np.log(train["price"])

# ── LOO: Find optimal single-match alpha per building ──
print("Analyzing per-building patterns...")

def floor_slope(g):
    if len(g) < 5 or g["floor"].std() < 1: return 0.0
    return np.polyfit(g["floor"], g["ppsf"], 1)[0]

bld_slopes = train.groupby("building").apply(floor_slope, include_groups=False).to_dict()

# Compute per-building PPSF consistency (coefficient of variation)
bld_cv_series = train.groupby("building")["ppsf"].agg(
    lambda x: x.std()/x.mean() if len(x) >= 5 and x.mean() > 0 else 0.15
)
bld_cv = bld_cv_series.to_dict()

print("Building CV distribution:")
cvs = bld_cv_series.values
for pct in [10, 25, 50, 75, 90]:
    print(f"  {pct}th percentile: {np.percentile(cvs, pct):.3f}")

# ── LOOKUPS ──
fa_grp = train.groupby("full_addr")
fa_stats = fa_grp.agg(
    p_mean=("price", "mean"), p_median=("price", "median"),
    count=("price", "count"), ppsf_median=("ppsf", "median"),
    floor_mean=("floor", "mean"),
)
fa_trimmed = fa_grp["price"].apply(lambda x: trim_mean(x, 0.1) if len(x) >= 4 else x.mean())
fa_stats = fa_stats.join(fa_trimmed.rename("p_trimmed"))

# Also compute in log space for 4+ groups
fa_log_mean = fa_grp["log_price"].apply(lambda x: np.exp(x.mean()) if len(x) >= 2 else np.exp(x.values[0]))
fa_stats = fa_stats.join(fa_log_mean.rename("p_log_mean"))

ao_stats = train.groupby("addr_only").agg(
    ppsf_median=("ppsf", "median"), ppsf_mean=("ppsf", "mean"),
    count=("price", "count"), floor_mean=("floor", "mean"),
)
ua5_stats = train.groupby("unit_area5").agg(
    ppsf_mean=("ppsf", "mean"), ppsf_median=("ppsf", "median"),
    floor_mean=("floor", "mean"), count=("price", "count"),
)
ua10_stats = train.groupby("unit_area10").agg(
    ppsf_mean=("ppsf", "mean"), ppsf_median=("ppsf", "median"),
    floor_mean=("floor", "mean"), count=("price", "count"),
)
unit_stats = train.groupby("unit_key").agg(
    ppsf_mean=("ppsf", "mean"), ppsf_median=("ppsf", "median"),
    floor_mean=("floor", "mean"), count=("price", "count"),
    area_mean=("area_sqft", "mean"),
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

# KNN
scaler = StandardScaler()
X_tr = scaler.fit_transform(train[["wgs_lat", "wgs_lon", "area_sqft", "floor"]].values)
X_te = scaler.transform(test[["wgs_lat", "wgs_lon", "area_sqft", "floor"]].values)
knn = KNeighborsRegressor(n_neighbors=10, weights="distance", n_jobs=-1)
knn.fit(X_tr, train["price"].values)
test["knn10"] = knn.predict(X_te)


def get_unit_ppsf_pred(row, area, floor_val, slope):
    """Best unit-level PPSF prediction. Returns (pred, source, count)."""
    # Try increasingly broad matching
    ua5k = row["unit_area5"]
    if ua5k in ua5_stats.index and ua5_stats.loc[ua5k]["count"] >= 2:
        d = ua5_stats.loc[ua5k]
        fadj = slope * (floor_val - d["floor_mean"]) if d["count"] >= 2 else 0
        return area * (d["ppsf_median"] + fadj), "ua5", int(d["count"])

    ua10k = row["unit_area10"]
    if ua10k in ua10_stats.index and ua10_stats.loc[ua10k]["count"] >= 2:
        d = ua10_stats.loc[ua10k]
        fadj = slope * (floor_val - d["floor_mean"]) if d["count"] >= 2 else 0
        return area * (d["ppsf_median"] + fadj), "ua10", int(d["count"])

    uk = row["unit_key"]
    if uk in unit_stats.index and unit_stats.loc[uk]["count"] >= 2:
        d = unit_stats.loc[uk]
        fadj = slope * (floor_val - d["floor_mean"])
        return area * (d["ppsf_median"] + fadj), "unit", int(d["count"])

    btk = row["bld_tower"]
    if btk in bt_stats.index and bt_stats.loc[btk]["count"] >= 3:
        d = bt_stats.loc[btk]
        fadj = slope * (floor_val - d["floor_mean"])
        return area * (d["ppsf_median"] + fadj), "bt", int(d["count"])

    bk = row["building"]
    if bk in bld_stats.index and bld_stats.loc[bk]["count"] >= 3:
        d = bld_stats.loc[bk]
        fadj = slope * (floor_val - d["floor_mean"]) if d["count"] >= 5 else 0
        return area * (d["ppsf_median"] + fadj), "bld", int(d["count"])

    return row["knn10"], "knn", 0


def fallback_chain(row, area, floor_val, slope):
    """Full fallback chain."""
    ao = row["addr_only"]
    if ao in ao_stats.index and ao_stats.loc[ao]["count"] >= 3:
        d = ao_stats.loc[ao]
        fadj = slope * (floor_val - d["floor_mean"])
        return area * (d["ppsf_median"] + fadj)
    pred, _, _ = get_unit_ppsf_pred(row, area, floor_val, slope)
    return pred


# ══════════════════════════════════════════════════
# SUBMISSION 1: Baseline (exact $1,450 reproduction)
# ══════════════════════════════════════════════════
def sub1_baseline():
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
            preds[i] = fallback_chain(row, area, fv, slope)
    return np.clip(preds, 2000, 500000)


# ══════════════════════════════════════════════════
# SUBMISSION 2: CV-adaptive single-match blending
# Low CV building (consistent pricing) -> trust single price more (alpha=0.9)
# High CV building (variable pricing) -> trust unit stats more (alpha=0.6)
# ══════════════════════════════════════════════════
def sub2_cv_adaptive():
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
                cv = bld_cv.get(bk, 0.15)
                # Map CV to alpha: low cv -> high alpha (trust direct)
                # CV 0.05 -> alpha 0.95, CV 0.20 -> alpha 0.70, CV 0.40 -> alpha 0.50
                alpha = max(0.50, min(0.95, 1.0 - cv * 1.5))
                unit_pred, src, cnt = get_unit_ppsf_pred(row, area, fv, slope)
                if cnt >= 3:
                    preds[i] = alpha * direct + (1 - alpha) * unit_pred
                else:
                    preds[i] = 0.8 * direct + 0.2 * row["knn10"]
        else:
            preds[i] = fallback_chain(row, area, fv, slope)
    return np.clip(preds, 2000, 500000)


# ══════════════════════════════════════════════════
# SUBMISSION 3: For 4+ matches use Hodges-Lehmann estimator
# (median of all pairwise means — more robust than trimmed mean)
# ══════════════════════════════════════════════════
def hodges_lehmann(x):
    n = len(x)
    if n < 2: return x[0] if n == 1 else 0
    if n > 30:  # Too many pairs, use sampling
        rng = np.random.RandomState(42)
        idx = rng.choice(n, size=min(n, 30), replace=False)
        x = x[idx]
        n = len(x)
    pairwise = []
    for i in range(n):
        for j in range(i, n):
            pairwise.append((x[i] + x[j]) / 2)
    return np.median(pairwise)

fa_hl = fa_grp["price"].apply(lambda x: hodges_lehmann(x.values) if len(x) >= 4 else x.median())
fa_stats = fa_stats.join(fa_hl.rename("p_hl"))

def sub3_hodges_lehmann():
    preds = np.zeros(len(test))
    for i in range(len(test)):
        row = test.iloc[i]
        area, fv = row["area_sqft"], row["floor"]
        fa = row["full_addr"]
        slope = bld_slopes.get(row["building"], 0.0)
        if fa in fa_stats.index:
            d = fa_stats.loc[fa]
            if d["count"] >= 4:
                preds[i] = d["p_hl"]
            elif d["count"] >= 2:
                preds[i] = d["p_median"]
            else:
                preds[i] = 0.8 * d["p_mean"] + 0.2 * row["knn10"]
        else:
            preds[i] = fallback_chain(row, area, fv, slope)
    return np.clip(preds, 2000, 500000)


# ══════════════════════════════════════════════════
# SUBMISSION 4: Blend trimmed + HL + median for 4+ (ensemble of estimators)
# ══════════════════════════════════════════════════
def sub4_estimator_blend():
    preds = np.zeros(len(test))
    for i in range(len(test)):
        row = test.iloc[i]
        area, fv = row["area_sqft"], row["floor"]
        fa = row["full_addr"]
        slope = bld_slopes.get(row["building"], 0.0)
        if fa in fa_stats.index:
            d = fa_stats.loc[fa]
            if d["count"] >= 4:
                # Blend 3 robust estimators
                preds[i] = (d["p_trimmed"] + d["p_hl"] + d["p_median"]) / 3
            elif d["count"] >= 2:
                preds[i] = d["p_median"]
            else:
                preds[i] = 0.8 * d["p_mean"] + 0.2 * row["knn10"]
        else:
            preds[i] = fallback_chain(row, area, fv, slope)
    return np.clip(preds, 2000, 500000)


# ══════════════════════════════════════════════════
# SUBMISSION 5: Everything combined
# - CV-adaptive for single matches
# - Estimator blend for 4+ matches
# - Improved fallback chain
# - Extreme outlier correction
# ══════════════════════════════════════════════════
def sub5_everything():
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
                # Estimator blend
                preds[i] = (d["p_trimmed"] + d["p_hl"] + d["p_median"]) / 3
            elif d["count"] >= 2:
                preds[i] = d["p_median"]
            else:
                direct = d["p_mean"]
                # Extreme outlier check first
                if bk in bld_stats.index:
                    bld_pred = area * bld_stats.loc[bk]["ppsf_median"]
                    ratio = direct / bld_pred if bld_pred > 0 else 1.0
                    if ratio > 1.5 or ratio < 0.67:
                        preds[i] = 0.5 * direct + 0.5 * bld_pred
                        continue

                # CV-adaptive blending
                cv = bld_cv.get(bk, 0.15)
                alpha = max(0.50, min(0.95, 1.0 - cv * 1.5))
                unit_pred, src, cnt = get_unit_ppsf_pred(row, area, fv, slope)
                if cnt >= 3:
                    preds[i] = alpha * direct + (1 - alpha) * unit_pred
                else:
                    preds[i] = 0.8 * direct + 0.2 * row["knn10"]
        else:
            preds[i] = fallback_chain(row, area, fv, slope)
    return np.clip(preds, 2000, 500000)


# ── RUN ALL ──
print("\nGenerating submissions...\n")
base = sub1_baseline()

subs = {
    "my_sub_baseline": base,
    "my_sub_cv_adaptive": sub2_cv_adaptive(),
    "my_sub_hodges_lehmann": sub3_hodges_lehmann(),
    "my_sub_estimator_blend": sub4_estimator_blend(),
    "my_sub_everything": sub5_everything(),
}

print(f"{'Name':30s} {'Changed':>8s} {'MeanDiff':>10s} {'MaxDiff':>10s}")
print("-" * 62)
for name, preds in subs.items():
    pd.DataFrame({"id": test["id"].astype(int), "price": preds.astype(int)}).to_csv(f"{name}.csv", index=False)
    diff = np.abs(preds - base)
    changed = (diff > 1).sum()
    print(f"{name:30s} {changed:8d} ${diff.mean():8,.0f} ${diff.max():8,.0f}")

print("\n=== TOP 3 TO SUBMIT ===")
print("1. my_sub_everything.csv      — All improvements combined")
print("2. my_sub_estimator_blend.csv  — 3-estimator blend for 4+ groups")
print("3. my_sub_cv_adaptive.csv      — Building-CV-aware single blending")
