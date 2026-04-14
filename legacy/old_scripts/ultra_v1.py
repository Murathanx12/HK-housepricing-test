"""
Hong Kong Rental Price Prediction — Ultra v1
==============================================
Optimized hardcoded lookup with Bayesian shrinkage blending.

Key improvements over LEGACY_winner_1450.py:
1. Geometric mean everywhere (FA agg + blend partner PPSF)
2. Split Bayesian shrinkage: k1=50 for count=1 (almost pure broader stats),
   k2=1.3 for count>=2 (adaptive blend)
3. unit_area5 as primary blend partner (more specific than unit_key)
4. Cascading fallback with geo mean: unit_area5 -> unit_key -> bld_tower -> building -> district
5. LOO RMSE ~1403 on count>=2 matched rows
"""

import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler

DATA_DIR = Path("./data")
K1 = 50    # For count=1 FA matches: w = 1/51 ~ 0.02 (almost ignore single observation)
K2 = 1.3   # For count>=2 FA matches: w = n/(n+1.3)

print("Loading...")
train = pd.read_csv(DATA_DIR / "HK_house_transactions.csv")
test = pd.read_csv(DATA_DIR / "test_features.csv")

# ── PARSE ──
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

# ── LOOKUP TABLES (all using geometric mean for PPSF where possible) ──
print("Building lookups...")

# Full address: geometric mean of price
fa_stats = train.groupby("full_addr").agg(
    p_geo_mean=("price", lambda x: np.exp(np.log(x).mean())),
    count=("price", "count"),
)

# Unit+area5: geo mean PPSF
ua_stats = train.groupby("unit_area5").agg(
    ppsf_geo=("ppsf", lambda x: np.exp(np.log(x).mean())),
    count=("price", "count"),
)

# Unit key: geo mean PPSF
unit_stats = train.groupby("unit_key").agg(
    ppsf_geo=("ppsf", lambda x: np.exp(np.log(x).mean())),
    count=("price", "count"),
)

# Building+tower
bt_stats = train.groupby("bld_tower").agg(
    ppsf_median=("ppsf", "median"),
    count=("price", "count"),
)

# Building+flat
bf_stats = train.groupby("bld_flat").agg(
    ppsf_median=("ppsf", "median"),
    count=("price", "count"),
)

# Building: geo mean PPSF
bld_stats = train.groupby("building").agg(
    ppsf_geo=("ppsf", lambda x: np.exp(np.log(x).mean())),
    ppsf_median=("ppsf", "median"),
    count=("price", "count"),
)

# District: geo mean PPSF
dist_stats = train.groupby("district").agg(
    ppsf_geo=("ppsf", lambda x: np.exp(np.log(x).mean())),
    ppsf_median=("ppsf", "median"),
)

# KNN for fallback (only used for unmatched rows without building match)
print("Building KNN...")
scaler = StandardScaler()
X_knn_tr = scaler.fit_transform(train[["wgs_lat", "wgs_lon", "area_sqft", "floor"]].values)
X_knn_te = scaler.transform(test[["wgs_lat", "wgs_lon", "area_sqft", "floor"]].values)

knn10 = KNeighborsRegressor(n_neighbors=10, weights="distance", n_jobs=-1)
knn10.fit(X_knn_tr, train["price"].values)
test["knn10"] = knn10.predict(X_knn_te)


def get_blend_pred(area, uak, uk, btk, bfk, bk, dk):
    """Cascading broader prediction for blending (uses geo mean)."""
    if uak in ua_stats.index and ua_stats.loc[uak]["count"] >= 2:
        return area * ua_stats.loc[uak]["ppsf_geo"]
    if uk in unit_stats.index:
        return area * unit_stats.loc[uk]["ppsf_geo"]
    if btk in bt_stats.index and bt_stats.loc[btk]["count"] >= 3:
        return area * bt_stats.loc[btk]["ppsf_median"]
    if bfk in bf_stats.index and bf_stats.loc[bfk]["count"] >= 3:
        return area * bf_stats.loc[bfk]["ppsf_median"]
    if bk in bld_stats.index:
        return area * bld_stats.loc[bk]["ppsf_geo"]
    if dk in dist_stats.index:
        return area * dist_stats.loc[dk]["ppsf_geo"]
    return None


def get_fallback_pred(row, area):
    """For rows with no full_addr match."""
    uak = row["unit_area5"]
    uk = row["unit_key"]
    btk = row["bld_tower"]
    bfk = row["bld_flat"]
    bk = row["building"]
    dk = row["district"]

    if uak in ua_stats.index and ua_stats.loc[uak]["count"] >= 2:
        return area * ua_stats.loc[uak]["ppsf_geo"]
    if uk in unit_stats.index:
        return area * unit_stats.loc[uk]["ppsf_geo"]
    if btk in bt_stats.index and bt_stats.loc[btk]["count"] >= 3:
        return area * bt_stats.loc[btk]["ppsf_median"]
    if bfk in bf_stats.index and bf_stats.loc[bfk]["count"] >= 3:
        return area * bf_stats.loc[bfk]["ppsf_median"]
    if bk in bld_stats.index:
        return area * bld_stats.loc[bk]["ppsf_geo"]

    # Last resort: KNN + district blend
    knn_p = row["knn10"]
    if dk in dist_stats.index:
        dp = area * dist_stats.loc[dk]["ppsf_median"]
        return 0.4 * knn_p + 0.6 * dp
    return knn_p


# ── PREDICT ──
print("\nPredicting...")
preds = np.zeros(len(test))
match_types = []

for i in range(len(test)):
    row = test.iloc[i]
    area = row["area_sqft"]
    fa = row["full_addr"]

    if fa in fa_stats.index:
        d = fa_stats.loc[fa]
        n = d["count"]

        # Full-address prediction: geometric mean of historical prices
        fa_pred = d["p_geo_mean"]

        # Broader blend partner
        blend_pred = get_blend_pred(
            area, row["unit_area5"], row["unit_key"],
            row["bld_tower"], row["bld_flat"], row["building"], row["district"]
        )

        if blend_pred is not None:
            # Split Bayesian shrinkage
            k = K1 if n == 1 else K2
            w = n / (n + k)
            preds[i] = w * fa_pred + (1 - w) * blend_pred
        else:
            preds[i] = fa_pred

        match_types.append(f"fa_n{int(n)}")
    else:
        preds[i] = get_fallback_pred(row, area)
        match_types.append("fallback")

preds = np.clip(preds, 2000, 500000)

# ── SAVE ──
submission = pd.DataFrame({"id": test["id"].astype(int), "price": preds.astype(int)})
submission.to_csv("my_submission.csv", index=False)

# Stats
mt = Counter(match_types)
print(f"\nSubmission saved: {len(submission)} rows")
print(f"Price: ${preds.min():,.0f} - ${preds.max():,.0f}, mean ${preds.mean():,.0f}")
print(f"\nMatch type distribution:")
fa_total = sum(v for k, v in mt.items() if k.startswith("fa_"))
fb = mt.get("fallback", 0)
n1 = sum(v for k, v in mt.items() if k == "fa_n1")
print(f"  full_addr match: {fa_total} ({100*fa_total/len(test):.1f}%)")
print(f"  - count=1: {n1} (using K1={K1}, w={1/(1+K1):.3f})")
print(f"  - count>=2: {fa_total - n1} (using K2={K2}, adaptive w)")
print(f"  fallback: {fb} ({100*fb/len(test):.1f}%)")
