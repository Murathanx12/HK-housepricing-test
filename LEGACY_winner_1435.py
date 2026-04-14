"""
Hong Kong Rental Price Prediction — $1,435 RMSE Winner
======================================================
EXACT code that produced the $1,435 leaderboard score.
DO NOT MODIFY THIS FILE.

Changes from $1,450:
  - For single-match rows where price/building_pred ratio > 1.5 or < 0.67,
    blend 50/50 with building median prediction (7 rows affected)
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
    count=("price", "count"), ppsf_median=("ppsf", "median"),
    floor_mean=("floor", "mean"),
)
fa_trimmed = fa_grp["price"].apply(
    lambda x: trim_mean(x, 0.1) if len(x) >= 4 else x.mean()
)
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
    ppsf_median=("ppsf", "median"), floor_mean=("floor", "mean"),
    count=("price", "count"),
)
bf_stats = train.groupby("bld_flat").agg(
    ppsf_median=("ppsf", "median"), floor_mean=("floor", "mean"),
    count=("price", "count"),
)
bld_stats = train.groupby("building").agg(
    ppsf_mean=("ppsf", "mean"), ppsf_median=("ppsf", "median"),
    floor_mean=("floor", "mean"), count=("price", "count"),
)
dist_stats = train.groupby("district").agg(ppsf_median=("ppsf", "median"))

scaler = StandardScaler()
X_tr = scaler.fit_transform(
    train[["wgs_lat", "wgs_lon", "area_sqft", "floor"]].values
)
X_te = scaler.transform(
    test[["wgs_lat", "wgs_lon", "area_sqft", "floor"]].values
)
knn = KNeighborsRegressor(n_neighbors=10, weights="distance", n_jobs=-1)
knn.fit(X_tr, train["price"].values)
test["knn10"] = knn.predict(X_te)


def fallback_pred(row, area, floor_val, slope):
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


# ── PREDICT ──
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
                    preds[i] = 0.5 * direct + 0.5 * bld_pred
                    fixes += 1
                else:
                    preds[i] = 0.8 * direct + 0.2 * row["knn10"]
            else:
                preds[i] = 0.8 * direct + 0.2 * row["knn10"]
    else:
        preds[i] = fallback_pred(row, area, fv, slope)

preds = np.clip(preds, 2000, 500000)

pd.DataFrame({
    "id": test["id"].astype(int),
    "price": preds.astype(int)
}).to_csv("my_submission.csv", index=False)

print(f"Saved my_submission.csv")
print(f"Fixes: {fixes} rows corrected")
print(f"Mean: ${preds.mean():,.0f}, Median: ${np.median(preds):,.0f}")
