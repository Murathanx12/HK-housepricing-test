"""
Hong Kong Rental Price Prediction — $1,355 RMSE Winner
======================================================
EXACT code that produces the $1,355 leaderboard score.
DO NOT MODIFY THIS FILE.

Config:
  - 4+ matches: trimmed mean (10% trim)
  - n=3 matches: plain mean
  - n=2 matches: mean (= median)
  - n=1 single match: 85% direct + 5% building_ppsf*area + 10% KNN
  - Fallback: ppsf_median cascade
"""

import pandas as pd, numpy as np
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

def floor_slope(g):
    if len(g) < 5 or g["floor"].std() < 1: return 0.0
    return np.polyfit(g["floor"], g["ppsf"], 1)[0]

bld_slopes = train.groupby("building").apply(floor_slope, include_groups=False).to_dict()

fa_grp = train.groupby("full_addr")
fa_stats = fa_grp.agg(
    p_mean=("price", "mean"), p_median=("price", "median"),
    count=("price", "count"), floor_mean=("floor", "mean"),
)
fa_trimmed = fa_grp["price"].apply(
    lambda x: trim_mean(x, 0.1) if len(x) >= 4 else x.mean()
)
fa_stats = fa_stats.join(fa_trimmed.rename("p_trimmed"))

ua_stats = train.groupby("unit_area5").agg(
    ppsf_mean=("ppsf","mean"), ppsf_median=("ppsf","median"),
    floor_mean=("floor","mean"), count=("price","count"),
)
unit_stats = train.groupby("unit_key").agg(
    ppsf_mean=("ppsf","mean"), ppsf_median=("ppsf","median"),
    floor_mean=("floor","mean"), count=("price","count"),
)
bt_stats = train.groupby("bld_tower").agg(
    ppsf_median=("ppsf","median"), floor_mean=("floor","mean"),
    count=("price","count"),
)
bf_stats = train.groupby("bld_flat").agg(
    ppsf_median=("ppsf","median"), floor_mean=("floor","mean"),
    count=("price","count"),
)
bld_stats = train.groupby("building").agg(
    ppsf_mean=("ppsf","mean"), ppsf_median=("ppsf","median"),
    floor_mean=("floor","mean"), count=("price","count"),
)
dist_stats = train.groupby("district").agg(ppsf_median=("ppsf","median"))

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

preds = np.zeros(len(test))
for i in range(len(test)):
    row = test.iloc[i]
    area, fv = row["area_sqft"], row["floor"]
    fa = row["full_addr"]
    slope = bld_slopes.get(row["building"], 0.0)
    bk = row["building"]

    if fa in fa_stats.index:
        d = fa_stats.loc[fa]
        n = int(d["count"])
        if n >= 4:
            preds[i] = d["p_trimmed"]
        elif n == 3:
            preds[i] = d["p_mean"]
        elif n == 2:
            preds[i] = d["p_mean"]
        else:
            direct = d["p_mean"]
            if bk in bld_stats.index:
                bp = area * bld_stats.loc[bk]["ppsf_median"]
                preds[i] = 0.85 * direct + 0.05 * bp + 0.10 * row["knn10"]
            else:
                preds[i] = 0.90 * direct + 0.10 * row["knn10"]
    else:
        uak = row["unit_area5"]
        uk = row["unit_key"]
        btk = row["bld_tower"]
        bfk = row["bld_flat"]
        dk = row["district"]
        if uak in ua_stats.index:
            d = ua_stats.loc[uak]
            base = d["ppsf_median"] if d["count"] >= 2 else d["ppsf_mean"]
            fadj = slope * (fv - d["floor_mean"]) if d["count"] >= 2 else 0
            preds[i] = area * (base + fadj)
        elif uk in unit_stats.index:
            d = unit_stats.loc[uk]
            base = d["ppsf_median"] if d["count"] >= 3 else d["ppsf_mean"]
            preds[i] = area * (base + slope * (fv - d["floor_mean"]))
        elif btk in bt_stats.index and bt_stats.loc[btk]["count"] >= 3:
            d = bt_stats.loc[btk]
            preds[i] = area * (d["ppsf_median"] + slope * (fv - d["floor_mean"]))
        elif bfk in bf_stats.index and bf_stats.loc[bfk]["count"] >= 3:
            d = bf_stats.loc[bfk]
            preds[i] = area * (d["ppsf_median"] + slope * (fv - d["floor_mean"]))
        elif bk in bld_stats.index:
            d = bld_stats.loc[bk]
            base = d["ppsf_median"] if d["count"] >= 3 else d["ppsf_mean"]
            fadj = slope * (fv - d["floor_mean"]) if d["count"] >= 5 else 0
            preds[i] = area * (base + fadj)
        else:
            kp = row["knn10"]
            if dk in dist_stats.index:
                preds[i] = 0.4 * kp + 0.6 * area * dist_stats.loc[dk]["ppsf_median"]
            else:
                preds[i] = kp

preds = np.clip(preds, 2000, 500000)
pd.DataFrame({"id": test["id"].astype(int), "price": preds.astype(int)}).to_csv(
    "my_submission.csv", index=False
)
print(f"Saved my_submission.csv")
print(f"Mean: ${preds.mean():,.0f}, Median: ${np.median(preds):,.0f}")
