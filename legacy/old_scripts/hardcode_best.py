"""
Hong Kong Rental Price Prediction — Best Hardcoded
====================================================
Pure lookup, no ML. Zero overfitting risk.
Uses hierarchical price lookup with floor adjustments.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import root_mean_squared_error

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
    df["area_bin"] = (df["area_sqft"] / 5).round() * 5  # Finer bins: every 5 sqft
    df["unit_area"] = df["unit_key"] + "|" + df["area_bin"].astype(str)
    df["full_addr"] = df["address"].fillna("") + "|" + df["area_sqft"].astype(str)

train["ppsf"] = train["price"] / train["area_sqft"]

# Floor slope per building
def floor_slope_ppsf(g):
    if len(g) < 5 or g["floor"].std() < 1: return 0.0
    return np.polyfit(g["floor"], g["ppsf"], 1)[0]

bld_slopes = train.groupby("building").apply(floor_slope_ppsf, include_groups=False).to_dict()

# Full address lookup (most specific)
full_addr_stats = train.groupby("full_addr").agg(
    price_median=("price", "median"), price_mean=("price", "mean"), count=("price", "count")
)

# Unit+area lookup
ua_stats = train.groupby("unit_area").agg(
    ppsf_median=("ppsf", "median"), ppsf_mean=("ppsf", "mean"),
    price_median=("price", "median"), floor_mean=("floor", "mean"), count=("price", "count")
)

# Unit lookup
unit_stats = train.groupby("unit_key").agg(
    ppsf_median=("ppsf", "median"), ppsf_mean=("ppsf", "mean"),
    price_median=("price", "median"), floor_mean=("floor", "mean"),
    area_mean=("area_sqft", "mean"), count=("price", "count")
)

# Building+tower
bt_stats = train.groupby("bld_tower").agg(
    ppsf_median=("ppsf", "median"), floor_mean=("floor", "mean"), count=("price", "count")
)

# Building+flat
bf_stats = train.groupby("bld_flat").agg(
    ppsf_median=("ppsf", "median"), floor_mean=("floor", "mean"), count=("price", "count")
)

# Building
bld_stats = train.groupby("building").agg(
    ppsf_median=("ppsf", "median"), ppsf_mean=("ppsf", "mean"),
    floor_mean=("floor", "mean"), count=("price", "count")
)

# District
dist_stats = train.groupby("district").agg(
    ppsf_median=("ppsf", "median"), floor_mean=("floor", "mean"), count=("price", "count")
)

# KNN fallback
scaler = StandardScaler()
X_knn_tr = scaler.fit_transform(train[["wgs_lat", "wgs_lon", "area_sqft", "floor"]].values)
X_knn_te = scaler.transform(test[["wgs_lat", "wgs_lon", "area_sqft", "floor"]].values)
knn = KNeighborsRegressor(n_neighbors=10, weights="distance", n_jobs=-1)
knn.fit(X_knn_tr, train["price"].values)
knn_prices = knn.predict(X_knn_te)

# Predict
predictions = np.zeros(len(test))
methods = []

for i in range(len(test)):
    row = test.iloc[i]
    area = row["area_sqft"]
    floor_val = row["floor"]

    fa = row["full_addr"]
    uak = row["unit_area"]
    uk = row["unit_key"]
    btk = row["bld_tower"]
    bfk = row["bld_flat"]
    bk = row["building"]
    dk = row["district"]

    slope = bld_slopes.get(bk, 0.0)

    # Level 0: exact full address + area match
    if fa in full_addr_stats.index:
        d = full_addr_stats.loc[fa]
        if d["count"] >= 2:
            predictions[i] = d["price_median"]
            methods.append("full_addr>=2")
            continue
        elif d["count"] == 1:
            predictions[i] = d["price_mean"]
            methods.append("full_addr=1")
            continue

    # Level 1: unit + area bin
    if uak in ua_stats.index:
        d = ua_stats.loc[uak]
        if d["count"] >= 2:
            base_ppsf = d["ppsf_median"]
            floor_adj = slope * (floor_val - d["floor_mean"])
            predictions[i] = area * (base_ppsf + floor_adj)
            methods.append("unit_area>=2")
            continue
        else:
            predictions[i] = area * d["ppsf_mean"]
            methods.append("unit_area=1")
            continue

    # Level 2: exact unit
    if uk in unit_stats.index:
        d = unit_stats.loc[uk]
        base_ppsf = d["ppsf_median"] if d["count"] >= 3 else d["ppsf_mean"]
        floor_adj = slope * (floor_val - d["floor_mean"])
        predictions[i] = area * (base_ppsf + floor_adj)
        methods.append("unit")
        continue

    # Level 3: building+tower
    if btk in bt_stats.index and bt_stats.loc[btk]["count"] >= 3:
        d = bt_stats.loc[btk]
        floor_adj = slope * (floor_val - d["floor_mean"])
        predictions[i] = area * (d["ppsf_median"] + floor_adj)
        methods.append("bld_tower")
        continue

    # Level 4: building+flat
    if bfk in bf_stats.index and bf_stats.loc[bfk]["count"] >= 3:
        d = bf_stats.loc[bfk]
        floor_adj = slope * (floor_val - d["floor_mean"])
        predictions[i] = area * (d["ppsf_median"] + floor_adj)
        methods.append("bld_flat")
        continue

    # Level 5: building
    if bk in bld_stats.index:
        d = bld_stats.loc[bk]
        floor_adj = slope * (floor_val - d["floor_mean"]) if d["count"] >= 5 else 0
        ppsf = d["ppsf_median"] if d["count"] >= 3 else d["ppsf_mean"]
        predictions[i] = area * (ppsf + floor_adj)
        methods.append("building")
        continue

    # Level 6: district + KNN blend
    if dk in dist_stats.index:
        d = dist_stats.loc[dk]
        dist_pred = area * d["ppsf_median"]
        predictions[i] = 0.5 * knn_prices[i] + 0.5 * dist_pred
        methods.append("knn+district")
    else:
        predictions[i] = knn_prices[i]
        methods.append("knn_only")

predictions = np.clip(predictions, 2000, 500000)

from collections import Counter
mc = Counter(methods)
print("Methods:")
for m, c in sorted(mc.items(), key=lambda x: -x[1]):
    print(f"  {m:20s}: {c:6d} ({c/len(test)*100:.1f}%)")

submission = pd.DataFrame({"id": test["id"].astype(int), "price": predictions.astype(int)})
submission.to_csv("hardcode_submission.csv", index=False)
print(f"\nSaved hardcode_submission.csv: {len(submission)} rows")
print(f"Price: ${predictions.min():,.0f} - ${predictions.max():,.0f}, mean ${predictions.mean():,.0f}")
