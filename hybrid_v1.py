"""
Hong Kong Rental Price Prediction — Hybrid Approach
=====================================================
Hardcoded lookups as primary features + light LightGBM correction.
Fast (~1 min) and should generalize well.
"""

import pandas as pd
import numpy as np
import re
import warnings
from pathlib import Path
from scipy.spatial import cKDTree
from sklearn.model_selection import KFold
from sklearn.metrics import root_mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
import lightgbm as lgb

warnings.filterwarnings("ignore")
np.random.seed(42)

DATA_DIR = Path("./data")

# ─────────────────────────────────────────────
# 1. LOAD
# ─────────────────────────────────────────────
print("Loading data...")
train = pd.read_csv(DATA_DIR / "HK_house_transactions.csv")
test = pd.read_csv(DATA_DIR / "test_features.csv")
mtr = pd.read_csv(DATA_DIR / "HK_mtr_station.csv")
parks = pd.read_csv(DATA_DIR / "HK_park.csv")
schools = pd.read_csv(DATA_DIR / "HK_school.csv")
malls = pd.read_csv(DATA_DIR / "HK_mall.csv")
hospitals = pd.read_csv(DATA_DIR / "HK_hospital.csv")
cbd = pd.read_csv(DATA_DIR / "HK_city_center.csv")

# ─────────────────────────────────────────────
# 2. PARSE
# ─────────────────────────────────────────────
print("Parsing...")

def get_building(addr):
    if pd.isna(addr): return "UNKNOWN"
    return addr.split(",")[0].strip()

def get_floor_desc(addr):
    if pd.isna(addr): return -1
    m = re.search(r'(Lower|Middle|Upper|High|Low)\s+Floor', addr)
    if not m: return -1
    return {"Lower": 0, "Low": 0, "Middle": 1, "Upper": 2, "High": 2}.get(m.group(1), -1)

for df in [train, test]:
    df["building"] = df["address"].apply(get_building)
    df["floor"] = pd.to_numeric(df["floor"], errors="coerce").fillna(10)
    df["Public_Housing"] = df["Public_Housing"].astype(int)
    df["flat_ord"] = df["Flat"].apply(lambda x: ord(str(x)[0].upper()) - 65 if pd.notna(x) and str(x)[0].isalpha() else -1)
    df["tower_num"] = pd.to_numeric(df["Tower"], errors="coerce").fillna(-1)
    df["phase_num"] = pd.to_numeric(df["Phase"], errors="coerce").fillna(-1)
    df["floor_desc"] = df["address"].apply(get_floor_desc)
    df["floor_cat"] = np.where(df["floor"] <= 7, 0, np.where(df["floor"] <= 14, 1, np.where(df["floor"] <= 22, 2, 3)))

    df["unit_key"] = df["building"] + "|" + df["Tower"].fillna("X").astype(str) + "|" + df["Flat"].fillna("X")
    df["unit_floorcat"] = df["unit_key"] + "|" + df["floor_cat"].astype(str)
    df["area_bin"] = (df["area_sqft"] / 10).round() * 10
    df["unit_area"] = df["unit_key"] + "|" + df["area_bin"].astype(str)
    df["bld_tower"] = df["building"] + "|T" + df["Tower"].fillna("X").astype(str)
    df["bld_flat"] = df["building"] + "|F" + df["Flat"].fillna("X")

train["price_per_sqft"] = train["price"] / train["area_sqft"]

# ─────────────────────────────────────────────
# 3. LEAVE-ONE-OUT TARGET ENCODING
# ─────────────────────────────────────────────
print("Computing LOO target encodings...")

global_ppsf_mean = train["price_per_sqft"].mean()
global_price_mean = train["price"].mean()

def loo_encode(train_df, test_df, group_col, target_col, prefix, min_count=1):
    """Leave-one-out encoding for train, regular encoding for test."""
    # For test: regular group stats
    grp = train_df.groupby(group_col)[target_col]
    test_stats = grp.agg(["mean", "median", "count"]).reset_index()
    test_stats.columns = [group_col, f"{prefix}_mean", f"{prefix}_median", f"{prefix}_count"]

    # For train: leave-one-out
    group_sum = grp.transform("sum")
    group_count = grp.transform("count")

    # LOO mean: (sum - current) / (count - 1)
    loo_mean = (group_sum - train_df[target_col]) / (group_count - 1)
    # Where count == 1, fall back to global mean
    global_mean = train_df[target_col].mean()
    loo_mean = loo_mean.fillna(global_mean)
    # Where count == 1, replace inf/nan
    loo_mean = loo_mean.replace([np.inf, -np.inf], global_mean)

    train_df[f"{prefix}_loo_mean"] = loo_mean

    # Regular stats for train too (for features that don't leak)
    merged = train_df[[group_col]].merge(test_stats, on=group_col, how="left")
    train_df[f"{prefix}_mean"] = merged[f"{prefix}_mean"].values
    train_df[f"{prefix}_median"] = merged[f"{prefix}_median"].values
    train_df[f"{prefix}_count"] = merged[f"{prefix}_count"].values

    # Test gets regular stats
    merged_te = test_df[[group_col]].merge(test_stats, on=group_col, how="left")
    test_df[f"{prefix}_mean"] = merged_te[f"{prefix}_mean"].values
    test_df[f"{prefix}_median"] = merged_te[f"{prefix}_median"].values
    test_df[f"{prefix}_count"] = merged_te[f"{prefix}_count"].values
    test_df[f"{prefix}_loo_mean"] = test_df[f"{prefix}_mean"]  # No LOO needed for test

# Price per sqft encodings
for group_col, prefix in [
    ("unit_area", "ua"), ("unit_floorcat", "ufc"), ("unit_key", "unit"),
    ("bld_tower", "bt"), ("bld_flat", "bf"), ("building", "bld"), ("district", "dist"),
]:
    loo_encode(train, test, group_col, "price_per_sqft", f"{prefix}_ppsf")

# Price encodings
for group_col, prefix in [
    ("unit_area", "ua"), ("unit_floorcat", "ufc"), ("unit_key", "unit"),
    ("bld_tower", "bt"), ("bld_flat", "bf"), ("building", "bld"), ("district", "dist"),
]:
    loo_encode(train, test, group_col, "price", f"{prefix}_price")

# Building extra stats
bld_extra = train.groupby("building").agg(
    bld_area_mean=("area_sqft", "mean"),
    bld_floor_mean=("floor", "mean"),
    bld_floor_min=("floor", "min"),
    bld_floor_max=("floor", "max"),
).reset_index()

def floor_slope(g):
    if len(g) < 5 or g["floor"].std() < 1: return 0.0
    return np.polyfit(g["floor"], g["price_per_sqft"], 1)[0]

bld_slope = train.groupby("building").apply(floor_slope, include_groups=False).reset_index()
bld_slope.columns = ["building", "bld_ppsf_floor_slope"]
bld_extra = bld_extra.merge(bld_slope, on="building", how="left").fillna(0)

for df in [train, test]:
    merged = df[["building"]].merge(bld_extra, on="building", how="left")
    for col in bld_extra.columns:
        if col != "building":
            df[col] = merged[col].values

# Cascading fallback for missing values
chains = [
    ("ua_ppsf", "ufc_ppsf", "unit_ppsf", "bt_ppsf", "bf_ppsf", "bld_ppsf", "dist_ppsf"),
    ("ufc_ppsf", "unit_ppsf", "bt_ppsf", "bf_ppsf", "bld_ppsf", "dist_ppsf"),
    ("unit_ppsf", "bt_ppsf", "bf_ppsf", "bld_ppsf", "dist_ppsf"),
    ("bt_ppsf", "bld_ppsf", "dist_ppsf"), ("bf_ppsf", "bld_ppsf", "dist_ppsf"),
    ("ua_price", "ufc_price", "unit_price", "bt_price", "bf_price", "bld_price", "dist_price"),
    ("ufc_price", "unit_price", "bt_price", "bf_price", "bld_price", "dist_price"),
    ("unit_price", "bt_price", "bf_price", "bld_price", "dist_price"),
]
for chain in chains:
    for suffix in ["_mean", "_median", "_loo_mean"]:
        for i in range(len(chain) - 1):
            col, fb = chain[i] + suffix, chain[i+1] + suffix
            if col in test.columns and fb in test.columns:
                test[col] = test[col].fillna(test[fb])
                train[col] = train[col].fillna(train[fb])

for df in [train, test]:
    for col in df.columns:
        if df[col].dtype in [np.float64, np.float32] and df[col].isna().any():
            df[col] = df[col].fillna(train[col].median() if col in train.columns else 0)

# ─────────────────────────────────────────────
# 4. KNN
# ─────────────────────────────────────────────
print("Computing KNN...")
scaler = StandardScaler()
X_knn_tr = scaler.fit_transform(train[["wgs_lat","wgs_lon","area_sqft","floor"]].values)
X_knn_te = scaler.transform(test[["wgs_lat","wgs_lon","area_sqft","floor"]].values)

for k in [3, 5, 10, 20]:
    knn = KNeighborsRegressor(n_neighbors=k, weights="distance", n_jobs=-1)
    knn.fit(X_knn_tr, train["price"].values)
    train[f"knn_{k}_price"] = knn.predict(X_knn_tr)
    test[f"knn_{k}_price"] = knn.predict(X_knn_te)

    knn2 = KNeighborsRegressor(n_neighbors=k, weights="distance", n_jobs=-1)
    knn2.fit(X_knn_tr, train["price_per_sqft"].values)
    train[f"knn_{k}_ppsf"] = knn2.predict(X_knn_tr)
    test[f"knn_{k}_ppsf"] = knn2.predict(X_knn_te)

# ─────────────────────────────────────────────
# 5. SPATIAL
# ─────────────────────────────────────────────
print("Spatial features...")
EARTH_R = 6371.0

def nearest_dist(df, plat, plon, name):
    tree = cKDTree(np.radians(np.column_stack([plat, plon])))
    d, _ = tree.query(np.radians(df[["wgs_lat","wgs_lon"]].values), k=1)
    df[f"dist_{name}"] = d * EARTH_R

def count_within(df, plat, plon, name, r):
    tree = cKDTree(np.radians(np.column_stack([plat, plon])))
    c = tree.query_ball_point(np.radians(df[["wgs_lat","wgs_lon"]].values), r=r/EARTH_R)
    df[f"cnt_{name}_{r}km"] = [len(x) for x in c]

def haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    d = lat2-lat1; dl = lon2-lon1
    a = np.sin(d/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dl/2)**2
    return 2*EARTH_R*np.arcsin(np.sqrt(a))

for df in [train, test]:
    df["dist_cbd"] = haversine(df["wgs_lat"].values, df["wgs_lon"].values, cbd["lat"].values[0], cbd["lon"].values[0])
    nearest_dist(df, mtr["lat"].values, mtr["lon"].values, "mtr")
    count_within(df, mtr["lat"].values, mtr["lon"].values, "mtr", 0.5)
    count_within(df, mtr["lat"].values, mtr["lon"].values, "mtr", 1.0)
    nearest_dist(df, parks["centroid_lat"].values, parks["centroid_lon"].values, "park")
    count_within(df, parks["centroid_lat"].values, parks["centroid_lon"].values, "park", 1.0)
    nearest_dist(df, schools["lat"].values, schools["lon"].values, "school")
    count_within(df, schools["lat"].values, schools["lon"].values, "school", 1.0)
    nearest_dist(df, malls["lat"].values, malls["lon"].values, "mall")
    count_within(df, malls["lat"].values, malls["lon"].values, "mall", 1.0)
    nearest_dist(df, hospitals["Latitude"].values, hospitals["Longitude"].values, "hospital")

# ─────────────────────────────────────────────
# 6. DERIVED FEATURES
# ─────────────────────────────────────────────
print("Derived features...")
for df in [train, test]:
    df["log_area"] = np.log1p(df["area_sqft"])
    df["area_vs_bld"] = df["area_sqft"] - df["bld_area_mean"]
    df["floor_vs_bld"] = df["floor"] - df["bld_floor_mean"]
    fr = df["bld_floor_max"] - df["bld_floor_min"]
    df["floor_pct"] = np.where(fr > 0, (df["floor"] - df["bld_floor_min"]) / fr, 0.5)

    # HARDCODED PREDICTIONS as features (using LOO means for train to avoid leakage!)
    df["pred_ua"] = df["area_sqft"] * df["ua_ppsf_loo_mean"]
    df["pred_ufc"] = df["area_sqft"] * df["ufc_ppsf_loo_mean"]
    df["pred_unit"] = df["area_sqft"] * df["unit_ppsf_loo_mean"]
    df["pred_bt"] = df["area_sqft"] * df["bt_ppsf_loo_mean"]
    df["pred_bf"] = df["area_sqft"] * df["bf_ppsf_loo_mean"]
    df["pred_bld"] = df["area_sqft"] * df["bld_ppsf_loo_mean"]
    df["pred_dist"] = df["area_sqft"] * df["dist_ppsf_loo_mean"]
    df["pred_bld_floor_adj"] = df["pred_bld"] + df["bld_ppsf_floor_slope"] * (df["floor"] - df["bld_floor_mean"]) * df["area_sqft"]
    df["pred_knn5"] = df["area_sqft"] * df["knn_5_ppsf"]

    # Direct price lookups (LOO)
    df["lookup_ua_price"] = df["ua_price_loo_mean"]
    df["lookup_ufc_price"] = df["ufc_price_loo_mean"]
    df["lookup_unit_price"] = df["unit_price_loo_mean"]
    df["lookup_bt_price"] = df["bt_price_loo_mean"]
    df["lookup_bld_price"] = df["bld_price_loo_mean"]

    df["area_x_floor"] = df["area_sqft"] * df["floor"]

    # Confidence features
    df["has_ua"] = (df["ua_ppsf_count"].fillna(0) >= 2).astype(int)
    df["has_unit"] = (df["unit_ppsf_count"].fillna(0) >= 1).astype(int)

# Encodings
for col in ["building", "district"]:
    le = LabelEncoder()
    le.fit(pd.concat([train[col], test[col]]).fillna("UNKNOWN"))
    train[f"{col}_code"] = le.transform(train[col].fillna("UNKNOWN"))
    test[f"{col}_code"] = le.transform(test[col].fillna("UNKNOWN"))

# ─────────────────────────────────────────────
# 7. FEATURES & TRAIN
# ─────────────────────────────────────────────
feature_cols = [
    # Core
    "area_sqft", "floor", "Public_Housing", "log_area",
    "wgs_lat", "wgs_lon", "flat_ord", "tower_num", "phase_num",
    "floor_desc", "floor_cat", "building_code", "district_code",

    # LOO target encodings (ppsf)
    "ua_ppsf_loo_mean", "ua_ppsf_count",
    "ufc_ppsf_loo_mean", "ufc_ppsf_count",
    "unit_ppsf_loo_mean", "unit_ppsf_median", "unit_ppsf_count",
    "bt_ppsf_loo_mean", "bt_ppsf_count",
    "bf_ppsf_loo_mean", "bf_ppsf_count",
    "bld_ppsf_loo_mean", "bld_ppsf_median", "bld_ppsf_count",
    "dist_ppsf_loo_mean", "dist_ppsf_median", "dist_ppsf_count",

    # Building
    "bld_area_mean", "bld_floor_mean", "bld_ppsf_floor_slope",

    # KNN
    "knn_3_price", "knn_5_price", "knn_10_price", "knn_20_price",
    "knn_3_ppsf", "knn_5_ppsf", "knn_10_ppsf", "knn_20_ppsf",

    # Hardcoded predictions (LOO)
    "pred_ua", "pred_ufc", "pred_unit", "pred_bt", "pred_bf",
    "pred_bld", "pred_dist", "pred_bld_floor_adj", "pred_knn5",

    # Direct price lookups (LOO)
    "lookup_ua_price", "lookup_ufc_price", "lookup_unit_price",
    "lookup_bt_price", "lookup_bld_price",

    # Derived
    "area_vs_bld", "floor_vs_bld", "floor_pct", "area_x_floor",
    "has_ua", "has_unit",

    # Spatial
    "dist_cbd", "dist_mtr", "dist_park", "dist_school", "dist_mall", "dist_hospital",
    "cnt_mtr_0.5km", "cnt_mtr_1.0km", "cnt_park_1.0km", "cnt_school_1.0km", "cnt_mall_1.0km",
]

feature_cols = [c for c in feature_cols if c in train.columns and c in test.columns]

X_train = train[feature_cols].values.astype(np.float32)
X_test = test[feature_cols].values.astype(np.float32)
y = train["price"].values.astype(np.float64)
y_log = np.log1p(y)

print(f"Features: {len(feature_cols)}")

# ─────────────────────────────────────────────
# 8. TRAIN MULTIPLE LightGBMs WITH DIFFERENT SEEDS
# ─────────────────────────────────────────────
N_FOLDS = 5
N_SEEDS = 5
print(f"\nTraining LightGBM ({N_FOLDS}-fold x {N_SEEDS} seeds)...")

all_oof = np.zeros(len(X_train))
all_test = np.zeros(len(X_test))

for seed_idx in range(N_SEEDS):
    seed = 42 + seed_idx * 111

    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    oof = np.zeros(len(X_train))
    test_preds = np.zeros(len(X_test))

    params = {
        "objective": "huber",
        "metric": "rmse",
        "learning_rate": 0.02,
        "num_leaves": 255,
        "min_child_samples": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.7,
        "reg_alpha": 0.05,
        "reg_lambda": 0.5,
        "n_estimators": 6000,
        "random_state": seed,
        "verbose": -1,
        "n_jobs": -1,
    }

    for fold, (tr_idx, val_idx) in enumerate(kf.split(X_train)):
        X_tr, X_val = X_train[tr_idx], X_train[val_idx]
        y_tr, y_val = y_log[tr_idx], y_log[val_idx]

        model = lgb.LGBMRegressor(**params)
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
                  callbacks=[lgb.early_stopping(200, verbose=False)])

        oof[val_idx] = model.predict(X_val)
        test_preds += model.predict(X_test) / N_FOLDS

    rmse = root_mean_squared_error(np.expm1(y_log), np.expm1(oof))
    print(f"  Seed {seed}: CV RMSE = {rmse:,.0f}")

    all_oof += oof / N_SEEDS
    all_test += test_preds / N_SEEDS

cv_rmse = root_mean_squared_error(np.expm1(y_log), np.expm1(all_oof))
print(f"\nEnsemble CV RMSE ({N_SEEDS} seeds): {cv_rmse:,.0f}")

predictions = np.expm1(all_test)
predictions = np.clip(predictions, 2000, 500000)

submission = pd.DataFrame({"id": test["id"].astype(int), "price": predictions.astype(int)})
submission.to_csv("my_submission.csv", index=False)
print(f"\nSubmission saved: {len(submission)} rows")
print(f"Price: ${predictions.min():,.0f} - ${predictions.max():,.0f}, mean ${predictions.mean():,.0f}")

# Feature importance (from last model)
imp = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
print("\nTop 20 features:")
for f, v in imp.head(20).items():
    print(f"  {f:30s} {v:6d}")
