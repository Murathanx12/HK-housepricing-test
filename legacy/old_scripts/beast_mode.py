"""
Hong Kong Rental Price Prediction — BEAST MODE
================================================
Multi-model stacking ensemble with aggressive target encoding.
Goal: Beat Helicea's $1,563 RMSE / 0.9918 R².
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

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("WARNING: xgboost not installed, skipping XGB models")

try:
    from catboost import CatBoostRegressor
    HAS_CB = True
except ImportError:
    HAS_CB = False
    print("WARNING: catboost not installed, skipping CatBoost models")

warnings.filterwarnings("ignore")
np.random.seed(42)

DATA_DIR = Path("./data")
N_FOLDS = 10  # More folds = better OOF estimates for stacking

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

print(f"Train: {len(train)}, Test: {len(test)}")

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

def get_estate(addr):
    """Extract estate name from address (text after first comma, before second)."""
    if pd.isna(addr): return "UNKNOWN"
    parts = addr.split(",")
    if len(parts) >= 2:
        return parts[1].strip()
    return "UNKNOWN"

for df in [train, test]:
    df["building"] = df["address"].apply(get_building)
    df["estate"] = df["address"].apply(get_estate)
    df["floor"] = pd.to_numeric(df["floor"], errors="coerce")
    df["Public_Housing"] = df["Public_Housing"].astype(int)
    df["flat_ord"] = df["Flat"].apply(lambda x: ord(str(x)[0].upper()) - 65 if pd.notna(x) and str(x)[0].isalpha() else -1)
    df["tower_num"] = pd.to_numeric(df["Tower"], errors="coerce").fillna(-1)
    df["phase_num"] = pd.to_numeric(df["Phase"], errors="coerce").fillna(-1)
    df["floor_desc"] = df["address"].apply(get_floor_desc)
    df["floor_cat"] = np.where(df["floor"] <= 7, 0, np.where(df["floor"] <= 14, 1, np.where(df["floor"] <= 22, 2, 3)))

    # Composite keys at various granularities
    df["unit_key"] = df["building"] + "|" + df["Tower"].fillna("X").astype(str) + "|" + df["Flat"].fillna("X")
    df["unit_floorcat"] = df["unit_key"] + "|" + df["floor_cat"].astype(str)
    df["area_bin"] = (df["area_sqft"] / 10).round() * 10
    df["unit_area"] = df["unit_key"] + "|" + df["area_bin"].astype(str)
    df["bld_tower"] = df["building"] + "|T" + df["Tower"].fillna("X").astype(str)
    df["bld_flat"] = df["building"] + "|F" + df["Flat"].fillna("X")

    # Fine-grained floor bins (every 3 floors)
    df["floor_bin3"] = (df["floor"] / 3).round() * 3
    df["unit_floorbin3"] = df["unit_key"] + "|" + df["floor_bin3"].astype(str)

train["price_per_sqft"] = train["price"] / train["area_sqft"]

# ─────────────────────────────────────────────
# 3. HIERARCHICAL STATS WITH SMOOTHING
# ─────────────────────────────────────────────
print("Computing hierarchical stats...")

global_price_mean = train["price"].mean()
global_ppsf_mean = train["price_per_sqft"].mean()
global_ppsf_median = train["price_per_sqft"].median()

def compute_stats(grp_col, prefix, min_count=1):
    stats = train.groupby(grp_col).agg(**{
        f"{prefix}_ppsf_mean": ("price_per_sqft", "mean"),
        f"{prefix}_ppsf_median": ("price_per_sqft", "median"),
        f"{prefix}_price_mean": ("price", "mean"),
        f"{prefix}_price_median": ("price", "median"),
        f"{prefix}_count": ("price", "count"),
    }).reset_index()
    if min_count > 1:
        n = stats[f"{prefix}_count"]
        smooth_w = n / (n + min_count)
        stats[f"{prefix}_ppsf_mean"] = smooth_w * stats[f"{prefix}_ppsf_mean"] + (1 - smooth_w) * global_ppsf_mean
        stats[f"{prefix}_price_mean"] = smooth_w * stats[f"{prefix}_price_mean"] + (1 - smooth_w) * global_price_mean
    return stats

# Finest to coarsest
ufb3_stats = compute_stats("unit_floorbin3", "ufb3", min_count=2)
ua_stats = compute_stats("unit_area", "ua", min_count=2)
ufc_stats = compute_stats("unit_floorcat", "ufc")
unit_stats = train.groupby("unit_key").agg(
    unit_ppsf_mean=("price_per_sqft", "mean"),
    unit_ppsf_median=("price_per_sqft", "median"),
    unit_ppsf_std=("price_per_sqft", "std"),
    unit_price_mean=("price", "mean"),
    unit_price_median=("price", "median"),
    unit_price_std=("price", "std"),
    unit_count=("price", "count"),
    unit_area_mean=("area_sqft", "mean"),
    unit_floor_mean=("floor", "mean"),
).reset_index()
unit_stats["unit_ppsf_std"] = unit_stats["unit_ppsf_std"].fillna(train["price_per_sqft"].std())
unit_stats["unit_price_std"] = unit_stats["unit_price_std"].fillna(train["price"].std())

bt_stats = compute_stats("bld_tower", "bt")
bf_stats = compute_stats("bld_flat", "bf")
estate_stats = compute_stats("estate", "est")

bld_stats = train.groupby("building").agg(
    bld_ppsf_mean=("price_per_sqft", "mean"),
    bld_ppsf_median=("price_per_sqft", "median"),
    bld_ppsf_std=("price_per_sqft", "std"),
    bld_ppsf_min=("price_per_sqft", "min"),
    bld_ppsf_max=("price_per_sqft", "max"),
    bld_ppsf_q25=("price_per_sqft", lambda x: x.quantile(0.25)),
    bld_ppsf_q75=("price_per_sqft", lambda x: x.quantile(0.75)),
    bld_price_mean=("price", "mean"),
    bld_price_median=("price", "median"),
    bld_area_mean=("area_sqft", "mean"),
    bld_area_min=("area_sqft", "min"),
    bld_area_max=("area_sqft", "max"),
    bld_floor_mean=("floor", "mean"),
    bld_floor_min=("floor", "min"),
    bld_floor_max=("floor", "max"),
    bld_count=("price", "count"),
).reset_index()
bld_stats["bld_ppsf_std"] = bld_stats["bld_ppsf_std"].fillna(train["price_per_sqft"].std())
bld_stats["bld_ppsf_iqr"] = bld_stats["bld_ppsf_q75"] - bld_stats["bld_ppsf_q25"]

def floor_slope(g):
    if len(g) < 3 or g["floor"].std() == 0: return 0.0
    return np.polyfit(g["floor"], g["price_per_sqft"], 1)[0]

bld_slope = train.groupby("building").apply(floor_slope, include_groups=False).reset_index()
bld_slope.columns = ["building", "bld_ppsf_floor_slope"]
bld_stats = bld_stats.merge(bld_slope, on="building", how="left").fillna(0)

dist_stats = compute_stats("district", "dist")

# Merge all stats
merge_list = [
    (ufb3_stats, "unit_floorbin3"), (ua_stats, "unit_area"),
    (ufc_stats, "unit_floorcat"), (unit_stats, "unit_key"),
    (bt_stats, "bld_tower"), (bf_stats, "bld_flat"),
    (estate_stats, "estate"),
    (bld_stats, "building"), (dist_stats, "district"),
]
for df in [train, test]:
    for stats, key in merge_list:
        merged = df[[key]].merge(stats, on=key, how="left")
        for col in stats.columns:
            if col != key:
                df[col] = merged[col].values

# Cascading fallback
chains = [
    ("ufb3", "ua", "ufc", "unit", "bt", "bf", "est", "bld", "dist"),
    ("ua", "ufc", "unit", "bt", "bf", "est", "bld", "dist"),
    ("ufc", "unit", "bt", "bf", "est", "bld", "dist"),
    ("unit", "bt", "bf", "est", "bld", "dist"),
    ("bt", "bld", "dist"), ("bf", "bld", "dist"),
    ("est", "bld", "dist"),
]
for chain in chains:
    for suffix in ["_ppsf_mean", "_ppsf_median", "_price_mean", "_price_median"]:
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
# 4. KNN (multiple feature spaces)
# ─────────────────────────────────────────────
print("Computing KNN features...")

# Standard KNN (lat, lon, area, floor)
scaler = StandardScaler()
X_knn_tr = scaler.fit_transform(train[["wgs_lat","wgs_lon","area_sqft","floor"]].fillna(0))
X_knn_te = scaler.transform(test[["wgs_lat","wgs_lon","area_sqft","floor"]].fillna(0))

for k in [3, 5, 10, 20, 50]:
    knn = KNeighborsRegressor(n_neighbors=k, weights="distance", n_jobs=-1)
    knn.fit(X_knn_tr, train["price"].values)
    train[f"knn_{k}_price"] = knn.predict(X_knn_tr)
    test[f"knn_{k}_price"] = knn.predict(X_knn_te)

    knn2 = KNeighborsRegressor(n_neighbors=k, weights="distance", n_jobs=-1)
    knn2.fit(X_knn_tr, train["price_per_sqft"].values)
    train[f"knn_{k}_ppsf"] = knn2.predict(X_knn_tr)
    test[f"knn_{k}_ppsf"] = knn2.predict(X_knn_te)

# Geo-only KNN (just location)
scaler2 = StandardScaler()
X_geo_tr = scaler2.fit_transform(train[["wgs_lat","wgs_lon"]].fillna(0))
X_geo_te = scaler2.transform(test[["wgs_lat","wgs_lon"]].fillna(0))

for k in [5, 20]:
    knn = KNeighborsRegressor(n_neighbors=k, weights="distance", n_jobs=-1)
    knn.fit(X_geo_tr, train["price_per_sqft"].values)
    train[f"geo_knn_{k}_ppsf"] = knn.predict(X_geo_tr)
    test[f"geo_knn_{k}_ppsf"] = knn.predict(X_geo_te)

# ─────────────────────────────────────────────
# 5. SPATIAL
# ─────────────────────────────────────────────
print("Spatial features...")
EARTH_R = 6371.0

def nearest_dist(df, plat, plon, name):
    tree = cKDTree(np.radians(np.column_stack([plat, plon])))
    d, _ = tree.query(np.radians(df[["wgs_lat","wgs_lon"]].values), k=1)
    df[f"dist_{name}"] = d * EARTH_R

def nearest_k_dist(df, plat, plon, name, k=3):
    tree = cKDTree(np.radians(np.column_stack([plat, plon])))
    d, _ = tree.query(np.radians(df[["wgs_lat","wgs_lon"]].values), k=k)
    df[f"dist_{name}_avg{k}"] = d.mean(axis=1) * EARTH_R

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
    nearest_k_dist(df, mtr["lat"].values, mtr["lon"].values, "mtr", k=3)
    count_within(df, mtr["lat"].values, mtr["lon"].values, "mtr", 0.5)
    count_within(df, mtr["lat"].values, mtr["lon"].values, "mtr", 1.0)
    count_within(df, mtr["lat"].values, mtr["lon"].values, "mtr", 2.0)
    nearest_dist(df, parks["centroid_lat"].values, parks["centroid_lon"].values, "park")
    count_within(df, parks["centroid_lat"].values, parks["centroid_lon"].values, "park", 0.5)
    count_within(df, parks["centroid_lat"].values, parks["centroid_lon"].values, "park", 1.0)
    nearest_dist(df, schools["lat"].values, schools["lon"].values, "school")
    count_within(df, schools["lat"].values, schools["lon"].values, "school", 0.5)
    count_within(df, schools["lat"].values, schools["lon"].values, "school", 1.0)
    nearest_dist(df, malls["lat"].values, malls["lon"].values, "mall")
    count_within(df, malls["lat"].values, malls["lon"].values, "mall", 0.5)
    count_within(df, malls["lat"].values, malls["lon"].values, "mall", 1.0)
    nearest_dist(df, hospitals["Latitude"].values, hospitals["Longitude"].values, "hospital")
    count_within(df, hospitals["Latitude"].values, hospitals["Longitude"].values, "hospital", 2.0)

# ─────────────────────────────────────────────
# 6. DERIVED FEATURES
# ─────────────────────────────────────────────
print("Derived features...")
for df in [train, test]:
    df["log_area"] = np.log1p(df["area_sqft"])
    df["sqrt_area"] = np.sqrt(df["area_sqft"])
    df["area_vs_bld"] = df["area_sqft"] - df["bld_area_mean"]
    df["floor_vs_bld"] = df["floor"] - df["bld_floor_mean"]
    fr = df["bld_floor_max"] - df["bld_floor_min"]
    df["floor_pct"] = np.where(fr > 0, (df["floor"] - df["bld_floor_min"]) / fr, 0.5)

    # Multiple prediction baselines
    df["pred_ufb3"] = df["area_sqft"] * df["ufb3_ppsf_mean"]
    df["pred_ua"] = df["area_sqft"] * df["ua_ppsf_mean"]
    df["pred_ufc"] = df["area_sqft"] * df["ufc_ppsf_mean"]
    df["pred_unit"] = df["area_sqft"] * df["unit_ppsf_mean"]
    df["pred_bt"] = df["area_sqft"] * df["bt_ppsf_mean"]
    df["pred_bf"] = df["area_sqft"] * df["bf_ppsf_mean"]
    df["pred_est"] = df["area_sqft"] * df["est_ppsf_mean"]
    df["pred_bld"] = df["area_sqft"] * df["bld_ppsf_mean"]
    df["pred_dist"] = df["area_sqft"] * df["dist_ppsf_mean"]
    df["pred_bld_floor_adj"] = df["pred_bld"] + df["bld_ppsf_floor_slope"] * (df["floor"] - df["bld_floor_mean"]) * df["area_sqft"]
    df["pred_knn5"] = df["area_sqft"] * df["knn_5_ppsf"]
    df["pred_knn10"] = df["area_sqft"] * df["knn_10_ppsf"]
    df["pred_geo5"] = df["area_sqft"] * df["geo_knn_5_ppsf"]

    # Unit-level adjustments
    df["area_vs_unit"] = df["area_sqft"] - df["unit_area_mean"].fillna(df["bld_area_mean"])
    df["floor_vs_unit"] = df["floor"] - df["unit_floor_mean"].fillna(df["bld_floor_mean"])

    df["ppsf_range_bld"] = df["bld_ppsf_max"] - df["bld_ppsf_min"]
    df["area_x_floor"] = df["area_sqft"] * df["floor"]
    df["log_area_x_floor"] = df["log_area"] * df["floor"]

    # Confidence-weighted prediction
    uc = df["unit_count"].fillna(0)
    df["pred_blended"] = np.where(
        uc >= 3,
        df["pred_unit"],
        np.where(uc >= 1, 0.7 * df["pred_unit"] + 0.3 * df["pred_bld"], df["pred_bld"])
    )

# Encodings
for col in ["building", "district", "estate"]:
    le = LabelEncoder()
    le.fit(pd.concat([train[col], test[col]]).fillna("UNKNOWN"))
    train[f"{col}_code"] = le.transform(train[col].fillna("UNKNOWN"))
    test[f"{col}_code"] = le.transform(test[col].fillna("UNKNOWN"))

# ─────────────────────────────────────────────
# 7. FEATURE LIST
# ─────────────────────────────────────────────
feature_cols = [
    # Core
    "area_sqft", "floor", "Public_Housing", "log_area", "sqrt_area",
    "wgs_lat", "wgs_lon", "flat_ord", "tower_num", "phase_num",
    "floor_desc", "floor_cat", "building_code", "district_code", "estate_code",

    # Finest granularity
    "ufb3_ppsf_mean", "ufb3_price_mean", "ufb3_count",
    "ua_ppsf_mean", "ua_price_mean", "ua_price_median", "ua_count",
    "ufc_ppsf_mean", "ufc_price_mean", "ufc_price_median", "ufc_count",

    # Unit
    "unit_ppsf_mean", "unit_ppsf_median", "unit_ppsf_std",
    "unit_price_mean", "unit_price_median", "unit_price_std", "unit_count",

    # Building+tower / Building+flat / Estate
    "bt_ppsf_mean", "bt_price_mean", "bt_count",
    "bf_ppsf_mean", "bf_price_mean", "bf_count",
    "est_ppsf_mean", "est_price_mean", "est_count",

    # Building
    "bld_ppsf_mean", "bld_ppsf_median", "bld_ppsf_std",
    "bld_ppsf_min", "bld_ppsf_max", "bld_ppsf_iqr",
    "bld_price_mean", "bld_price_median",
    "bld_area_mean", "bld_area_min", "bld_area_max",
    "bld_floor_mean", "bld_count", "bld_ppsf_floor_slope",

    # District
    "dist_ppsf_mean", "dist_ppsf_median", "dist_price_mean", "dist_count",

    # KNN
    "knn_3_price", "knn_5_price", "knn_10_price", "knn_20_price", "knn_50_price",
    "knn_3_ppsf", "knn_5_ppsf", "knn_10_ppsf", "knn_20_ppsf", "knn_50_ppsf",
    "geo_knn_5_ppsf", "geo_knn_20_ppsf",

    # Predictions
    "pred_ufb3", "pred_ua", "pred_ufc", "pred_unit", "pred_bt", "pred_bf",
    "pred_est", "pred_bld", "pred_dist", "pred_bld_floor_adj",
    "pred_knn5", "pred_knn10", "pred_geo5", "pred_blended",

    # Derived
    "area_vs_bld", "floor_vs_bld", "floor_pct", "ppsf_range_bld",
    "area_x_floor", "log_area_x_floor", "area_vs_unit", "floor_vs_unit",

    # Spatial
    "dist_cbd", "dist_mtr", "dist_mtr_avg3", "dist_park", "dist_school", "dist_mall", "dist_hospital",
    "cnt_mtr_0.5km", "cnt_mtr_1.0km", "cnt_mtr_2.0km",
    "cnt_park_0.5km", "cnt_park_1.0km",
    "cnt_school_0.5km", "cnt_school_1.0km",
    "cnt_mall_0.5km", "cnt_mall_1.0km",
    "cnt_hospital_2.0km",
]

# Filter to columns that actually exist
feature_cols = [c for c in feature_cols if c in train.columns and c in test.columns]

X_train = train[feature_cols].values.astype(np.float32)
X_test = test[feature_cols].values.astype(np.float32)
y = train["price"].values.astype(np.float64)
y_log = np.log1p(y)

print(f"Features: {len(feature_cols)}")

# ─────────────────────────────────────────────
# 8. MULTI-MODEL ENSEMBLE WITH STACKING
# ─────────────────────────────────────────────
print("\n=== STAGE 1: Base models ===")

kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

# Store OOF predictions for stacking
oof_preds = {}
test_preds_dict = {}

# --- LightGBM configs ---
lgb_configs = [
    {
        "name": "lgb_huber_v1",
        "params": {
            "objective": "huber", "metric": "rmse",
            "learning_rate": 0.015, "num_leaves": 255,
            "min_child_samples": 5, "subsample": 0.8,
            "colsample_bytree": 0.7, "reg_alpha": 0.05,
            "reg_lambda": 0.5, "n_estimators": 8000,
            "random_state": 42, "verbose": -1, "n_jobs": -1,
        }
    },
    {
        "name": "lgb_huber_v2",
        "params": {
            "objective": "huber", "metric": "rmse",
            "learning_rate": 0.01, "num_leaves": 511,
            "min_child_samples": 10, "subsample": 0.75,
            "colsample_bytree": 0.6, "reg_alpha": 0.1,
            "reg_lambda": 1.0, "n_estimators": 10000,
            "random_state": 123, "verbose": -1, "n_jobs": -1,
        }
    },
    {
        "name": "lgb_mse_v1",
        "params": {
            "objective": "regression", "metric": "rmse",
            "learning_rate": 0.02, "num_leaves": 255,
            "min_child_samples": 5, "subsample": 0.8,
            "colsample_bytree": 0.7, "reg_alpha": 0.05,
            "reg_lambda": 0.5, "n_estimators": 8000,
            "random_state": 42, "verbose": -1, "n_jobs": -1,
        }
    },
    {
        "name": "lgb_huber_v3",
        "params": {
            "objective": "huber", "metric": "rmse",
            "learning_rate": 0.02, "num_leaves": 127,
            "min_child_samples": 3, "subsample": 0.85,
            "colsample_bytree": 0.75, "reg_alpha": 0.02,
            "reg_lambda": 0.3, "n_estimators": 8000,
            "random_state": 777, "verbose": -1, "n_jobs": -1,
        }
    },
    {
        "name": "lgb_huber_v4",
        "params": {
            "objective": "huber", "metric": "rmse",
            "learning_rate": 0.025, "num_leaves": 383,
            "min_child_samples": 7, "subsample": 0.7,
            "colsample_bytree": 0.65, "reg_alpha": 0.08,
            "reg_lambda": 0.8, "n_estimators": 8000,
            "random_state": 2024, "verbose": -1, "n_jobs": -1,
        }
    },
]

for cfg in lgb_configs:
    name = cfg["name"]
    params = cfg["params"]
    print(f"\nTraining {name}...")

    oof = np.zeros(len(X_train))
    tpreds = np.zeros(len(X_test))

    for fold, (tr_idx, val_idx) in enumerate(kf.split(X_train)):
        X_tr, X_val = X_train[tr_idx], X_train[val_idx]
        y_tr, y_val = y_log[tr_idx], y_log[val_idx]

        model = lgb.LGBMRegressor(**params)
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
                  callbacks=[lgb.early_stopping(200, verbose=False)])

        oof[val_idx] = model.predict(X_val)
        tpreds += model.predict(X_test) / N_FOLDS

    rmse = root_mean_squared_error(np.expm1(y_log), np.expm1(oof))
    print(f"  {name} CV RMSE: {rmse:,.0f}")
    oof_preds[name] = oof
    test_preds_dict[name] = tpreds

# --- XGBoost ---
if HAS_XGB:
    xgb_configs = [
        {
            "name": "xgb_v1",
            "params": {
                "objective": "reg:pseudohubererror", "eval_metric": "rmse",
                "learning_rate": 0.02, "max_depth": 8,
                "min_child_weight": 5, "subsample": 0.8,
                "colsample_bytree": 0.7, "reg_alpha": 0.05,
                "reg_lambda": 0.5, "n_estimators": 8000,
                "random_state": 42, "verbosity": 0, "n_jobs": -1,
                "tree_method": "hist",
            }
        },
        {
            "name": "xgb_v2",
            "params": {
                "objective": "reg:squarederror", "eval_metric": "rmse",
                "learning_rate": 0.015, "max_depth": 10,
                "min_child_weight": 3, "subsample": 0.75,
                "colsample_bytree": 0.65, "reg_alpha": 0.1,
                "reg_lambda": 1.0, "n_estimators": 8000,
                "random_state": 123, "verbosity": 0, "n_jobs": -1,
                "tree_method": "hist",
            }
        },
    ]

    for cfg in xgb_configs:
        name = cfg["name"]
        params = cfg["params"]
        print(f"\nTraining {name}...")

        oof = np.zeros(len(X_train))
        tpreds = np.zeros(len(X_test))

        for fold, (tr_idx, val_idx) in enumerate(kf.split(X_train)):
            X_tr, X_val = X_train[tr_idx], X_train[val_idx]
            y_tr, y_val = y_log[tr_idx], y_log[val_idx]

            model = xgb.XGBRegressor(**params, early_stopping_rounds=200)
            model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)

            oof[val_idx] = model.predict(X_val)
            tpreds += model.predict(X_test) / N_FOLDS

        oof_real = np.expm1(np.clip(oof, -20, 20))
        rmse = root_mean_squared_error(np.expm1(y_log), oof_real)
        print(f"  {name} CV RMSE: {rmse:,.0f}")
        oof_preds[name] = oof
        test_preds_dict[name] = tpreds

# --- CatBoost ---
if HAS_CB:
    cb_configs = [
        {
            "name": "cb_v1",
            "params": {
                "loss_function": "Huber:delta=1.0",
                "eval_metric": "RMSE",
                "learning_rate": 0.03,
                "depth": 8,
                "l2_leaf_reg": 3,
                "iterations": 8000,
                "random_seed": 42,
                "verbose": 0,
            }
        },
        {
            "name": "cb_v2",
            "params": {
                "loss_function": "RMSE",
                "eval_metric": "RMSE",
                "learning_rate": 0.02,
                "depth": 10,
                "l2_leaf_reg": 5,
                "iterations": 8000,
                "random_seed": 123,
                "verbose": 0,
            }
        },
    ]

    for cfg in cb_configs:
        name = cfg["name"]
        params = cfg["params"]
        print(f"\nTraining {name}...")

        oof = np.zeros(len(X_train))
        tpreds = np.zeros(len(X_test))

        for fold, (tr_idx, val_idx) in enumerate(kf.split(X_train)):
            X_tr, X_val = X_train[tr_idx], X_train[val_idx]
            y_tr, y_val = y_log[tr_idx], y_log[val_idx]

            model = CatBoostRegressor(**params)
            model.fit(X_tr, y_tr, eval_set=(X_val, y_val),
                      early_stopping_rounds=200, verbose=0)

            oof[val_idx] = model.predict(X_val)
            tpreds += model.predict(X_test) / N_FOLDS

        rmse = root_mean_squared_error(np.expm1(y_log), np.expm1(oof))
        print(f"  {name} CV RMSE: {rmse:,.0f}")
        oof_preds[name] = oof
        test_preds_dict[name] = tpreds

# ─────────────────────────────────────────────
# 9. STACKING (Level 2)
# ─────────────────────────────────────────────
print(f"\n=== STAGE 2: Stacking ({len(oof_preds)} base models) ===")

# Build stacking features
stack_train = np.column_stack([oof_preds[k] for k in sorted(oof_preds.keys())])
stack_test = np.column_stack([test_preds_dict[k] for k in sorted(test_preds_dict.keys())])

# Add key original features to stacking
key_feats = ["area_sqft", "floor", "unit_ppsf_mean", "bld_ppsf_mean", "unit_count", "pred_unit", "pred_bld"]
key_idx = [feature_cols.index(f) for f in key_feats if f in feature_cols]
stack_train = np.column_stack([stack_train, X_train[:, key_idx]])
stack_test = np.column_stack([stack_test, X_test[:, key_idx]])

print(f"Stacking features: {stack_train.shape[1]}")

# Meta-learner: Ridge regression (simple to avoid overfitting)
from sklearn.linear_model import Ridge

kf2 = KFold(n_splits=5, shuffle=True, random_state=99)
meta_oof = np.zeros(len(stack_train))
meta_test = np.zeros(len(stack_test))

for fold, (tr_idx, val_idx) in enumerate(kf2.split(stack_train)):
    meta = Ridge(alpha=1.0)
    meta.fit(stack_train[tr_idx], y_log[tr_idx])
    meta_oof[val_idx] = meta.predict(stack_train[val_idx])
    meta_test += meta.predict(stack_test) / 5

meta_rmse = root_mean_squared_error(np.expm1(y_log), np.expm1(meta_oof))
print(f"Meta-learner (Ridge) CV RMSE: {meta_rmse:,.0f}")

# Also try LightGBM as meta-learner
meta_oof_lgb = np.zeros(len(stack_train))
meta_test_lgb = np.zeros(len(stack_test))

for fold, (tr_idx, val_idx) in enumerate(kf2.split(stack_train)):
    meta = lgb.LGBMRegressor(
        objective="huber", learning_rate=0.05, num_leaves=15,
        min_child_samples=20, n_estimators=1000, reg_lambda=1.0,
        verbose=-1, n_jobs=-1, random_state=42
    )
    meta.fit(stack_train[tr_idx], y_log[tr_idx],
             eval_set=[(stack_train[val_idx], y_log[val_idx])],
             callbacks=[lgb.early_stopping(50, verbose=False)])
    meta_oof_lgb[val_idx] = meta.predict(stack_train[val_idx])
    meta_test_lgb += meta.predict(stack_test) / 5

meta_rmse_lgb = root_mean_squared_error(np.expm1(y_log), np.expm1(meta_oof_lgb))
print(f"Meta-learner (LGB) CV RMSE: {meta_rmse_lgb:,.0f}")

# ─────────────────────────────────────────────
# 10. OPTIMAL BLEND
# ─────────────────────────────────────────────
print("\n=== STAGE 3: Finding optimal blend ===")

# Simple average of all base models
avg_oof = np.mean([oof_preds[k] for k in oof_preds], axis=0)
avg_test = np.mean([test_preds_dict[k] for k in test_preds_dict], axis=0)
avg_rmse = root_mean_squared_error(np.expm1(y_log), np.expm1(avg_oof))
print(f"Simple average CV RMSE: {avg_rmse:,.0f}")

# Try different blend weights between avg, ridge-meta, lgb-meta
best_rmse = float("inf")
best_w = None
for w1 in np.arange(0, 1.05, 0.05):
    for w2 in np.arange(0, 1.05 - w1, 0.05):
        w3 = 1 - w1 - w2
        if w3 < 0: continue
        blend = w1 * avg_oof + w2 * meta_oof + w3 * meta_oof_lgb
        rmse = root_mean_squared_error(np.expm1(y_log), np.expm1(blend))
        if rmse < best_rmse:
            best_rmse = rmse
            best_w = (w1, w2, w3)

print(f"Best blend weights: avg={best_w[0]:.2f}, ridge={best_w[1]:.2f}, lgb={best_w[2]:.2f}")
print(f"Best blend CV RMSE: {best_rmse:,.0f}")

final_log = best_w[0] * avg_test + best_w[1] * meta_test + best_w[2] * meta_test_lgb
final_oof_log = best_w[0] * avg_oof + best_w[1] * meta_oof + best_w[2] * meta_oof_lgb

# ─────────────────────────────────────────────
# 11. POST-PROCESSING: Direct lookup blending
# ─────────────────────────────────────────────
print("\n=== STAGE 4: Post-processing ===")

predictions = np.expm1(final_log)

# For test rows with high-confidence unit matches, blend toward historical median
test_unit_counts = test["unit_key"].map(train.groupby("unit_key")["price"].count()).fillna(0).values
test_unit_medians = test["unit_key"].map(train.groupby("unit_key")["price"].median()).fillna(0).values
test_unit_means = test["unit_key"].map(train.groupby("unit_key")["price"].mean()).fillna(0).values

# Also try unit_area level (same unit, same area bin)
test_ua_medians = test["unit_area"].map(train.groupby("unit_area")["price"].median()).fillna(0).values
test_ua_counts = test["unit_area"].map(train.groupby("unit_area")["price"].count()).fillna(0).values

# Blend model prediction with direct lookup based on confidence
for i in range(len(predictions)):
    ua_c = test_ua_counts[i]
    u_c = test_unit_counts[i]

    if ua_c >= 3:
        # Very high confidence: same unit + same area bin, blend heavily
        alpha = min(0.35, ua_c * 0.05)
        predictions[i] = (1 - alpha) * predictions[i] + alpha * test_ua_medians[i]
    elif u_c >= 5:
        alpha = min(0.20, u_c * 0.03)
        predictions[i] = (1 - alpha) * predictions[i] + alpha * test_unit_medians[i]

predictions = np.clip(predictions, 2000, 500000)

# Final CV estimate
final_cv = root_mean_squared_error(np.expm1(y_log), np.expm1(final_oof_log))
print(f"\nFinal CV RMSE (before post-processing): {final_cv:,.0f}")

submission = pd.DataFrame({"id": test["id"].astype(int), "price": predictions.astype(int)})
submission.to_csv("my_submission.csv", index=False)
print(f"\nSubmission saved: {len(submission)} rows")
print(f"Price: ${predictions.min():,.0f} - ${predictions.max():,.0f}, mean ${predictions.mean():,.0f}")

# Show per-model performance
print("\n=== Model Summary ===")
for name in sorted(oof_preds.keys()):
    rmse = root_mean_squared_error(np.expm1(y_log), np.expm1(oof_preds[name]))
    print(f"  {name:20s}  CV RMSE: {rmse:,.0f}")
print(f"  {'Simple Average':20s}  CV RMSE: {avg_rmse:,.0f}")
print(f"  {'Ridge Meta':20s}  CV RMSE: {meta_rmse:,.0f}")
print(f"  {'LGB Meta':20s}  CV RMSE: {meta_rmse_lgb:,.0f}")
print(f"  {'Best Blend':20s}  CV RMSE: {best_rmse:,.0f}")
