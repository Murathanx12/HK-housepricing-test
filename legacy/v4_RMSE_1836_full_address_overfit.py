"""
Hong Kong Rental Price Prediction — v5
========================================
Back to v3 base (scored $1,746) + unit_area stats.
NO full address stats (caused overfitting / outlier blowup).
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
N_FOLDS = 5

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
    df["floor"] = pd.to_numeric(df["floor"], errors="coerce")
    df["Public_Housing"] = df["Public_Housing"].astype(int)
    df["flat_ord"] = df["Flat"].apply(lambda x: ord(str(x)[0].upper()) - 65 if pd.notna(x) and str(x)[0].isalpha() else -1)
    df["tower_num"] = pd.to_numeric(df["Tower"], errors="coerce").fillna(-1)
    df["phase_num"] = pd.to_numeric(df["Phase"], errors="coerce").fillna(-1)
    df["floor_desc"] = df["address"].apply(get_floor_desc)
    df["floor_cat"] = np.where(df["floor"] <= 10, 0, np.where(df["floor"] <= 17, 1, 2))

    # Keys
    df["unit_key"] = df["building"] + "|" + df["Tower"].fillna("X").astype(str) + "|" + df["Flat"].fillna("X")
    df["unit_floorcat"] = df["unit_key"] + "|" + df["floor_cat"].astype(str)
    df["area_bin"] = (df["area_sqft"] / 10).round() * 10
    df["unit_area"] = df["unit_key"] + "|" + df["area_bin"].astype(str)
    df["bld_tower"] = df["building"] + "|T" + df["Tower"].fillna("X").astype(str)
    df["bld_flat"] = df["building"] + "|F" + df["Flat"].fillna("X")

train["price_per_sqft"] = train["price"] / train["area_sqft"]

# ─────────────────────────────────────────────
# 3. HIERARCHICAL STATS (with smoothing for specific levels)
# ─────────────────────────────────────────────
print("Computing stats...")

global_price_mean = train["price"].mean()
global_ppsf_mean = train["price_per_sqft"].mean()

def compute_stats(grp_col, prefix, min_count=1):
    """Compute stats with optional smoothing for small groups."""
    stats = train.groupby(grp_col).agg(**{
        f"{prefix}_ppsf_mean": ("price_per_sqft", "mean"),
        f"{prefix}_ppsf_median": ("price_per_sqft", "median"),
        f"{prefix}_price_mean": ("price", "mean"),
        f"{prefix}_price_median": ("price", "median"),
        f"{prefix}_count": ("price", "count"),
    }).reset_index()
    # Smooth means toward global for small groups
    if min_count > 1:
        n = stats[f"{prefix}_count"]
        smooth_w = n / (n + min_count)
        stats[f"{prefix}_ppsf_mean"] = smooth_w * stats[f"{prefix}_ppsf_mean"] + (1 - smooth_w) * global_ppsf_mean
        stats[f"{prefix}_price_mean"] = smooth_w * stats[f"{prefix}_price_mean"] + (1 - smooth_w) * global_price_mean
    return stats

# unit_area: smoothed (some bins have few observations)
ua_stats = compute_stats("unit_area", "ua", min_count=2)

# unit_floorcat: the sweet spot from v3
ufc_stats = compute_stats("unit_floorcat", "ufc")

# Unit
unit_stats = train.groupby("unit_key").agg(
    unit_ppsf_mean=("price_per_sqft", "mean"),
    unit_ppsf_median=("price_per_sqft", "median"),
    unit_ppsf_std=("price_per_sqft", "std"),
    unit_price_mean=("price", "mean"),
    unit_price_median=("price", "median"),
    unit_count=("price", "count"),
).reset_index()
unit_stats["unit_ppsf_std"] = unit_stats["unit_ppsf_std"].fillna(train["price_per_sqft"].std())

# Building+tower
bt_stats = compute_stats("bld_tower", "bt")

# Building+flat
bf_stats = compute_stats("bld_flat", "bf")

# Building
bld_stats = train.groupby("building").agg(
    bld_ppsf_mean=("price_per_sqft", "mean"),
    bld_ppsf_median=("price_per_sqft", "median"),
    bld_ppsf_std=("price_per_sqft", "std"),
    bld_ppsf_min=("price_per_sqft", "min"),
    bld_ppsf_max=("price_per_sqft", "max"),
    bld_price_mean=("price", "mean"),
    bld_price_median=("price", "median"),
    bld_area_mean=("area_sqft", "mean"),
    bld_floor_mean=("floor", "mean"),
    bld_floor_min=("floor", "min"),
    bld_floor_max=("floor", "max"),
    bld_count=("price", "count"),
).reset_index()
bld_stats["bld_ppsf_std"] = bld_stats["bld_ppsf_std"].fillna(train["price_per_sqft"].std())

def floor_slope(g):
    if len(g) < 3 or g["floor"].std() == 0: return 0.0
    return np.polyfit(g["floor"], g["price_per_sqft"], 1)[0]

bld_slope = train.groupby("building").apply(floor_slope, include_groups=False).reset_index()
bld_slope.columns = ["building", "bld_ppsf_floor_slope"]
bld_stats = bld_stats.merge(bld_slope, on="building", how="left").fillna(0)

# District
dist_stats = compute_stats("district", "dist")

# Merge all
merge_list = [
    (ua_stats, "unit_area"), (ufc_stats, "unit_floorcat"), (unit_stats, "unit_key"),
    (bt_stats, "bld_tower"), (bf_stats, "bld_flat"),
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
    ("ua", "ufc", "unit", "bt", "bf", "bld", "dist"),
    ("ufc", "unit", "bt", "bf", "bld", "dist"),
    ("unit", "bt", "bf", "bld", "dist"),
    ("bt", "bld", "dist"), ("bf", "bld", "dist"),
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
# 4. KNN
# ─────────────────────────────────────────────
print("Computing KNN...")
scaler = StandardScaler()
X_knn_tr = scaler.fit_transform(train[["wgs_lat","wgs_lon","area_sqft","floor"]].fillna(0))
X_knn_te = scaler.transform(test[["wgs_lat","wgs_lon","area_sqft","floor"]].fillna(0))

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
# 6. DERIVED
# ─────────────────────────────────────────────
print("Derived features...")
for df in [train, test]:
    df["log_area"] = np.log1p(df["area_sqft"])
    df["area_vs_bld"] = df["area_sqft"] - df["bld_area_mean"]
    df["floor_vs_bld"] = df["floor"] - df["bld_floor_mean"]
    fr = df["bld_floor_max"] - df["bld_floor_min"]
    df["floor_pct"] = np.where(fr > 0, (df["floor"] - df["bld_floor_min"]) / fr, 0.5)

    df["pred_ua"] = df["area_sqft"] * df["ua_ppsf_mean"]
    df["pred_ufc"] = df["area_sqft"] * df["ufc_ppsf_mean"]
    df["pred_unit"] = df["area_sqft"] * df["unit_ppsf_mean"]
    df["pred_bt"] = df["area_sqft"] * df["bt_ppsf_mean"]
    df["pred_bf"] = df["area_sqft"] * df["bf_ppsf_mean"]
    df["pred_bld"] = df["area_sqft"] * df["bld_ppsf_mean"]
    df["pred_dist"] = df["area_sqft"] * df["dist_ppsf_mean"]
    df["pred_bld_floor_adj"] = df["pred_bld"] + df["bld_ppsf_floor_slope"] * (df["floor"] - df["bld_floor_mean"]) * df["area_sqft"]
    df["pred_knn5"] = df["area_sqft"] * df["knn_5_ppsf"]

    df["ppsf_range_bld"] = df["bld_ppsf_max"] - df["bld_ppsf_min"]
    df["area_x_floor"] = df["area_sqft"] * df["floor"]

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

    # Unit+area (new, smoothed)
    "ua_ppsf_mean", "ua_price_mean", "ua_price_median", "ua_count",

    # Unit+floorcat (v3 sweet spot)
    "ufc_ppsf_mean", "ufc_price_mean", "ufc_price_median", "ufc_count",

    # Unit
    "unit_ppsf_mean", "unit_ppsf_median", "unit_ppsf_std",
    "unit_price_mean", "unit_price_median", "unit_count",

    # Building+tower / Building+flat
    "bt_ppsf_mean", "bt_price_mean", "bt_count",
    "bf_ppsf_mean", "bf_price_mean", "bf_count",

    # Building
    "bld_ppsf_mean", "bld_ppsf_median", "bld_ppsf_std",
    "bld_ppsf_min", "bld_ppsf_max",
    "bld_price_mean", "bld_price_median",
    "bld_area_mean", "bld_floor_mean", "bld_count", "bld_ppsf_floor_slope",

    # District
    "dist_ppsf_mean", "dist_ppsf_median", "dist_price_mean", "dist_count",

    # KNN
    "knn_3_price", "knn_5_price", "knn_10_price", "knn_20_price",
    "knn_3_ppsf", "knn_5_ppsf", "knn_10_ppsf", "knn_20_ppsf",

    # Predictions
    "pred_ua", "pred_ufc", "pred_unit", "pred_bt", "pred_bf",
    "pred_bld", "pred_dist", "pred_bld_floor_adj", "pred_knn5",

    # Derived
    "area_vs_bld", "floor_vs_bld", "floor_pct", "ppsf_range_bld", "area_x_floor",

    # Spatial
    "dist_cbd", "dist_mtr", "dist_park", "dist_school", "dist_mall", "dist_hospital",
    "cnt_mtr_0.5km", "cnt_mtr_1.0km", "cnt_park_1.0km", "cnt_school_1.0km", "cnt_mall_1.0km",
]

X_train = train[feature_cols].values.astype(np.float32)
X_test = test[feature_cols].values.astype(np.float32)
y = train["price"].values.astype(np.float64)
y_log = np.log1p(y)

print(f"Features: {len(feature_cols)}")
print("\nTraining LightGBM (5-fold)...")

kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
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
    "random_state": 42,
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

    rmse = root_mean_squared_error(np.expm1(y_val), np.expm1(oof[val_idx]))
    print(f"  Fold {fold+1}: RMSE = {rmse:,.0f} (iters: {model.best_iteration_})")

cv_rmse = root_mean_squared_error(np.expm1(y_log), np.expm1(oof))
print(f"\nOverall CV RMSE: {cv_rmse:,.0f}")

predictions = np.expm1(test_preds)
predictions = np.clip(predictions, 2000, 500000)

submission = pd.DataFrame({"id": test["id"].astype(int), "price": predictions.astype(int)})
submission.to_csv("my_submission.csv", index=False)
print(f"\nSubmission saved: {len(submission)} rows")
print(f"Price: ${predictions.min():,.0f} - ${predictions.max():,.0f}, mean ${predictions.mean():,.0f}")

imp = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
print("\nTop 15 features:")
for f, v in imp.head(15).items():
    print(f"  {f:30s} {v:6d}")
