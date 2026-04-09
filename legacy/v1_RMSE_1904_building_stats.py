"""
Hong Kong Rental Price Prediction — v3
=======================================
Key improvements over v2:
  - KNN price lookup: for each test apt, find K most similar training apts and use their prices
  - Building+Tower+Flat granularity for sub-building matching
  - Floor premium modeling within buildings
  - Building-floor regression slope as feature
  - Better handling of categorical features
"""

import pandas as pd
import numpy as np
import warnings
from pathlib import Path
from scipy.spatial import cKDTree
from sklearn.model_selection import KFold
from sklearn.metrics import root_mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
import lightgbm as lgb

warnings.filterwarnings("ignore")
np.random.seed(42)

DATA_DIR = Path("./data")
N_FOLDS = 5

# ─────────────────────────────────────────────
# 1. LOAD DATA
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
# 2. PARSE ADDRESS COMPONENTS
# ─────────────────────────────────────────────
print("Parsing addresses...")

def get_building(addr):
    if pd.isna(addr): return "UNKNOWN"
    return addr.split(",")[0].strip()

for df in [train, test]:
    df["building"] = df["address"].apply(get_building)
    df["floor"] = pd.to_numeric(df["floor"], errors="coerce")
    df["Public_Housing"] = df["Public_Housing"].astype(int)

    # Extract flat letter as ordinal (A=0, B=1, etc.)
    df["flat_ord"] = df["Flat"].apply(lambda x: ord(str(x)[0].upper()) - 65 if pd.notna(x) and str(x)[0].isalpha() else -1)
    # Tower as numeric
    df["tower_num"] = pd.to_numeric(df["Tower"], errors="coerce").fillna(-1)
    # Phase as numeric
    df["phase_num"] = pd.to_numeric(df["Phase"], errors="coerce").fillna(-1)

train["price_per_sqft"] = train["price"] / train["area_sqft"]

# ─────────────────────────────────────────────
# 3. BUILDING-LEVEL STATS
# ─────────────────────────────────────────────
print("Computing building stats...")

bld_stats = train.groupby("building").agg(
    bld_ppsf_mean=("price_per_sqft", "mean"),
    bld_ppsf_median=("price_per_sqft", "median"),
    bld_ppsf_std=("price_per_sqft", "std"),
    bld_ppsf_min=("price_per_sqft", "min"),
    bld_ppsf_max=("price_per_sqft", "max"),
    bld_price_mean=("price", "mean"),
    bld_price_median=("price", "median"),
    bld_price_std=("price", "std"),
    bld_area_mean=("area_sqft", "mean"),
    bld_area_std=("area_sqft", "std"),
    bld_floor_mean=("floor", "mean"),
    bld_floor_min=("floor", "min"),
    bld_floor_max=("floor", "max"),
    bld_count=("price", "count"),
).reset_index()

bld_stats["bld_ppsf_std"] = bld_stats["bld_ppsf_std"].fillna(train["price_per_sqft"].std())
bld_stats["bld_price_std"] = bld_stats["bld_price_std"].fillna(train["price"].std())
bld_stats["bld_area_std"] = bld_stats["bld_area_std"].fillna(train["area_sqft"].std())

# Floor premium per building (regression slope: price = a + b*floor)
def floor_slope(g):
    if len(g) < 3 or g["floor"].std() == 0:
        return 0.0
    return np.polyfit(g["floor"], g["price"], 1)[0]

bld_floor_slope = train.groupby("building").apply(floor_slope, include_groups=False).reset_index()
bld_floor_slope.columns = ["building", "bld_floor_slope"]
bld_stats = bld_stats.merge(bld_floor_slope, on="building", how="left")
bld_stats["bld_floor_slope"] = bld_stats["bld_floor_slope"].fillna(0)

for df in [train, test]:
    df_merged = df.merge(bld_stats, on="building", how="left")
    for col in bld_stats.columns:
        if col != "building":
            df[col] = df_merged[col]

# Fill missing for test rows without building match
global_defaults = {
    "bld_ppsf_mean": train["price_per_sqft"].mean(),
    "bld_ppsf_median": train["price_per_sqft"].median(),
    "bld_ppsf_std": train["price_per_sqft"].std(),
    "bld_ppsf_min": train["price_per_sqft"].min(),
    "bld_ppsf_max": train["price_per_sqft"].max(),
    "bld_price_mean": train["price"].mean(),
    "bld_price_median": train["price"].median(),
    "bld_price_std": train["price"].std(),
    "bld_area_mean": train["area_sqft"].mean(),
    "bld_area_std": train["area_sqft"].std(),
    "bld_floor_mean": train["floor"].mean(),
    "bld_floor_min": 1,
    "bld_floor_max": 50,
    "bld_count": 0,
    "bld_floor_slope": 0,
}
for col, val in global_defaults.items():
    test[col] = test[col].fillna(val)

# ─────────────────────────────────────────────
# 4. BUILDING+TOWER STATS (finer granularity)
# ─────────────────────────────────────────────
print("Computing building+tower stats...")

train["bld_tower"] = train["building"] + "_T" + train["Tower"].fillna("X").astype(str)
test["bld_tower"] = test["building"] + "_T" + test["Tower"].fillna("X").astype(str)

bt_stats = train.groupby("bld_tower").agg(
    bt_ppsf_mean=("price_per_sqft", "mean"),
    bt_price_mean=("price", "mean"),
    bt_count=("price", "count"),
).reset_index()

for df in [train, test]:
    df_merged = df.merge(bt_stats, on="bld_tower", how="left")
    for col in bt_stats.columns:
        if col != "bld_tower":
            df[col] = df_merged[col]

# Fall back to building-level stats when tower stats unavailable
for df in [train, test]:
    df["bt_ppsf_mean"] = df["bt_ppsf_mean"].fillna(df["bld_ppsf_mean"])
    df["bt_price_mean"] = df["bt_price_mean"].fillna(df["bld_price_mean"])
    df["bt_count"] = df["bt_count"].fillna(0)

# ─────────────────────────────────────────────
# 5. DISTRICT-LEVEL STATS
# ─────────────────────────────────────────────
print("Computing district stats...")

dist_stats = train.groupby("district").agg(
    dist_ppsf_mean=("price_per_sqft", "mean"),
    dist_ppsf_median=("price_per_sqft", "median"),
    dist_price_mean=("price", "mean"),
    dist_price_median=("price", "median"),
    dist_count=("price", "count"),
).reset_index()

for df in [train, test]:
    df_merged = df.merge(dist_stats, on="district", how="left")
    for col in dist_stats.columns:
        if col != "district":
            df[col] = df_merged[col]
            df[col] = df[col].fillna(df[col].median())

# ─────────────────────────────────────────────
# 6. KNN PRICE FEATURES
# ─────────────────────────────────────────────
print("Computing KNN price features...")

# Build KNN on (lat, lon, area_sqft_normalized, floor_normalized)
from sklearn.preprocessing import StandardScaler

knn_features = ["wgs_lat", "wgs_lon", "area_sqft", "floor"]
scaler = StandardScaler()
X_knn_train = scaler.fit_transform(train[knn_features].fillna(0))
X_knn_test = scaler.transform(test[knn_features].fillna(0))

for k in [3, 5, 10, 20]:
    knn = KNeighborsRegressor(n_neighbors=k, weights="distance", n_jobs=-1)
    knn.fit(X_knn_train, train["price"].values)
    train[f"knn_{k}_price"] = knn.predict(X_knn_train)
    test[f"knn_{k}_price"] = knn.predict(X_knn_test)

    knn_ppsf = KNeighborsRegressor(n_neighbors=k, weights="distance", n_jobs=-1)
    knn_ppsf.fit(X_knn_train, train["price_per_sqft"].values)
    train[f"knn_{k}_ppsf"] = knn_ppsf.predict(X_knn_train)
    test[f"knn_{k}_ppsf"] = knn_ppsf.predict(X_knn_test)

# ─────────────────────────────────────────────
# 7. SPATIAL FEATURES
# ─────────────────────────────────────────────
print("Engineering spatial features...")

EARTH_R = 6371.0

def add_nearest_dist(df, poi_lat, poi_lon, name):
    coords = np.radians(np.column_stack([poi_lat, poi_lon]))
    tree = cKDTree(coords)
    pts = np.radians(df[["wgs_lat", "wgs_lon"]].values)
    d, _ = tree.query(pts, k=1)
    df[f"dist_{name}"] = d * EARTH_R

def add_count_within(df, poi_lat, poi_lon, name, r_km):
    coords = np.radians(np.column_stack([poi_lat, poi_lon]))
    tree = cKDTree(coords)
    pts = np.radians(df[["wgs_lat", "wgs_lon"]].values)
    counts = tree.query_ball_point(pts, r=r_km / EARTH_R)
    df[f"cnt_{name}_{r_km}km"] = [len(c) for c in counts]

def haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return 2 * EARTH_R * np.arcsin(np.sqrt(a))

for df in [train, test]:
    df["dist_cbd"] = haversine(df["wgs_lat"].values, df["wgs_lon"].values,
                               cbd["lat"].values[0], cbd["lon"].values[0])
    add_nearest_dist(df, mtr["lat"].values, mtr["lon"].values, "mtr")
    add_count_within(df, mtr["lat"].values, mtr["lon"].values, "mtr", 0.5)
    add_count_within(df, mtr["lat"].values, mtr["lon"].values, "mtr", 1.0)
    add_nearest_dist(df, parks["centroid_lat"].values, parks["centroid_lon"].values, "park")
    add_count_within(df, parks["centroid_lat"].values, parks["centroid_lon"].values, "park", 1.0)
    add_nearest_dist(df, schools["lat"].values, schools["lon"].values, "school")
    add_count_within(df, schools["lat"].values, schools["lon"].values, "school", 1.0)
    add_nearest_dist(df, malls["lat"].values, malls["lon"].values, "mall")
    add_count_within(df, malls["lat"].values, malls["lon"].values, "mall", 1.0)
    add_nearest_dist(df, hospitals["Latitude"].values, hospitals["Longitude"].values, "hospital")

# ─────────────────────────────────────────────
# 8. DERIVED FEATURES
# ─────────────────────────────────────────────
print("Creating derived features...")

for df in [train, test]:
    df["log_area"] = np.log1p(df["area_sqft"])

    # Deviation from building averages
    df["area_vs_bld"] = df["area_sqft"] - df["bld_area_mean"]
    df["floor_vs_bld"] = df["floor"] - df["bld_floor_mean"]

    # Floor position within building (0=lowest, 1=highest)
    floor_range = df["bld_floor_max"] - df["bld_floor_min"]
    df["floor_pct_in_bld"] = np.where(
        floor_range > 0,
        (df["floor"] - df["bld_floor_min"]) / floor_range,
        0.5
    )

    # Predicted price from building ppsf * area
    df["pred_bld_ppsf"] = df["area_sqft"] * df["bld_ppsf_mean"]
    df["pred_bt_ppsf"] = df["area_sqft"] * df["bt_ppsf_mean"]
    df["pred_dist_ppsf"] = df["area_sqft"] * df["dist_ppsf_mean"]

    # Floor-adjusted building prediction
    df["pred_bld_floor_adj"] = df["pred_bld_ppsf"] + df["bld_floor_slope"] * (df["floor"] - df["bld_floor_mean"])

    # Interactions
    df["area_x_floor"] = df["area_sqft"] * df["floor"]
    df["area_x_dist_cbd"] = df["area_sqft"] * df["dist_cbd"]

    # KNN-derived predictions
    df["pred_knn5_ppsf"] = df["area_sqft"] * df["knn_5_ppsf"]
    df["pred_knn10_ppsf"] = df["area_sqft"] * df["knn_10_ppsf"]

# ─────────────────────────────────────────────
# 9. ENCODE CATEGORICALS
# ─────────────────────────────────────────────
print("Encoding categoricals...")

from sklearn.preprocessing import LabelEncoder

for col in ["building", "district"]:
    le = LabelEncoder()
    combined = pd.concat([train[col], test[col]]).fillna("UNKNOWN")
    le.fit(combined)
    train[f"{col}_code"] = le.transform(train[col].fillna("UNKNOWN"))
    test[f"{col}_code"] = le.transform(test[col].fillna("UNKNOWN"))

# ─────────────────────────────────────────────
# 10. FEATURE LIST
# ─────────────────────────────────────────────
feature_cols = [
    # Core
    "area_sqft", "floor", "Public_Housing", "log_area",
    "wgs_lat", "wgs_lon",
    "flat_ord", "tower_num", "phase_num",

    # Encoded
    "building_code", "district_code",

    # Building-level
    "bld_ppsf_mean", "bld_ppsf_median", "bld_ppsf_std",
    "bld_ppsf_min", "bld_ppsf_max",
    "bld_price_mean", "bld_price_median", "bld_price_std",
    "bld_area_mean", "bld_area_std",
    "bld_floor_mean", "bld_floor_min", "bld_floor_max",
    "bld_count", "bld_floor_slope",

    # Building+tower
    "bt_ppsf_mean", "bt_price_mean", "bt_count",

    # District-level
    "dist_ppsf_mean", "dist_ppsf_median",
    "dist_price_mean", "dist_price_median", "dist_count",

    # KNN features
    "knn_3_price", "knn_5_price", "knn_10_price", "knn_20_price",
    "knn_3_ppsf", "knn_5_ppsf", "knn_10_ppsf", "knn_20_ppsf",

    # Predictions as features
    "pred_bld_ppsf", "pred_bt_ppsf", "pred_dist_ppsf",
    "pred_bld_floor_adj",
    "pred_knn5_ppsf", "pred_knn10_ppsf",

    # Derived
    "area_vs_bld", "floor_vs_bld", "floor_pct_in_bld",
    "area_x_floor", "area_x_dist_cbd",

    # Spatial
    "dist_cbd", "dist_mtr", "dist_park", "dist_school", "dist_mall", "dist_hospital",
    "cnt_mtr_0.5km", "cnt_mtr_1.0km", "cnt_park_1.0km",
    "cnt_school_1.0km", "cnt_mall_1.0km",
]

X_train = train[feature_cols].values.astype(np.float32)
X_test = test[feature_cols].values.astype(np.float32)
y = train["price"].values.astype(np.float64)
y_log = np.log1p(y)

print(f"Features: {len(feature_cols)}")

# ─────────────────────────────────────────────
# 11. TRAIN LIGHTGBM
# ─────────────────────────────────────────────
print("\nTraining LightGBM (5-fold)...")

kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
oof = np.zeros(len(X_train))
test_preds = np.zeros(len(X_test))

lgb_params = {
    "objective": "regression",
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

    model = lgb.LGBMRegressor(**lgb_params)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(200, verbose=False)],
    )

    oof[val_idx] = model.predict(X_val)
    test_preds += model.predict(X_test) / N_FOLDS

    fold_rmse = root_mean_squared_error(np.expm1(y_val), np.expm1(oof[val_idx]))
    print(f"  Fold {fold+1}: RMSE = {fold_rmse:,.0f} (iters: {model.best_iteration_})")

cv_rmse = root_mean_squared_error(np.expm1(y_log), np.expm1(oof))
print(f"\nOverall CV RMSE: {cv_rmse:,.0f}")

# ─────────────────────────────────────────────
# 12. SUBMISSION
# ─────────────────────────────────────────────
predictions = np.expm1(test_preds)
predictions = np.clip(predictions, 2000, 500000)

submission = pd.DataFrame({
    "id": test["id"].astype(int),
    "price": predictions.astype(int),
})
submission.to_csv("my_submission.csv", index=False)

print(f"\nSubmission saved: {len(submission)} rows")
print(f"Price range: ${predictions.min():,.0f} – ${predictions.max():,.0f}")
print(f"Mean: ${predictions.mean():,.0f}")

# Feature importance
imp = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
print("\nTop 20 features:")
for f, v in imp.head(20).items():
    print(f"  {f:30s} {v:6d}")
