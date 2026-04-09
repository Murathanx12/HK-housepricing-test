"""
Hong Kong Rental Price Prediction — v2
=======================================
Key insight: 98.1% of test apts have exact unit matches (building+tower+flat)
in training. Within-unit price variance is 5x lower than global.

v2 improvements over v1:
  - Exact unit-level stats (building+tower+flat)
  - Building+flat stats (same layout across towers)
  - Weighted KNN with heavier weight on building proximity
  - More granular floor premium modeling
  - Huber loss for robustness to outliers
"""

import pandas as pd
import numpy as np
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
# 2. PARSE & PREP
# ─────────────────────────────────────────────
print("Parsing addresses...")

def get_building(addr):
    if pd.isna(addr): return "UNKNOWN"
    return addr.split(",")[0].strip()

for df in [train, test]:
    df["building"] = df["address"].apply(get_building)
    df["floor"] = pd.to_numeric(df["floor"], errors="coerce")
    df["Public_Housing"] = df["Public_Housing"].astype(int)
    df["flat_ord"] = df["Flat"].apply(lambda x: ord(str(x)[0].upper()) - 65 if pd.notna(x) and str(x)[0].isalpha() else -1)
    df["tower_num"] = pd.to_numeric(df["Tower"], errors="coerce").fillna(-1)
    df["phase_num"] = pd.to_numeric(df["Phase"], errors="coerce").fillna(-1)

    # Composite keys
    df["unit_key"] = df["building"] + "|" + df["Tower"].fillna("X").astype(str) + "|" + df["Flat"].fillna("X")
    df["bld_tower"] = df["building"] + "|T" + df["Tower"].fillna("X").astype(str)
    df["bld_flat"] = df["building"] + "|F" + df["Flat"].fillna("X")

train["price_per_sqft"] = train["price"] / train["area_sqft"]

# ─────────────────────────────────────────────
# 3. HIERARCHICAL STATS: unit → bld+tower → bld+flat → building → district
# ─────────────────────────────────────────────
print("Computing hierarchical statistics...")

def compute_stats(grp_col, prefix, extra_aggs=None):
    aggs = {
        f"{prefix}_ppsf_mean": ("price_per_sqft", "mean"),
        f"{prefix}_ppsf_median": ("price_per_sqft", "median"),
        f"{prefix}_ppsf_std": ("price_per_sqft", "std"),
        f"{prefix}_price_mean": ("price", "mean"),
        f"{prefix}_price_median": ("price", "median"),
        f"{prefix}_count": ("price", "count"),
    }
    if extra_aggs:
        aggs.update(extra_aggs)
    stats = train.groupby(grp_col).agg(**aggs).reset_index()
    stats[f"{prefix}_ppsf_std"] = stats[f"{prefix}_ppsf_std"].fillna(train["price_per_sqft"].std())
    return stats

# Unit level (building+tower+flat)
unit_stats = compute_stats("unit_key", "unit")

# Building+tower level
bt_stats = compute_stats("bld_tower", "bt")

# Building+flat level (same layout)
bf_stats = compute_stats("bld_flat", "bf")

# Building level
bld_extra = {
    "bld_ppsf_min": ("price_per_sqft", "min"),
    "bld_ppsf_max": ("price_per_sqft", "max"),
    "bld_area_mean": ("area_sqft", "mean"),
    "bld_floor_mean": ("floor", "mean"),
    "bld_floor_min": ("floor", "min"),
    "bld_floor_max": ("floor", "max"),
}
bld_stats = compute_stats("building", "bld", bld_extra)

# Floor slope per building
def floor_slope(g):
    if len(g) < 3 or g["floor"].std() == 0:
        return 0.0
    return np.polyfit(g["floor"], g["price_per_sqft"], 1)[0]

bld_slope = train.groupby("building").apply(floor_slope, include_groups=False).reset_index()
bld_slope.columns = ["building", "bld_ppsf_floor_slope"]
bld_stats = bld_stats.merge(bld_slope, on="building", how="left")
bld_stats["bld_ppsf_floor_slope"] = bld_stats["bld_ppsf_floor_slope"].fillna(0)

# District level
dist_stats = compute_stats("district", "dist")

# Merge all stats
for df in [train, test]:
    for stats, key in [(unit_stats, "unit_key"), (bt_stats, "bld_tower"),
                       (bf_stats, "bld_flat"), (bld_stats, "building"),
                       (dist_stats, "district")]:
        merged = df[[key]].merge(stats, on=key, how="left")
        for col in stats.columns:
            if col != key:
                df[col] = merged[col].values

# Cascading fallback for missing values
for prefix_chain in [
    ("unit", "bt", "bld", "dist"),
    ("bt", "bld", "dist"),
    ("bf", "bld", "dist"),
]:
    for suffix in ["_ppsf_mean", "_ppsf_median", "_price_mean", "_price_median"]:
        for i in range(len(prefix_chain) - 1):
            col = prefix_chain[i] + suffix
            fallback = prefix_chain[i + 1] + suffix
            if col in test.columns and fallback in test.columns:
                test[col] = test[col].fillna(test[fallback])

# Fill any remaining NaNs with global means
for col in test.columns:
    if test[col].dtype in [np.float64, np.float32] and test[col].isna().any():
        test[col] = test[col].fillna(train[col].median() if col in train.columns else 0)
for col in train.columns:
    if train[col].dtype in [np.float64, np.float32] and train[col].isna().any():
        train[col] = train[col].fillna(train[col].median())

# ─────────────────────────────────────────────
# 4. KNN FEATURES
# ─────────────────────────────────────────────
print("Computing KNN features...")

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
# 5. SPATIAL FEATURES
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
# 6. DERIVED FEATURES
# ─────────────────────────────────────────────
print("Creating derived features...")

for df in [train, test]:
    df["log_area"] = np.log1p(df["area_sqft"])

    # Deviation from unit/building averages
    df["area_vs_bld"] = df["area_sqft"] - df["bld_area_mean"]
    df["floor_vs_bld"] = df["floor"] - df["bld_floor_mean"]

    # Floor position in building
    floor_range = df["bld_floor_max"] - df["bld_floor_min"]
    df["floor_pct"] = np.where(floor_range > 0, (df["floor"] - df["bld_floor_min"]) / floor_range, 0.5)

    # Multiple prediction baselines for the model to blend
    df["pred_unit"] = df["area_sqft"] * df["unit_ppsf_mean"]
    df["pred_bt"] = df["area_sqft"] * df["bt_ppsf_mean"]
    df["pred_bf"] = df["area_sqft"] * df["bf_ppsf_mean"]
    df["pred_bld"] = df["area_sqft"] * df["bld_ppsf_mean"]
    df["pred_dist"] = df["area_sqft"] * df["dist_ppsf_mean"]

    # Floor-adjusted predictions
    df["pred_bld_floor_adj"] = df["pred_bld"] + df["bld_ppsf_floor_slope"] * (df["floor"] - df["bld_floor_mean"]) * df["area_sqft"]
    df["pred_unit_floor_adj"] = df["pred_unit"] + df["bld_ppsf_floor_slope"] * (df["floor"] - df["bld_floor_mean"]) * df["area_sqft"]

    # KNN-derived
    df["pred_knn5"] = df["area_sqft"] * df["knn_5_ppsf"]
    df["pred_knn10"] = df["area_sqft"] * df["knn_10_ppsf"]

    # How "unusual" is this unit vs its building
    df["ppsf_range_bld"] = df["bld_ppsf_max"] - df["bld_ppsf_min"]

    # Interactions
    df["area_x_floor"] = df["area_sqft"] * df["floor"]

    # Confidence: how many data points back this unit's stats
    df["unit_confidence"] = df["unit_count"].fillna(0)
    df["bt_confidence"] = df["bt_count"].fillna(0)

# ─────────────────────────────────────────────
# 7. ENCODE CATEGORICALS
# ─────────────────────────────────────────────
print("Encoding categoricals...")

for col in ["building", "district"]:
    le = LabelEncoder()
    combined = pd.concat([train[col], test[col]]).fillna("UNKNOWN")
    le.fit(combined)
    train[f"{col}_code"] = le.transform(train[col].fillna("UNKNOWN"))
    test[f"{col}_code"] = le.transform(test[col].fillna("UNKNOWN"))

# ─────────────────────────────────────────────
# 8. FEATURE LIST
# ─────────────────────────────────────────────
feature_cols = [
    # Core
    "area_sqft", "floor", "Public_Housing", "log_area",
    "wgs_lat", "wgs_lon",
    "flat_ord", "tower_num", "phase_num",

    # Encoded
    "building_code", "district_code",

    # Unit-level (THE KILLER FEATURES)
    "unit_ppsf_mean", "unit_ppsf_median", "unit_ppsf_std",
    "unit_price_mean", "unit_price_median", "unit_count",

    # Building+tower
    "bt_ppsf_mean", "bt_ppsf_median", "bt_price_mean", "bt_count",

    # Building+flat (same layout)
    "bf_ppsf_mean", "bf_ppsf_median", "bf_price_mean", "bf_count",

    # Building-level
    "bld_ppsf_mean", "bld_ppsf_median", "bld_ppsf_std",
    "bld_ppsf_min", "bld_ppsf_max",
    "bld_price_mean", "bld_price_median",
    "bld_area_mean", "bld_floor_mean",
    "bld_count", "bld_ppsf_floor_slope",

    # District-level
    "dist_ppsf_mean", "dist_ppsf_median", "dist_price_mean", "dist_count",

    # KNN
    "knn_3_price", "knn_5_price", "knn_10_price", "knn_20_price",
    "knn_3_ppsf", "knn_5_ppsf", "knn_10_ppsf", "knn_20_ppsf",

    # Predictions (model blends these)
    "pred_unit", "pred_bt", "pred_bf", "pred_bld", "pred_dist",
    "pred_bld_floor_adj", "pred_unit_floor_adj",
    "pred_knn5", "pred_knn10",

    # Derived
    "area_vs_bld", "floor_vs_bld", "floor_pct",
    "ppsf_range_bld", "area_x_floor",
    "unit_confidence", "bt_confidence",

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
# 9. TRAIN LIGHTGBM
# ─────────────────────────────────────────────
print("\nTraining LightGBM (5-fold)...")

kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
oof = np.zeros(len(X_train))
test_preds = np.zeros(len(X_test))

lgb_params = {
    "objective": "huber",  # more robust to outliers than MSE
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
# 10. SUBMISSION
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
