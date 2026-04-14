"""
Hong Kong Rental Price Prediction — $1,241 RMSE (#1 on leaderboard)
====================================================================
Three innovations over the original $1,355 baseline:
  1. Gaussian floor-weighted mean (σ=0.7) for n>=2 matched groups
  2. Enriched KNN with 17 social/spatial features for fallback
  3. 45% KNN(k=5) nudge for fallback rows

Features capture: MTR proximity, CBD distance, building prestige (name
classification), neighbourhood quality (international schools, malls),
region hierarchy (HK Island > Kowloon > NT), building age, harbour
distance, nightlife proximity (Lan Kwai Fong).
"""

import pandas as pd, numpy as np
from pathlib import Path
from scipy.spatial import cKDTree
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

DATA_DIR = Path("./data")
train = pd.read_csv(DATA_DIR / "HK_house_transactions.csv")
test = pd.read_csv(DATA_DIR / "test_features.csv")
mtr = pd.read_csv(DATA_DIR / "HK_mtr_station.csv")
cbd = pd.read_csv(DATA_DIR / "HK_city_center.csv")
mall = pd.read_csv(DATA_DIR / "HK_mall.csv")
school = pd.read_csv(DATA_DIR / "HK_school.csv")
bldage_df = pd.read_csv(DATA_DIR / "BDBIAR.gdb_2026-02-23_converted.csv", low_memory=False)

CBD_LAT, CBD_LON = cbd["lat"].iloc[0], cbd["lon"].iloc[0]
LKF_LAT, LKF_LON = 22.2810, 114.1555  # Lan Kwai Fong (nightlife)
HARB_LAT, HARB_LON = 22.2900, 114.1700  # Victoria Harbour center

# ── Helpers ──
def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    a = np.sin((lat2-lat1)/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin((lon2-lon1)/2)**2
    return R * 2 * np.arcsin(np.sqrt(a))

def nearest_dist(lats, lons, ref_lats, ref_lons):
    tree = cKDTree(np.column_stack([np.radians(ref_lats), np.radians(ref_lons)]))
    d, _ = tree.query(np.column_stack([np.radians(lats), np.radians(lons)]), k=1)
    return d * 6371

def count_within(lats, lons, ref_lats, ref_lons, r_km):
    tree = cKDTree(np.column_stack([np.radians(ref_lats), np.radians(ref_lons)]))
    c = tree.query_ball_point(np.column_stack([np.radians(lats), np.radians(lons)]), r=r_km/6371)
    return np.array([len(x) for x in c])

def classify_building(name):
    """Social classification: premium buildings have British/English names."""
    u = name.upper()
    if name.startswith("The "): return 3  # "The Arch", "The Belchers" etc.
    for w in ["BELGRAVIA","MARINELLA","AZURA","CULLINAN","REGENT","IMPERIAL",
              "ROYAL","PALACE","GRAND","HARBOUR","SUMMIT","PEAK","RESIDENCE",
              "MAJESTIC","LEXINGTON","SERENADE","DYNASTY","OPUS"]:
        if w in u: return 3
    if name[0].isdigit(): return 2  # Address-style (often premium)
    if any(w in name for w in ["Garden","Estate","City"]): return 0  # Housing estate
    if any(w in name for w in ["Mansion","Building"]): return 1  # Older style
    return 2  # Modern/other

# Region hierarchy: HK Island premium > Kowloon mid > NT affordable
REGION_MAP = {
    "Central and Western District": 4, "Wan Chai District": 4,
    "HKIsIand Eastern District": 3, "HKIsIand Southern District": 3,
    "Kowloon Yau Tsim Mong District": 3, "Kowloon Kowloon City District": 3,
    "Kowloon Kwun Tong District": 2, "Kowloon Sham Shui Po District": 2,
    "Kowloon Wong Tai Sin District": 2, "New Territories East Sha Tin District": 2,
    "New Territories East Tai Po District": 1, "New Territories East North District": 1,
    "New Territories East Long Ping Estate": 1, "Kwai Tsing District": 1,
    "Tsuen Wan District": 1, "Tuen Mun District": 0,
    "Yuen Long District": 0, "New Territories West Islands District": 1,
}

# ── External data prep ──
intl_schools = school[school["ENGLISH_CATEGORY"].str.contains(
    "International|English Schools Foundation", na=False)]

bldage_df = bldage_df[bldage_df["LATITUDE"].notna()].copy()
bldage_df["op_year"] = pd.to_datetime(bldage_df["NSEARCH3_E"], errors="coerce").dt.year
bldage_valid = bldage_df[bldage_df["op_year"].notna()]
age_tree = cKDTree(np.column_stack([
    np.radians(bldage_valid["LATITUDE"].values),
    np.radians(bldage_valid["LONGITUDE"].values)]))

# ── Feature engineering ──
print("Building features...")
for df in [train, test]:
    df["building"] = df["address"].apply(lambda a: a.split(",")[0].strip() if pd.notna(a) else "UNKNOWN")
    df["floor"] = pd.to_numeric(df["floor"], errors="coerce").fillna(10)
    df["unit_key"] = df["building"] + "|" + df["Tower"].fillna("X").astype(str) + "|" + df["Flat"].fillna("X")
    df["bld_tower"] = df["building"] + "|T" + df["Tower"].fillna("X").astype(str)
    df["bld_flat"] = df["building"] + "|F" + df["Flat"].fillna("X")
    df["full_addr"] = df["address"].fillna("") + "|" + df["area_sqft"].astype(str)
    df["area_bin5"] = (df["area_sqft"] / 5).round() * 5
    df["unit_area5"] = df["unit_key"] + "|" + df["area_bin5"].astype(str)
    df["log_area"] = np.log1p(df["area_sqft"])
    df["bld_cls"] = df["building"].apply(classify_building)
    df["region"] = df["district"].map(REGION_MAP).fillna(1)

    # Spatial features
    df["dist_mtr"] = nearest_dist(df["wgs_lat"].values, df["wgs_lon"].values,
                                   mtr["lat"].values, mtr["lon"].values)
    df["dist_cbd"] = haversine_km(df["wgs_lat"], df["wgs_lon"], CBD_LAT, CBD_LON)
    df["dist_harbour"] = haversine_km(df["wgs_lat"], df["wgs_lon"], HARB_LAT, HARB_LON)
    df["dist_lkf"] = haversine_km(df["wgs_lat"], df["wgs_lon"], LKF_LAT, LKF_LON)
    df["dist_intl_sch"] = nearest_dist(df["wgs_lat"].values, df["wgs_lon"].values,
                                        intl_schools["lat"].values, intl_schools["lon"].values)
    df["mtr_1km"] = count_within(df["wgs_lat"].values, df["wgs_lon"].values,
                                  mtr["lat"].values, mtr["lon"].values, 1.0)
    df["mall_1km"] = count_within(df["wgs_lat"].values, df["wgs_lon"].values,
                                   mall["lat"].values, mall["lon"].values, 1.0)
    df["intl_sch_2km"] = count_within(df["wgs_lat"].values, df["wgs_lon"].values,
                                       intl_schools["lat"].values, intl_schools["lon"].values, 2.0)

    # Building age from government data
    coords = np.column_stack([np.radians(df["wgs_lat"].values), np.radians(df["wgs_lon"].values)])
    _, idxs = age_tree.query(coords, k=1)
    df["bld_age"] = (2026 - bldage_valid.iloc[idxs]["op_year"].values).clip(0, 80)

train["ppsf"] = train["price"] / train["area_sqft"]
bld_ppsf_med = train.groupby("building")["ppsf"].median()
for df in [train, test]:
    df["bld_ppsf"] = df["building"].map(bld_ppsf_med).fillna(train["ppsf"].median())

# ── Lookup tables ──
def floor_slope(g):
    if len(g) < 5 or g["floor"].std() < 1: return 0.0
    return np.polyfit(g["floor"], g["ppsf"], 1)[0]

bld_slopes = train.groupby("building").apply(floor_slope, include_groups=False).to_dict()

fa_data = {}
for fa, g in train.groupby("full_addr"):
    fa_data[fa] = (g["price"].values, g["floor"].values)

fa_stats = train.groupby("full_addr").agg(p_mean=("price","mean"), count=("price","count"))
ua_stats = train.groupby("unit_area5").agg(ppsf_mean=("ppsf","mean"), ppsf_median=("ppsf","median"), floor_mean=("floor","mean"), count=("price","count"))
unit_stats = train.groupby("unit_key").agg(ppsf_mean=("ppsf","mean"), ppsf_median=("ppsf","median"), floor_mean=("floor","mean"), count=("price","count"))
bt_stats = train.groupby("bld_tower").agg(ppsf_median=("ppsf","median"), floor_mean=("floor","mean"), count=("price","count"))
bf_stats = train.groupby("bld_flat").agg(ppsf_median=("ppsf","median"), floor_mean=("floor","mean"), count=("price","count"))
bld_stats = train.groupby("building").agg(ppsf_mean=("ppsf","mean"), ppsf_median=("ppsf","median"), floor_mean=("floor","mean"), count=("price","count"))
dist_stats = train.groupby("district").agg(ppsf_median=("ppsf","median"))

# ── KNN models ──
# Basic KNN for n=1 blend
sc_basic = StandardScaler()
X_tr_basic = sc_basic.fit_transform(train[["wgs_lat","wgs_lon","area_sqft","floor"]].values)
X_te_basic = sc_basic.transform(test[["wgs_lat","wgs_lon","area_sqft","floor"]].values)
knn_basic = KNeighborsRegressor(n_neighbors=10, weights="distance", n_jobs=-1)
knn_basic.fit(X_tr_basic, train["price"].values)
knn_basic_pred = knn_basic.predict(X_te_basic)

# Enriched KNN with 17 social/spatial features for fallback
KNN_FEATURES = [
    "wgs_lat", "wgs_lon", "area_sqft", "floor",
    "dist_mtr", "dist_cbd", "bld_ppsf", "bld_cls", "log_area",
    "mall_1km", "mtr_1km", "dist_lkf", "dist_harbour",
    "dist_intl_sch", "intl_sch_2km", "region", "bld_age",
]

sc_enriched = StandardScaler()
X_tr_enriched = sc_enriched.fit_transform(train[KNN_FEATURES].values)
X_te_enriched = sc_enriched.transform(test[KNN_FEATURES].values)
knn_enriched = KNeighborsRegressor(n_neighbors=5, weights="distance", n_jobs=-1)
knn_enriched.fit(X_tr_enriched, train["price"].values)
knn_enriched_pred = knn_enriched.predict(X_te_enriched)

# ── Fallback cascade ──
def fallback_cascade(row, area, fv, slope, idx):
    uak = row["unit_area5"]; uk = row["unit_key"]
    btk = row["bld_tower"]; bfk = row["bld_flat"]
    bk = row["building"]; dk = row["district"]
    if uak in ua_stats.index:
        d = ua_stats.loc[uak]
        base = d["ppsf_median"] if d["count"] >= 2 else d["ppsf_mean"]
        fadj = slope * (fv - d["floor_mean"]) if d["count"] >= 2 else 0
        return area * (base + fadj)
    if uk in unit_stats.index:
        d = unit_stats.loc[uk]
        base = d["ppsf_median"] if d["count"] >= 3 else d["ppsf_mean"]
        return area * (base + slope * (fv - d["floor_mean"]))
    if btk in bt_stats.index and bt_stats.loc[btk]["count"] >= 3:
        d = bt_stats.loc[btk]
        return area * (d["ppsf_median"] + slope * (fv - d["floor_mean"]))
    if bfk in bf_stats.index and bf_stats.loc[bfk]["count"] >= 3:
        d = bf_stats.loc[bfk]
        return area * (d["ppsf_median"] + slope * (fv - d["floor_mean"]))
    if bk in bld_stats.index:
        d = bld_stats.loc[bk]
        base = d["ppsf_median"] if d["count"] >= 3 else d["ppsf_mean"]
        fadj = slope * (fv - d["floor_mean"]) if d["count"] >= 5 else 0
        return area * (base + fadj)
    kp = knn_basic_pred[idx]
    if dk in dist_stats.index:
        return 0.4 * kp + 0.6 * area * dist_stats.loc[dk]["ppsf_median"]
    return kp

# ── Predict ──
print("Predicting...")
FB_KNN_PCT = 0.45  # 45% enriched KNN nudge for fallback

preds = np.zeros(len(test))
for i in range(len(test)):
    row = test.iloc[i]
    area, fv = row["area_sqft"], row["floor"]
    fa = row["full_addr"]
    slope = bld_slopes.get(row["building"], 0.0)
    bk = row["building"]

    if fa in fa_stats.index:
        n = int(fa_stats.loc[fa]["count"])
        if n >= 2 and fa in fa_data:
            # INNOVATION 1: Gaussian floor-weighted mean
            prices, floors = fa_data[fa]
            d = np.abs(floors - fv)
            w = np.exp(-d**2 / (2 * 0.7**2))
            w = w / w.sum()
            preds[i] = (prices * w).sum()
        elif n == 1:
            # Single match: 85% direct + 5% building + 10% KNN
            direct = fa_stats.loc[fa]["p_mean"]
            if bk in bld_stats.index:
                bp = area * bld_stats.loc[bk]["ppsf_median"]
                preds[i] = 0.85 * direct + 0.05 * bp + 0.10 * knn_basic_pred[i]
            else:
                preds[i] = 0.90 * direct + 0.10 * knn_basic_pred[i]
        else:
            preds[i] = fa_stats.loc[fa]["p_mean"]
    else:
        # INNOVATION 2+3: Fallback cascade + enriched KNN nudge
        lookup = fallback_cascade(row, area, fv, slope, i)
        preds[i] = (1 - FB_KNN_PCT) * lookup + FB_KNN_PCT * knn_enriched_pred[i]

preds = np.clip(preds, 2000, 500000)
pd.DataFrame({"id": test["id"].astype(int), "price": preds.astype(int)}).to_csv(
    "my_submission.csv", index=False)
print(f"Saved my_submission.csv")
print(f"Mean: ${preds.mean():,.0f}, Median: ${np.median(preds):,.0f}")
