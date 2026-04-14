"""
Hunting the last $$ — within-group noise reduction
====================================================
673 groups have high price variance but NO floor variation.
Floor-weighting can't help these. Try: outlier rejection,
mode prediction, median for high-variance groups.
"""

import pandas as pd, numpy as np
from pathlib import Path
from scipy.stats import trim_mean
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

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
    df["full_addr"] = df["address"].fillna("") + "|" + df["area_sqft"].astype(str)
    df["area_bin5"] = (df["area_sqft"] / 5).round() * 5
    df["unit_area5"] = df["unit_key"] + "|" + df["area_bin5"].astype(str)

train["ppsf"] = train["price"] / train["area_sqft"]

def floor_slope(g):
    if len(g) < 5 or g["floor"].std() < 1: return 0.0
    return np.polyfit(g["floor"], g["ppsf"], 1)[0]

bld_slopes = train.groupby("building").apply(floor_slope, include_groups=False).to_dict()

fa_data = {}
for fa, g in train.groupby("full_addr"):
    fa_data[fa] = (g["price"].values, g["floor"].values)
fa_stats = train.groupby("full_addr").agg(
    p_mean=("price","mean"), p_median=("price","median"),
    count=("price","count"), p_std=("price","std"),
    floor_std=("floor","std"),
)
fa_stats["p_std"] = fa_stats["p_std"].fillna(0)
fa_stats["floor_std"] = fa_stats["floor_std"].fillna(0)

# Precompute special aggregations
fa_mode = train.groupby("full_addr")["price"].agg(lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else x.mean())
fa_reject = {}  # full_addr -> price with outlier removed
for fa, g in train.groupby("full_addr"):
    prices = g["price"].values
    if len(prices) >= 3:
        med = np.median(prices)
        dists = np.abs(prices - med)
        # Remove the one furthest from median
        keep = prices[np.argsort(dists)[:-1]]
        fa_reject[fa] = keep.mean()
    else:
        fa_reject[fa] = prices.mean()

ua_stats = train.groupby("unit_area5").agg(ppsf_mean=("ppsf","mean"), ppsf_median=("ppsf","median"), floor_mean=("floor","mean"), count=("price","count"))
unit_stats = train.groupby("unit_key").agg(ppsf_mean=("ppsf","mean"), ppsf_median=("ppsf","median"), floor_mean=("floor","mean"), count=("price","count"))
bt_stats = train.groupby("bld_tower").agg(ppsf_median=("ppsf","median"), floor_mean=("floor","mean"), count=("price","count"))
bf_stats = train.groupby("bld_flat").agg(ppsf_median=("ppsf","median"), floor_mean=("floor","mean"), count=("price","count"))
bld_stats = train.groupby("building").agg(ppsf_mean=("ppsf","mean"), ppsf_median=("ppsf","median"), floor_mean=("floor","mean"), count=("price","count"))
dist_stats = train.groupby("district").agg(ppsf_median=("ppsf","median"))

scaler = StandardScaler()
X_tr = scaler.fit_transform(train[["wgs_lat","wgs_lon","area_sqft","floor"]].values)
X_te = scaler.transform(test[["wgs_lat","wgs_lon","area_sqft","floor"]].values)
knn = KNeighborsRegressor(n_neighbors=10, weights="distance", n_jobs=-1)
knn.fit(X_tr, train["price"].values)
test["knn10"] = knn.predict(X_te)

def fb(row, area, fv, slope):
    uak,uk,btk,bfk,bk,dk = row["unit_area5"],row["unit_key"],row["bld_tower"],row["bld_flat"],row["building"],row["district"]
    if uak in ua_stats.index:
        d=ua_stats.loc[uak]; base=d["ppsf_median"] if d["count"]>=2 else d["ppsf_mean"]
        fadj=slope*(fv-d["floor_mean"]) if d["count"]>=2 else 0; return area*(base+fadj)
    if uk in unit_stats.index:
        d=unit_stats.loc[uk]; base=d["ppsf_median"] if d["count"]>=3 else d["ppsf_mean"]
        return area*(base+slope*(fv-d["floor_mean"]))
    if btk in bt_stats.index and bt_stats.loc[btk]["count"]>=3:
        d=bt_stats.loc[btk]; return area*(d["ppsf_median"]+slope*(fv-d["floor_mean"]))
    if bfk in bf_stats.index and bf_stats.loc[bfk]["count"]>=3:
        d=bf_stats.loc[bfk]; return area*(d["ppsf_median"]+slope*(fv-d["floor_mean"]))
    if bk in bld_stats.index:
        d=bld_stats.loc[bk]; base=d["ppsf_median"] if d["count"]>=3 else d["ppsf_mean"]
        fadj=slope*(fv-d["floor_mean"]) if d["count"]>=5 else 0; return area*(base+fadj)
    kp=row["knn10"]
    if dk in dist_stats.index: return 0.4*kp+0.6*area*dist_stats.loc[dk]["ppsf_median"]
    return kp

def gauss_w(floors, test_floor, sigma=0.7):
    d = np.abs(floors - test_floor)
    w = np.exp(-d**2 / (2 * sigma**2))
    return w / w.sum()

def gen(n_agg="fw_mean", high_var_agg=None, high_var_thresh=2000):
    """
    n_agg: aggregation for n>=2 matched rows
      "fw_mean": Gaussian floor-weighted mean (baseline $1,300)
      "fw_median": floor-weighted, but use median for same-floor groups
      "fw_reject": floor-weighted, but reject outlier for high-var same-floor groups
      "fw_mode": floor-weighted, but use mode for same-floor groups
    high_var_agg: override for high-variance (>thresh) same-floor groups only
      None: use n_agg for everything
      "median": use median for high-var same-floor groups
      "reject": use reject-outlier for these
      "mode": use mode for these
    """
    preds = np.zeros(len(test))
    for i in range(len(test)):
        row = test.iloc[i]; area, fv = row["area_sqft"], row["floor"]
        fa = row["full_addr"]; slope = bld_slopes.get(row["building"], 0.0)
        bk = row["building"]

        if fa in fa_stats.index:
            n = int(fa_stats.loc[fa]["count"])
            if n >= 2 and fa in fa_data:
                prices, floors = fa_data[fa]
                p_std = fa_stats.loc[fa]["p_std"]
                f_std = fa_stats.loc[fa]["floor_std"]

                # Check if this is a high-variance same-floor group
                is_high_var_same_floor = (p_std > high_var_thresh and f_std < 0.5 and n >= 3)

                if is_high_var_same_floor and high_var_agg is not None:
                    if high_var_agg == "median":
                        preds[i] = fa_stats.loc[fa]["p_median"]
                    elif high_var_agg == "reject":
                        preds[i] = fa_reject.get(fa, prices.mean())
                    elif high_var_agg == "mode":
                        preds[i] = fa_mode.get(fa, prices.mean())
                    else:
                        w = gauss_w(floors, fv)
                        preds[i] = (prices * w).sum()
                else:
                    # Standard floor-weighted mean
                    w = gauss_w(floors, fv)
                    preds[i] = (prices * w).sum()
            elif n == 1:
                direct = fa_stats.loc[fa]["p_mean"]
                if bk in bld_stats.index:
                    bp = area * bld_stats.loc[bk]["ppsf_median"]
                    preds[i] = 0.85 * direct + 0.05 * bp + 0.10 * row["knn10"]
                else:
                    preds[i] = 0.90 * direct + 0.10 * row["knn10"]
            else:
                preds[i] = fa_stats.loc[fa]["p_mean"]
        else:
            preds[i] = fb(row, area, fv, slope)
    return np.clip(preds, 2000, 500000).astype(int)


# ============================================================
print("Generating probes...\n")

base = gen()  # $1,300 baseline

configs = [
    ("1_base", "fw_mean", None, 2000),
    # Median for ALL n>=2 (replaces floor-weighted for same-floor groups)
    ("2_all_median", "fw_median", "median", 0),
    # Reject outlier for high-var same-floor groups (std>2K)
    ("3_reject_2k", "fw_mean", "reject", 2000),
    # Reject outlier for high-var same-floor (std>1K)
    ("4_reject_1k", "fw_mean", "reject", 1000),
    # Mode for high-var same-floor
    ("5_mode_2k", "fw_mean", "mode", 2000),
    # Mode for high-var same-floor (std>1K)
    ("6_mode_1k", "fw_mean", "mode", 1000),
    # Median for high-var same-floor (std>2K)
    ("7_median_2k", "fw_mean", "median", 2000),
    # Median for high-var same-floor (std>1K)
    ("8_median_1k", "fw_mean", "median", 1000),
    # Reject with lower threshold (std>3K)
    ("9_reject_3k", "fw_mean", "reject", 3000),
    # Median with std>3K
    ("10_median_3k", "fw_mean", "median", 3000),
]

print(f"{'#':>2} {'Name':20s} {'Changed':>8s} {'AvgDiff':>9s}")
print("-" * 43)

for idx, (name, nagg, hvagg, thresh) in enumerate(configs):
    p = gen(n_agg=nagg, high_var_agg=hvagg, high_var_thresh=thresh)
    diff = np.abs(p.astype(float) - base.astype(float))
    changed = (diff > 0).sum()
    avg = diff[diff > 0].mean() if changed > 0 else 0
    print(f"{idx+1:2d} {name:20s} {changed:8d} {avg:9.0f}")
    pd.DataFrame({"id": test["id"].astype(int), "price": p}).to_csv(f"{idx+1}.csv", index=False)

print("\n=== STRATEGY ===")
print("Groups with high price variance + same floor = unexplained noise")
print("Testing if median/mode/reject-outlier handles these better than mean")
print("Reject = remove the price furthest from group median, then average rest")
print("Mode = use the most frequently occurring price in the group")
