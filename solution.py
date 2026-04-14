"""
BREAKTHROUGH: Floor-weighted mean = $1,324 RMSE (#1!)
=====================================================
Weighting group members by floor proximity to test row.
Now iterate on the weighting function to push even lower.
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

fa_grp = train.groupby("full_addr")
fa_stats = fa_grp.agg(p_mean=("price","mean"), count=("price","count"), floor_mean=("floor","mean"))
fa_trimmed = fa_grp["price"].apply(lambda x: trim_mean(x, 0.1) if len(x) >= 4 else x.mean())
fa_stats = fa_stats.join(fa_trimmed.rename("p_trimmed"))

# Pre-compute group data for floor-weighted calculations
fa_data = {}  # full_addr -> (prices, floors)
for fa, group in train.groupby("full_addr"):
    fa_data[fa] = (group["price"].values, group["floor"].values)

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


def gen(weight_func="inverse", weight_power=1.0, min_n_for_fw=2,
        n1_sd=0.85, n1_sb=0.05, n1_sk=0.10, fw_for_n1=False):
    """
    weight_func: how to weight group members by floor proximity
      "inverse": w = 1 / (1 + |floor_diff|^power)
      "gaussian": w = exp(-floor_diff^2 / (2*sigma^2)) where sigma=power
      "nearest": w = 1 for nearest floor, 0 for others
      "plain": w = 1 (original mean — baseline)
    """
    preds = np.zeros(len(test))
    for i in range(len(test)):
        row = test.iloc[i]; area, fv = row["area_sqft"], row["floor"]
        fa = row["full_addr"]; slope = bld_slopes.get(row["building"], 0.0); bk = row["building"]

        if fa in fa_stats.index:
            d = fa_stats.loc[fa]; n = int(d["count"])

            if n >= min_n_for_fw and fa in fa_data:
                prices, floors = fa_data[fa]
                if weight_func == "plain":
                    # Plain mean (baseline)
                    if n >= 4:
                        preds[i] = trim_mean(prices, 0.1)
                    else:
                        preds[i] = prices.mean()
                elif weight_func == "nearest":
                    # Use price from nearest floor
                    nearest_idx = np.argmin(np.abs(floors - fv))
                    preds[i] = prices[nearest_idx]
                else:
                    # Weighted mean
                    floor_diff = np.abs(floors - fv)
                    if weight_func == "inverse":
                        w = 1.0 / (1.0 + floor_diff ** weight_power)
                    elif weight_func == "gaussian":
                        sigma = weight_power
                        w = np.exp(-floor_diff**2 / (2 * sigma**2))
                    else:
                        w = np.ones(len(prices))
                    w = w / w.sum()
                    preds[i] = (prices * w).sum()
            elif n >= 4:
                preds[i] = d["p_trimmed"]
            elif n >= 2:
                preds[i] = d["p_mean"]
            else:
                # n=1
                direct = d["p_mean"]
                if fw_for_n1:
                    preds[i] = direct  # pure direct for n=1 when fw is on
                else:
                    if bk in bld_stats.index:
                        bp = area * bld_stats.loc[bk]["ppsf_median"]
                        preds[i] = n1_sd * direct + n1_sb * bp + n1_sk * row["knn10"]
                    else:
                        preds[i] = (n1_sd + n1_sb) * direct + n1_sk * row["knn10"]
        else:
            preds[i] = fb(row, area, fv, slope)

    return np.clip(preds, 2000, 500000).astype(int)


# ============================================================
# Generate variants to iterate on the breakthrough
# ============================================================
print("Generating variants...\n")

baseline = gen(weight_func="plain")  # $1,355 baseline

configs = [
    # The winner: inverse distance, power=1
    ("1_inv_p1_n2", "inverse", 1.0, 2, 0.85, 0.05, 0.10, False),
    # Sharper weighting (closer floors matter MORE)
    ("2_inv_p2_n2", "inverse", 2.0, 2, 0.85, 0.05, 0.10, False),
    ("3_inv_p3_n2", "inverse", 3.0, 2, 0.85, 0.05, 0.10, False),
    ("4_inv_p05_n2", "inverse", 0.5, 2, 0.85, 0.05, 0.10, False),
    # Gaussian weighting
    ("5_gauss_s3_n2", "gaussian", 3.0, 2, 0.85, 0.05, 0.10, False),
    ("6_gauss_s5_n2", "gaussian", 5.0, 2, 0.85, 0.05, 0.10, False),
    ("7_gauss_s1_n2", "gaussian", 1.0, 2, 0.85, 0.05, 0.10, False),
    # Nearest floor only
    ("8_nearest_n2", "nearest", 1.0, 2, 0.85, 0.05, 0.10, False),
    # Winner + pure direct for n=1 (remove KNN/bld for n=1)
    ("9_inv_p1_n2_pured", "inverse", 1.0, 2, 1.0, 0.0, 0.0, True),
    # Winner but only for n>=3 (leave n=2 as plain mean)
    ("10_inv_p1_n3", "inverse", 1.0, 3, 0.85, 0.05, 0.10, False),
]

print(f"{'#':>2s} {'Name':35s} {'Changed':>8s} {'AvgDiff':>9s}")
print("-" * 58)

for idx, (name, wf, wp, mn, sd, sb, sk, fwn1) in enumerate(configs):
    p = gen(weight_func=wf, weight_power=wp, min_n_for_fw=mn,
            n1_sd=sd, n1_sb=sb, n1_sk=sk, fw_for_n1=fwn1)
    diff = np.abs(p.astype(float) - baseline.astype(float))
    changed = (diff > 0).sum()
    avg_d = diff[diff > 0].mean() if changed > 0 else 0
    print(f"{idx+1:2d} {name:35s} {changed:8d} {avg_d:9.0f}")
    pd.DataFrame({"id": test["id"].astype(int), "price": p}).to_csv(f"{idx+1}.csv", index=False)

# Save the winner as my_submission.csv
p = gen(weight_func="inverse", weight_power=1.0, min_n_for_fw=2)
pd.DataFrame({"id": test["id"].astype(int), "price": p}).to_csv("my_submission.csv", index=False)
print(f"\nmy_submission.csv = 1_inv_p1_n2 (the $1,324 winner)")
