"""
$1,300 RMSE baseline — now push lower by applying floor-weighting
to ALL prediction levels, not just n>=2 matched rows.
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

# Pre-compute group data for floor-weighted calculations at ALL levels
fa_data = {}
for fa, g in train.groupby("full_addr"):
    fa_data[fa] = (g["price"].values, g["floor"].values)

ua_data = {}
for ua, g in train.groupby("unit_area5"):
    ua_data[ua] = (g["ppsf"].values, g["floor"].values, g["area_sqft"].values)

uk_data = {}
for uk, g in train.groupby("unit_key"):
    uk_data[uk] = (g["ppsf"].values, g["floor"].values, g["area_sqft"].values)

bt_data = {}
for bt, g in train.groupby("bld_tower"):
    bt_data[bt] = (g["ppsf"].values, g["floor"].values)

bf_data = {}
for bf, g in train.groupby("bld_flat"):
    bf_data[bf] = (g["ppsf"].values, g["floor"].values)

bld_data = {}
for bld, g in train.groupby("building"):
    bld_data[bld] = (g["ppsf"].values, g["floor"].values)

# Standard lookups for fallback
fa_stats = train.groupby("full_addr").agg(p_mean=("price","mean"), count=("price","count"))
ua_stats = train.groupby("unit_area5").agg(ppsf_median=("ppsf","median"), floor_mean=("floor","mean"), count=("price","count"))
unit_stats = train.groupby("unit_key").agg(ppsf_median=("ppsf","median"), floor_mean=("floor","mean"), count=("price","count"))
bt_stats = train.groupby("bld_tower").agg(ppsf_median=("ppsf","median"), floor_mean=("floor","mean"), count=("price","count"))
bf_stats = train.groupby("bld_flat").agg(ppsf_median=("ppsf","median"), floor_mean=("floor","mean"), count=("price","count"))
bld_stats = train.groupby("building").agg(ppsf_median=("ppsf","median"), floor_mean=("floor","mean"), count=("price","count"))
dist_stats = train.groupby("district").agg(ppsf_median=("ppsf","median"))

scaler = StandardScaler()
X_tr = scaler.fit_transform(train[["wgs_lat","wgs_lon","area_sqft","floor"]].values)
X_te = scaler.transform(test[["wgs_lat","wgs_lon","area_sqft","floor"]].values)
knn = KNeighborsRegressor(n_neighbors=10, weights="distance", n_jobs=-1)
knn.fit(X_tr, train["price"].values)
test["knn10"] = knn.predict(X_te)


def gauss_w(floors, test_floor, sigma):
    """Gaussian floor weights."""
    d = np.abs(floors - test_floor)
    w = np.exp(-d**2 / (2 * sigma**2))
    return w / w.sum()


def gauss_weighted_ppsf(ppsf_arr, floor_arr, test_floor, sigma):
    """Floor-weighted ppsf."""
    w = gauss_w(floor_arr, test_floor, sigma)
    return (ppsf_arr * w).sum()


def gen(sigma=0.7, fb_sigma=None, n1_sd=0.85, n1_sb=0.05, n1_sk=0.10,
        n1_bld_floor_weighted=False, fb_floor_weighted=False):
    """
    sigma: Gaussian sigma for n>=2 full_addr groups
    fb_sigma: sigma for fallback groups (None = use standard cascade)
    n1_bld_floor_weighted: use floor-weighted building ppsf for n=1 blend
    fb_floor_weighted: use floor-weighted ppsf in fallback cascade
    """
    if fb_sigma is None:
        fb_sigma = sigma

    preds = np.zeros(len(test))
    for i in range(len(test)):
        row = test.iloc[i]
        area, fv = row["area_sqft"], row["floor"]
        fa = row["full_addr"]
        bk = row["building"]

        if fa in fa_stats.index:
            n = int(fa_stats.loc[fa]["count"])

            if n >= 2 and fa in fa_data:
                prices, floors = fa_data[fa]
                w = gauss_w(floors, fv, sigma)
                preds[i] = (prices * w).sum()
            elif n == 1:
                direct = fa_stats.loc[fa]["p_mean"]
                if bk in bld_stats.index:
                    if n1_bld_floor_weighted and bk in bld_data:
                        ppsfs, floors = bld_data[bk]
                        bld_ppsf = gauss_weighted_ppsf(ppsfs, floors, fv, fb_sigma)
                    else:
                        bld_ppsf = bld_stats.loc[bk]["ppsf_median"]
                    bp = area * bld_ppsf
                    preds[i] = n1_sd * direct + n1_sb * bp + n1_sk * row["knn10"]
                else:
                    preds[i] = (n1_sd + n1_sb) * direct + n1_sk * row["knn10"]
            else:
                preds[i] = fa_stats.loc[fa]["p_mean"]
        else:
            # Fallback cascade — optionally with floor weighting
            uak = row["unit_area5"]; uk = row["unit_key"]
            btk = row["bld_tower"]; bfk = row["bld_flat"]; dk = row["district"]

            if fb_floor_weighted:
                # Floor-weighted fallback
                if uak in ua_data:
                    ppsfs, floors, areas = ua_data[uak]
                    if len(ppsfs) >= 2:
                        wp = gauss_weighted_ppsf(ppsfs, floors, fv, fb_sigma)
                        preds[i] = area * wp
                    else:
                        preds[i] = area * ppsfs.mean()
                elif uk in uk_data:
                    ppsfs, floors, areas = uk_data[uk]
                    if len(ppsfs) >= 3:
                        wp = gauss_weighted_ppsf(ppsfs, floors, fv, fb_sigma)
                        preds[i] = area * wp
                    else:
                        preds[i] = area * ppsfs.mean()
                elif btk in bt_data and len(bt_data[btk][0]) >= 3:
                    ppsfs, floors = bt_data[btk]
                    wp = gauss_weighted_ppsf(ppsfs, floors, fv, fb_sigma)
                    preds[i] = area * wp
                elif bfk in bf_data and len(bf_data[bfk][0]) >= 3:
                    ppsfs, floors = bf_data[bfk]
                    wp = gauss_weighted_ppsf(ppsfs, floors, fv, fb_sigma)
                    preds[i] = area * wp
                elif bk in bld_data:
                    ppsfs, floors = bld_data[bk]
                    if len(ppsfs) >= 3:
                        wp = gauss_weighted_ppsf(ppsfs, floors, fv, fb_sigma)
                    else:
                        wp = ppsfs.mean()
                    preds[i] = area * wp
                else:
                    kp = row["knn10"]
                    if dk in dist_stats.index:
                        preds[i] = 0.4*kp + 0.6*area*dist_stats.loc[dk]["ppsf_median"]
                    else:
                        preds[i] = kp
            else:
                # Standard fallback (same as baseline)
                if uak in ua_stats.index:
                    d=ua_stats.loc[uak]; preds[i]=area*(d["ppsf_median"] if d["count"]>=2 else d["ppsf_median"])
                elif uk in unit_stats.index:
                    d=unit_stats.loc[uk]; preds[i]=area*d["ppsf_median"]
                elif btk in bt_stats.index and bt_stats.loc[btk]["count"]>=3:
                    preds[i]=area*bt_stats.loc[btk]["ppsf_median"]
                elif bfk in bf_stats.index and bf_stats.loc[bfk]["count"]>=3:
                    preds[i]=area*bf_stats.loc[bfk]["ppsf_median"]
                elif bk in bld_stats.index:
                    d=bld_stats.loc[bk]; preds[i]=area*(d["ppsf_median"] if d["count"]>=3 else d["ppsf_median"])
                else:
                    kp=row["knn10"]
                    if dk in dist_stats.index:
                        preds[i]=0.4*kp+0.6*area*dist_stats.loc[dk]["ppsf_median"]
                    else:
                        preds[i]=kp

    return np.clip(preds, 2000, 500000).astype(int)


# ============================================================
print("Generating variants...\n")

baseline = gen(sigma=0.7, fb_floor_weighted=False, n1_bld_floor_weighted=False)

configs = [
    # 1. Current best: Gaussian sigma=0.7, standard fallback
    ("1_s07_baseline", 0.7, None, 0.85, 0.05, 0.10, False, False),
    # 2. + floor-weighted fallback cascade
    ("2_s07_fw_fb", 0.7, 3.0, 0.85, 0.05, 0.10, False, True),
    # 3. + floor-weighted building for n=1 blend
    ("3_s07_fw_n1bld", 0.7, 3.0, 0.85, 0.05, 0.10, True, False),
    # 4. + both: floor-weighted fb AND n=1 building
    ("4_s07_fw_both", 0.7, 3.0, 0.85, 0.05, 0.10, True, True),
    # 5. Pure direct for n=1 + fw fb
    ("5_s07_pured_fwfb", 0.7, 3.0, 1.0, 0.0, 0.0, False, True),
    # 6. fw fb with tighter sigma
    ("6_s07_fwfb_s1", 0.7, 1.0, 0.85, 0.05, 0.10, False, True),
    # 7. fw everything, tighter sigma
    ("7_s07_fwall_s1", 0.7, 1.0, 0.85, 0.05, 0.10, True, True),
    # 8. n=1 different blend: 90/0/10 (no building, more direct)
    ("8_s07_n1_90_0_10", 0.7, None, 0.90, 0.00, 0.10, False, False),
    # 9. n=1 different: 80/10/10 (more building)
    ("9_s07_n1_80_10_10", 0.7, None, 0.80, 0.10, 0.10, False, False),
    # 10. Everything combined: fw fb + fw n1 bld + n=1 90/0/10
    ("10_s07_combo", 0.7, 3.0, 0.90, 0.00, 0.10, True, True),
]

print(f"{'#':>2s} {'Name':30s} {'Changed':>8s} {'AvgDiff':>9s}")
print("-" * 53)

for idx, (name, s, fbs, sd, sb, sk, n1fw, fbfw) in enumerate(configs):
    p = gen(sigma=s, fb_sigma=fbs, n1_sd=sd, n1_sb=sb, n1_sk=sk,
            n1_bld_floor_weighted=n1fw, fb_floor_weighted=fbfw)
    diff = np.abs(p.astype(float) - baseline.astype(float))
    changed = (diff > 0).sum()
    avg_d = diff[diff > 0].mean() if changed > 0 else 0
    print(f"{idx+1:2d} {name:30s} {changed:8d} {avg_d:9.0f}")
    pd.DataFrame({"id": test["id"].astype(int), "price": p}).to_csv(f"{idx+1}.csv", index=False)

pd.DataFrame({"id": test["id"].astype(int), "price": baseline}).to_csv("my_submission.csv", index=False)
print(f"\nmy_submission.csv = sigma=0.7 baseline ($1,300)")
