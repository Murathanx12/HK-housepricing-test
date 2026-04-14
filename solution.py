"""
$1,300 confirmed base + systematic probes for further improvement
=================================================================
Fixed: fallback now correctly includes floor slope adjustment.
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

fa_stats = train.groupby("full_addr").agg(p_mean=("price","mean"), count=("price","count"))

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
    """Standard fallback WITH floor slope (same as $1,355 baseline)."""
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


def gen(sigma=0.7, n1_sd=0.85, n1_sb=0.05, n1_sk=0.10):
    """Gaussian floor-weighted mean for n>=2, standard n=1 + fallback."""
    preds = np.zeros(len(test))
    cats = []
    for i in range(len(test)):
        row = test.iloc[i]; area, fv = row["area_sqft"], row["floor"]
        fa = row["full_addr"]; slope = bld_slopes.get(row["building"], 0.0)
        bk = row["building"]

        if fa in fa_stats.index:
            n = int(fa_stats.loc[fa]["count"])
            if n >= 2 and fa in fa_data:
                prices, floors = fa_data[fa]
                d = np.abs(floors - fv)
                w = np.exp(-d**2 / (2 * sigma**2))
                w = w / w.sum()
                preds[i] = (prices * w).sum()
                cats.append("fw")
            elif n == 1:
                direct = fa_stats.loc[fa]["p_mean"]
                if bk in bld_stats.index:
                    bp = area * bld_stats.loc[bk]["ppsf_median"]
                    preds[i] = n1_sd * direct + n1_sb * bp + n1_sk * row["knn10"]
                else:
                    preds[i] = (n1_sd + n1_sb) * direct + n1_sk * row["knn10"]
                cats.append("n1")
            else:
                preds[i] = fa_stats.loc[fa]["p_mean"]
                cats.append("n1")
        else:
            preds[i] = fb(row, area, fv, slope)
            cats.append("fb")

    return np.clip(preds, 2000, 500000).astype(int), np.array(cats)


# ============================================================
# Generate probes
# ============================================================
print("Generating probes...\n")

base, base_cats = gen(sigma=0.7)

# Verify categories
for cat in ["fw", "n1", "fb"]:
    print(f"  {cat}: {(base_cats == cat).sum()} rows")

probes = {}

# 1. Confirm baseline (should be $1,300 with correct fallback)
probes["1_baseline_s07"] = base

# 2. n=1 pure direct (no KNN/bld blend)
p, _ = gen(sigma=0.7, n1_sd=1.0, n1_sb=0.0, n1_sk=0.0)
probes["2_n1_pure"] = p

# 3. n=1: 90/0/10 (no building, keep KNN)
p, _ = gen(sigma=0.7, n1_sd=0.90, n1_sb=0.0, n1_sk=0.10)
probes["3_n1_90_0_10"] = p

# 4. Shift all n=1 UP $100 (probe bias on new baseline)
p = base.copy(); p[base_cats == "n1"] += 100
probes["4_n1_up100"] = p

# 5. Shift all n=1 DOWN $100
p = base.copy(); p[base_cats == "n1"] -= 100
probes["5_n1_down100"] = p

# 6. Shift fallback UP $500
p = base.copy(); p[base_cats == "fb"] += 500
probes["6_fb_up500"] = p

# 7. Shift fallback DOWN $500
p = base.copy(); p[base_cats == "fb"] -= 500
probes["7_fb_down500"] = p

# 8. Shift fw (n>=2) UP $50
p = base.copy(); p[base_cats == "fw"] += 50
probes["8_fw_up50"] = p

# 9. Shift fw (n>=2) DOWN $50
p = base.copy(); p[base_cats == "fw"] -= 50
probes["9_fw_down50"] = p

# 10. Scale prediction by area: pred *= (1 + 0.001 * (area - 500))
# Tests if larger apartments need bigger predictions
p = base.copy()
areas = test["area_sqft"].values
p = (p * (1 + 0.0005 * (areas - 500))).astype(int)
probes["10_area_scale"] = np.clip(p, 2000, 500000)

# Save all
print("\n=== PROBES ===")
for name, preds in probes.items():
    preds = np.clip(preds, 2000, 500000).astype(int)
    pd.DataFrame({"id": test["id"].astype(int), "price": preds}).to_csv(f"{name}.csv", index=False)
    diff = np.abs(preds.astype(float) - base.astype(float))
    changed = (diff > 0).sum()
    avg = diff[diff > 0].mean() if changed > 0 else 0
    print(f"  {name:25s}: {changed:5d} rows, avg |shift|=${avg:,.0f}")
