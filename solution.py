"""
Hong Kong Rental Price Prediction — Size-Adjusted Outlier Detection
====================================================================
Key insight: building median ppsf is too coarse. Large units legitimately
have different ppsf. Use per-building ppsf~area regression to get
SIZE-ADJUSTED expected ppsf, then detect outliers from that.
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

# ── Lookups ──
def floor_slope(g):
    if len(g) < 5 or g["floor"].std() < 1: return 0.0
    return np.polyfit(g["floor"], g["ppsf"], 1)[0]

bld_slopes = train.groupby("building").apply(floor_slope, include_groups=False).to_dict()

fa_grp = train.groupby("full_addr")
fa_stats = fa_grp.agg(p_mean=("price","mean"), p_median=("price","median"), count=("price","count"), floor_mean=("floor","mean"))
fa_trimmed = fa_grp["price"].apply(lambda x: trim_mean(x, 0.1) if len(x) >= 4 else x.mean())
fa_stats = fa_stats.join(fa_trimmed.rename("p_trimmed"))

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

# ── Size-adjusted building models ──
bld_models = {}      # building -> polyfit coefficients (ppsf ~ area)
bld_resid_std = {}   # building -> residual std

for bld, g in train.groupby("building"):
    if len(g) >= 10 and g["area_sqft"].std() > 10:
        try:
            coeffs = np.polyfit(g["area_sqft"], g["ppsf"], 1)
            predicted = np.polyval(coeffs, g["area_sqft"])
            resid_std = (g["ppsf"] - predicted).std()
            if resid_std > 0:
                bld_models[bld] = coeffs
                bld_resid_std[bld] = resid_std
        except:
            pass

print(f"Buildings with size-adjusted model: {len(bld_models)}")

# Also build ppsf ~ area + floor models for buildings with enough data
bld_models_2d = {}
bld_resid_std_2d = {}
for bld, g in train.groupby("building"):
    if len(g) >= 20 and g["area_sqft"].std() > 10 and g["floor"].std() > 1:
        try:
            X = np.column_stack([g["area_sqft"], g["floor"]])
            y = g["ppsf"].values
            # Linear: ppsf = a*area + b*floor + c
            A = np.column_stack([X, np.ones(len(X))])
            coeffs, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
            predicted = A @ coeffs
            resid_std = (y - predicted).std()
            if resid_std > 0:
                bld_models_2d[bld] = coeffs
                bld_resid_std_2d[bld] = resid_std
        except:
            pass

print(f"Buildings with area+floor model: {len(bld_models_2d)}")


def get_adj_z_and_expected(bld, area, floor_val, ppsf_direct):
    """Get size-adjusted z-score and expected price."""
    # Try 2D model first (area + floor)
    if bld in bld_models_2d:
        coeffs = bld_models_2d[bld]
        exp_ppsf = coeffs[0] * area + coeffs[1] * floor_val + coeffs[2]
        z = (ppsf_direct - exp_ppsf) / bld_resid_std_2d[bld]
        return z, exp_ppsf * area
    # Fall back to 1D model (area only)
    if bld in bld_models:
        exp_ppsf = np.polyval(bld_models[bld], area)
        z = (ppsf_direct - exp_ppsf) / bld_resid_std[bld]
        return z, exp_ppsf * area
    return 0.0, None


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


def gen(z_thresh=2.5, outlier_sd=0.50, outlier_sb=0.25, outlier_sk=0.25,
        normal_sd=0.85, normal_sb=0.05, normal_sk=0.10, use_adj_price=False):
    """
    use_adj_price: if True, outlier correction blends toward size-adjusted
                   expected price instead of flat building median.
    """
    preds = np.zeros(len(test))
    n_outliers = 0
    for i in range(len(test)):
        row = test.iloc[i]
        area, fv = row["area_sqft"], row["floor"]
        fa = row["full_addr"]
        slope = bld_slopes.get(row["building"], 0.0)
        bk = row["building"]

        if fa in fa_stats.index:
            d = fa_stats.loc[fa]; n = int(d["count"])
            if n >= 4:
                preds[i] = d["p_trimmed"]
            elif n == 3:
                preds[i] = d["p_mean"]
            elif n == 2:
                preds[i] = d["p_mean"]
            else:
                direct = d["p_mean"]
                knn_p = row["knn10"]
                ppsf_direct = direct / area

                # Size-adjusted z-score
                adj_z, adj_exp_price = get_adj_z_and_expected(bk, area, fv, ppsf_direct)

                if abs(adj_z) > z_thresh and adj_exp_price is not None:
                    # Outlier detected with size-adjusted model
                    if use_adj_price:
                        bp = adj_exp_price  # Size-adjusted expected price
                    elif bk in bld_stats.index:
                        bp = area * bld_stats.loc[bk]["ppsf_median"]
                    else:
                        bp = direct
                    preds[i] = outlier_sd * direct + outlier_sb * bp + outlier_sk * knn_p
                    n_outliers += 1
                else:
                    # Normal n=1
                    if bk in bld_stats.index:
                        bp = area * bld_stats.loc[bk]["ppsf_median"]
                        preds[i] = normal_sd * direct + normal_sb * bp + normal_sk * knn_p
                    else:
                        preds[i] = (normal_sd + normal_sb) * direct + normal_sk * knn_p
        else:
            preds[i] = fb(row, area, fv, slope)

    preds = np.clip(preds, 2000, 500000)
    return preds.astype(int), n_outliers


# ── Generate variants ──
print("\nGenerating variants...\n")
baseline = gen(z_thresh=999)  # No outlier correction = baseline $1,355

configs = []

# Size-adjusted z-score with various thresholds and correction strengths
for z in [2.0, 2.5, 3.0, 3.5]:
    for osd, osb, osk in [(0.50, 0.25, 0.25), (0.40, 0.30, 0.30), (0.60, 0.20, 0.20),
                           (0.70, 0.15, 0.15), (0.30, 0.35, 0.35), (0.20, 0.40, 0.40)]:
        # Use flat building ppsf for correction
        configs.append((f"adj_z{z}_{osd:.0%}d_{osb:.0%}b_{osk:.0%}k",
                        z, osd, osb, osk, False))
        # Use size-adjusted price for correction
        configs.append((f"adj_z{z}_{osd:.0%}d_{osb:.0%}a_{osk:.0%}k_adjp",
                        z, osd, osb, osk, True))

print(f"{'Name':50s} {'#out':>5s} {'Changed':>8s} {'AvgDiff':>9s}")
print("-" * 75)

results = []
for name, z, osd, osb, osk, use_adj in configs:
    p, n_out = gen(z_thresh=z, outlier_sd=osd, outlier_sb=osb, outlier_sk=osk, use_adj_price=use_adj)
    diff = np.abs(p.astype(float) - baseline[0].astype(float))
    changed = (diff > 0).sum()
    avg_d = diff[diff > 0].mean() if changed > 0 else 0
    results.append((name, n_out, changed, avg_d, p))
    if n_out > 0:
        print(f"{name:50s} {n_out:5d} {changed:8d} {avg_d:9.0f}")

# Save best candidates
print("\n" + "="*75)
print("SAVING BEST CANDIDATES")
print("="*75)

saved = 0
for name, n_out, changed, avg_d, p in results:
    if n_out > 0 and saved < 20:
        pd.DataFrame({"id": test["id"].astype(int), "price": p}).to_csv(f"sub_{name}.csv", index=False)
        saved += 1

# My pick: z=2.5, 50/25/25, using size-adjusted price for correction
for name, n_out, changed, avg_d, p in results:
    if "adj_z2.5_50%" in name and "adjp" in name:
        pd.DataFrame({"id": test["id"].astype(int), "price": p}).to_csv("my_submission.csv", index=False)
        print(f"\nmy_submission.csv = {name} ({n_out} outliers)")
        break

# Also save the gentlest versions
for name, n_out, changed, avg_d, p in results:
    if "adj_z3.5" in name and "70%" in name and "adjp" in name:
        pd.DataFrame({"id": test["id"].astype(int), "price": p}).to_csv("sub_gentlest.csv", index=False)
        print(f"sub_gentlest.csv = {name} ({n_out} outliers)")
        break

print(f"\nSaved {saved} variants total")
