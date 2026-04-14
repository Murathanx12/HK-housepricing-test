"""
Break the $1,355 plateau.

CV analysis showed: n=1 luxury is the bottleneck.
80% of MSE from rows where we have no good answer.

New strategy: use CV on MATCHED rows only (exclude fallback)
to get a proxy that correlates with leaderboard.
Then sweep all params and find what actually moves the needle.

Also try: data-quality-aware blending for n=1.
"""

import pandas as pd, numpy as np
from pathlib import Path
from scipy.stats import trim_mean
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold

DATA_DIR = Path("./data")
train_full = pd.read_csv(DATA_DIR / "HK_house_transactions.csv")
test = pd.read_csv(DATA_DIR / "test_features.csv")

def get_building(addr):
    if pd.isna(addr): return "UNKNOWN"
    return addr.split(",")[0].strip()

for df in [train_full, test]:
    df["building"] = df["address"].apply(get_building)
    df["floor"] = pd.to_numeric(df["floor"], errors="coerce").fillna(10)
    df["unit_key"] = df["building"]+"|"+df["Tower"].fillna("X").astype(str)+"|"+df["Flat"].fillna("X")
    df["bld_tower"] = df["building"]+"|T"+df["Tower"].fillna("X").astype(str)
    df["bld_flat"] = df["building"]+"|F"+df["Flat"].fillna("X")
    df["full_addr"] = df["address"].fillna("")+"|"+df["area_sqft"].astype(str)
    df["area_bin5"] = (df["area_sqft"]/5).round()*5
    df["unit_area5"] = df["unit_key"]+"|"+df["area_bin5"].astype(str)
train_full["ppsf"] = train_full["price"]/train_full["area_sqft"]


def build_and_predict(train_df, predict_df, n1_sd=0.85, n1_sb=0.05, n1_sk=0.10,
                      n3_agg="mean", m4_agg="trimmed",
                      n1_luxury_sd=None, n1_luxury_sb=None, n1_luxury_sk=None,
                      luxury_thresh=50000, n1_small_bld_pure=False):
    """Full pipeline. Returns (predictions, match_types)."""
    tr = train_df.copy()
    tr["ppsf"] = tr["price"]/tr["area_sqft"]

    def floor_slope(g):
        if len(g)<5 or g["floor"].std()<1: return 0.0
        return np.polyfit(g["floor"],g["ppsf"],1)[0]

    bld_slopes = tr.groupby("building").apply(floor_slope, include_groups=False).to_dict()
    fa_grp = tr.groupby("full_addr")
    fa_stats = fa_grp.agg(p_mean=("price","mean"),p_median=("price","median"),count=("price","count"),floor_mean=("floor","mean"))
    fa_trimmed = fa_grp["price"].apply(lambda x: trim_mean(x,0.1) if len(x)>=4 else x.mean())
    fa_stats = fa_stats.join(fa_trimmed.rename("p_trimmed"))
    ua_stats = tr.groupby("unit_area5").agg(ppsf_mean=("ppsf","mean"),ppsf_median=("ppsf","median"),floor_mean=("floor","mean"),count=("price","count"))
    unit_stats = tr.groupby("unit_key").agg(ppsf_mean=("ppsf","mean"),ppsf_median=("ppsf","median"),floor_mean=("floor","mean"),count=("price","count"))
    bt_stats = tr.groupby("bld_tower").agg(ppsf_median=("ppsf","median"),floor_mean=("floor","mean"),count=("price","count"))
    bf_stats = tr.groupby("bld_flat").agg(ppsf_median=("ppsf","median"),floor_mean=("floor","mean"),count=("price","count"))
    bld_stats = tr.groupby("building").agg(ppsf_mean=("ppsf","mean"),ppsf_median=("ppsf","median"),floor_mean=("floor","mean"),count=("price","count"))
    dist_stats = tr.groupby("district").agg(ppsf_median=("ppsf","median"))

    sc = StandardScaler()
    X_tr = sc.fit_transform(tr[["wgs_lat","wgs_lon","area_sqft","floor"]].values)
    X_te = sc.transform(predict_df[["wgs_lat","wgs_lon","area_sqft","floor"]].values)
    knn = KNeighborsRegressor(n_neighbors=10,weights="distance",n_jobs=-1)
    knn.fit(X_tr, tr["price"].values)
    knn_preds = knn.predict(X_te)

    preds = np.zeros(len(predict_df))
    mtypes = []
    for i in range(len(predict_df)):
        row = predict_df.iloc[i]
        area,fv = row["area_sqft"],row["floor"]
        fa = row["full_addr"]
        slope = bld_slopes.get(row["building"],0.0)
        bk = row["building"]

        if fa in fa_stats.index:
            d = fa_stats.loc[fa]; n = int(d["count"])
            if n >= 4:
                preds[i] = d["p_trimmed"] if m4_agg=="trimmed" else d["p_mean"]
                mtypes.append("4+")
            elif n == 3:
                preds[i] = d["p_mean"] if n3_agg=="mean" else d["p_median"]
                mtypes.append("n3")
            elif n == 2:
                preds[i] = d["p_mean"]
                mtypes.append("n2")
            else:
                direct = d["p_mean"]
                # Data-quality-aware blend for n=1
                bld_n = bld_stats.loc[bk]["count"] if bk in bld_stats.index else 0

                # Use luxury blend if applicable
                is_luxury = (n1_luxury_sd is not None and direct > luxury_thresh)
                # Use pure direct if building has <3 training rows
                is_small_bld = (n1_small_bld_pure and bld_n < 3)

                if is_small_bld:
                    preds[i] = direct
                    mtypes.append("n1_pure")
                elif is_luxury:
                    sd, sb, sk = n1_luxury_sd, n1_luxury_sb, n1_luxury_sk
                    if bk in bld_stats.index:
                        bp = area*bld_stats.loc[bk]["ppsf_median"]
                        preds[i] = sd*direct + sb*bp + sk*knn_preds[i]
                    else:
                        preds[i] = (sd+sb)*direct + sk*knn_preds[i]
                    mtypes.append("n1_lux")
                else:
                    if bk in bld_stats.index:
                        bp = area*bld_stats.loc[bk]["ppsf_median"]
                        preds[i] = n1_sd*direct + n1_sb*bp + n1_sk*knn_preds[i]
                    else:
                        preds[i] = (n1_sd+n1_sb)*direct + n1_sk*knn_preds[i]
                    mtypes.append("n1")
        else:
            uak,uk,btk,bfk,dk = row["unit_area5"],row["unit_key"],row["bld_tower"],row["bld_flat"],row["district"]
            if uak in ua_stats.index:
                d=ua_stats.loc[uak]; base=d["ppsf_median"] if d["count"]>=2 else d["ppsf_mean"]
                fadj=slope*(fv-d["floor_mean"]) if d["count"]>=2 else 0; preds[i]=area*(base+fadj)
            elif uk in unit_stats.index:
                d=unit_stats.loc[uk]; base=d["ppsf_median"] if d["count"]>=3 else d["ppsf_mean"]
                preds[i]=area*(base+slope*(fv-d["floor_mean"]))
            elif btk in bt_stats.index and bt_stats.loc[btk]["count"]>=3:
                d=bt_stats.loc[btk]; preds[i]=area*(d["ppsf_median"]+slope*(fv-d["floor_mean"]))
            elif bfk in bf_stats.index and bf_stats.loc[bfk]["count"]>=3:
                d=bf_stats.loc[bfk]; preds[i]=area*(d["ppsf_median"]+slope*(fv-d["floor_mean"]))
            elif bk in bld_stats.index:
                d=bld_stats.loc[bk]; base=d["ppsf_median"] if d["count"]>=3 else d["ppsf_mean"]
                fadj=slope*(fv-d["floor_mean"]) if d["count"]>=5 else 0; preds[i]=area*(base+fadj)
            else:
                if dk in dist_stats.index:
                    preds[i]=0.4*knn_preds[i]+0.6*area*dist_stats.loc[dk]["ppsf_median"]
                else:
                    preds[i]=knn_preds[i]
            mtypes.append("fb")

    return np.clip(preds, 2000, 500000), mtypes


# ══════════════════════════════════════════════════
# CV ON MATCHED ROWS ONLY (proxy for leaderboard)
# ══════════════════════════════════════════════════
print("="*60)
print("CV ON MATCHED ROWS ONLY")
print("="*60)

def cv_matched(configs, n_folds=5, seed=42):
    """Run CV and compute RMSE only on matched rows (excluding fallback)."""
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    results = {}

    for name, kwargs in configs.items():
        matched_errors = []
        all_errors = []
        for train_idx, val_idx in kf.split(train_full):
            tr = train_full.iloc[train_idx]
            va = train_full.iloc[val_idx]
            preds, mtypes = build_and_predict(tr, va, **kwargs)
            actual = va["price"].values
            errors = preds - actual

            for j, mt in enumerate(mtypes):
                if mt != "fb":
                    matched_errors.append(errors[j]**2)
                all_errors.append(errors[j]**2)

        rmse_matched = np.sqrt(np.mean(matched_errors))
        rmse_all = np.sqrt(np.mean(all_errors))
        results[name] = (rmse_matched, rmse_all, len(matched_errors))

    return results

# Define configs to test
configs = {
    "baseline_85_5_10_mean3": dict(n1_sd=0.85,n1_sb=0.05,n1_sk=0.10,n3_agg="mean"),
    "pure_direct_100": dict(n1_sd=1.0,n1_sb=0.0,n1_sk=0.0,n3_agg="mean"),
    "n1_90_5_5": dict(n1_sd=0.90,n1_sb=0.05,n1_sk=0.05,n3_agg="mean"),
    "n1_88_5_7": dict(n1_sd=0.88,n1_sb=0.05,n1_sk=0.07,n3_agg="mean"),
    "n1_85_10_5": dict(n1_sd=0.85,n1_sb=0.10,n1_sk=0.05,n3_agg="mean"),
    "n1_80_10_10": dict(n1_sd=0.80,n1_sb=0.10,n1_sk=0.10,n3_agg="mean"),
    "median_n3": dict(n1_sd=0.85,n1_sb=0.05,n1_sk=0.10,n3_agg="median"),
    "small_bld_pure": dict(n1_sd=0.85,n1_sb=0.05,n1_sk=0.10,n3_agg="mean",n1_small_bld_pure=True),
    "lux_75_10_15": dict(n1_sd=0.85,n1_sb=0.05,n1_sk=0.10,n3_agg="mean",
                         n1_luxury_sd=0.75,n1_luxury_sb=0.10,n1_luxury_sk=0.15),
    "lux_90_5_5": dict(n1_sd=0.85,n1_sb=0.05,n1_sk=0.10,n3_agg="mean",
                        n1_luxury_sd=0.90,n1_luxury_sb=0.05,n1_luxury_sk=0.05),
    "lux_pure": dict(n1_sd=0.85,n1_sb=0.05,n1_sk=0.10,n3_agg="mean",
                     n1_luxury_sd=1.0,n1_luxury_sb=0.0,n1_luxury_sk=0.0),
    "combo_pure_lux_small": dict(n1_sd=0.85,n1_sb=0.05,n1_sk=0.10,n3_agg="mean",
                                  n1_small_bld_pure=True,
                                  n1_luxury_sd=1.0,n1_luxury_sb=0.0,n1_luxury_sk=0.0),
    "mean4": dict(n1_sd=0.85,n1_sb=0.05,n1_sk=0.10,n3_agg="mean",m4_agg="mean"),
}

print("Running 5-fold CV (matched rows only)...\n")
results = cv_matched(configs)

# Show results sorted by matched RMSE
print(f"{'Config':35s} {'Matched RMSE':>13s} {'All RMSE':>10s} {'N_matched':>10s}")
print("-"*72)
for name, (rm, ra, nm) in sorted(results.items(), key=lambda x: x[1][0]):
    # Known LB score for calibration
    lb = ""
    if name == "baseline_85_5_10_mean3": lb = " (LB=$1,355)"
    elif name == "median_n3": lb = " (LB=$1,393)"
    elif name == "pure_direct_100": lb = " (LB=$1,356)"
    print(f"{name:35s} ${rm:>10,.0f} ${ra:>8,.0f} {nm:>10d}{lb}")

# ══════════════════════════════════════════════════
# CALIBRATE AND PREDICT
# ══════════════════════════════════════════════════
print(f"\n{'='*60}")
print("CALIBRATION")
print("="*60)

# Use known LB scores to calibrate
known = {
    "baseline_85_5_10_mean3": 1355,
    "median_n3": 1393,
    "pure_direct_100": 1356,
}

cv_vals = [results[k][0] for k in known]
lb_vals = [known[k] for k in known]
a, b = np.polyfit(cv_vals, lb_vals, 1)
print(f"Calibration: LB = {a:.4f} * CV_matched + {b:.0f}")

for name, (rm, ra, nm) in sorted(results.items(), key=lambda x: x[1][0]):
    predicted_lb = a * rm + b
    lb_actual = known.get(name, "")
    actual_str = f" (actual=${lb_actual})" if lb_actual else ""
    print(f"  {name:35s}: CV_m=${rm:,.0f} -> predicted_LB=${predicted_lb:,.0f}{actual_str}")

# ══════════════════════════════════════════════════
# GENERATE SUBMISSIONS FOR TOP CONFIGS
# ══════════════════════════════════════════════════
print(f"\n{'='*60}")
print("GENERATING SUBMISSIONS")
print("="*60)

# Pick top 5 by predicted LB
ranked = sorted(results.items(), key=lambda x: a*x[1][0]+b)
baseline_preds, _ = build_and_predict(train_full, test, **configs["baseline_85_5_10_mean3"])

for rank, (name, (rm, ra, nm)) in enumerate(ranked[:6]):
    predicted_lb = a * rm + b
    preds, _ = build_and_predict(train_full, test, **configs[name])
    diff = np.abs(preds - baseline_preds)
    changed = (diff > 1).sum()
    fname = f"sub_{name}.csv"
    pd.DataFrame({"id":test["id"].astype(int),"price":preds.astype(int)}).to_csv(fname, index=False)
    print(f"  {rank+1}. {fname:40s} pred_LB=${predicted_lb:,.0f} ({changed} diff)")

pd.DataFrame({"id":test["id"].astype(int),"price":baseline_preds.astype(int)}).to_csv("my_submission.csv",index=False)
