"""
Internal test engine + outlier hunt.

We have the EXACT evaluation formula: RMSE on id-matched predictions vs hidden labels.
We can't get the labels, but we CAN build a holdout that mimics the real test.

Strategy:
1. The real test has 8,633 rows from a pool of ~47K transactions
2. Split training 80/20, predict the 20% holdout, compute RMSE
3. Calibrate: find a split whose RMSE matches the leaderboard
4. Use that split to test improvements INTERNALLY
5. Hunt for the outlier rows that dominate RMSE
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
                      n3_agg="mean", m4_agg="trimmed"):
    """Build all lookups from train_df, predict predict_df. Returns predictions."""
    train_df = train_df.copy()
    train_df["ppsf"] = train_df["price"] / train_df["area_sqft"]

    def floor_slope(g):
        if len(g)<5 or g["floor"].std()<1: return 0.0
        return np.polyfit(g["floor"],g["ppsf"],1)[0]

    bld_slopes = train_df.groupby("building").apply(floor_slope, include_groups=False).to_dict()
    fa_grp = train_df.groupby("full_addr")
    fa_stats = fa_grp.agg(p_mean=("price","mean"),p_median=("price","median"),
        count=("price","count"),floor_mean=("floor","mean"))
    fa_trimmed = fa_grp["price"].apply(lambda x: trim_mean(x,0.1) if len(x)>=4 else x.mean())
    fa_stats = fa_stats.join(fa_trimmed.rename("p_trimmed"))

    ua_stats = train_df.groupby("unit_area5").agg(ppsf_mean=("ppsf","mean"),ppsf_median=("ppsf","median"),floor_mean=("floor","mean"),count=("price","count"))
    unit_stats = train_df.groupby("unit_key").agg(ppsf_mean=("ppsf","mean"),ppsf_median=("ppsf","median"),floor_mean=("floor","mean"),count=("price","count"))
    bt_stats = train_df.groupby("bld_tower").agg(ppsf_median=("ppsf","median"),floor_mean=("floor","mean"),count=("price","count"))
    bf_stats = train_df.groupby("bld_flat").agg(ppsf_median=("ppsf","median"),floor_mean=("floor","mean"),count=("price","count"))
    bld_stats = train_df.groupby("building").agg(ppsf_mean=("ppsf","mean"),ppsf_median=("ppsf","median"),floor_mean=("floor","mean"),count=("price","count"))
    dist_stats = train_df.groupby("district").agg(ppsf_median=("ppsf","median"))

    sc = StandardScaler()
    X_tr = sc.fit_transform(train_df[["wgs_lat","wgs_lon","area_sqft","floor"]].values)
    X_te = sc.transform(predict_df[["wgs_lat","wgs_lon","area_sqft","floor"]].values)
    knn = KNeighborsRegressor(n_neighbors=10,weights="distance",n_jobs=-1)
    knn.fit(X_tr, train_df["price"].values)
    knn_preds = knn.predict(X_te)

    preds = np.zeros(len(predict_df))
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
            elif n == 3:
                preds[i] = d["p_mean"] if n3_agg=="mean" else d["p_median"]
            elif n == 2:
                preds[i] = d["p_mean"]
            else:
                direct = d["p_mean"]
                if bk in bld_stats.index:
                    bp = area*bld_stats.loc[bk]["ppsf_median"]
                    preds[i] = n1_sd*direct + n1_sb*bp + n1_sk*knn_preds[i]
                else:
                    preds[i] = (n1_sd+n1_sb)*direct + n1_sk*knn_preds[i]
        else:
            # Fallback cascade
            uak,uk,btk,bfk = row["unit_area5"],row["unit_key"],row["bld_tower"],row["bld_flat"]
            dk = row["district"]
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

    return np.clip(preds, 2000, 500000)


# ══════════════════════════════════════════════════
# INTERNAL CROSS-VALIDATION
# ══════════════════════════════════════════════════
print("="*60)
print("INTERNAL CROSS-VALIDATION (5-fold)")
print("="*60)

kf = KFold(n_splits=5, shuffle=True, random_state=42)
fold_rmses = []
all_errors = []
all_indices = []

for fold, (train_idx, val_idx) in enumerate(kf.split(train_full)):
    tr = train_full.iloc[train_idx].copy()
    va = train_full.iloc[val_idx].copy()

    preds = build_and_predict(tr, va)
    actual = va["price"].values
    errors = preds - actual
    sq_errors = errors**2
    rmse = np.sqrt(sq_errors.mean())
    mae = np.abs(errors).mean()
    fold_rmses.append(rmse)
    all_errors.extend(errors.tolist())
    all_indices.extend(val_idx.tolist())

    print(f"  Fold {fold+1}: RMSE=${rmse:,.0f}, MAE=${mae:,.0f}")

overall_rmse = np.sqrt(np.mean(np.array(all_errors)**2))
overall_mae = np.mean(np.abs(all_errors))
print(f"\n  Overall CV RMSE: ${overall_rmse:,.0f}")
print(f"  Overall CV MAE:  ${overall_mae:,.0f}")
print(f"  RMSE/MAE ratio:  {overall_rmse/overall_mae:.2f}")
print(f"  Leaderboard:     RMSE=$1,355, MAE=$489, ratio=2.77")

# ══════════════════════════════════════════════════
# HUNT THE OUTLIERS
# ══════════════════════════════════════════════════
print(f"\n{'='*60}")
print("HUNTING THE OUTLIER ROWS")
print("="*60)

errors_arr = np.array(all_errors)
indices_arr = np.array(all_indices)

# Sort by absolute error
sorted_idx = np.argsort(np.abs(errors_arr))[::-1]
top_errors = sorted_idx[:50]

print(f"\nTop 50 worst predictions (by absolute error):")
print(f"{'Rank':>4s} {'Error':>10s} {'Actual':>10s} {'Pred':>10s} {'Area':>6s} {'PPSF':>6s} {'N_grp':>5s} {'Building':>30s}")

# Count how many rows contribute to majority of MSE
total_mse = (errors_arr**2).sum()
cumulative = 0
for rank, idx in enumerate(top_errors):
    i = indices_arr[idx]
    row = train_full.iloc[i]
    err = errors_arr[idx]
    actual = row["price"]
    pred = actual + err  # since error = pred - actual... wait, need to get pred
    fa = row["full_addr"]
    n_grp = len(train_full[train_full["full_addr"]==fa])
    cumulative += err**2

    if rank < 30:
        print(f"{rank+1:4d} ${err:>+9,.0f} ${actual:>9,.0f} ${actual+err:>9,.0f} "
              f"{row['area_sqft']:>5.0f} ${row['ppsf']:>5.1f} {n_grp:>5d} "
              f"{row['building'][:30]}")

    if rank == 49:
        print(f"\nTop 50 rows contribute {100*cumulative/total_mse:.1f}% of total MSE")
    if rank == 99:
        print(f"Top 100 rows contribute {100*cumulative/total_mse:.1f}% of total MSE")
    if rank == 199:
        print(f"Top 200 rows contribute {100*cumulative/total_mse:.1f}% of total MSE")

# What categories are these in?
print(f"\n--- Error by match type ---")
# Rerun one fold to get detailed categorization
tr = train_full.iloc[kf.split(train_full).__next__()[0]].copy()
va = train_full.iloc[kf.split(train_full).__next__()[1]].copy()
tr["ppsf"] = tr["price"]/tr["area_sqft"]

fa_counts = tr.groupby("full_addr")["price"].count()
va["match_n"] = va["full_addr"].map(fa_counts).fillna(0).astype(int)

preds_va = build_and_predict(tr, va)
va["pred"] = preds_va
va["error"] = preds_va - va["price"]
va["abs_error"] = np.abs(va["error"])
va["sq_error"] = va["error"]**2

for label, lo, hi in [("4+",4,999),("n=3",3,3),("n=2",2,2),("n=1",1,1),("fallback",0,0)]:
    mask = (va["match_n"]>=lo) & (va["match_n"]<=hi)
    if mask.sum()==0: continue
    subset = va[mask]
    rmse = np.sqrt(subset["sq_error"].mean())
    mae = subset["abs_error"].mean()
    pct_mse = 100*subset["sq_error"].sum()/va["sq_error"].sum()
    print(f"  {label:10s}: n={mask.sum():5d}, RMSE=${rmse:>7,.0f}, MAE=${mae:>6,.0f}, "
          f"MSE_contrib={pct_mse:>5.1f}%")

# What are the characteristics of high-error rows?
print(f"\n--- High-error row characteristics ---")
high_err = va[va["abs_error"] > 5000]
low_err = va[va["abs_error"] <= 5000]
print(f"Rows with |error| > $5,000: {len(high_err)} ({100*len(high_err)/len(va):.1f}%)")
print(f"Their MSE contribution: {100*high_err['sq_error'].sum()/va['sq_error'].sum():.1f}%")
print(f"\nHigh-error rows breakdown:")
print(f"  Mean price: ${high_err['price'].mean():,.0f} vs ${low_err['price'].mean():,.0f}")
print(f"  Mean area: {high_err['area_sqft'].mean():.0f} vs {low_err['area_sqft'].mean():.0f}")
print(f"  Mean match_n: {high_err['match_n'].mean():.1f} vs {low_err['match_n'].mean():.1f}")

# District breakdown of high errors
print(f"\nHigh-error by district (top 5):")
for d, cnt in high_err["district"].value_counts().head(5).items():
    total_d = len(va[va["district"]==d])
    print(f"  {d:40s}: {cnt:3d} high-err / {total_d:4d} total ({100*cnt/total_d:.1f}%)")

# Price bracket breakdown
print(f"\nHigh-error by price bracket:")
for lo, hi, label in [(0,15000,"<15K"),(15000,25000,"15-25K"),(25000,50000,"25-50K"),
                        (50000,100000,"50-100K"),(100000,999999,">100K")]:
    mask_price = (va["price"]>=lo) & (va["price"]<hi)
    n_total = mask_price.sum()
    n_high = ((va["abs_error"]>5000) & mask_price).sum()
    if n_total > 0:
        print(f"  {label:10s}: {n_high:4d}/{n_total:4d} high-error ({100*n_high/n_total:.1f}%)")

# ══════════════════════════════════════════════════
# GENERATE BEST SUBMISSION
# ══════════════════════════════════════════════════
print(f"\n{'='*60}")
print("GENERATING SUBMISSION")
print("="*60)

# Use full training to predict test
preds = build_and_predict(train_full, test)
pd.DataFrame({"id":test["id"].astype(int),"price":preds.astype(int)}).to_csv("my_submission.csv",index=False)
print(f"Saved my_submission.csv (baseline $1,355 config)")
