"""Generate adaptive submissions: pure direct for normal, building correction for extreme."""

import pandas as pd, numpy as np
from pathlib import Path
from scipy.stats import trim_mean
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler

DATA_DIR = Path("./data")
train = pd.read_csv(DATA_DIR / "HK_house_transactions.csv")
test = pd.read_csv(DATA_DIR / "test_features.csv")

def get_building(addr):
    if pd.isna(addr): return "UNKNOWN"
    return addr.split(",")[0].strip()

for df in [train, test]:
    df["building"] = df["address"].apply(get_building)
    df["floor"] = pd.to_numeric(df["floor"], errors="coerce").fillna(10)
    df["unit_key"] = df["building"]+"|"+df["Tower"].fillna("X").astype(str)+"|"+df["Flat"].fillna("X")
    df["bld_tower"] = df["building"]+"|T"+df["Tower"].fillna("X").astype(str)
    df["bld_flat"] = df["building"]+"|F"+df["Flat"].fillna("X")
    df["full_addr"] = df["address"].fillna("")+"|"+df["area_sqft"].astype(str)
    df["area_bin5"] = (df["area_sqft"]/5).round()*5
    df["unit_area5"] = df["unit_key"]+"|"+df["area_bin5"].astype(str)
train["ppsf"] = train["price"]/train["area_sqft"]

def floor_slope(g):
    if len(g)<5 or g["floor"].std()<1: return 0.0
    return np.polyfit(g["floor"],g["ppsf"],1)[0]

bld_slopes = train.groupby("building").apply(floor_slope, include_groups=False).to_dict()
fa_grp = train.groupby("full_addr")
fa_stats = fa_grp.agg(p_mean=("price","mean"),p_median=("price","median"),count=("price","count"),floor_mean=("floor","mean"))
fa_trimmed = fa_grp["price"].apply(lambda x: trim_mean(x,0.1) if len(x)>=4 else x.mean())
fa_stats = fa_stats.join(fa_trimmed.rename("p_trimmed"))
ua_stats = train.groupby("unit_area5").agg(ppsf_mean=("ppsf","mean"),ppsf_median=("ppsf","median"),floor_mean=("floor","mean"),count=("price","count"))
unit_stats = train.groupby("unit_key").agg(ppsf_mean=("ppsf","mean"),ppsf_median=("ppsf","median"),floor_mean=("floor","mean"),count=("price","count"))
bt_stats = train.groupby("bld_tower").agg(ppsf_median=("ppsf","median"),floor_mean=("floor","mean"),count=("price","count"))
bf_stats = train.groupby("bld_flat").agg(ppsf_median=("ppsf","median"),floor_mean=("floor","mean"),count=("price","count"))
bld_stats = train.groupby("building").agg(ppsf_mean=("ppsf","mean"),ppsf_median=("ppsf","median"),floor_mean=("floor","mean"),count=("price","count"))
dist_stats = train.groupby("district").agg(ppsf_median=("ppsf","median"))

sc = StandardScaler()
X_tr = sc.fit_transform(train[["wgs_lat","wgs_lon","area_sqft","floor"]].values)
X_te = sc.transform(test[["wgs_lat","wgs_lon","area_sqft","floor"]].values)
knn = KNeighborsRegressor(n_neighbors=10,weights="distance",n_jobs=-1)
knn.fit(X_tr, train["price"].values)
test["knn10"] = knn.predict(X_te)

def fb(row,area,fv,slope):
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

# Baseline ($1,355)
base = np.zeros(len(test))
for i in range(len(test)):
    row=test.iloc[i]; area,fv=row["area_sqft"],row["floor"]; fa=row["full_addr"]
    slope=bld_slopes.get(row["building"],0.0); bk=row["building"]
    if fa in fa_stats.index:
        d=fa_stats.loc[fa]; n=int(d["count"])
        if n>=4: base[i]=d["p_trimmed"]
        elif n==3: base[i]=d["p_mean"]
        elif n==2: base[i]=d["p_mean"]
        else:
            direct=d["p_mean"]
            if bk in bld_stats.index:
                bp=area*bld_stats.loc[bk]["ppsf_median"]
                base[i]=0.85*direct+0.05*bp+0.10*row["knn10"]
            else: base[i]=0.90*direct+0.10*row["knn10"]
    else: base[i]=fb(row,area,fv,slope)
base = np.clip(base, 2000, 500000)

def gen_adaptive(nsd,nsb,nsk, esd,esb,esk, et):
    preds = np.zeros(len(test))
    n_ext = 0
    for i in range(len(test)):
        row=test.iloc[i]; area,fv=row["area_sqft"],row["floor"]; fa=row["full_addr"]
        slope=bld_slopes.get(row["building"],0.0); bk=row["building"]
        if fa in fa_stats.index:
            d=fa_stats.loc[fa]; n=int(d["count"])
            if n>=4: preds[i]=d["p_trimmed"]
            elif n==3: preds[i]=d["p_mean"]
            elif n==2: preds[i]=d["p_mean"]
            else:
                direct=d["p_mean"]
                if bk in bld_stats.index:
                    bp=area*bld_stats.loc[bk]["ppsf_median"]
                    ratio=direct/bp if bp>0 else 1.0
                    if ratio>et or ratio<1.0/et:
                        preds[i]=esd*direct+esb*bp+esk*row["knn10"]
                        n_ext+=1
                    else:
                        preds[i]=nsd*direct+nsb*bp+nsk*row["knn10"]
                else:
                    preds[i]=(nsd+nsb)*direct+nsk*row["knn10"]
        else: preds[i]=fb(row,area,fv,slope)
    return np.clip(preds,2000,500000), n_ext

configs = [
    # Normal: pure direct. Extreme: building correction.
    ("adapt_100_x70_t15",   1.0,0.0,0.0,   0.70,0.15,0.15, 1.5),    # 7 extreme
    ("adapt_100_x75_t13",   1.0,0.0,0.0,   0.75,0.15,0.10, 1.3),    # ~60 extreme
    ("adapt_100_x80_t13",   1.0,0.0,0.0,   0.80,0.10,0.10, 1.3),    # lighter
    # Normal: 95/0/5. Extreme: stronger correction.
    ("adapt_95_x75_t13",    0.95,0.0,0.05,  0.75,0.15,0.10, 1.3),
    # Normal: 90/0/10. Extreme: stronger correction.
    ("adapt_90_x75_t13",    0.90,0.0,0.10,  0.75,0.15,0.10, 1.3),
    ("adapt_90_x70_t15",    0.90,0.0,0.10,  0.70,0.15,0.15, 1.5),
]

print("Adaptive submissions:")
for name, nsd,nsb,nsk, esd,esb,esk, et in configs:
    p, n_ext = gen_adaptive(nsd,nsb,nsk, esd,esb,esk, et)
    diff = np.abs(p - base)
    changed = (diff > 1).sum()
    pd.DataFrame({"id":test["id"].astype(int),"price":p.astype(int)}).to_csv(f"sub_{name}.csv",index=False)
    print(f"  {name:25s}: {changed:5d} diff, {n_ext:3d} extreme")

print("\nDone!")
