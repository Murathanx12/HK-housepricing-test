"""
Hong Kong Rental Price Prediction — Broader Matching + Floor Adjustment
========================================================================
Key hypothesis: JigsawBlock uses broader address groups (strip floor band)
combined with floor-level price adjustment. This trades specificity (higher
MAE) for robustness (lower RMSE through better averaging).

n=1 rows that become n=3 with broader matching get a GROUP MEAN instead of
a single noisy observation, plus a floor adjustment to stay precise.
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

def strip_floor_band(addr):
    """Remove floor band text for broader matching."""
    if pd.isna(addr): return ""
    parts = addr.split(",")
    filtered = [p.strip() for p in parts
                if not any(x in p.upper() for x in [
                    "UPPER FLOOR", "MIDDLE FLOOR", "LOWER FLOOR",
                    "HIGH FLOOR", "LOW FLOOR", "VERY HIGH FLOOR"])]
    return ", ".join(filtered)

for df in [train, test]:
    df["building"] = df["address"].apply(get_building)
    df["floor"] = pd.to_numeric(df["floor"], errors="coerce").fillna(10)
    df["unit_key"] = df["building"] + "|" + df["Tower"].fillna("X").astype(str) + "|" + df["Flat"].fillna("X")
    df["bld_tower"] = df["building"] + "|T" + df["Tower"].fillna("X").astype(str)
    df["bld_flat"] = df["building"] + "|F" + df["Flat"].fillna("X")
    df["full_addr"] = df["address"].fillna("") + "|" + df["area_sqft"].astype(str)
    df["area_bin5"] = (df["area_sqft"] / 5).round() * 5
    df["unit_area5"] = df["unit_key"] + "|" + df["area_bin5"].astype(str)
    # Broader key: strip floor band
    df["broad_addr"] = df["address"].apply(strip_floor_band) + "|" + df["area_sqft"].astype(str)

train["ppsf"] = train["price"] / train["area_sqft"]

# ── Standard lookups ──
def floor_slope(g):
    if len(g) < 5 or g["floor"].std() < 1: return 0.0
    return np.polyfit(g["floor"], g["ppsf"], 1)[0]

bld_slopes = train.groupby("building").apply(floor_slope, include_groups=False).to_dict()

fa_grp = train.groupby("full_addr")
fa_stats = fa_grp.agg(p_mean=("price","mean"), p_median=("price","median"),
                       count=("price","count"), floor_mean=("floor","mean"))
fa_trimmed = fa_grp["price"].apply(lambda x: trim_mean(x, 0.1) if len(x) >= 4 else x.mean())
fa_stats = fa_stats.join(fa_trimmed.rename("p_trimmed"))

# ── Broad lookups (floor band stripped) ──
ba_grp = train.groupby("broad_addr")
ba_stats = ba_grp.agg(p_mean=("price","mean"), p_median=("price","median"),
                       count=("price","count"), floor_mean=("floor","mean"),
                       ppsf_mean=("ppsf","mean"), ppsf_median=("ppsf","median"))
ba_trimmed = ba_grp["price"].apply(lambda x: trim_mean(x, 0.1) if len(x) >= 4 else x.mean())
ba_stats = ba_stats.join(ba_trimmed.rename("p_trimmed"))

# ── Other lookups ──
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


def gen_baseline():
    """Exact $1,355 reproduction."""
    preds = np.zeros(len(test))
    for i in range(len(test)):
        row = test.iloc[i]; area, fv = row["area_sqft"], row["floor"]
        fa = row["full_addr"]; slope = bld_slopes.get(row["building"], 0.0); bk = row["building"]
        if fa in fa_stats.index:
            d = fa_stats.loc[fa]; n = int(d["count"])
            if n >= 4: preds[i] = d["p_trimmed"]
            elif n == 3: preds[i] = d["p_mean"]
            elif n == 2: preds[i] = d["p_mean"]
            else:
                direct = d["p_mean"]
                if bk in bld_stats.index:
                    bp = area * bld_stats.loc[bk]["ppsf_median"]
                    preds[i] = 0.85 * direct + 0.05 * bp + 0.10 * row["knn10"]
                else:
                    preds[i] = 0.90 * direct + 0.10 * row["knn10"]
        else:
            preds[i] = fb(row, area, fv, slope)
    return np.clip(preds, 2000, 500000).astype(int)


def gen_broad(n1_min_broad=2, n1_use_fadj=True, n1_blend_alpha=1.0,
              fb_use_broad=True, n2_use_broad=False):
    """
    Broader matching: for n=1 rows where the broad_addr group has n>=n1_min_broad,
    use the broader group mean (+ floor adjustment) instead of direct.

    n1_blend_alpha: 1.0 = full broad, 0.0 = full direct (baseline)
    """
    preds = np.zeros(len(test))
    cats = []
    for i in range(len(test)):
        row = test.iloc[i]; area, fv = row["area_sqft"], row["floor"]
        fa = row["full_addr"]; ba = row["broad_addr"]
        slope = bld_slopes.get(row["building"], 0.0); bk = row["building"]

        if fa in fa_stats.index:
            d = fa_stats.loc[fa]; n = int(d["count"])
            if n >= 4:
                preds[i] = d["p_trimmed"]; cats.append("4+")
            elif n == 3:
                preds[i] = d["p_mean"]; cats.append("n3")
            elif n == 2:
                if n2_use_broad and ba in ba_stats.index and ba_stats.loc[ba]["count"] >= 4:
                    # Use broader group for n=2 if broader group is larger
                    bd = ba_stats.loc[ba]
                    broad_pred = bd["p_trimmed"]
                    if n1_use_fadj:
                        broad_pred += slope * (fv - bd["floor_mean"]) * area
                    preds[i] = (1-n1_blend_alpha) * d["p_mean"] + n1_blend_alpha * broad_pred
                    cats.append("n2_broad")
                else:
                    preds[i] = d["p_mean"]; cats.append("n2")
            else:
                # n=1: check if broader group exists
                direct = d["p_mean"]
                if ba in ba_stats.index and ba_stats.loc[ba]["count"] >= n1_min_broad:
                    bd = ba_stats.loc[ba]
                    bn = int(bd["count"])
                    if bn >= 4:
                        broad_pred = bd["p_trimmed"]
                    else:
                        broad_pred = bd["p_mean"]
                    if n1_use_fadj:
                        broad_pred += slope * (fv - bd["floor_mean"]) * area
                    preds[i] = (1-n1_blend_alpha) * direct + n1_blend_alpha * broad_pred
                    cats.append("n1_broad")
                else:
                    # Standard n=1 blend
                    if bk in bld_stats.index:
                        bp = area * bld_stats.loc[bk]["ppsf_median"]
                        preds[i] = 0.85 * direct + 0.05 * bp + 0.10 * row["knn10"]
                    else:
                        preds[i] = 0.90 * direct + 0.10 * row["knn10"]
                    cats.append("n1")
        else:
            # Fallback: try broad match first
            if fb_use_broad and ba in ba_stats.index:
                bd = ba_stats.loc[ba]
                bn = int(bd["count"])
                ppsf_est = bd["ppsf_median"] if bn >= 2 else bd["ppsf_mean"]
                fadj = slope * (fv - bd["floor_mean"]) if bn >= 2 else 0
                preds[i] = area * (ppsf_est + fadj)
                cats.append("fb_broad")
            else:
                preds[i] = fb(row, area, fv, slope)
                cats.append("fb")

    return np.clip(preds, 2000, 500000).astype(int), cats


# ── Generate variants ──
print("Generating variants...\n")
baseline = gen_baseline()

configs = [
    # (name, n1_min_broad, use_fadj, blend_alpha, fb_broad, n2_broad)
    # Full replacement with floor adjustment
    ("broad_full_fadj_n2", 2, True, 1.0, True, False),
    ("broad_full_fadj_n3", 3, True, 1.0, True, False),
    ("broad_full_fadj_n4", 4, True, 1.0, True, False),
    # Full replacement without floor adjustment
    ("broad_full_nofadj_n2", 2, False, 1.0, True, False),
    ("broad_full_nofadj_n3", 3, 1.0, False, True, False),
    # Partial blend (50% broad + 50% direct)
    ("broad_50pct_fadj_n2", 2, True, 0.50, True, False),
    ("broad_50pct_fadj_n3", 3, True, 0.50, True, False),
    # Gentle blend (30% broad + 70% direct)
    ("broad_30pct_fadj_n2", 2, True, 0.30, True, False),
    ("broad_30pct_fadj_n3", 3, True, 0.30, True, False),
    # Only n=1 (no fallback broad matching)
    ("broad_n1only_fadj_n2", 2, True, 1.0, False, False),
    ("broad_n1only_fadj_n3", 3, True, 1.0, False, False),
    # Also apply to n=2 (use broader group for n=2 if broad group is bigger)
    ("broad_all_fadj_n2_n2b", 2, True, 1.0, True, True),
    ("broad_all_fadj_n3_n2b", 3, True, 1.0, True, True),
    # Gentle n=1 + broad fallback
    ("broad_20pct_fadj_n2_fb", 2, True, 0.20, True, False),
    ("broad_10pct_fadj_n2_fb", 2, True, 0.10, True, False),
]

print(f"{'#':>2s} {'Name':40s} {'Changed':>8s} {'AvgDiff':>9s} {'n1_broad':>9s} {'fb_broad':>9s}")
print("-" * 82)

results = []
for idx, (name, n1min, fadj, alpha, fbb, n2b) in enumerate(configs):
    p, cats = gen_broad(n1_min_broad=n1min, n1_use_fadj=fadj,
                        n1_blend_alpha=alpha, fb_use_broad=fbb, n2_use_broad=n2b)
    diff = np.abs(p.astype(float) - baseline.astype(float))
    changed = (diff > 0).sum()
    avg_d = diff[diff > 0].mean() if changed > 0 else 0
    n1b = sum(1 for c in cats if c == "n1_broad")
    fbb_count = sum(1 for c in cats if c == "fb_broad")
    results.append((idx+1, name, changed, avg_d, n1b, fbb_count, p))
    print(f"{idx+1:2d} {name:40s} {changed:8d} {avg_d:9.0f} {n1b:9d} {fbb_count:9d}")

    pd.DataFrame({"id": test["id"].astype(int), "price": p}).to_csv(f"{idx+1}.csv", index=False)

# Pick my_submission
pd.DataFrame({"id": test["id"].astype(int), "price": results[0][6]}).to_csv("my_submission.csv", index=False)
print(f"\nmy_submission.csv = 1 ({configs[0][0]})")
print(f"\nSaved {len(configs)} numbered submissions (1.csv through {len(configs)}.csv)")
