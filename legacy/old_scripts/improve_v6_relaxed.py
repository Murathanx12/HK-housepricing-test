"""
Key insight: full_addr = address|area_sqft uses EXACT area.
If we relax area matching (round to 5 or 10), more rows become multi-match.
This converts single-match rows into multi-match, giving better median estimates.

Also: multi-level blended prediction for fallback rows.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import trim_mean
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler

DATA_DIR = Path("./data")

print("Loading...")
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
    df["area_bin10"] = (df["area_sqft"] / 10).round() * 10
    df["area_bin20"] = (df["area_sqft"] / 20).round() * 20
    df["unit_area5"] = df["unit_key"] + "|" + df["area_bin5"].astype(str)
    # NEW: address with rounded area (relaxed matching)
    df["addr_area5"] = df["address"].fillna("") + "|" + df["area_bin5"].astype(str)
    df["addr_area10"] = df["address"].fillna("") + "|" + df["area_bin10"].astype(str)
    df["addr_area20"] = df["address"].fillna("") + "|" + df["area_bin20"].astype(str)
    # Address only (no area)
    df["addr_only"] = df["address"].fillna("")

train["ppsf"] = train["price"] / train["area_sqft"]

# ── Check how relaxed matching helps ──
print("\n=== MATCHING COVERAGE ===")
for key_name, key_col in [("full_addr (exact)", "full_addr"),
                           ("addr_area5 (round 5)", "addr_area5"),
                           ("addr_area10 (round 10)", "addr_area10"),
                           ("addr_area20 (round 20)", "addr_area20"),
                           ("addr_only (no area)", "addr_only")]:
    train_groups = train.groupby(key_col)["price"].count()
    matched = test[key_col].isin(train_groups.index).sum()
    # How many test rows have 2+ matches?
    multi = 0
    single = 0
    for val in test[key_col]:
        if val in train_groups.index:
            c = train_groups[val]
            if c >= 2: multi += 1
            else: single += 1
    print(f"  {key_name:30s}: matched={matched} ({100*matched/len(test):.1f}%), "
          f"multi(2+)={multi}, single={single}")

# ── Build lookup tables for different matching levels ──
print("\nBuilding all lookup tables...")

def floor_slope(g):
    if len(g) < 5 or g["floor"].std() < 1: return 0.0
    return np.polyfit(g["floor"], g["ppsf"], 1)[0]

bld_slopes = train.groupby("building").apply(floor_slope, include_groups=False).to_dict()

def build_stats(group_col):
    grp = train.groupby(group_col)
    stats = grp.agg(
        p_mean=("price", "mean"), p_median=("price", "median"),
        count=("price", "count"),
        ppsf_mean=("ppsf", "mean"), ppsf_median=("ppsf", "median"),
        floor_mean=("floor", "mean"),
    )
    trimmed = grp["price"].apply(lambda x: trim_mean(x, 0.1) if len(x) >= 4 else x.mean())
    stats = stats.join(trimmed.rename("p_trimmed"))
    return stats

fa_stats = build_stats("full_addr")
aa5_stats = build_stats("addr_area5")
aa10_stats = build_stats("addr_area10")
aa20_stats = build_stats("addr_area20")
ao_stats = build_stats("addr_only")

ua_stats = train.groupby("unit_area5").agg(
    ppsf_mean=("ppsf", "mean"), ppsf_median=("ppsf", "median"),
    floor_mean=("floor", "mean"), count=("price", "count"),
)
unit_stats = train.groupby("unit_key").agg(
    ppsf_mean=("ppsf", "mean"), ppsf_median=("ppsf", "median"),
    floor_mean=("floor", "mean"), count=("price", "count"),
)
bt_stats = train.groupby("bld_tower").agg(
    ppsf_median=("ppsf", "median"), floor_mean=("floor", "mean"), count=("price", "count"),
)
bf_stats = train.groupby("bld_flat").agg(
    ppsf_median=("ppsf", "median"), floor_mean=("floor", "mean"), count=("price", "count"),
)
bld_stats = train.groupby("building").agg(
    ppsf_mean=("ppsf", "mean"), ppsf_median=("ppsf", "median"),
    floor_mean=("floor", "mean"), count=("price", "count"),
)
dist_stats = train.groupby("district").agg(
    ppsf_median=("ppsf", "median"),
)

# KNN
print("Building KNN...")
scaler = StandardScaler()
X_knn_tr = scaler.fit_transform(train[["wgs_lat", "wgs_lon", "area_sqft", "floor"]].values)
X_knn_te = scaler.transform(test[["wgs_lat", "wgs_lon", "area_sqft", "floor"]].values)
knn10 = KNeighborsRegressor(n_neighbors=10, weights="distance", n_jobs=-1)
knn10.fit(X_knn_tr, train["price"].values)
test["knn10"] = knn10.predict(X_knn_te)


def predict_from_stats(stats, key, area, floor_val, slope, use_ppsf=False):
    """Get prediction from a stats table. Returns (pred, count) or (None, 0)."""
    if key not in stats.index:
        return None, 0
    d = stats.loc[key]
    n = int(d["count"])
    if use_ppsf:
        base = d["ppsf_median"] if n >= 2 else d["ppsf_mean"]
        fadj = slope * (floor_val - d["floor_mean"]) if n >= 2 else 0
        return area * (base + fadj), n
    else:
        if n >= 4:
            return d["p_trimmed"], n
        elif n >= 2:
            return d["p_median"], n
        else:
            return d["p_mean"], n


def get_fallback_pred(row, area, floor_val, slope):
    uak = row["unit_area5"]
    uk = row["unit_key"]
    btk = row["bld_tower"]
    bfk = row["bld_flat"]
    bk = row["building"]
    dk = row["district"]

    if uak in ua_stats.index:
        d = ua_stats.loc[uak]
        base = d["ppsf_median"] if d["count"] >= 2 else d["ppsf_mean"]
        fadj = slope * (floor_val - d["floor_mean"]) if d["count"] >= 2 else 0
        return area * (base + fadj)
    if uk in unit_stats.index:
        d = unit_stats.loc[uk]
        base = d["ppsf_median"] if d["count"] >= 3 else d["ppsf_mean"]
        fadj = slope * (floor_val - d["floor_mean"])
        return area * (base + fadj)
    if btk in bt_stats.index and bt_stats.loc[btk]["count"] >= 3:
        d = bt_stats.loc[btk]
        fadj = slope * (floor_val - d["floor_mean"])
        return area * (d["ppsf_median"] + fadj)
    if bfk in bf_stats.index and bf_stats.loc[bfk]["count"] >= 3:
        d = bf_stats.loc[bfk]
        fadj = slope * (floor_val - d["floor_mean"])
        return area * (d["ppsf_median"] + fadj)
    if bk in bld_stats.index:
        d = bld_stats.loc[bk]
        base = d["ppsf_median"] if d["count"] >= 3 else d["ppsf_mean"]
        fadj = slope * (floor_val - d["floor_mean"]) if d["count"] >= 5 else 0
        return area * (base + fadj)
    knn_p = row["knn10"]
    if dk in dist_stats.index:
        dp = area * dist_stats.loc[dk]["ppsf_median"]
        return 0.4 * knn_p + 0.6 * dp
    return knn_p


# ══════════════════════════════════════════════════
# BASELINE: Original $1,450 winner
# ══════════════════════════════════════════════════
def variant_baseline():
    preds = np.zeros(len(test))
    for i in range(len(test)):
        row = test.iloc[i]
        area, floor_val = row["area_sqft"], row["floor"]
        fa = row["full_addr"]
        slope = bld_slopes.get(row["building"], 0.0)
        if fa in fa_stats.index:
            d = fa_stats.loc[fa]
            if d["count"] >= 4:
                preds[i] = d["p_trimmed"]
            elif d["count"] >= 2:
                preds[i] = d["p_median"]
            else:
                preds[i] = 0.8 * d["p_mean"] + 0.2 * row["knn10"]
        else:
            preds[i] = get_fallback_pred(row, area, floor_val, slope)
    return np.clip(preds, 2000, 500000)


# ══════════════════════════════════════════════════
# V1: Use addr_area10 instead of full_addr (relaxed area)
# This merges groups where area differs by up to ~5 sqft
# ══════════════════════════════════════════════════
def variant_relaxed_area10():
    preds = np.zeros(len(test))
    for i in range(len(test)):
        row = test.iloc[i]
        area, floor_val = row["area_sqft"], row["floor"]
        slope = bld_slopes.get(row["building"], 0.0)

        # Try exact first
        fa = row["full_addr"]
        if fa in fa_stats.index and fa_stats.loc[fa]["count"] >= 2:
            d = fa_stats.loc[fa]
            if d["count"] >= 4:
                preds[i] = d["p_trimmed"]
            else:
                preds[i] = d["p_median"]
            continue

        # Try relaxed (area rounded to 10)
        aa10 = row["addr_area10"]
        if aa10 in aa10_stats.index:
            d = aa10_stats.loc[aa10]
            if d["count"] >= 4:
                # Use ppsf * actual area (since area is rounded in the key)
                preds[i] = area * d["ppsf_median"]
            elif d["count"] >= 2:
                preds[i] = area * d["ppsf_median"]
            else:
                preds[i] = 0.8 * (area * d["ppsf_mean"]) + 0.2 * row["knn10"]
            continue

        # Exact single match
        if fa in fa_stats.index:
            d = fa_stats.loc[fa]
            preds[i] = 0.8 * d["p_mean"] + 0.2 * row["knn10"]
            continue

        preds[i] = get_fallback_pred(row, area, floor_val, slope)
    return np.clip(preds, 2000, 500000)


# ══════════════════════════════════════════════════
# V2: Hierarchical: exact → addr_area5 → addr_area10 → fallback
# For single matches, use the relaxed key to get more data points
# ══════════════════════════════════════════════════
def variant_hierarchical():
    preds = np.zeros(len(test))
    for i in range(len(test)):
        row = test.iloc[i]
        area, floor_val = row["area_sqft"], row["floor"]
        slope = bld_slopes.get(row["building"], 0.0)
        fa = row["full_addr"]

        # Exact multi-match: use as before
        if fa in fa_stats.index and fa_stats.loc[fa]["count"] >= 2:
            d = fa_stats.loc[fa]
            preds[i] = d["p_trimmed"] if d["count"] >= 4 else d["p_median"]
            continue

        # For single exact match: blend with relaxed level
        if fa in fa_stats.index:  # count == 1
            direct = fa_stats.loc[fa]["p_mean"]

            # Try relaxed addr_area5
            aa5 = row["addr_area5"]
            if aa5 in aa5_stats.index and aa5_stats.loc[aa5]["count"] >= 3:
                relaxed = area * aa5_stats.loc[aa5]["ppsf_median"]
                preds[i] = 0.7 * direct + 0.3 * relaxed
                continue

            # Try addr_area10
            aa10 = row["addr_area10"]
            if aa10 in aa10_stats.index and aa10_stats.loc[aa10]["count"] >= 3:
                relaxed = area * aa10_stats.loc[aa10]["ppsf_median"]
                preds[i] = 0.7 * direct + 0.3 * relaxed
                continue

            preds[i] = 0.8 * direct + 0.2 * row["knn10"]
            continue

        # No exact match — try relaxed
        for key, stats_table in [("addr_area5", aa5_stats), ("addr_area10", aa10_stats)]:
            k = row[key]
            if k in stats_table.index:
                d = stats_table.loc[k]
                if d["count"] >= 2:
                    preds[i] = area * d["ppsf_median"]
                else:
                    preds[i] = 0.8 * (area * d["ppsf_mean"]) + 0.2 * row["knn10"]
                break
        else:
            preds[i] = get_fallback_pred(row, area, floor_val, slope)
    return np.clip(preds, 2000, 500000)


# ══════════════════════════════════════════════════
# V3: For single matches, use addr_only ppsf as sanity check
# addr_only = same address, any area. Get ppsf and multiply by test area.
# Only use when the single match differs from addr_only prediction.
# ══════════════════════════════════════════════════
def variant_addr_ppsf():
    preds = np.zeros(len(test))
    for i in range(len(test)):
        row = test.iloc[i]
        area, floor_val = row["area_sqft"], row["floor"]
        slope = bld_slopes.get(row["building"], 0.0)
        fa = row["full_addr"]

        if fa in fa_stats.index:
            d = fa_stats.loc[fa]
            if d["count"] >= 4:
                preds[i] = d["p_trimmed"]
            elif d["count"] >= 2:
                preds[i] = d["p_median"]
            else:
                # Single match: also check addr_only for more data
                direct = d["p_mean"]
                ao = row["addr_only"]
                if ao in ao_stats.index and ao_stats.loc[ao]["count"] >= 4:
                    ao_pred = area * ao_stats.loc[ao]["ppsf_median"]
                    # Blend: trust direct more, but use addr_only to stabilize
                    preds[i] = 0.75 * direct + 0.25 * ao_pred
                else:
                    preds[i] = 0.8 * direct + 0.2 * row["knn10"]
        else:
            # For fallback: try addr_only first
            ao = row["addr_only"]
            if ao in ao_stats.index and ao_stats.loc[ao]["count"] >= 3:
                preds[i] = area * ao_stats.loc[ao]["ppsf_median"]
            else:
                preds[i] = get_fallback_pred(row, area, floor_val, slope)
    return np.clip(preds, 2000, 500000)


# ══════════════════════════════════════════════════
# V4: Multi-level weighted average for all rows
# ══════════════════════════════════════════════════
def variant_multilevel():
    preds = np.zeros(len(test))
    for i in range(len(test)):
        row = test.iloc[i]
        area, floor_val = row["area_sqft"], row["floor"]
        slope = bld_slopes.get(row["building"], 0.0)
        fa = row["full_addr"]

        if fa in fa_stats.index:
            d = fa_stats.loc[fa]
            n = int(d["count"])

            if n >= 4:
                preds[i] = d["p_trimmed"]
            elif n >= 2:
                # Blend exact median with addr_only ppsf
                exact = d["p_median"]
                ao = row["addr_only"]
                if ao in ao_stats.index and ao_stats.loc[ao]["count"] >= 5:
                    ao_pred = area * ao_stats.loc[ao]["ppsf_median"]
                    preds[i] = 0.85 * exact + 0.15 * ao_pred
                else:
                    preds[i] = exact
            else:
                # Single match: multi-level blend
                direct = d["p_mean"]
                levels = [(direct, 5)]  # (pred, weight)

                ao = row["addr_only"]
                if ao in ao_stats.index and ao_stats.loc[ao]["count"] >= 3:
                    levels.append((area * ao_stats.loc[ao]["ppsf_median"], 3))

                aa5 = row["addr_area5"]
                if aa5 in aa5_stats.index and aa5_stats.loc[aa5]["count"] >= 2:
                    levels.append((area * aa5_stats.loc[aa5]["ppsf_median"], 2))

                levels.append((row["knn10"], 1))

                total_w = sum(w for _, w in levels)
                preds[i] = sum(p * w for p, w in levels) / total_w
        else:
            preds[i] = get_fallback_pred(row, area, floor_val, slope)
    return np.clip(preds, 2000, 500000)


# ── RUN ALL ──
print("\nGenerating submissions...")
variants = [
    ("sub_v6_baseline", variant_baseline),
    ("sub_v6_relaxed10", variant_relaxed_area10),
    ("sub_v6_hier", variant_hierarchical),
    ("sub_v6_addr_ppsf", variant_addr_ppsf),
    ("sub_v6_multilevel", variant_multilevel),
]

results = {}
for name, func in variants:
    preds = func()
    results[name] = preds
    pd.DataFrame({"id": test["id"].astype(int), "price": preds.astype(int)}).to_csv(f"{name}.csv", index=False)

base = results["sub_v6_baseline"]
print(f"\n{'Name':25s} {'Changed':>8s} {'MeanDiff':>10s} {'MaxDiff':>10s}")
print("-" * 56)
for name, preds in results.items():
    diff = np.abs(preds - base)
    changed = (diff > 1).sum()
    print(f"{name:25s} {changed:8d} ${diff.mean():8,.0f} ${diff.max():8,.0f}")

# Check what changed WHERE
print("\n=== CHANGES BY MATCH TYPE ===")
fa_count_map = fa_stats["count"].to_dict()
test["_fa_count"] = test["full_addr"].map(fa_count_map).fillna(0).astype(int)

for name, preds in results.items():
    if name == "sub_v6_baseline": continue
    diff = np.abs(preds - base)
    for label, lo, hi in [("4+", 4, 9999), ("2-3", 2, 3), ("1", 1, 1), ("0-fb", 0, 0)]:
        mask = (test["_fa_count"] >= lo) & (test["_fa_count"] <= hi)
        if mask.sum() > 0 and diff[mask].mean() > 0.5:
            ch = (diff[mask] > 1).sum()
            print(f"  {name}: {label} — {ch} changed, mean_diff=${diff[mask].mean():,.0f}")

print("\n=== SUBMIT PRIORITY ===")
print("1. sub_v6_hier.csv        — Hierarchical relaxed matching for singles")
print("2. sub_v6_addr_ppsf.csv   — addr_only PPSF for singles + fallback")
print("3. sub_v6_multilevel.csv  — Multi-level weighted blend")
print("4. sub_v6_relaxed10.csv   — Relaxed area matching")
