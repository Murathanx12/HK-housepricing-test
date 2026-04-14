"""Deep analysis of training data for HK rental prediction."""
import pandas as pd, numpy as np
from scipy.stats import trim_mean, skew, kurtosis
from collections import Counter

DATA_DIR = "./data"
train = pd.read_csv(f"{DATA_DIR}/HK_house_transactions.csv")
test = pd.read_csv(f"{DATA_DIR}/test_features.csv")

def get_building(addr):
    if pd.isna(addr): return "UNKNOWN"
    return addr.split(",")[0].strip()

for df in [train, test]:
    df["building"] = df["address"].apply(get_building)
    df["floor"] = pd.to_numeric(df["floor"], errors="coerce").fillna(10)
    df["unit_key"] = df["building"]+"|"+df["Tower"].fillna("X").astype(str)+"|"+df["Flat"].fillna("X")
    df["bld_tower"] = df["building"]+"|T"+df["Tower"].fillna("X").astype(str)
    df["full_addr"] = df["address"].fillna("")+"|"+df["area_sqft"].astype(str)
    df["area_bin5"] = (df["area_sqft"]/5).round()*5
    df["unit_area5"] = df["unit_key"]+"|"+df["area_bin5"].astype(str)

train["ppsf"] = train["price"]/train["area_sqft"]

# ==================================================
# Q1: n=3 groups - why does mean beat median?
# ==================================================
print("="*70)
print("Q1: ANALYZING n=3 GROUPS")
print("="*70)

fa_grp = train.groupby("full_addr")
fa_stats = fa_grp.agg(p_mean=("price","mean"), p_median=("price","median"),
    p_std=("price","std"), count=("price","count"), p_min=("price","min"), p_max=("price","max"))

# Which full_addrs appear in test?
test_fas = set(test["full_addr"])
fa_in_test = fa_stats[fa_stats.index.isin(test_fas)]

n3_groups = fa_in_test[fa_in_test["count"]==3]
print(f"\nn=3 groups appearing in test: {len(n3_groups)}")
print(f"Test rows in n=3 groups: {test['full_addr'].isin(n3_groups.index).sum()}")

# Distribution analysis of n=3 groups
n3_groups = n3_groups.copy()
n3_groups["cv"] = n3_groups["p_std"] / n3_groups["p_mean"]
n3_groups["range_pct"] = (n3_groups["p_max"] - n3_groups["p_min"]) / n3_groups["p_mean"]
n3_groups["mean_med_diff"] = (n3_groups["p_mean"] - n3_groups["p_median"]) / n3_groups["p_mean"]

print(f"\nCV (coeff of variation) stats for n=3 groups:")
print(f"  Mean CV: {n3_groups['cv'].mean():.4f}")
print(f"  Median CV: {n3_groups['cv'].median():.4f}")
print(f"  Range/Mean: {n3_groups['range_pct'].mean():.4f}")
print(f"  Mean-Median diff (as % of mean): {n3_groups['mean_med_diff'].mean():.4f}")

# Look at individual n=3 groups - which ones have biggest mean-median difference
print(f"\nn=3 groups with LARGEST absolute mean-median diff (top 20):")
n3_groups["abs_diff"] = np.abs(n3_groups["p_mean"] - n3_groups["p_median"])
n3_top = n3_groups.nlargest(20, "abs_diff")
for fa, row in n3_top.iterrows():
    prices = train[train["full_addr"]==fa]["price"].values
    addr_short = fa.split("|")[0][:50]
    print(f"  {addr_short:50s} prices={sorted(prices)} mean={row['p_mean']:.0f} med={row['p_median']:.0f} diff={row['abs_diff']:.0f}")

# Why mean beats median for n=3
print(f"\nSkewness of n=3 groups:")
skews = []
for fa in n3_groups.index:
    prices = train[train["full_addr"]==fa]["price"].values
    if len(prices) == 3:
        skews.append(skew(prices))
print(f"  Mean skew: {np.mean(skews):.4f}")
print(f"  Median skew: {np.median(skews):.4f}")
print(f"  % positive skew: {(np.array(skews) > 0).mean():.2%}")
print(f"  % negative skew: {(np.array(skews) < 0).mean():.2%}")
print(f"  % zero skew: {(np.array(skews) == 0).mean():.2%}")

# ==================================================
# Q2: n=4-9 groups - better estimator?
# ==================================================
print("\n" + "="*70)
print("Q2: ANALYZING n=4-9 GROUPS (trimmed_mean = plain mean)")
print("="*70)

for n in range(4, 12):
    ng = fa_in_test[fa_in_test["count"]==n]
    test_rows = test["full_addr"].isin(ng.index).sum()
    if len(ng) > 0:
        cv = ng["p_std"] / ng["p_mean"]
        print(f"  n={n:2d}: {len(ng):4d} groups, {test_rows:4d} test rows, mean CV={cv.mean():.4f}, median CV={cv.median():.4f}")

# For specific n values, LOO comparison
print("\nLOO analysis per group size:")
for n_target in [4, 5, 6, 7, 8, 9, 10, 15, 20]:
    mean_errors = []
    median_errors = []
    trimmed_errors = []
    winsor_errors = []

    for fa, group_df in fa_grp:
        prices = group_df["price"].values
        if len(prices) != n_target:
            continue
        for i in range(len(prices)):
            true_val = prices[i]
            others = np.delete(prices, i)
            mean_errors.append((true_val - np.mean(others))**2)
            median_errors.append((true_val - np.median(others))**2)
            if len(others) >= 4:
                trimmed_errors.append((true_val - trim_mean(others, 0.1))**2)
            else:
                trimmed_errors.append((true_val - np.mean(others))**2)
            # Winsorized mean: clip extreme values to 5th/95th percentile then mean
            p5, p95 = np.percentile(others, [10, 90])
            winsor = np.clip(others, p5, p95)
            winsor_errors.append((true_val - np.mean(winsor))**2)

    if len(mean_errors) > 0:
        print(f"  n={n_target:2d}: LOO_RMSE mean={np.sqrt(np.mean(mean_errors)):>8.0f} "
              f"median={np.sqrt(np.mean(median_errors)):>8.0f} "
              f"trimmed={np.sqrt(np.mean(trimmed_errors)):>8.0f} "
              f"winsor={np.sqrt(np.mean(winsor_errors)):>8.0f} "
              f"(N={len(mean_errors)} rows)")

# ==================================================
# Q3: Single-match rows analysis
# ==================================================
print("\n" + "="*70)
print("Q3: SINGLE-MATCH ROWS ANALYSIS")
print("="*70)

n1_test = test[test["full_addr"].isin(fa_stats[fa_stats["count"]==1].index)]
print(f"Single-match test rows: {len(n1_test)}")

# Breakdown by district
print(f"\nSingle-match by district:")
for d, cnt in n1_test["district"].value_counts().head(15).items():
    dist_train = train[train["district"]==d]
    print(f"  {d:40s}: {cnt:4d} test rows, train ppsf median=${dist_train['ppsf'].median():.1f}")

# Price level analysis
n1_prices = []
n1_bld_sizes_list = []
bld_counts = train.groupby("building").size()
for idx, row in n1_test.iterrows():
    fa = row["full_addr"]
    if fa in fa_stats.index:
        n1_prices.append(fa_stats.loc[fa, "p_mean"])
        n1_bld_sizes_list.append(bld_counts.get(row["building"], 0))
n1_prices = np.array(n1_prices)
n1_bld_sizes_arr = np.array(n1_bld_sizes_list)

print(f"\nSingle-match price distribution:")
for pct in [10, 25, 50, 75, 90]:
    print(f"  P{pct}: ${np.percentile(n1_prices, pct):,.0f}")

# Building size breakdown
print(f"\nSingle-match buildings by training population:")
n1_bld_sizes = n1_test["building"].map(bld_counts).fillna(0)
print(f"  Buildings with 1-5 training rows: {(n1_bld_sizes <= 5).sum()}")
print(f"  Buildings with 6-20 training rows: {((n1_bld_sizes > 5) & (n1_bld_sizes <= 20)).sum()}")
print(f"  Buildings with 21-50 training rows: {((n1_bld_sizes > 20) & (n1_bld_sizes <= 50)).sum()}")
print(f"  Buildings with 50+ training rows: {(n1_bld_sizes > 50).sum()}")

# PPSF CV by building size for single-match
print(f"\nSingle-match: building size vs ppsf CV:")
for bsize_min, bsize_max in [(1,5), (6,20), (21,50), (51,500)]:
    mask = (n1_bld_sizes >= bsize_min) & (n1_bld_sizes <= bsize_max)
    blds = n1_test[mask]["building"].unique()
    if len(blds) == 0: continue
    cvs = []
    for bld in blds:
        g = train[train["building"]==bld]
        if len(g) >= 3:
            cvs.append(g["ppsf"].std() / g["ppsf"].mean())
    if cvs:
        print(f"  Bld size {bsize_min}-{bsize_max}: {mask.sum()} rows, mean ppsf CV={np.mean(cvs):.4f}")

# ==================================================
# Q4: Fallback rows
# ==================================================
print("\n" + "="*70)
print("Q4: FALLBACK (UNMATCHED) ROWS")
print("="*70)

unmatched = test[~test["full_addr"].isin(fa_stats.index)]
print(f"Unmatched test rows: {len(unmatched)}")

print(f"\nUnmatched by building (top 20):")
for bld, cnt in unmatched["building"].value_counts().head(20).items():
    if bld in bld_counts.index:
        tc = bld_counts[bld]
        tp = train[train["building"]==bld]["ppsf"].median()
        print(f"  {bld:45s}: {cnt:3d} test, {tc:4d} train rows, median ppsf=${tp:.1f}")
    else:
        print(f"  {bld:45s}: {cnt:3d} test, NO training data!")

print(f"\nUnmatched by district:")
for d, cnt in unmatched["district"].value_counts().items():
    print(f"  {d:40s}: {cnt:4d}")

# Fallback level distribution
ua_idx = set(train.groupby("unit_area5").groups.keys())
uk_idx = set(train.groupby("unit_key").groups.keys())
bt_idx = set(train.groupby("bld_tower").groups.keys())
bld_idx = set(train.groupby("building").groups.keys())

fb_level = []
for idx, row in unmatched.iterrows():
    if row["unit_area5"] in ua_idx:
        fb_level.append("unit_area5")
    elif row["unit_key"] in uk_idx:
        fb_level.append("unit_key")
    elif row["bld_tower"] in bt_idx:
        fb_level.append("bld_tower")
    elif row["building"] in bld_idx:
        fb_level.append("building")
    else:
        fb_level.append("KNN_only")

fb_counts = Counter(fb_level)
print(f"\nFallback level distribution:")
for level, cnt in sorted(fb_counts.items(), key=lambda x: -x[1]):
    print(f"  {level:20s}: {cnt:4d} rows")

# Buildings with NO training data
no_train_blds = unmatched[~unmatched["building"].isin(bld_idx)]["building"].unique()
print(f"\nBuildings with ZERO training rows: {len(no_train_blds)}")
for bld in no_train_blds[:20]:
    rows = unmatched[unmatched["building"]==bld]
    print(f"  {bld:45s}: {len(rows)} test rows, areas={sorted(rows['area_sqft'].values)}, dist={rows['district'].iloc[0]}")

# High price fallback rows
print(f"\nFallback rows with large area (likely high predicted price):")
big_unmatched = unmatched[unmatched["area_sqft"] > 1000].sort_values("area_sqft", ascending=False)
for idx, row in big_unmatched.head(15).iterrows():
    bld = row["building"]
    if bld in bld_idx:
        tp = train[train["building"]==bld]["ppsf"].median()
        est = tp * row["area_sqft"]
        print(f"  {bld:40s} area={row['area_sqft']:5.0f} est_price=${est:>8,.0f} dist={row['district']}")
    else:
        print(f"  {bld:40s} area={row['area_sqft']:5.0f} NO BUILDING DATA dist={row['district']}")

# ==================================================
# Q5: Price distribution
# ==================================================
print("\n" + "="*70)
print("Q5: PRICE DISTRIBUTION ANALYSIS")
print("="*70)

print(f"\nOverall price distribution:")
for pct in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
    print(f"  P{pct:2d}: ${np.percentile(train['price'], pct):>10,.0f}")
print(f"  Mean: ${train['price'].mean():>10,.0f}")
print(f"  Std: ${train['price'].std():>10,.0f}")
print(f"  Skew: {skew(train['price']):.2f}")

print(f"\nPPSF by district (sorted by median):")
for d in train.groupby("district")["ppsf"].median().sort_values(ascending=False).index:
    g = train[train["district"]==d]
    print(f"  {d:40s} n={len(g):5d} med_ppsf=${g['ppsf'].median():>6.1f} CV={g['ppsf'].std()/g['ppsf'].mean():.3f}")

# ==================================================
# Q6: Data quality issues
# ==================================================
print("\n" + "="*70)
print("Q6: DATA QUALITY ISSUES")
print("="*70)

# Extreme ppsf
print(f"\nExtreme PPSF values:")
extreme_low = train[train["ppsf"] < 5].sort_values("ppsf")
print(f"  PPSF < $5/sqft: {len(extreme_low)} rows")
for idx, row in extreme_low.head(10).iterrows():
    print(f"    price=${row['price']:>8,.0f} area={row['area_sqft']:>5.0f} ppsf=${row['ppsf']:.1f} {row['building'][:40]}")

extreme_high = train[train["ppsf"] > 200].sort_values("ppsf", ascending=False)
print(f"\n  PPSF > $200/sqft: {len(extreme_high)} rows")
for idx, row in extreme_high.head(10).iterrows():
    print(f"    price=${row['price']:>8,.0f} area={row['area_sqft']:>5.0f} ppsf=${row['ppsf']:.1f} {row['building'][:40]}")

# Within-group outliers
print(f"\nWithin-group outliers (price > 3 std from group mean, n>=5):")
outlier_examples = []
for fa, group_df in fa_grp:
    if len(group_df) < 5: continue
    prices = group_df["price"]
    z = np.abs((prices - prices.mean()) / prices.std())
    outliers = group_df[z > 3]
    if len(outliers) > 0:
        for idx, row in outliers.iterrows():
            outlier_examples.append((fa, row["price"], prices.mean(), prices.std(), len(group_df)))

print(f"  Total within-group outliers: {len(outlier_examples)}")
outlier_examples.sort(key=lambda x: abs(x[1] - x[2]), reverse=True)
for fa, price, mean, std, n in outlier_examples[:15]:
    addr_short = fa.split("|")[0][:45]
    in_test = "TEST" if fa in test_fas else "    "
    print(f"  {in_test} {addr_short:45s} price=${price:>8,.0f} mean=${mean:>8,.0f} diff=${price-mean:>+8,.0f} (n={n})")

# High-variance groups in test
print(f"\nHigh-variance groups (CV > 0.3, n >= 3) in test:")
high_var = fa_stats[(fa_stats["count"] >= 3) & (fa_stats["p_std"]/fa_stats["p_mean"] > 0.3)]
high_var_in_test = high_var[high_var.index.isin(test_fas)]
print(f"  Total: {len(high_var_in_test)}")
for fa in high_var_in_test.index[:15]:
    prices = sorted(train[train["full_addr"]==fa]["price"].values)
    addr_short = fa.split("|")[0][:45]
    print(f"  {addr_short:45s} prices={prices}")

# ==================================================
# Q7: WHAT HAVEN'T WE TRIED?
# ==================================================
print("\n" + "="*70)
print("Q7: POTENTIAL IMPROVEMENTS")
print("="*70)

# 1. Address-only matching for unmatched rows
print("\n--- Unmatched rows with address-only match (diff area) ---")
addr_match_diff_area = 0
addr_match_examples = []
for idx, row in unmatched.iterrows():
    addr = row["address"]
    matches = train[train["address"]==addr]
    if len(matches) > 0:
        addr_match_diff_area += 1
        if len(addr_match_examples) < 10:
            addr_match_examples.append((addr[:55], row["area_sqft"],
                sorted(matches["area_sqft"].unique().tolist()), sorted(matches["price"].values.tolist())))

print(f"Unmatched rows with same address but different area: {addr_match_diff_area}")
for ex in addr_match_examples:
    print(f"  {ex[0]}")
    print(f"    Test area: {ex[1]}, Train areas: {ex[2][:5]}, Train prices: {ex[3][:5]}")

# 2. n=2 groups with large spread
print(f"\n--- n=2 groups with large spread ---")
n2_groups = fa_in_test[fa_in_test["count"]==2].copy()
n2_groups["spread"] = (n2_groups["p_max"] - n2_groups["p_min"]) / n2_groups["p_mean"]
print(f"n=2 groups in test: {len(n2_groups)}")
print(f"n=2 with spread > 30%: {(n2_groups['spread'] > 0.3).sum()}")
print(f"n=2 with spread > 50%: {(n2_groups['spread'] > 0.5).sum()}")
big_spread_n2 = n2_groups[n2_groups["spread"] > 0.3].sort_values("spread", ascending=False)
for fa, row in big_spread_n2.head(15).iterrows():
    prices = sorted(train[train["full_addr"]==fa]["price"].values)
    addr_short = fa.split("|")[0][:45]
    print(f"  {addr_short:45s} prices={prices} spread={row['spread']:.2f}")

# 3. n=1 outlier detection
print(f"\n--- n=1 outlier detection (training price vs building median) ---")
n1_test_outliers = []
for idx, row in n1_test.iterrows():
    fa = row["full_addr"]
    if fa not in fa_stats.index: continue
    train_price = fa_stats.loc[fa, "p_mean"]
    bld = row["building"]
    bld_data = train[train["building"]==bld]
    if len(bld_data) < 5: continue
    bld_ppsf_med = bld_data["ppsf"].median()
    expected = bld_ppsf_med * row["area_sqft"]
    ratio = train_price / expected
    if ratio > 1.5 or ratio < 0.67:
        n1_test_outliers.append((row["address"][:50], train_price, expected, ratio, len(bld_data), row["area_sqft"]))

print(f"n=1 TEST rows where training price is >1.5x or <0.67x building estimate: {len(n1_test_outliers)}")
n1_test_outliers.sort(key=lambda x: abs(x[3] - 1.0), reverse=True)
for ex in n1_test_outliers[:25]:
    print(f"  {ex[0]:50s} train_p=${ex[1]:>8,.0f} bld_est=${ex[2]:>8,.0f} ratio={ex[3]:.2f} (n_bld={ex[4]}, area={ex[5]})")

over = sum(1 for x in n1_test_outliers if x[3] > 1.5)
under = sum(1 for x in n1_test_outliers if x[3] < 0.67)
print(f"  Over-priced in training: {over}")
print(f"  Under-priced in training: {under}")

# 4. Floor description in address - can we use it?
print(f"\n--- Floor description in address ---")
for df_name, df in [("train", train), ("test", test)]:
    floor_desc = df["address"].str.extract(r'(Upper|Middle|Lower|High|Low)\s+Floor', expand=False)
    print(f"  {df_name}: {floor_desc.value_counts().to_dict()}")

# Check if floor description matches numeric floor
for desc, floor_range in [("Lower", (1, 10)), ("Middle", (11, 20)), ("Upper", (21, 50)), ("High", (30, 60)), ("Low", (1, 10))]:
    mask = train["address"].str.contains(f"{desc} Floor", na=False)
    if mask.sum() > 0:
        floors = train[mask]["floor"]
        print(f"  {desc} Floor: mean floor={floors.mean():.1f}, std={floors.std():.1f}, range=[{floors.min():.0f}-{floors.max():.0f}]")

# 5. Public housing analysis
print(f"\n--- Public housing ---")
pub_train = train[train["Public_Housing"]==True]
pub_test = test[test["Public_Housing"]==True]
print(f"Public housing train: {len(pub_train)} ({len(pub_train)/len(train):.1%})")
print(f"Public housing test: {len(pub_test)} ({len(pub_test)/len(test):.1%})")
if len(pub_train) > 0:
    print(f"  Public ppsf: mean=${pub_train['ppsf'].mean():.1f}, med=${pub_train['ppsf'].median():.1f}")
    priv = train[train["Public_Housing"]==False]
    print(f"  Private ppsf: mean=${priv['ppsf'].mean():.1f}, med=${priv['ppsf'].median():.1f}")

# 6. Key insight: for matched groups, are there systematic patterns in WHICH test row
# from a group has the highest/lowest price?
print(f"\n--- Floor premium analysis for matched groups ---")
# For groups with n>=4, do higher floors consistently get higher prices?
floor_premium_groups = 0
floor_premium_positive = 0
for fa, group_df in fa_grp:
    if len(group_df) < 4: continue
    if group_df["floor"].std() < 2: continue
    corr = np.corrcoef(group_df["floor"], group_df["price"])[0,1]
    if not np.isnan(corr):
        floor_premium_groups += 1
        if corr > 0:
            floor_premium_positive += 1
print(f"Groups with floor variation (n>=4, floor_std>2): {floor_premium_groups}")
print(f"  Positive floor-price correlation: {floor_premium_positive} ({floor_premium_positive/max(floor_premium_groups,1):.1%})")

# 7. For test rows in matched groups, does the test row floor differ from training mean?
print(f"\n--- Test floor vs training floor for matched rows ---")
floor_diffs = []
for idx, row in test.iterrows():
    fa = row["full_addr"]
    if fa in fa_stats.index:
        train_floor_mean = fa_stats.loc[fa, "count"]  # Need actual floor mean
        pass

# Recompute with floor
fa_floor = fa_grp.agg(floor_mean=("floor","mean"), floor_std=("floor","std"),
                       price_mean=("price","mean"), count=("price","count"))
floor_diffs = []
for idx, row in test.iterrows():
    fa = row["full_addr"]
    if fa not in fa_floor.index: continue
    d = fa_floor.loc[fa]
    if d["count"] >= 2 and d["floor_std"] > 0:
        floor_diffs.append(row["floor"] - d["floor_mean"])
floor_diffs = np.array(floor_diffs)
print(f"Test-train floor differences (matched rows with floor variation):")
print(f"  N: {len(floor_diffs)}")
print(f"  Mean diff: {floor_diffs.mean():.2f}")
print(f"  Std: {floor_diffs.std():.2f}")
print(f"  % test floor HIGHER than train mean: {(floor_diffs > 0).mean():.2%}")
print(f"  % test floor LOWER than train mean: {(floor_diffs < 0).mean():.2%}")
print(f"  % test floor SAME as train mean: {(floor_diffs == 0).mean():.2%}")

# 8. CRITICAL: Are there addresses in test that are ALMOST matching but differ by trivial things?
print(f"\n--- Near-miss matching analysis ---")
# Check if any unmatched test addresses differ from a training address only by whitespace/case
unmatched_check = 0
for idx, row in unmatched.head(100).iterrows():
    addr_norm = row["address"].lower().strip().replace("  "," ") if pd.notna(row["address"]) else ""
    # Check if normalized version matches
    # This is expensive, so just check a sample

# Better: check if unmatched test rows have buildings in training
unmatched_bld_match = unmatched["building"].isin(bld_idx).sum()
unmatched_bld_no_match = (~unmatched["building"].isin(bld_idx)).sum()
print(f"Unmatched rows with building in training: {unmatched_bld_match}")
print(f"Unmatched rows with NO building in training: {unmatched_bld_no_match}")

# For those with building match, why didn't full_addr match?
# It's because area_sqft is different, or address text is different
print(f"\nWhy don't unmatched rows with known building match?")
sample = unmatched[unmatched["building"].isin(bld_idx)].head(20)
for idx, row in sample.iterrows():
    # Find closest training row by address
    bld_train = train[train["building"]==row["building"]]
    # Check if address (without area) exists
    addr_match = bld_train[bld_train["address"]==row["address"]]
    if len(addr_match) > 0:
        print(f"  ADDR MATCH: {row['address'][:50]} test_area={row['area_sqft']} train_areas={sorted(addr_match['area_sqft'].unique())[:5]}")
    else:
        # Find most similar address
        pass

# 9. WEIGHTED estimator for n=3 based on recency/floor proximity
print(f"\n--- Weighted mean for n=3 based on floor proximity ---")
# If test floor is closer to one training observation, weight it higher
n3_floor_potential = 0
for fa in n3_groups.index:
    group_df = train[train["full_addr"]==fa]
    test_here = test[test["full_addr"]==fa]
    if len(test_here) == 0: continue
    floors_train = group_df["floor"].values
    prices_train = group_df["price"].values
    for _, trow in test_here.iterrows():
        floor_test = trow["floor"]
        if np.std(floors_train) > 2:
            n3_floor_potential += 1
print(f"n=3 test rows where training floors vary (std>2): {n3_floor_potential}")
print(f"n=3 total test rows: {test['full_addr'].isin(n3_groups.index).sum()}")

# 10. Check PPSF outliers at group level that affect predictions
print(f"\n--- Groups where one outlier price dominates the mean ---")
problematic = []
for fa in fa_in_test.index:
    group_df = train[train["full_addr"]==fa]
    n = len(group_df)
    if n < 3 or n > 20: continue
    prices = group_df["price"].values
    mean_price = np.mean(prices)
    # Remove each price and see max impact on mean
    for i, p in enumerate(prices):
        others = np.delete(prices, i)
        alt_mean = np.mean(others)
        impact = abs(mean_price - alt_mean)
        if impact > 2000:
            problematic.append((fa[:50], n, p, mean_price, alt_mean, impact))
            break

print(f"Groups where removing one price changes mean by >$2000: {len(problematic)}")
problematic.sort(key=lambda x: -x[5])
for ex in problematic[:15]:
    print(f"  {ex[0]:50s} n={ex[1]} outlier_p=${ex[2]:>8,.0f} mean=${ex[3]:>8,.0f} mean_without=${ex[4]:>8,.0f} impact=${ex[5]:>6,.0f}")

print("\n\nDONE.")
