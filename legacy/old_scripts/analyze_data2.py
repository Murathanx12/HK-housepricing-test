"""Second round: deep dive on actionable findings."""
import pandas as pd, numpy as np
from scipy.stats import trim_mean
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

fa_grp = train.groupby("full_addr")
fa_stats = fa_grp.agg(p_mean=("price","mean"), p_median=("price","median"),
    p_std=("price","std"), count=("price","count"), p_min=("price","min"), p_max=("price","max"))
test_fas = set(test["full_addr"])
fa_in_test = fa_stats[fa_stats.index.isin(test_fas)]

# ==================================================
# FINDING 1: 28 "near-miss" unmatched rows with area off by 1-5 sqft
# These are currently using fallback (unit_area5 etc) but could use direct address match
# ==================================================
print("="*70)
print("FINDING 1: NEAR-MISS ADDRESS MATCHING")
print("="*70)

unmatched = test[~test["full_addr"].isin(fa_stats.index)]
near_misses = []
for idx, row in unmatched.iterrows():
    addr = row["address"]
    test_area = row["area_sqft"]
    matches = train[train["address"]==addr]
    if len(matches) > 0:
        closest_area = matches.loc[(matches["area_sqft"] - test_area).abs().idxmin(), "area_sqft"]
        area_diff = abs(closest_area - test_area)
        if area_diff <= 20:  # Within 20 sqft
            # What full_addr would this map to?
            new_fa = addr + "|" + str(int(closest_area))
            if new_fa in fa_stats.index:
                d = fa_stats.loc[new_fa]
                near_misses.append((idx, row["id"], addr[:55], test_area, closest_area, area_diff,
                                   d["p_mean"], d["count"], new_fa))

print(f"Near-miss matches (area diff <= 20 sqft): {len(near_misses)}")
for nm in near_misses:
    print(f"  id={nm[1]:5.0f} area_diff={nm[5]:2.0f} train_mean=${nm[6]:>8,.0f} n={nm[7]:.0f} {nm[2]}")

# How much would this help? These 28 rows would go from fallback to direct match
# Estimated improvement: depends on how bad fallback is for these

# What about matching by address with area tolerance?
print(f"\nAddress-only matching with area PPSF scaling:")
for idx, row in unmatched.iterrows():
    addr = row["address"]
    test_area = row["area_sqft"]
    matches = train[train["address"]==addr]
    if len(matches) > 0:
        # Use the PPSF from matching address, scale by test area
        ppsf_mean = matches["ppsf"].mean()
        estimated = ppsf_mean * test_area
        # Current fallback estimate would be building-level
        pass  # Already counted above

# ==================================================
# FINDING 2: For n=4-9, winsorized mean is consistently better or similar
# ==================================================
print("\n" + "="*70)
print("FINDING 2: WINSORIZED MEAN FOR N=7-9")
print("="*70)

# From LOO results:
# n=7: winsor=827 vs mean=837 (1.2% better)
# n=8: winsor=2015 vs mean=2105 (4.3% better!)
# n=9: winsor=1495 vs mean=1530 (2.3% better)
# But for n=4-6 it's similar or worse

# Let's compute exactly how many test rows this affects and potential impact
for n in range(4, 12):
    ng = fa_in_test[fa_in_test["count"]==n]
    test_rows = test["full_addr"].isin(ng.index).sum()
    if len(ng) == 0: continue

    # Compare mean vs winsorized mean
    mean_vals = []
    winsor_vals = []
    for fa in ng.index:
        prices = train[train["full_addr"]==fa]["price"].values
        mean_vals.append(np.mean(prices))
        p10, p90 = np.percentile(prices, [10, 90])
        winsor_vals.append(np.mean(np.clip(prices, p10, p90)))

    diffs = np.abs(np.array(mean_vals) - np.array(winsor_vals))
    print(f"  n={n}: {test_rows} test rows, mean |diff|=${diffs.mean():.0f}, max |diff|=${diffs.max():.0f}, "
          f"rows with diff>0: {(diffs > 0.01).sum()}")

# ==================================================
# FINDING 3: n=1 outliers - 5 rows with bad training data
# ==================================================
print("\n" + "="*70)
print("FINDING 3: n=1 OUTLIER CORRECTION IMPACT")
print("="*70)

bld_stats = train.groupby("building").agg(ppsf_median=("ppsf","median"), count=("price","count"))
n1_test = test[test["full_addr"].isin(fa_stats[fa_stats["count"]==1].index)]

outlier_impact = []
for idx, row in n1_test.iterrows():
    fa = row["full_addr"]
    if fa not in fa_stats.index: continue
    train_price = fa_stats.loc[fa, "p_mean"]
    bld = row["building"]
    if bld not in bld_stats.index: continue
    bld_data = bld_stats.loc[bld]
    if bld_data["count"] < 5: continue
    expected = bld_data["ppsf_median"] * row["area_sqft"]
    ratio = train_price / expected

    # Current prediction with 85/5/10 blend
    current = 0.85 * train_price + 0.05 * expected + 0.10 * row.get("knn10", train_price)
    # If we corrected to building estimate
    corrected = expected
    impact = abs(current - corrected)
    outlier_impact.append((fa[:50], train_price, expected, ratio, impact, bld_data["count"]))

outlier_impact.sort(key=lambda x: -x[4])
print(f"Top 20 n=1 rows by potential correction impact:")
for ex in outlier_impact[:20]:
    print(f"  {ex[0]:50s} train=${ex[1]:>8,.0f} bld_est=${ex[2]:>8,.0f} ratio={ex[3]:.2f} impact=${ex[4]:>6,.0f} n_bld={ex[5]}")

# ==================================================
# FINDING 4: Estimate RMSE contribution by category
# ==================================================
print("\n" + "="*70)
print("FINDING 4: ESTIMATED RMSE BY CATEGORY")
print("="*70)

# Use LOO-like analysis to estimate error by category
# For n>=2 matched groups: LOO gives us error estimates
# For n=1: compare to building estimate

# LOO for each category
categories = {"n>=10": [], "n=4-9": [], "n=3": [], "n=2": [], "n=1": []}
for fa, group_df in fa_grp:
    if fa not in test_fas: continue
    prices = group_df["price"].values
    n = len(prices)
    for i in range(len(prices)):
        true_val = prices[i]
        others = np.delete(prices, i)
        if len(others) == 0: continue

        if n >= 10:
            pred = trim_mean(others, 0.1)
            categories["n>=10"].append((true_val - pred)**2)
        elif n >= 4:
            pred = np.mean(others)  # trimmed_mean = mean for n<10
            categories["n=4-9"].append((true_val - pred)**2)
        elif n == 3:
            pred = np.mean(others)
            categories["n=3"].append((true_val - pred)**2)
        elif n == 2:
            pred = others[0]  # only one value
            categories["n=2"].append((true_val - pred)**2)

# Test row counts per category
cat_test_counts = {"n>=10": 0, "n=4-9": 0, "n=3": 0, "n=2": 0, "n=1": 0, "unmatched": 0}
for idx, row in test.iterrows():
    fa = row["full_addr"]
    if fa in fa_stats.index:
        n = int(fa_stats.loc[fa, "count"])
        if n >= 10: cat_test_counts["n>=10"] += 1
        elif n >= 4: cat_test_counts["n=4-9"] += 1
        elif n == 3: cat_test_counts["n=3"] += 1
        elif n == 2: cat_test_counts["n=2"] += 1
        else: cat_test_counts["n=1"] += 1
    else:
        cat_test_counts["unmatched"] += 1

print(f"\n{'Category':15s} {'Test Rows':>10s} {'LOO RMSE':>10s} {'Contribution':>15s}")
print("-"*55)
total_test = 8633
for cat in ["n>=10", "n=4-9", "n=3", "n=2", "n=1", "unmatched"]:
    n_rows = cat_test_counts[cat]
    if cat in categories and len(categories[cat]) > 0:
        rmse = np.sqrt(np.mean(categories[cat]))
        # Contribution to overall RMSE: (n/N) * RMSE^2
        contrib = n_rows / total_test * rmse**2
        print(f"{cat:15s} {n_rows:>10d} ${rmse:>9,.0f} {contrib:>14,.0f}")
    else:
        print(f"{cat:15s} {n_rows:>10d} {'N/A':>10s}")

# ==================================================
# FINDING 5: Breakdown of what changes between mean and median for n=3
# ==================================================
print("\n" + "="*70)
print("FINDING 5: n=3 MEAN vs MEDIAN - DETAILED")
print("="*70)

# For n=3, mean = (a+b+c)/3, median = middle value
# mean - median = (a + c - 2*b) / 3 where a<=b<=c
# If distribution is symmetric, mean ≈ median
# If right-skewed (c far from b), mean > median
# If left-skewed (a far from b), mean < median

n3_fas = fa_in_test[fa_in_test["count"]==3].index
total_mean_higher = 0
total_median_higher = 0
total_equal = 0
abs_diffs = []

for fa in n3_fas:
    prices = sorted(train[train["full_addr"]==fa]["price"].values)
    mean_v = np.mean(prices)
    median_v = prices[1]  # middle value
    diff = mean_v - median_v
    abs_diffs.append(abs(diff))
    if diff > 0.01: total_mean_higher += 1
    elif diff < -0.01: total_median_higher += 1
    else: total_equal += 1

print(f"n=3 groups: {len(n3_fas)}")
print(f"  Mean > Median: {total_mean_higher} ({total_mean_higher/len(n3_fas):.1%})")
print(f"  Mean < Median: {total_median_higher} ({total_median_higher/len(n3_fas):.1%})")
print(f"  Mean = Median: {total_equal} ({total_equal/len(n3_fas):.1%})")
print(f"  Mean |diff|: ${np.mean(abs_diffs):,.0f}")
print(f"  Median |diff|: ${np.median(abs_diffs):,.0f}")
print(f"  Max |diff|: ${np.max(abs_diffs):,.0f}")

# Key: for 814 test rows, $38 RMSE improvement from mean->median
# RMSE change: 1393 -> 1355
# Impact on squared errors: 1393^2 * 8633 - 1355^2 * 8633 = delta
delta_sq = (1393**2 - 1355**2) * 8633
print(f"\nTotal squared error reduction from mean (vs median) for n=3: {delta_sq:,.0f}")
print(f"Per n=3 test row: {delta_sq/814:,.0f}")
print(f"Implied per-row RMSE improvement: ${np.sqrt(delta_sq/814):,.0f}")

# ==================================================
# FINDING 6: Check if PPSF-based prediction is better than price-based for ANY category
# ==================================================
print("\n" + "="*70)
print("FINDING 6: PPSF vs PRICE-BASED PREDICTION (LOO)")
print("="*70)

# For matched groups, current approach: use group mean/median/trimmed of PRICE
# Alternative: use group mean of PPSF * area
# These are the same only if all areas in the group are identical
# Since full_addr includes area, they SHOULD be identical... let's verify

area_varies = 0
for fa, group_df in fa_grp:
    if fa not in test_fas: continue
    if len(group_df) < 2: continue
    if group_df["area_sqft"].std() > 0.01:
        area_varies += 1
        if area_varies <= 3:
            print(f"  AREA VARIES: {fa[:50]} areas={group_df['area_sqft'].values}")

print(f"Groups in test where area varies: {area_varies}")
# Expected: 0 since full_addr = address|area

# ==================================================
# FINDING 7: For the 570 fallback rows, can we use PPSF from same building + different flat?
# ==================================================
print("\n" + "="*70)
print("FINDING 7: FALLBACK IMPROVEMENT - SAME BUILDING PPSF")
print("="*70)

# Current fallback: unit_area5 -> unit_key -> bld_tower -> building -> KNN
# The top fallback buildings (Lohas Park=28, Taikoo Shing=16, etc) have LOTS of training data
# The issue is they're large complexes with multiple towers/phases

# For Lohas Park specifically: 28 test rows, 857 train rows
# Let's see if Phase/Tower matching helps
lohas_test = unmatched[unmatched["building"]=="Lohas Park"]
lohas_train = train[train["building"]=="Lohas Park"]
print(f"\nLohas Park analysis:")
print(f"  Test rows: {len(lohas_test)}")
print(f"  Train rows: {len(lohas_train)}")
print(f"  Train PPSF range: ${lohas_train['ppsf'].min():.1f} - ${lohas_train['ppsf'].max():.1f}")
print(f"  Train PPSF std: ${lohas_train['ppsf'].std():.1f}")

# Check Phase distribution
print(f"  Test phases: {lohas_test['Phase'].value_counts().to_dict()}")
print(f"  Train phases: {lohas_train['Phase'].value_counts().head(10).to_dict()}")

# What if we parse the phase from the address?
def get_phase_from_addr(addr):
    if pd.isna(addr): return None
    import re
    m = re.search(r'Phase\s+(\d+\w*)', addr)
    if m: return m.group(1)
    return None

lohas_test_phases = lohas_test["address"].apply(get_phase_from_addr)
lohas_train_phases = lohas_train["address"].apply(get_phase_from_addr)
print(f"\n  Test phases from address: {lohas_test_phases.value_counts().to_dict()}")

# PPSF by phase for Lohas Park
for phase in sorted(lohas_train_phases.dropna().unique()):
    mask = lohas_train_phases == phase
    if mask.sum() >= 5:
        g = lohas_train[mask]
        print(f"  Phase {phase:5s}: n={len(g):3d} ppsf=${g['ppsf'].median():.1f} ({g['ppsf'].min():.0f}-{g['ppsf'].max():.0f})")

# ==================================================
# FINDING 8: Are there any easy wins in the clipping?
# ==================================================
print("\n" + "="*70)
print("FINDING 8: CLIPPING ANALYSIS")
print("="*70)

# Current clip: [2000, 500000]
print(f"Training price range: ${train['price'].min():,.0f} - ${train['price'].max():,.0f}")
print(f"Prices below 5000: {(train['price'] < 5000).sum()}")
print(f"Prices below 3000: {(train['price'] < 3000).sum()}")
print(f"Prices above 200000: {(train['price'] > 200000).sum()}")
print(f"Prices above 300000: {(train['price'] > 300000).sum()}")

# ==================================================
# FINDING 9: Comprehensive error budget
# ==================================================
print("\n" + "="*70)
print("FINDING 9: ERROR BUDGET ESTIMATION")
print("="*70)

# Estimate: how much error comes from each category?
# Total RMSE = 1355
# RMSE^2 * N = total squared error = 1355^2 * 8633 = 15.85B

# For n>=4 (2829 test rows): LOO RMSE ~$800-1500 depending on n
# For n=3 (814): LOO RMSE ~$1700 (from our calculation)
# For n=2 (1112): LOO RMSE = price spread/sqrt(2)
# For n=1 (2704): hardest to estimate
# For unmatched (570): probably worst

# n=2 LOO analysis
n2_errors = []
for fa, group_df in fa_grp:
    if fa not in test_fas: continue
    if len(group_df) != 2: continue
    prices = group_df["price"].values
    for i in range(2):
        pred = prices[1-i]  # only other value
        n2_errors.append((prices[i] - pred)**2)
if n2_errors:
    print(f"n=2 LOO RMSE: ${np.sqrt(np.mean(n2_errors)):,.0f} ({len(n2_errors)} LOO samples)")

# n=1: compare to building median * area
n1_errors = []
for idx, row in n1_test.iterrows():
    fa = row["full_addr"]
    if fa not in fa_stats.index: continue
    train_price = fa_stats.loc[fa, "p_mean"]
    bld = row["building"]
    if bld not in bld_stats.index: continue
    bld_est = bld_stats.loc[bld, "ppsf_median"] * row["area_sqft"]
    # With 85/5/10 blend, prediction ≈ 0.85 * train_price + 0.05 * bld_est + 0.10 * knn
    # If true price = train_price, error ≈ 0 for direct part
    # But true price is NOT the training price - it's a DIFFERENT observation
    # We'd need to know the within-unit variance
    pass

# Estimate within-unit variance from groups
within_unit_var_by_n = {}
for fa, group_df in fa_grp:
    n = len(group_df)
    if n < 2: continue
    var = group_df["price"].var()
    if n not in within_unit_var_by_n:
        within_unit_var_by_n[n] = []
    within_unit_var_by_n[n].append(var)

print(f"\nWithin-unit variance by group size:")
for n in sorted(within_unit_var_by_n.keys())[:15]:
    vars_list = within_unit_var_by_n[n]
    mean_var = np.mean(vars_list)
    print(f"  n={n:2d}: mean_var={mean_var:>12,.0f} RMSE=${np.sqrt(mean_var):>8,.0f} ({len(vars_list)} groups)")

# Overall within-unit RMSE
all_vars = []
for n, vars_list in within_unit_var_by_n.items():
    all_vars.extend(vars_list)
print(f"\nOverall within-unit RMSE: ${np.sqrt(np.mean(all_vars)):,.0f}")

# This is the FLOOR for prediction error - even with perfect matching,
# price varies by this much within the same unit

# ==================================================
# FINDING 10: Can we do better than mean for n=2?
# ==================================================
print("\n" + "="*70)
print("FINDING 10: n=2 ALTERNATIVES")
print("="*70)

# For n=2, mean = median. But we could weight by building PPSF
# If one of the two prices is closer to building estimate, weight it higher
n2_test = test[test["full_addr"].isin(fa_in_test[fa_in_test["count"]==2].index)]
print(f"n=2 test rows: {len(n2_test)}")

# LOO: for n=2 groups, compare different estimators
n2_loo_mean = []
n2_loo_bld_weighted = []
n2_loo_recency = []

for fa, group_df in fa_grp:
    if fa not in test_fas: continue
    if len(group_df) != 2: continue
    bld = group_df.iloc[0]["building"]
    if bld not in bld_stats.index: continue
    bld_ppsf = bld_stats.loc[bld, "ppsf_median"]
    area = group_df.iloc[0]["area_sqft"]
    bld_est = bld_ppsf * area

    prices = group_df["price"].values
    for i in range(2):
        true_val = prices[i]
        other = prices[1-i]

        # Plain mean (current)
        n2_loo_mean.append((true_val - other)**2)

        # Building-weighted: blend other with building estimate
        for w in [0.9, 0.85, 0.8, 0.7]:
            pass  # Test specific weights

        # 90/10 with building
        blended = 0.9 * other + 0.1 * bld_est
        n2_loo_bld_weighted.append((true_val - blended)**2)

print(f"\nn=2 LOO comparison:")
print(f"  Mean (100% other): RMSE=${np.sqrt(np.mean(n2_loo_mean)):,.0f}")
print(f"  90/10 bld blend:   RMSE=${np.sqrt(np.mean(n2_loo_bld_weighted)):,.0f}")

# Test more weights
for w_direct in [0.95, 0.90, 0.85, 0.80, 0.70]:
    errors = []
    for fa, group_df in fa_grp:
        if fa not in test_fas: continue
        if len(group_df) != 2: continue
        bld = group_df.iloc[0]["building"]
        if bld not in bld_stats.index: continue
        bld_est = bld_stats.loc[bld, "ppsf_median"] * group_df.iloc[0]["area_sqft"]
        prices = group_df["price"].values
        for i in range(2):
            pred = w_direct * prices[1-i] + (1-w_direct) * bld_est
            errors.append((prices[i] - pred)**2)
    print(f"  {w_direct:.0%}/{1-w_direct:.0%} blend: RMSE=${np.sqrt(np.mean(errors)):,.0f}")

# ==================================================
# FINDING 11: For n>=4 groups, can REMOVING outliers help?
# ==================================================
print("\n" + "="*70)
print("FINDING 11: OUTLIER REMOVAL FOR n>=4")
print("="*70)

# LOO comparing mean vs mean-after-removing-outliers (IQR method)
for n_min, n_max in [(4,6), (7,9), (10,20), (21,100)]:
    mean_errors = []
    iqr_errors = []
    for fa, group_df in fa_grp:
        if fa not in test_fas: continue
        n = len(group_df)
        if n < n_min or n > n_max: continue
        prices = group_df["price"].values
        for i in range(n):
            true_val = prices[i]
            others = np.delete(prices, i)

            # Plain mean
            mean_errors.append((true_val - np.mean(others))**2)

            # IQR-filtered mean
            q1, q3 = np.percentile(others, [25, 75])
            iqr = q3 - q1
            if iqr > 0:
                filtered = others[(others >= q1 - 1.5*iqr) & (others <= q3 + 1.5*iqr)]
                if len(filtered) > 0:
                    iqr_errors.append((true_val - np.mean(filtered))**2)
                else:
                    iqr_errors.append((true_val - np.mean(others))**2)
            else:
                iqr_errors.append((true_val - np.mean(others))**2)

    if mean_errors:
        print(f"  n={n_min}-{n_max}: mean RMSE=${np.sqrt(np.mean(mean_errors)):>6,.0f} "
              f"IQR-filtered RMSE=${np.sqrt(np.mean(iqr_errors)):>6,.0f} "
              f"({len(mean_errors)} samples)")

# ==================================================
# FINDING 12: Address parsing - are we missing Phase/Block info?
# ==================================================
print("\n" + "="*70)
print("FINDING 12: PHASE-SPECIFIC MATCHING FOR LARGE ESTATES")
print("="*70)

# For large estates (Lohas Park, South Horizons, etc), can we parse phase from address
# and create a building+phase key for better fallback?
import re

def get_estate_phase(addr):
    if pd.isna(addr): return None, None
    parts = addr.split(",")
    estate = parts[0].strip()
    phase = None
    for part in parts:
        m = re.search(r'Phase\s+(\d+\w*)', part)
        if m:
            phase = m.group(1)
            break
    return estate, phase

large_estates = train.groupby("building").size()
large_estates = large_estates[large_estates > 100].index

for estate in ["Lohas Park", "South Horizons", "Taikoo Shing", "Kingswood Villas",
               "Caribbean Coast", "Coastal Skyline", "Century Link"]:
    if estate not in large_estates: continue
    train_e = train[train["building"]==estate]
    test_e = unmatched[unmatched["building"]==estate]
    if len(test_e) == 0: continue

    # Parse phases
    train_phases = train_e["address"].apply(lambda x: get_estate_phase(x)[1])
    test_phases = test_e["address"].apply(lambda x: get_estate_phase(x)[1])

    # PPSF by phase
    print(f"\n{estate}: {len(test_e)} unmatched test, {len(train_e)} train")
    phase_ppsf = {}
    for phase in sorted(train_phases.dropna().unique()):
        mask = train_phases == phase
        g = train_e[mask]
        if len(g) >= 5:
            phase_ppsf[phase] = g["ppsf"].median()

    # Overall building ppsf
    overall_ppsf = train_e["ppsf"].median()

    if phase_ppsf:
        max_ppsf = max(phase_ppsf.values())
        min_ppsf = min(phase_ppsf.values())
        print(f"  PPSF range across phases: ${min_ppsf:.1f} - ${max_ppsf:.1f} (overall: ${overall_ppsf:.1f})")
        print(f"  Phase spread: {(max_ppsf-min_ppsf)/overall_ppsf:.1%}")

        # Show unmatched test rows and their phases
        for idx, row in test_e.iterrows():
            phase = get_estate_phase(row["address"])[1]
            phase_est = phase_ppsf.get(phase, overall_ppsf) if phase else overall_ppsf
            overall_est = overall_ppsf * row["area_sqft"]
            phase_est_price = phase_est * row["area_sqft"]
            diff = phase_est_price - overall_est
            if abs(diff) > 500:
                print(f"    Phase {phase}: area={row['area_sqft']} overall_est=${overall_est:,.0f} phase_est=${phase_est_price:,.0f} diff=${diff:+,.0f}")

print("\n\nDONE.")
