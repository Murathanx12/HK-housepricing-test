"""Third round: quantify RMSE impact of top ideas."""
import pandas as pd, numpy as np
from scipy.stats import trim_mean

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
    df["full_addr"] = df["address"].fillna("")+"|"+df["area_sqft"].astype(str)

train["ppsf"] = train["price"]/train["area_sqft"]
fa_grp = train.groupby("full_addr")
fa_stats = fa_grp.agg(p_mean=("price","mean"), p_median=("price","median"),
    count=("price","count"))
test_fas = set(test["full_addr"])
bld_stats = train.groupby("building").agg(ppsf_median=("ppsf","median"), count=("price","count"))

# ==================================================
# IDEA A: n=2 blending with building estimate
# ==================================================
print("="*70)
print("IDEA A: n=2 BUILDING BLEND")
print("="*70)

# LOO says 70/30 blend beats pure mean for n=2 (RMSE $2064 vs $2281)
# BUT LOO is unreliable! The n=3 example showed LOO can be wrong.
# n=3 LOO said median is better, but leaderboard said mean is better.
# So we CANNOT trust LOO for n=2 either.

# What we CAN do: estimate the POTENTIAL impact
# 1280 test rows in n=2 groups
# If we do 95/5 blend (conservative): avg change per row is small
# Let's compute actual prediction changes

n2_in_test = test[test["full_addr"].isin(fa_stats[fa_stats["count"]==2].index)]
print(f"n=2 test rows: {len(n2_in_test)}")

changes = []
for idx, row in n2_in_test.iterrows():
    fa = row["full_addr"]
    current_pred = fa_stats.loc[fa, "p_mean"]  # mean of 2 values
    bld = row["building"]
    if bld not in bld_stats.index: continue
    bld_est = bld_stats.loc[bld, "ppsf_median"] * row["area_sqft"]
    # With 95/5 blend
    new_pred = 0.95 * current_pred + 0.05 * bld_est
    changes.append(abs(new_pred - current_pred))

changes = np.array(changes)
print(f"Mean prediction change (95/5): ${changes.mean():.0f}")
print(f"Max prediction change: ${changes.max():.0f}")
print(f"Rows with change > $500: {(changes > 500).sum()}")
print(f"Rows with change > $1000: {(changes > 1000).sum()}")

# Impact on RMSE: if mean absolute change is $X for N rows,
# max RMSE improvement ≈ X * sqrt(N/total)
# But it could also make things WORSE
potential = changes.mean() * np.sqrt(len(changes) / 8633)
print(f"Maximum potential RMSE impact: ${potential:.0f}")
print(f"WARNING: LOO says this helps but LOO was WRONG for n=3!")

# ==================================================
# IDEA B: Near-miss address matching (21 rows)
# ==================================================
print("\n" + "="*70)
print("IDEA B: NEAR-MISS MATCHING (21 rows)")
print("="*70)

# These 21 rows currently use fallback. Matching them to the correct full_addr
# would give them direct price data.
# Impact: depends on how bad fallback currently is for these rows.

# Estimate fallback prediction for these rows
unmatched = test[~test["full_addr"].isin(fa_stats.index)]
near_miss_impact = []
for idx, row in unmatched.iterrows():
    addr = row["address"]
    test_area = row["area_sqft"]
    matches = train[train["address"]==addr]
    if len(matches) == 0: continue
    closest_idx = (matches["area_sqft"] - test_area).abs().idxmin()
    closest_area = matches.loc[closest_idx, "area_sqft"]
    if abs(closest_area - test_area) > 20: continue

    # The "correct" answer from address matching
    new_fa = addr + "|" + str(int(closest_area))
    if new_fa not in fa_stats.index: continue
    addr_pred = fa_stats.loc[new_fa, "p_mean"]

    # Current fallback estimate (approximate: building ppsf * area)
    bld = row["building"]
    if bld in bld_stats.index:
        fallback_pred = bld_stats.loc[bld, "ppsf_median"] * test_area
    else:
        fallback_pred = addr_pred  # No way to estimate

    diff = abs(addr_pred - fallback_pred)
    near_miss_impact.append(diff)

print(f"Near-miss rows: {len(near_miss_impact)}")
print(f"Mean |diff| between address match and fallback: ${np.mean(near_miss_impact):,.0f}")
print(f"Max |diff|: ${np.max(near_miss_impact):,.0f}")

# Impact on overall RMSE
# If each of these rows' error changes by ~$X, impact on RMSE is:
# delta_RMSE ≈ X^2 * N / (2 * RMSE * total_N)
mean_diff = np.mean(near_miss_impact)
delta = mean_diff**2 * len(near_miss_impact) / (2 * 1355 * 8633)
print(f"Estimated RMSE impact: ~${delta:.1f}")
print("NOTE: Only 21 rows, impact is likely VERY small (~$1)")

# ==================================================
# IDEA C: Phase-specific PPSF for Lohas Park fallback
# ==================================================
print("\n" + "="*70)
print("IDEA C: PHASE-SPECIFIC MATCHING")
print("="*70)

import re
def get_phase_from_addr(addr):
    if pd.isna(addr): return None
    m = re.search(r'Phase\s+(\d+\w*)', addr)
    if m: return m.group(1)
    return None

# Lohas Park: 28 test rows, PPSF spread 37.6% across phases
# If we use phase-specific PPSF instead of building-wide, changes can be $3-6K
# That's 28 rows with ~$4K change
delta_lohas = 4000**2 * 28 / (2 * 1355 * 8633)
print(f"Lohas Park: 28 rows, ~$4K avg change")
print(f"Estimated RMSE impact: ~${delta_lohas:.1f}")

# All large estates combined
print(f"\nAll phase-variant estates combined: ~50 rows, ~$3K avg change")
delta_all = 3000**2 * 50 / (2 * 1355 * 8633)
print(f"Estimated RMSE impact: ~${delta_all:.1f}")

# ==================================================
# IDEA D: Correct n=1 outlier rows
# ==================================================
print("\n" + "="*70)
print("IDEA D: n=1 OUTLIER CORRECTION")
print("="*70)

# Currently: 85% direct + 5% bld + 10% KNN
# For rows where training price is far from building estimate,
# we could increase building weight

# But we already tried this (outlier correction was the path from 1450 to 1402)
# The current 85/5/10 IS the best found blend for ALL singles
# The question: could DIFFERENT blends for outlier vs non-outlier singles help?

n1_test_rows = test[test["full_addr"].isin(fa_stats[fa_stats["count"]==1].index)]
print(f"n=1 test rows: {len(n1_test_rows)}")

# How many have ratio > 1.3 or < 0.77?
outlier_ratios = []
for idx, row in n1_test_rows.iterrows():
    fa = row["full_addr"]
    train_price = fa_stats.loc[fa, "p_mean"]
    bld = row["building"]
    if bld not in bld_stats.index: continue
    bld_est = bld_stats.loc[bld, "ppsf_median"] * row["area_sqft"]
    ratio = train_price / bld_est if bld_est > 0 else 1.0
    outlier_ratios.append((idx, ratio, train_price, bld_est, bld_stats.loc[bld, "count"]))

ratios = np.array([x[1] for x in outlier_ratios])
print(f"n=1 rows with building data: {len(ratios)}")
print(f"  ratio > 1.3: {(ratios > 1.3).sum()}")
print(f"  ratio > 1.2: {(ratios > 1.2).sum()}")
print(f"  ratio < 0.8: {(ratios < 0.8).sum()}")
print(f"  ratio < 0.7: {(ratios < 0.7).sum()}")
print(f"  ratio 0.8-1.2 (normal): {((ratios >= 0.8) & (ratios <= 1.2)).sum()}")

# For the ~130 outlier n=1 rows (ratio outside 0.8-1.2),
# if we used 50/25/25 instead of 85/5/10:
outlier_n1 = [(r, x[2], x[3]) for x, r in zip(outlier_ratios, ratios) if r > 1.2 or r < 0.8]
print(f"\nOutlier n=1 rows: {len(outlier_n1)}")
if outlier_n1:
    changes = []
    for ratio, tp, be in outlier_n1:
        current = 0.85 * tp + 0.05 * be + 0.10 * tp  # approximate KNN ≈ train price
        proposed = 0.50 * tp + 0.25 * be + 0.25 * tp  # more building weight
        changes.append(abs(current - proposed))
    print(f"Mean prediction change (50/25/25 vs 85/5/10): ${np.mean(changes):,.0f}")
    print(f"This is the KEY question: are these training prices outliers, or is the UNIT genuinely different?")

# ==================================================
# IDEA E: What's the theoretical RMSE floor?
# ==================================================
print("\n" + "="*70)
print("IDEA E: THEORETICAL RMSE FLOOR")
print("="*70)

# Within-unit variance = $1,419 RMSE (from previous analysis)
# This means even with PERFECT unit matching, irreducible noise is ~$1,419
# But: some of this variance is from unit-specific temporal trends
# Our RMSE of $1,355 is BELOW the within-unit RMSE!
# This suggests our predictions are ALREADY very close to optimal

print(f"Within-unit RMSE (irreducible for matched groups): $1,419")
print(f"Current leaderboard RMSE: $1,355")
print(f"Current RMSE is BELOW the within-unit noise!")
print(f"This means:")
print(f"  1. Some of the 'noise' is systematic (floor effects, trends)")
print(f"  2. OR the test set is drawn from the same transactions (less noisy)")
print(f"  3. OR our RMSE is already near-optimal for matched rows")
print(f"")
print(f"Since 93.4% of test rows are matched, and within-unit RMSE is ~$1,419,")
print(f"the remaining improvement must come from the 6.6% unmatched + careful fine-tuning")

# ==================================================
# IDEA F: What if test prices come from the SAME distribution as training?
# ==================================================
print("\n" + "="*70)
print("IDEA F: TEST IS SAMPLED FROM SAME POOL?")
print("="*70)

# If test rows are random samples from the same groups as training,
# then for n=3 groups (3 training + 1 test), the best estimator is just the mean
# This explains why mean beats median!
# For n=3, the test price is the 4th observation from the same distribution
# Mean of 3 is the MLE for the population mean (assuming normal)
# Median of 3 is less efficient

# For n=2 (2 training + 1 test = 3 observations),
# the best prediction is the mean of the 2, which is what we do
# But blending with building gives prior information
# The question is: does the building prior HELP?

# For n=1 (1 training + 1 test = 2 observations),
# using 85% direct + priors makes sense because 1 observation is noisy

# KEY INSIGHT: the optimal blend depends on within-unit vs between-unit variance
# If within-unit variance is HIGH relative to between-unit, use more prior (building)
# If within-unit variance is LOW, use more direct

# Compute within-unit vs between-unit variance
within_vars = []
between_vars = []

for bld, bld_group in train.groupby("building"):
    if len(bld_group) < 10: continue
    bld_ppsf_mean = bld_group["ppsf"].mean()

    for fa, unit_group in bld_group.groupby("full_addr"):
        if len(unit_group) < 2: continue
        within_vars.append(unit_group["ppsf"].var())
        between_vars.append((unit_group["ppsf"].mean() - bld_ppsf_mean)**2)

print(f"Within-unit PPSF variance: {np.mean(within_vars):.2f}")
print(f"Between-unit PPSF variance: {np.mean(between_vars):.2f}")
ratio_var = np.mean(within_vars) / (np.mean(within_vars) + np.mean(between_vars))
print(f"Within / Total: {ratio_var:.3f}")
print(f"This means {ratio_var:.1%} of variance is within-unit (noise)")
print(f"And {1-ratio_var:.1%} is between-unit (signal)")
print(f"Optimal shrinkage toward building: ~{ratio_var:.1%}")
print(f"For n=1: optimal blend might be ~{1-ratio_var:.0%} direct + {ratio_var:.0%} building")

# ==================================================
# IDEA G: n=8 has anomalously high within-unit variance
# ==================================================
print("\n" + "="*70)
print("IDEA G: n=8 ANOMALY INVESTIGATION")
print("="*70)

# n=8 LOO RMSE was $2,105 (mean) vs $2,015 (winsor) - both much higher than n=7 ($837) or n=9 ($1,530)
# What's special about n=8 groups?

fa_in_test_8 = fa_stats[(fa_stats.index.isin(test_fas)) & (fa_stats["count"]==8)]
print(f"n=8 groups in test: {len(fa_in_test_8)}")

# Find the high-variance n=8 groups
for fa in fa_in_test_8.index:
    group_df = train[train["full_addr"]==fa]
    prices = group_df["price"].values
    if np.std(prices) > 3000:
        print(f"  {fa[:55]} prices={sorted(prices)} std=${np.std(prices):,.0f}")

# ==================================================
# SUMMARY: Ranked ideas by estimated impact
# ==================================================
print("\n" + "="*70)
print("SUMMARY: RANKED IDEAS BY ESTIMATED IMPACT")
print("="*70)

ideas = [
    ("A: n=2 bld blend", "1280 rows", "$2-10", "HIGH RISK: LOO unreliable", "Would need leaderboard test"),
    ("B: Near-miss matching", "21 rows", "~$1", "LOW RISK: better match", "Easy to implement"),
    ("C: Phase-specific fallback", "~50 rows", "~$2", "MEDIUM: phase PPSF more accurate", "Moderate effort"),
    ("D: n=1 outlier blend", "~130 rows", "~$5-20", "MEDIUM RISK: need right threshold", "Would need leaderboard test"),
    ("E: Winsorized mean n=7-9", "~920 rows", "~$3-5", "MEDIUM RISK: LOO says helps for n=8", "Would need leaderboard test"),
]

print(f"\n{'Idea':35s} {'Rows':>10s} {'Est RMSE':>10s} {'Risk':>30s}")
print("-"*90)
for name, rows, est, risk, note in ideas:
    print(f"{name:35s} {rows:>10s} {est:>10s} {risk:>30s}")
    print(f"{'':35s} Note: {note}")

# The BIG question: where's the $50+ improvement?
print(f"\n{'='*70}")
print("WHERE IS THE $50+ IMPROVEMENT?")
print("="*70)
print("""
With RMSE at $1,355 and within-unit noise at ~$1,419, we are ALREADY
below the irreducible noise level for matched groups. This means:

1. The easy gains are DONE. The current approach is near-optimal.

2. Remaining improvements must come from:
   a) Reducing errors on unmatched rows (570 rows, ~7%)
   b) Reducing errors on n=1 rows where training price is an outlier
   c) Systematic small improvements across many categories

3. A $50 improvement would require reducing total squared error by:
   (1355^2 - 1305^2) * 8633 = 1.15B
   That's $1,328 avg squared error reduction PER test row.

4. The ONLY realistic path to $50+ is:
   - Finding a SYSTEMATIC bias in current predictions
   - OR a fundamentally new signal (e.g., temporal trends if data has dates)
   - OR an error in our address matching/preprocessing

5. MOST PROMISING: n=2 blending (1280 rows). If the building blend
   reduces per-row RMSE by even $100 on these rows, that's worth $15 overall.
   But LOO says 70/30 is optimal, which is a HUGE change from current 100/0.
   Given that LOO was WRONG for n=3 (said median > mean), we can't trust it.

6. SAFEST BET: Try a very small n=2 blend (97/3 or 95/5) on the leaderboard.
""")
