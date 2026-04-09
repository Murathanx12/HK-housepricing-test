"""
Final blend: hardcoded lookup for high-confidence, ML for the rest.
Produces multiple submission files to try on the leaderboard.
"""

import pandas as pd
import numpy as np

# Load both predictions
hybrid = pd.read_csv("my_submission.csv")  # LOO + LightGBM (CV RMSE 991)
hardcode = pd.read_csv("hardcode_submission.csv")  # Pure lookup

# Load test data for confidence info
test = pd.read_csv("data/test_features.csv")
train = pd.read_csv("data/HK_house_transactions.csv")

def get_building(addr):
    if pd.isna(addr): return "UNKNOWN"
    return addr.split(",")[0].strip()

for df in [train, test]:
    df["building"] = df["address"].apply(get_building)
    df["unit_key"] = df["building"] + "|" + df["Tower"].fillna("X").astype(str) + "|" + df["Flat"].fillna("X")
    df["area_bin"] = (df["area_sqft"] / 5).round() * 5
    df["unit_area"] = df["unit_key"] + "|" + df["area_bin"].astype(str)
    df["full_addr"] = df["address"].fillna("") + "|" + df["area_sqft"].astype(str)

# Count matches at each level
full_addr_counts = train.groupby("full_addr")["price"].count()
unit_area_counts = train.groupby("unit_area")["price"].count()
unit_counts = train.groupby("unit_key")["price"].count()

test["fa_count"] = test["full_addr"].map(full_addr_counts).fillna(0).astype(int)
test["ua_count"] = test["unit_area"].map(unit_area_counts).fillna(0).astype(int)
test["u_count"] = test["unit_key"].map(unit_counts).fillna(0).astype(int)

# Strategy: blend based on confidence
h_prices = hybrid["price"].values.astype(float)
hc_prices = hardcode["price"].values.astype(float)

# Blend 1: Hardcode-dominant (trust lookups more)
blend1 = np.zeros(len(test))
for i in range(len(test)):
    fa_c = test.iloc[i]["fa_count"]
    ua_c = test.iloc[i]["ua_count"]
    u_c = test.iloc[i]["u_count"]

    if fa_c >= 3:
        # Very high confidence - use hardcoded
        blend1[i] = 0.85 * hc_prices[i] + 0.15 * h_prices[i]
    elif fa_c >= 1:
        # Good confidence
        blend1[i] = 0.7 * hc_prices[i] + 0.3 * h_prices[i]
    elif ua_c >= 2:
        blend1[i] = 0.5 * hc_prices[i] + 0.5 * h_prices[i]
    else:
        # Low confidence - trust ML more
        blend1[i] = 0.3 * hc_prices[i] + 0.7 * h_prices[i]

# Blend 2: Simple average
blend2 = 0.5 * hc_prices + 0.5 * h_prices

# Blend 3: ML-dominant
blend3 = 0.3 * hc_prices + 0.7 * h_prices

# Blend 4: Pure hardcoded (already have this)
# Blend 5: Pure ML (already have this)

# Save all blends
for name, preds in [
    ("blend_hc_dominant.csv", blend1),
    ("blend_equal.csv", blend2),
    ("blend_ml_dominant.csv", blend3),
]:
    preds_clipped = np.clip(preds, 2000, 500000)
    sub = pd.DataFrame({"id": test["id"].astype(int), "price": preds_clipped.astype(int)})
    sub.to_csv(name, index=False)

# Compare all submissions
print("=== Submission Comparison ===")
print(f"{'Name':30s}  {'Mean':>10s}  {'Median':>10s}  {'Min':>10s}  {'Max':>10s}")
for name, preds in [
    ("my_submission.csv (ML+LOO)", h_prices),
    ("hardcode_submission.csv", hc_prices),
    ("blend_hc_dominant.csv", blend1),
    ("blend_equal.csv", blend2),
    ("blend_ml_dominant.csv", blend3),
]:
    p = np.clip(preds, 2000, 500000)
    print(f"  {name:30s}  ${p.mean():>9,.0f}  ${np.median(p):>9,.0f}  ${p.min():>9,.0f}  ${p.max():>9,.0f}")

print(f"\nConfidence breakdown:")
print(f"  Full addr >= 3: {(test['fa_count'] >= 3).sum()} ({(test['fa_count'] >= 3).sum()/len(test)*100:.1f}%)")
print(f"  Full addr 1-2:  {((test['fa_count'] >= 1) & (test['fa_count'] < 3)).sum()} ({((test['fa_count'] >= 1) & (test['fa_count'] < 3)).sum()/len(test)*100:.1f}%)")
print(f"  Unit area >= 2: {((test['fa_count'] == 0) & (test['ua_count'] >= 2)).sum()}")
print(f"  Low confidence: {((test['fa_count'] == 0) & (test['ua_count'] < 2)).sum()}")

# Also check: how different are the two base predictions?
diff = np.abs(h_prices - hc_prices)
print(f"\nDifference between ML and Hardcoded:")
print(f"  Mean abs diff: ${diff.mean():,.0f}")
print(f"  Median abs diff: ${np.median(diff):,.0f}")
print(f"  Max abs diff: ${diff.max():,.0f}")
print(f"  Within $500: {(diff < 500).sum()} ({(diff < 500).sum()/len(diff)*100:.1f}%)")
print(f"  Within $1000: {(diff < 1000).sum()} ({(diff < 1000).sum()/len(diff)*100:.1f}%)")
print(f"  Within $2000: {(diff < 2000).sum()} ({(diff < 2000).sum()/len(diff)*100:.1f}%)")

print("\n=== FILES READY TO SUBMIT ===")
print("Try these in order (most likely to beat $1,563 RMSE):")
print("  1. my_submission.csv        — Hybrid LOO+LightGBM (CV RMSE: 991)")
print("  2. blend_hc_dominant.csv    — 85% hardcoded / 15% ML for high-confidence")
print("  3. blend_equal.csv          — 50/50 blend")
print("  4. hardcode_submission.csv  — Pure hardcoded lookup")
print("  5. blend_ml_dominant.csv    — 30% hardcoded / 70% ML")
