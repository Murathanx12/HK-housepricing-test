"""
LEGACY — 1st Place Winner: RMSE $1,553, MAE $719, R² 0.9919
==============================================================
blend_equal.csv = 50% hardcode_submission.csv + 50% my_submission.csv (LOO+LightGBM)

This file documents the winning approach. The actual code that produced the
two base CSVs was:
  - hardcode_best.py  → hardcode_submission.csv (pure hierarchical lookup)
  - hybrid_v1.py      → my_submission.csv (LOO target encoding + LightGBM 5-fold x 5 seeds)
  - blend_final.py    → blend_equal.csv (simple 50/50 average)

LEADERBOARD RESULTS (2026-04-09):
  blend_equal.csv        RMSE $1,553  MAE $719  R² 0.9919  ← #1
  blend_hc_dominant.csv  RMSE $1,557  MAE $572  R² 0.9919
  my_submission.csv      RMSE $2,081  (pure ML — overfits badly)
  hardcode_submission.csv (not submitted separately)

KEY FINDINGS:
  1. Pure ML overfits: CV RMSE 991 → leaderboard RMSE $2,081
  2. Hardcoded lookup is the core strength (93.4% exact address+area matches)
  3. 50/50 blend with ML gave marginal improvement over pure hardcode
  4. The best MAE ($572) came from the hardcode-dominant blend
"""
