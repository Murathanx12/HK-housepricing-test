# HK Rental Price Prediction — Instructions for Claude

## Current State
- **#2 on leaderboard**: $1,355 RMSE, $493 MAE
- **#1 JigsawBlock**: $1,347 RMSE, $556 MAE (higher MAE = broader matching)
- MSE gap: 187M total (need to improve ~30 rows by ~$2.5K each)

## Architecture: Pure Lookup (NO ML)
93.4% of test rows have exact full_addr matches. Lookup beats every ML approach.
The backbone is in `LEGACY_winner_1355.py` — NEVER modify that file.

## Proven Rules (violation = worse score)
1. **Direct price is always right** for n=1 exact matches
2. **Mean > median** for n=3 groups ($38 improvement)
3. **Never blend ML** into predictions (LGB blend = $1,915 disaster)
4. **5% building weight** for n=1 is the sweet spot
5. **No internal validation works** — only leaderboard gives truth
6. **Building median ppsf is too coarse** for outlier detection

## What FAILED (do not retry)
- ML blending (any amount): $1,373-$1,915
- Z-score outlier correction (flat or size-adjusted): $1,357-$1,411
- Heavier building blend (>5%): $1,357-$1,418
- Shrinkage, rounding, post-processing: all worse
- Fuzzy address matching: neutral ($1,355)

## What to Try Next
1. **Broader matching with floor adjustment**: Strip floor band from address
   to create larger groups (n=1→n=3), apply floor slope correction.
   Matches JigsawBlock's profile (higher MAE, lower RMSE).
2. **Probe-based optimization**: Change specific rows, submit, check RMSE.
3. **External data**: Building age, MTR walking distance.

## Naming Convention
Submissions should be numbered: `1.csv`, `2.csv`, etc.
