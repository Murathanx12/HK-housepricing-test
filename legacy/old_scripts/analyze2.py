import pandas as pd
import numpy as np

train = pd.read_csv('./data/HK_house_transactions.csv')
test = pd.read_csv('./data/test_features.csv')

def get_building(addr):
    if pd.isna(addr): return 'UNK'
    return addr.split(',')[0].strip()

for df in [train, test]:
    df['building'] = df['address'].apply(get_building)
    df['unit_key'] = df['building'] + '|' + df['Tower'].fillna('X').astype(str) + '|' + df['Flat'].fillna('X')
    df['floor_cat'] = np.where(df['floor'] <= 10, 0, np.where(df['floor'] <= 17, 1, 2))
    df['unit_fc'] = df['unit_key'] + '|' + df['floor_cat'].astype(str)

train['ppsf'] = train['price'] / train['area_sqft']

# 1. How many area_sqft values exist per unit? Is area constant or varying?
unit_area = train.groupby('unit_key')['area_sqft'].agg(['mean','std','nunique','count'])
multi = unit_area[unit_area['count'] >= 2]
print("=== Area variation within same unit ===")
print(f"Units with multiple area values: {(multi['nunique']>1).sum()}/{len(multi)}")
print(f"Avg area std within unit: {multi['std'].mean():.1f}")

# 2. Could Block help differentiate?
print("\n=== Block patterns ===")
train['bld_block'] = train['building'] + '|' + train['Block'].fillna('X').astype(str)
test['bld_block'] = test['building'] + '|' + test['Block'].fillna('X').astype(str)
bb_overlap = set(test['bld_block']) & set(train['bld_block'])
print(f"Building+Block overlap: {len(bb_overlap)}")
print(f"Test rows with bld+block match: {test['bld_block'].isin(bb_overlap).sum()}/{len(test)}")

# 3. What about building+flat+floor_cat (ignoring tower)?
train['bf_fc'] = train['building'] + '|' + train['Flat'].fillna('X') + '|' + train['floor_cat'].astype(str)
test['bf_fc'] = test['building'] + '|' + test['Flat'].fillna('X') + '|' + test['floor_cat'].astype(str)
bffc_overlap = set(test['bf_fc']) & set(train['bf_fc'])
print(f"\nBuilding+Flat+FloorCat overlap: {test['bf_fc'].isin(bffc_overlap).sum()}/{len(test)}")

# 4. Exact floor match within unit
train['unit_floor'] = train['unit_key'] + '|' + train['floor'].astype(str)
test['unit_floor'] = test['unit_key'] + '|' + test['floor'].astype(str)
uf_overlap = set(test['unit_floor']) & set(train['unit_floor'])
print(f"Unit+ExactFloor overlap: {test['unit_floor'].isin(uf_overlap).sum()}/{len(test)}")
uf_stats = train.groupby('unit_floor')['price'].agg(['std','count'])
uf_multi = uf_stats[uf_stats['count'] >= 2]
print(f"Unit+ExactFloor with 2+ txns: {len(uf_multi)}, avg price std: ${uf_multi['std'].mean():.0f}")

# 5. What if we use area_sqft rounded as part of unit key?
train['area_bin'] = (train['area_sqft'] / 10).round() * 10
test['area_bin'] = (test['area_sqft'] / 10).round() * 10
train['unit_area'] = train['unit_key'] + '|' + train['area_bin'].astype(str)
test['unit_area'] = test['unit_key'] + '|' + test['area_bin'].astype(str)
ua_overlap = set(test['unit_area']) & set(train['unit_area'])
print(f"\nUnit+AreaBin overlap: {test['unit_area'].isin(ua_overlap).sum()}/{len(test)}")
ua_stats = train.groupby('unit_area')['price'].agg(['std','count'])
ua_multi = ua_stats[ua_stats['count'] >= 2]
print(f"Unit+AreaBin with 2+ txns: {len(ua_multi)}, avg price std: ${ua_multi['std'].mean():.0f}")

# 6. Address - full address match?
full_overlap = set(test['address']) & set(train['address'])
print(f"\nFull address overlap: {test['address'].isin(full_overlap).sum()}/{len(test)}")
fa_stats = train.groupby('address')['price'].agg(['std','count'])
fa_multi = fa_stats[fa_stats['count'] >= 2]
print(f"Full address with 2+ txns: {len(fa_multi)}, avg price std: ${fa_multi['std'].mean():.0f}")
