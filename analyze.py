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

train['ppsf'] = train['price'] / train['area_sqft']

# 1. Within-unit price variation
unit_stats = train.groupby('unit_key').agg(
    ppsf_std=('ppsf','std'), count=('ppsf','count'),
    price_std=('price','std'), price_mean=('price','mean')
)
multi = unit_stats[unit_stats['count'] >= 2]
print(f'Units with 2+ txns: {len(multi)}')
print(f'Avg within-unit price std: ${multi["price_std"].mean():.0f}')
print(f'Avg within-unit price mean: ${multi["price_mean"].mean():.0f}')
print(f'Within-unit CV: {multi["price_std"].mean()/multi["price_mean"].mean()*100:.1f}%')

# 2. Floor as within-unit differentiator
train['bld_flat'] = train['building'] + '|' + train['Flat'].fillna('X')
bf_groups = train.groupby('bld_flat').filter(lambda x: len(x) >= 5)
corr = bf_groups.groupby('bld_flat').apply(lambda x: x['floor'].corr(x['price']), include_groups=False)
print(f'\nFloor-Price corr within bld+flat (mean): {corr.mean():.3f}')

# 3. Address parsing
print('\nSample addresses:')
for a in train['address'].head(10):
    print(f'  {a}')

# 4. Area within same unit
bf_area_std = train.groupby('unit_key')['area_sqft'].std()
print(f'\nArea std within same unit: mean={bf_area_std.mean():.2f}')

# 5. Floor values
print(f'\nFloor value counts (top 10):')
print(train['floor'].value_counts().head(15))

# 6. How many test units have NO match at all?
test_no_bld = ~test['building'].isin(set(train['building']))
print(f'\nTest with no building match: {test_no_bld.sum()}')
test_no_unit = ~test['unit_key'].isin(set(train['unit_key']))
print(f'Test with no unit match: {test_no_unit.sum()}')

# 7. Check address for floor description (Lower/Middle/Upper)
import re
for a in train['address'].sample(20, random_state=42):
    floor_desc = re.search(r'(Lower|Middle|Upper|High|Low)\s+Floor', a)
    if floor_desc:
        print(f'  {floor_desc.group()} -> {a}')
