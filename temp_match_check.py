import pandas as pd
cands = pd.read_csv('logs/final_candidates.csv')
print('cands code sample:', cands['code'].astype(str).str.zfill(4).head(10).tolist())
all_prices = pd.read_parquet('data/raw/prices_10y.parquet')
print('all_prices Code sample:', pd.Series(all_prices['Code'].astype(str).str.zfill(4).unique()).head(20).tolist())
print('available count:', len(set(all_prices['Code'].astype(str).str.zfill(4).unique())))
print('sample intersection count:', len(set(cands['code'].astype(str).str.zfill(4)).intersection(set(all_prices['Code'].astype(str).str.zfill(4).unique()))))
