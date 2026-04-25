import pandas as pd
cands = pd.read_csv('logs/final_candidates.csv')
print('cands code sample:', cands['code'].astype(str).str.zfill(4).head(3).tolist())
all_prices = pd.read_parquet('data/raw/prices_10y.parquet')
print('prices Code sample:', all_prices['Code'].astype(str).str.zfill(4).head(3).tolist())
print('prices unique Code sample:', pd.Series(all_prices['Code'].astype(str).str.zfill(4).unique()).head(10).tolist())
