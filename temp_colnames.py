import pandas as pd
path = 'data/raw/prices_10y.parquet'
df = pd.read_parquet(path)
print(list(df.columns))
