import sys
sys.stdout.reconfigure(encoding="utf-8")
import pandas as pd
import numpy as np

df = pd.read_parquet("data/raw/margin_interest.parquet")
print(f"=== margin_interest ===")
print(f"行数: {len(df):,}  銘柄数: {df['Code'].nunique():,}")
print(f"期間: {df['Date'].min()} ~ {df['Date'].max()}")
print(f"列: {list(df.columns)}")
print(f"\nIssType の分布:")
print(df["IssType"].value_counts().head())

# 数値変換
for col in ["LongVol","ShrtVol","LongNegVol","ShrtNegVol"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# 信用倍率 = 買い残 / 売り残
df["margin_ratio"] = df["LongVol"] / df["ShrtVol"].replace(0, np.nan)

print(f"\n=== 信用倍率（LongVol/ShrtVol）分布 ===")
valid = df[df["margin_ratio"].notna() & df["ShrtVol"] > 0]
print(f"有効サンプル: {len(valid):,}件")
print(valid["margin_ratio"].describe().round(2))

# 一般信用（IssType=1）に絞る
df1 = df[df["IssType"].astype(str) == "1"].copy()
print(f"\n=== IssType=1（一般信用）サンプル ===")
print(f"行数: {len(df1):,}  銘柄数: {df1['Code'].nunique():,}")

# サンプル銘柄（トヨタ）
sample = df1[df1["Code"].str.startswith("7203")].sort_values("Date").tail(5)
print(f"\nサンプル（トヨタ系）信用倍率推移:")
cols = ["Date","Code","LongVol","ShrtVol","margin_ratio"]
print(sample[cols].to_string(index=False))

# 信用倍率の偏りを確認（低信用倍率 = 売り残過多 = 逆張り候補？）
print(f"\n=== 信用倍率 閾値別の銘柄数（直近データ） ===")
latest = df1.sort_values("Date").groupby("Code").last().reset_index()
latest["margin_ratio"] = pd.to_numeric(latest["LongVol"], errors="coerce") / \
                          pd.to_numeric(latest["ShrtVol"], errors="coerce").replace(0, np.nan)
for thresh in [0.5, 1.0, 2.0, 5.0, 10.0]:
    n = (latest["margin_ratio"] < thresh).sum()
    print(f"  倍率 < {thresh:5.1f}: {n:4d}銘柄 ({n/len(latest)*100:.0f}%)")
