import sys
sys.stdout.reconfigure(encoding="utf-8")
import pandas as pd

df = pd.read_parquet("data/raw/indices.parquet")
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
for col in ["O","H","L","C"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

print(f"=== indices ===")
print(f"行数: {len(df):,}  コード数: {df['Code'].nunique():,}")
print(f"期間: {df['Date'].min().date()} ~ {df['Date'].max().date()}")

# コード一覧（先頭30件）
codes = df.groupby("Code").agg(count=("Date","count"), last=("Date","max")).reset_index()
codes = codes.sort_values("Code")
print(f"\n=== インデックスコード一覧（先頭30） ===")
print(codes.head(30).to_string(index=False))

# TOPIX（1306 ETFとは別に指数コードを確認）
topix_like = codes[codes["Code"].str.contains("TOPIX|0000|9999|1306", na=False)]
print(f"\n=== TOPIX関連コード ===")
print(topix_like.to_string(index=False))

# 業種別TOPIXコードを確認
sector_codes = codes[codes["Code"].str.len() >= 4]
print(f"\n=== 4桁以上コード（業種別指数候補） ===")
print(sector_codes.to_string(index=False))
