import sys
sys.stdout.reconfigure(encoding="utf-8")
import pandas as pd

df = pd.read_parquet("data/raw/fins_summary.parquet")
df["DiscDate"] = pd.to_datetime(df["DiscDate"], errors="coerce")

print("=== DocType 値の分布 ===")
print(df["DocType"].value_counts().head(10))

print("\n=== CurPerType 値の分布 ===")
print(df["CurPerType"].value_counts().head(10))

# 年次決算のみ抽出（CurPerType=FY が通期）
annual = df[df["CurPerType"] == "FY"].copy()
# 数値列を変換（parquet保存時にstr化されているため）
for col in ["Sales", "OP", "NP", "EPS", "Eq", "TA", "EqAR", "CFO"]:
    if col in annual.columns:
        annual[col] = pd.to_numeric(annual[col], errors="coerce")

print(f"\n=== 年次決算（CurPerType=FY）===")
print(f"行数: {len(annual):,}  銘柄数: {annual['Code'].nunique():,}")
print(f"期間: {annual['DiscDate'].min().date()} ~ {annual['DiscDate'].max().date()}")

# サンプル1銘柄で主要指標を確認（トヨタ 7203）
sample = annual[annual["Code"].str.startswith("7203")].sort_values("DiscDate").tail(5)
print("\n=== サンプル（トヨタ系）主要財務指標 ===")
cols = ["DiscDate", "Code", "Sales", "OP", "NP", "Eq", "EqAR", "CFO"]
cols = [c for c in cols if c in sample.columns]
print(sample[cols].to_string(index=False))

# ROE計算
annual["ROE"] = annual["NP"] / annual["Eq"] * 100

print("\n=== ROE 分布（年次・直近5年） ===")
recent = annual[annual["DiscDate"] >= "2020-01-01"]
print(f"有効サンプル: {recent['ROE'].dropna().__len__():,}件")
print(recent["ROE"].describe().round(1))

print("\n=== EqAR（自己資本比率）分布 ===")
print(recent["EqAR"].describe().round(1))

print("\n=== 品質フィルター（ROE>10% & EqAR>30% & CFO>0）===")
latest = annual.sort_values("DiscDate").groupby("Code").last().reset_index()
latest["ROE"] = latest["NP"] / latest["Eq"] * 100
quality = latest[
    (latest["ROE"] > 10) &
    (latest["EqAR"] > 30) &
    (latest["CFO"] > 0)
]
total = latest["ROE"].notna().sum()
print(f"有効銘柄: {total:,}  通過: {len(quality):,}銘柄 ({len(quality)/total*100:.0f}%)")
