import sys
sys.stdout.reconfigure(encoding="utf-8")
import pandas as pd

targets = [
    ("prices_20y",      "data/raw/prices_20y.parquet"),
    ("financials",      "data/raw/financials.parquet"),
    ("fins_summary",    "data/raw/fins_summary.parquet"),
    ("dividends",       "data/raw/dividends.parquet"),
    ("indices",         "data/raw/indices.parquet"),
    ("mkt_breakdown",   "data/raw/mkt_breakdown.parquet"),
    ("investor_types",  "data/raw/investor_types.parquet"),
    ("short_ratio",     "data/raw/short_ratio.parquet"),
    ("margin_interest", "data/raw/margin_interest.parquet"),
]

for name, path in targets:
    try:
        df = pd.read_parquet(path)
        code_col = next((c for c in df.columns if c in ("Code", "Code5", "code")), None)
        n_codes = df[code_col].nunique() if code_col else "-"
        date_col = next((c for c in df.columns if "Date" in c or "date" in c), None)
        if date_col:
            dates = pd.to_datetime(df[date_col], errors="coerce")
            period = f"{dates.min().date()} ~ {dates.max().date()}"
        else:
            period = "-"
        print(f"\n=== {name} ===")
        print(f"  行数     : {len(df):,}")
        print(f"  銘柄数   : {n_codes}")
        print(f"  期間     : {period}")
        print(f"  列数     : {len(df.columns)}")
        print(f"  列一覧   : {list(df.columns)}")
    except Exception as e:
        print(f"\n=== {name} === ERROR: {e}")
