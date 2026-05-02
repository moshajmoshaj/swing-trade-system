"""
fetch_all_premium.py
Premium プランで取得できる全データを一括取得・保存するスクリプト。
Premium 契約期間中に 1 回実行し、ローカルに永続保存する。

実行方法:
    python fetch_all_premium.py

出力先: data/raw/
    prices_20y.parquet       株価20年分（prices_10y.parquet は上書きしない）
    financials.parquet       財務諸表フル（P&L・BS・CF）
    fins_summary.parquet     財務サマリー（決算日・EPS等）
    dividends.parquet        配当金情報
    investor_types.parquet   投資部門別売買状況
    mkt_breakdown.parquet    売買内訳データ
    indices.parquet          業種別・主要指数四本値
    short_ratio.parquet      業種別空売り比率
    margin_interest.parquet  信用取引週末残高

注意:
    - 既存ファイルはスキップ（--force で上書き）
    - 取得失敗は警告のみ（Standard プランで実行した場合など）
    - EQ_BARS_MINUTE（分足）は容量過大のため除外
    - デリバティブ系は現物専業のため除外
"""
import argparse
import io
import os
import sys
import time
sys.stdout.reconfigure(encoding="utf-8")

from pathlib import Path
import requests
import pandas as pd
from dotenv import load_dotenv
import jquantsapi
from jquantsapi.enums import BulkEndpoint

load_dotenv()

OUTPUT_DIR = Path("data/raw")

# (BulkEndpoint, 保存ファイル名, プライム絞り込みフラグ, 説明)
TARGETS = [
    (BulkEndpoint.EQ_BARS_DAILY,       "prices_20y.parquet",       True,  "株価20年分"),
    (BulkEndpoint.FIN_DETAILS,         "financials.parquet",       True,  "財務諸表フル（P&L・BS・CF）"),
    (BulkEndpoint.FIN_SUMMARY,         "fins_summary.parquet",     True,  "財務サマリー（決算日・EPS等）"),
    (BulkEndpoint.FIN_DIVIDEND,        "dividends.parquet",        True,  "配当金情報"),
    (BulkEndpoint.EQ_INVESTOR_TYPES,   "investor_types.parquet",   False, "投資部門別売買状況"),
    (BulkEndpoint.MKT_BREAKDOWN,       "mkt_breakdown.parquet",    True,  "売買内訳データ"),
    (BulkEndpoint.IDX_BARS_DAILY,      "indices.parquet",          False, "業種別・主要指数四本値"),
    (BulkEndpoint.MKT_SHORT_RATIO,     "short_ratio.parquet",      False, "業種別空売り比率"),
    (BulkEndpoint.MKT_MARGIN_INTEREST, "margin_interest.parquet",  False, "信用取引週末残高"),
]


def get_client() -> jquantsapi.ClientV2:
    api_key = os.getenv("JQUANTS_REFRESH_TOKEN")
    if not api_key:
        print("エラー: JQUANTS_REFRESH_TOKEN が未設定です。")
        sys.exit(1)
    return jquantsapi.ClientV2(api_key=api_key)


def fetch_prime_codes(client: jquantsapi.ClientV2) -> set[str]:
    master = client.get_eq_master()
    prime = master[master["MktNm"] == "プライム"]
    codes = set(prime["Code"].dropna().astype(str).tolist())
    print(f"  東証プライム銘柄数: {len(codes)} 銘柄")
    return codes


def download_bulk(client: jquantsapi.ClientV2, endpoint: BulkEndpoint) -> list[pd.DataFrame]:
    bulk_list = client.get_bulk_list(endpoint=endpoint)
    total = len(bulk_list)
    total_mb = bulk_list["Size"].sum() / 1024 / 1024
    print(f"  ファイル数: {total}  圧縮合計: {total_mb:.1f} MB")

    t0 = time.time()
    results: list[pd.DataFrame] = []
    failed: list[str] = []

    for i, row in enumerate(bulk_list.itertuples(), 1):
        key = row.Key
        try:
            url = client.get_bulk(key=key)
            resp = requests.get(url, timeout=60)
            resp.raise_for_status()
            df = pd.read_csv(io.BytesIO(resp.content), compression="gzip", low_memory=False)
            results.append(df)
        except Exception as e:
            print(f"  失敗: {key}  ({e})")
            failed.append(key)

        if i % 20 == 0 or i == total:
            elapsed = time.time() - t0
            eta = (elapsed / i) * (total - i)
            print(f"  [{i:3d}/{total}]  経過: {elapsed:.0f}秒  残り推定: {eta:.0f}秒")

    if failed:
        print(f"  警告: {len(failed)} ファイル取得失敗")
    return results


def postprocess_prices(df: pd.DataFrame, prime_codes: set[str]) -> pd.DataFrame:
    df["Code"] = df["Code"].astype(str).str.strip()
    df = df[df["Code"].isin(prime_codes)].copy()
    for raw, adj in [("O", "Open"), ("H", "High"), ("L", "Low"), ("C", "Close")]:
        if raw in df.columns and "AdjFactor" in df.columns:
            df[adj] = df[raw] * df["AdjFactor"]
    if "Vo" in df.columns:
        df = df.rename(columns={"Vo": "Volume"})
    df["Code4"] = df["Code"].str[:4]
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.sort_values(["Code", "Date"]).reset_index(drop=True)
    return df


def filter_prime(df: pd.DataFrame, prime_codes: set[str]) -> pd.DataFrame:
    code_cols = [c for c in df.columns if c.lower() in ("code", "code5")]
    if not code_cols:
        return df
    col = code_cols[0]
    df[col] = df[col].astype(str).str.strip()
    return df[df[col].isin(prime_codes)].reset_index(drop=True)


def fetch_one(
    client: jquantsapi.ClientV2,
    endpoint: BulkEndpoint,
    output_path: Path,
    prime_codes: set[str],
    filter_to_prime: bool,
    label: str,
    force: bool,
) -> bool:
    print(f"\n{'='*60}")
    print(f"  取得: {label}  ({endpoint.value})")
    print(f"  出力: {output_path}")
    print(f"{'='*60}")

    if output_path.exists() and not force:
        size_mb = output_path.stat().st_size / 1024 / 1024
        print(f"  スキップ（既存: {size_mb:.1f} MB）  --force で上書き")
        return True

    try:
        dfs = download_bulk(client, endpoint)
    except Exception as e:
        print(f"  エラー: Bulk一覧取得失敗 ({e})")
        print(f"  → Standard プランでは取得できない可能性があります")
        return False

    if not dfs:
        print("  エラー: データが空です")
        return False

    combined = pd.concat(dfs, ignore_index=True)
    print(f"  結合後: {len(combined):,} 行  ({combined.shape[1]} 列)")

    if endpoint == BulkEndpoint.EQ_BARS_DAILY:
        combined = postprocess_prices(combined, prime_codes)
    elif filter_to_prime:
        combined = filter_prime(combined, prime_codes)

    print(f"  フィルタ後: {len(combined):,} 行")

    # object列の混在型をstrに統一（pyarrow変換エラー回避）
    for col in combined.select_dtypes(include='object').columns:
        combined[col] = combined[col].astype(str)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(output_path)

    size_mb = output_path.stat().st_size / 1024 / 1024
    print(f"  保存完了: {size_mb:.1f} MB")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Premium プラン全データ一括取得")
    parser.add_argument("--force", action="store_true", help="既存ファイルを上書き")
    args = parser.parse_args()

    print("=" * 60)
    print("  Premium プラン全データ一括取得")
    print(f"  出力先: {OUTPUT_DIR.resolve()}")
    if args.force:
        print("  モード: 強制上書き（--force）")
    print("=" * 60)

    client = get_client()

    print("\n銘柄マスタ取得中...")
    prime_codes = fetch_prime_codes(client)

    results: list[tuple[str, bool]] = []
    total_start = time.time()

    for endpoint, filename, filter_prime_flag, label in TARGETS:
        output_path = OUTPUT_DIR / filename
        ok = fetch_one(
            client=client,
            endpoint=endpoint,
            output_path=output_path,
            prime_codes=prime_codes,
            filter_to_prime=filter_prime_flag,
            label=label,
            force=args.force,
        )
        results.append((label, ok))

    elapsed = time.time() - total_start

    print(f"\n{'='*60}")
    print(f"  完了サマリー  （総所要時間: {elapsed:.0f}秒）")
    print(f"{'='*60}")
    for label, ok in results:
        status = "✅ 成功" if ok else "❌ 失敗"
        print(f"  {status}  {label}")

    failed = [label for label, ok in results if not ok]
    if failed:
        print(f"\n  ⚠ {len(failed)} 件失敗。Standard プランでは取得できないデータの可能性があります。")
        print("  Premium にアップグレード後に再実行してください。")
    else:
        print("\n  全データ取得完了。Standard プランに戻してください。")


if __name__ == "__main__":
    main()
