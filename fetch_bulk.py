"""
Bulk API で全銘柄・10年分を高速一括取得
- /bulk/list → 138ファイル（月次+日次 gzip CSV）
- /bulk/get  → 各ファイルのダウンロードURL取得
- 直接メモリに読み込み → プライム絞り込み → parquet保存
- 推定所要時間: 2〜4分（51分 → 大幅短縮）
"""
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

OUTPUT = Path("data/raw/prices_10y.parquet")


def get_client() -> jquantsapi.ClientV2:
    api_key = os.getenv("JQUANTS_REFRESH_TOKEN")
    if not api_key:
        print("エラー: JQUANTS_REFRESH_TOKEN が未設定です。")
        sys.exit(1)
    return jquantsapi.ClientV2(api_key=api_key)


def main() -> None:
    print("=" * 64)
    print("  全銘柄 10年分価格データ取得（Bulk API）")
    print(f"  出力: {OUTPUT}")
    print("=" * 64)

    client = get_client()

    # プライム銘柄コード取得
    print("\n銘柄マスタ取得中...")
    master   = client.get_eq_master()
    prime_df = master[master["MktNm"] == "プライム"].copy()
    prime5   = set(prime_df["Code"].dropna().tolist())
    print(f"  東証プライム銘柄数: {len(prime5)} 銘柄")

    # Bulk ファイル一覧取得
    print("\nBulkファイル一覧取得中...")
    bulk_list = client.get_bulk_list(endpoint=BulkEndpoint.EQ_BARS_DAILY)
    total_files = len(bulk_list)
    total_mb    = bulk_list["Size"].sum() / 1024 / 1024
    print(f"  ファイル数: {total_files}  圧縮合計: {total_mb:.1f} MB")
    print(f"  期間: {bulk_list['Key'].iloc[0]} ～ {bulk_list['Key'].iloc[-1]}")

    # 全ファイルをダウンロード・読み込み
    print(f"\n全ファイルダウンロード中...")
    t0 = time.time()
    results: list[pd.DataFrame] = []
    failed:  list[str] = []

    for i, row in enumerate(bulk_list.itertuples(), 1):
        key = row.Key
        try:
            # ダウンロードURL取得（API呼び出し）
            url = client.get_bulk(key=key)

            # gzip CSV をメモリに直接読み込み
            resp = requests.get(url, timeout=60)
            resp.raise_for_status()
            df = pd.read_csv(io.BytesIO(resp.content), compression="gzip")
            results.append(df)

        except Exception as e:
            print(f"  失敗: {key}  ({e})")
            failed.append(key)

        if i % 20 == 0 or i == total_files:
            elapsed = time.time() - t0
            eta = (elapsed / i) * (total_files - i)
            print(f"  [{i:3d}/{total_files}]  経過: {elapsed:.0f}秒  残り推定: {eta:.0f}秒")

    elapsed_total = time.time() - t0
    print(f"\nダウンロード完了: {len(results)}ファイル  所要: {elapsed_total:.0f}秒")

    if not results:
        print("エラー: データが取得できませんでした。")
        sys.exit(1)

    # 結合
    print("\nデータ結合中...")
    combined = pd.concat(results, ignore_index=True)

    # Code を文字列に統一（歴史ファイルはint64・直近ファイルはobject）
    combined["Code"] = combined["Code"].astype(str).str.strip()
    prime5_str = {str(c) for c in prime5}
    print(f"  全銘柄合計: {len(combined):,} 行  {combined['Code'].nunique():,} 銘柄")

    # プライム絞り込み
    print("プライム銘柄に絞り込み中...")
    df_prime = combined[combined["Code"].isin(prime5_str)].copy()

    # 調整済み価格を計算（O/H/L/C × AdjFactor）
    for raw, adj in [("O", "Open"), ("H", "High"), ("L", "Low"), ("C", "Close")]:
        df_prime[adj] = df_prime[raw] * df_prime["AdjFactor"]
    df_prime = df_prime.rename(columns={"Vo": "Volume"})
    df_prime["Code4"] = df_prime["Code"].str[:4]
    df_prime["Date"]  = pd.to_datetime(df_prime["Date"], errors="coerce")
    df_prime = df_prime.sort_values(["Code", "Date"]).reset_index(drop=True)
    print(f"  絞り込み後: {len(df_prime):,} 行  {df_prime['Code'].nunique():,} 銘柄")

    # 保存
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    df_prime.to_parquet(OUTPUT)

    size_bytes = OUTPUT.stat().st_size
    size_mb    = size_bytes / 1024 / 1024

    print("\n" + "=" * 64)
    print("  保存完了")
    print("=" * 64)
    print(f"  ファイル      : {OUTPUT}")
    print(f"  ファイルサイズ: {size_mb:.1f} MB  ({size_bytes:,} bytes)")
    print(f"  銘柄数        : {df_prime['Code'].nunique():,} 銘柄")
    print(f"  データ件数    : {len(df_prime):,} 行")
    print(f"  期間（最小）  : {df_prime['Date'].min().date()}")
    print(f"  期間（最大）  : {df_prime['Date'].max().date()}")
    print(f"  所要時間      : {elapsed_total:.0f}秒")
    print("=" * 64)


if __name__ == "__main__":
    main()
