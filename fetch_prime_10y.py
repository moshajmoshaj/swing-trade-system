"""
東証プライム全銘柄 10年分価格データ再取得スクリプト
対象期間: 2016-04-25 ～ 2026-04-24（Standardプランカバー範囲）
手法: 銘柄ごと逐次取得（0.5秒間隔・120req/min制限準拠）
出力先: data/raw/prices_all_prime.parquet
"""
import os
import sys
import time
sys.stdout.reconfigure(encoding="utf-8")

from pathlib import Path
import pandas as pd
from dotenv import load_dotenv
import jquantsapi

load_dotenv()

DATA_START  = "20160425"   # Standardプランカバー開始日
DATA_END    = "20260424"
PRIME_CACHE = Path("data/raw/prices_all_prime.parquet")
SLEEP_SEC   = 0.5          # 120req/min = 2req/s → 0.5秒間隔


def get_client() -> jquantsapi.ClientV2:
    api_key = os.getenv("JQUANTS_REFRESH_TOKEN")
    if not api_key:
        print("エラー: JQUANTS_REFRESH_TOKEN が未設定です。")
        sys.exit(1)
    return jquantsapi.ClientV2(api_key=api_key)


def fetch_one(client: jquantsapi.ClientV2, code5: str) -> pd.DataFrame:
    try:
        df = client.get_eq_bars_daily(
            code=code5,
            from_yyyymmdd=DATA_START,
            to_yyyymmdd=DATA_END,
        )
        if df.empty:
            return pd.DataFrame()
        df = df.rename(columns={
            "AdjO": "Open", "AdjH": "High", "AdjL": "Low",
            "AdjC": "Close", "AdjVo": "Volume",
        })
        df["Date"]  = pd.to_datetime(df["Date"])
        df["Code4"] = code5[:-1]
        return df.sort_values("Date").reset_index(drop=True)
    except Exception as e:
        print(f"  取得失敗: {code5}  ({e})")
        return pd.DataFrame()


def main() -> None:
    print("=" * 64)
    print("  東証プライム全銘柄 10年分価格データ取得")
    print(f"  期間: {DATA_START} ～ {DATA_END}")
    print(f"  手法: 銘柄ごと逐次取得（{SLEEP_SEC}秒間隔）")
    print(f"  出力: {PRIME_CACHE}")
    print("=" * 64)

    # 少し待機（直前のリクエストによるレート制限リセット）
    print("\n30秒待機（レートリセット）...")
    time.sleep(30)

    client = get_client()

    print("銘柄マスタ取得中...")
    master   = client.get_eq_master()
    prime_df = master[master["MktNm"] == "プライム"].copy()
    codes5   = prime_df["Code"].dropna().tolist()
    print(f"  東証プライム銘柄数: {len(codes5)} 銘柄")
    print(f"  推定所要時間: {len(codes5) * SLEEP_SEC / 60:.0f}分")

    t0 = time.time()
    results: list[pd.DataFrame] = []
    failed:  list[str] = []

    for i, c5 in enumerate(codes5, 1):
        df_one = fetch_one(client, c5)
        if not df_one.empty:
            results.append(df_one)
        else:
            failed.append(c5)
        time.sleep(SLEEP_SEC)

        if i % 200 == 0 or i == len(codes5):
            elapsed = time.time() - t0
            eta = (elapsed / i) * (len(codes5) - i)
            print(f"  [{i:4d}/{len(codes5)}]  成功: {len(results)}  失敗: {len(failed)}"
                  f"  経過: {elapsed:.0f}秒  残り: {eta:.0f}秒")

    elapsed_total = time.time() - t0
    print(f"\n取得完了: {len(results)} 銘柄  所要: {elapsed_total:.0f}秒")

    if not results:
        print("エラー: データが取得できませんでした。")
        sys.exit(1)

    combined = pd.concat(results, ignore_index=True)
    PRIME_CACHE.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(PRIME_CACHE)

    size_bytes = PRIME_CACHE.stat().st_size
    size_mb    = size_bytes / 1024 / 1024

    print("\n" + "=" * 64)
    print("  保存完了")
    print("=" * 64)
    print(f"  ファイル      : {PRIME_CACHE}")
    print(f"  ファイルサイズ: {size_mb:.1f} MB  ({size_bytes:,} bytes)")
    print(f"  銘柄数        : {combined['Code4'].nunique():,} 銘柄")
    print(f"  データ件数    : {len(combined):,} 行")
    print(f"  期間（最小）  : {combined['Date'].min().date()}")
    print(f"  期間（最大）  : {combined['Date'].max().date()}")
    if failed:
        print(f"  取得失敗      : {len(failed)} 銘柄  {failed[:10]}")
    print(f"  所要時間      : {elapsed_total:.0f}秒")
    print("=" * 64)


if __name__ == "__main__":
    main()
