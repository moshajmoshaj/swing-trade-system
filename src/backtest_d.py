import os
import sys
import time
sys.stdout.reconfigure(encoding="utf-8")

from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = "Noto Sans JP"
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker
import pandas as pd
from dotenv import load_dotenv
import jquantsapi

sys.path.insert(0, str(Path(__file__).parent))
from indicators import add_indicators
from strategy_d import generate_signals
from utils.risk import calc_position_size

load_dotenv()

INITIAL_CAPITAL = 1_000_000
ATR_STOP_MULT   = 1.5  # 戦略D: 損切り ATR × 1.5
ATR_TP_MULT     = 3.0  # 戦略D: 利確 ATR × 3（高期待値）
MAX_HOLD_DAYS   = 5    # 戦略D: 最大保有5日（短期）
PRIME_CACHE     = Path("data/raw/prices_10y.parquet")
CANDIDATE_CSV   = Path("logs/strategy_d_candidates.csv")

MIN_TRADES   = 3
MIN_WIN_RATE = 50.0
MIN_PNL      = 0
MAX_DD_LIMIT = -5.0


def _get_client() -> jquantsapi.ClientV2:
    api_key = os.getenv("JQUANTS_REFRESH_TOKEN")
    if not api_key:
        print("エラー: JQUANTS_REFRESH_TOKEN が未設定です。")
        sys.exit(1)
    return jquantsapi.ClientV2(api_key=api_key)


def _fetch_one(client: jquantsapi.ClientV2, code5: str, start: str, end: str) -> pd.DataFrame:
    try:
        df = client.get_eq_bars_daily(code=code5, from_yyyymmdd=start, to_yyyymmdd=end)
        if df.empty:
            return pd.DataFrame()
        df = df.rename(columns={
            "AdjO": "Open", "AdjH": "High", "AdjL": "Low",
            "AdjC": "Close", "AdjVo": "Volume",
        })
        df["Date"]  = pd.to_datetime(df["Date"])
        df["Code4"] = code5[:-1]
        return df.sort_values("Date").reset_index(drop=True)
    except Exception:
        return pd.DataFrame()


def _split_by_code(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    result = {}
    for code4, group in df.groupby("Code4"):
        result[str(code4)] = group.drop(columns=["Code4"]).sort_values("Date").reset_index(drop=True)
    return result


def fetch_prime_ohlcv(data_start: str, end: str) -> dict[str, pd.DataFrame]:
    """
    東証プライム全銘柄を逐次フェッチ（レート制限対策）→ parquetキャッシュ。
    既存キャッシュがあれば不足分のみ追加取得してマージ保存。
    """
    PRIME_CACHE.parent.mkdir(parents=True, exist_ok=True)

    client = _get_client()

    print("東証プライム銘柄リスト取得中...")
    master       = client.get_eq_master()
    prime_df     = master[master["MktNm"] == "プライム"].copy()
    all_codes5   = prime_df["Code"].dropna().tolist()
    print(f"  対象: {len(all_codes5)} 銘柄")

    cached_codes4: set[str] = set()
    cached_df: pd.DataFrame | None = None
    if PRIME_CACHE.exists():
        cached_df     = pd.read_parquet(PRIME_CACHE)
        cached_df["Date"] = pd.to_datetime(cached_df["Date"])
        # prices_10y.parquet の生列を削除して列を正規化
        _raw = [c for c in ["Code", "O", "H", "L", "C", "Vo", "Va", "UL", "LL", "AdjFactor"]
                if c in cached_df.columns]
        cached_df = cached_df.drop(columns=_raw)
        cached_codes4 = set(cached_df["Code4"].unique())
        print(f"  既存キャッシュ: {len(cached_codes4)}銘柄")

    missing_codes5 = [c for c in all_codes5 if c[:-1] not in cached_codes4]

    if not missing_codes5:
        print(f"キャッシュ完全一致: {len(cached_codes4)}銘柄  読み込み完了")
        return _split_by_code(cached_df)

    print(f"\n不足銘柄を逐次取得中: {len(missing_codes5)}銘柄  {data_start} ～ {end}")
    print("  ※レート制限対策のため0.35秒間隔。初回全銘柄は約7分かかります。")
    t0         = time.time()
    results: list[pd.DataFrame] = []
    done_count = 0
    total      = len(missing_codes5)

    for c5 in missing_codes5:
        df_one = _fetch_one(client, c5, data_start, end)
        done_count += 1
        if not df_one.empty:
            results.append(df_one)
        time.sleep(0.35)
        if done_count % 200 == 0 or done_count == total:
            elapsed = time.time() - t0
            eta = (elapsed / done_count) * (total - done_count) if done_count else 0
            print(f"  [{done_count:4d}/{total}] 取得: {len(results)}銘柄  経過: {elapsed:.0f}秒  残り推定: {eta:.0f}秒")

    elapsed_total = time.time() - t0
    print(f"\n取得完了: {len(results)} 銘柄  所要時間: {elapsed_total:.0f}秒")

    new_df = pd.concat(results, ignore_index=True) if results else pd.DataFrame()
    if cached_df is not None and not new_df.empty:
        combined = pd.concat([cached_df, new_df], ignore_index=True)
    elif cached_df is not None:
        combined = cached_df
    else:
        combined = new_df

    combined.to_parquet(PRIME_CACHE)
    size_mb     = PRIME_CACHE.stat().st_size / 1024 / 1024
    total_codes = combined["Code4"].nunique()
    print(f"キャッシュ保存: {PRIME_CACHE}  ({size_mb:.1f} MB)  合計 {total_codes}銘柄")

    return _split_by_code(combined)


def run_backtest(df: pd.DataFrame, backtest_start: str = "") -> dict:
    df = add_indicators(df)
    df = generate_signals(df)

    start_dt = pd.Timestamp(backtest_start) if backtest_start else df["Date"].iloc[0]

    capital  = float(INITIAL_CAPITAL)
    equity   = [capital]
    dates    = [start_dt]
    trades   = []
    in_trade = False

    for i, row in df.iterrows():
        if row["Date"] <= start_dt:
            continue

        if in_trade:
            hold_days += 1
            hi, lo, cl = row["High"], row["Low"], row["Close"]
            exit_price = exit_reason = None

            if lo <= stop_loss:
                exit_price, exit_reason = stop_loss, "損切り"
            elif hi >= take_profit:
                exit_price, exit_reason = take_profit, "利確"
            elif hold_days >= MAX_HOLD_DAYS:
                exit_price, exit_reason = cl, "期間満了"

            if exit_price is not None:
                pnl = (exit_price - entry_price) * shares
                capital += pnl
                trades.append({
                    "entry_date": entry_date, "exit_date": row["Date"],
                    "entry_price": entry_price, "exit_price": exit_price,
                    "shares": shares, "pnl": pnl, "reason": exit_reason,
                })
                in_trade = False

        prev = df.iloc[i - 1]
        if not in_trade and prev["signal"] == 1 and pd.notna(prev["ATR14"]):
            entry_price = row["Open"]
            atr = prev["ATR14"]
            shares, _ = calc_position_size(capital, atr, entry_price)
            stop_loss = entry_price - atr * ATR_STOP_MULT
            take_profit = entry_price + atr * ATR_TP_MULT
            if shares > 0 and entry_price * shares <= capital:
                in_trade   = True
                entry_date = row["Date"]
                hold_days  = 0

        equity.append(capital)
        dates.append(row["Date"])

    total    = len(trades)
    wins     = [t for t in trades if t["pnl"] > 0]
    win_rate = len(wins) / total * 100 if total > 0 else 0
    avg_pnl  = sum(t["pnl"] for t in trades) / total if total > 0 else 0
    final_pnl = capital - INITIAL_CAPITAL

    eq_s   = pd.Series(equity)
    max_dd = ((eq_s - eq_s.cummax()) / eq_s.cummax() * 100).min()

    return {
        "trades": trades, "equity": equity, "dates": dates,
        "total": total, "win_rate": win_rate, "avg_pnl": avg_pnl,
        "max_dd": max_dd, "final_pnl": final_pnl, "final_cap": capital,
    }


def _backtest_worker(args: tuple) -> tuple:
    """ProcessPoolExecutor用ワーカー。
    Windows spawn 方式のため module-level で定義が必須。"""
    code4, df, bt_start_str = args
    try:
        return code4, run_backtest(df, backtest_start=bt_start_str)
    except Exception:
        return code4, None


def save_candidate_chart(top_results: dict[str, dict], names: dict[str, str]) -> None:
    path = "logs/strategy_d_equity.png"
    fig, ax = plt.subplots(figsize=(14, 6))
    for code4, res in top_results.items():
        label = f"{code4} {names.get(code4, '')}"
        ax.plot(res["dates"], res["equity"], linewidth=1.2, label=label)
    ax.set_title("戦略D 運用候補銘柄 損益曲線（上位20銘柄）", fontsize=14)
    ax.set_xlabel("日付")
    ax.set_ylabel("資産（円）")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=45)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.legend(fontsize=6, loc="upper left", ncol=3)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"損益曲線を保存しました: {path}")


def main() -> None:
    today        = datetime.today()
    end          = today.strftime("%Y%m%d")
    bt_start     = today - timedelta(days=365 * 2)
    data_start   = (bt_start - timedelta(days=250)).strftime("%Y%m%d")
    bt_start_str = bt_start.strftime("%Y-%m-%d")

    print("=" * 60)
    print("  戦略D バックテスト（ギャップアップ翌日戦略）")
    print(f"  バックテスト期間: {bt_start_str} ～ {today.strftime('%Y-%m-%d')}")
    print("=" * 60)

    price_data = fetch_prime_ohlcv(data_start, end)
    print(f"\nバックテスト対象: {len(price_data)} 銘柄")

    try:
        client   = _get_client()
        master   = client.get_eq_master()
        name_map = dict(zip(master["Code"].str[:-1], master["CoName"]))
    except Exception:
        name_map = {}

    print("バックテスト実行中...")
    t0 = time.time()

    # 並列処理用リストを事前構築
    stock_list = [(c, df, bt_start_str) for c, df in price_data.items() if len(df) >= 260]
    skipped = len(price_data) - len(stock_list)
    print(f"  並列処理対象: {len(stock_list)} 銘柄（workers=4, chunksize=50）  スキップ: {skipped}")

    all_results: dict[str, dict] = {}
    with ProcessPoolExecutor(max_workers=4) as executor:
        for i, (code4, result) in enumerate(
            executor.map(_backtest_worker, stock_list, chunksize=50), 1
        ):
            if result is not None:
                all_results[code4] = result
            if i % 300 == 0 or i == len(stock_list):
                elapsed = time.time() - t0
                print(f"  [{i:4d}/{len(stock_list)}] 完了  経過: {elapsed:.0f}秒")

    elapsed_bt = time.time() - t0
    print(f"\nバックテスト完了: {len(all_results)}銘柄  所要: {elapsed_bt:.0f}秒  スキップ: {skipped}銘柄")

    # 結果フィルタリング
    filtered = {
        c: r for c, r in all_results.items()
        if r["total"] >= MIN_TRADES and r["win_rate"] >= MIN_WIN_RATE
           and r["final_pnl"] >= MIN_PNL and r["max_dd"] >= MAX_DD_LIMIT
    }
    print(f"\nフィルタ後: {len(filtered)} 銘柄（最小取引数: {MIN_TRADES}、最小勝率: {MIN_WIN_RATE}%）")

    if not filtered:
        print("警告: フィルタを満たす銘柄がありません")
        return

    sorted_by_pnl = sorted(filtered.items(), key=lambda x: x[1]["final_pnl"], reverse=True)
    top_20 = dict(sorted_by_pnl[:20])

    # CSVに出力
    results_list = []
    for code4, res in sorted_by_pnl:
        results_list.append({
            "Code": code4,
            "Name": name_map.get(code4, ""),
            "Trades": res["total"],
            "WinRate": f"{res['win_rate']:.1f}%",
            "AvgPnL": f"{res['avg_pnl']:.0f}",
            "FinalPnL": f"{res['final_pnl']:.0f}",
            "FinalCapital": f"{res['final_cap']:.0f}",
            "MaxDD": f"{res['max_dd']:.1f}%",
        })

    results_df = pd.DataFrame(results_list)
    CANDIDATE_CSV.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(CANDIDATE_CSV, index=False, encoding="utf-8-sig")
    print(f"\n結果を保存: {CANDIDATE_CSV}")
    print("\n上位10銘柄:")
    print(results_df.head(10).to_string(index=False))

    # グラフを保存
    if top_20:
        save_candidate_chart(top_20, name_map)


if __name__ == "__main__":
    main()
