"""
複合運用シミュレーション：戦略A・C・D の組み合わせ運用
- 各戦略の上位候補銘柄を統合
- 複合シグナルで複数ポジション保持
- リスク管理：最大5ポジション、資金配分均等
"""
import os
import sys
sys.stdout.reconfigure(encoding="utf-8")

from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
import jquantsapi

sys.path.insert(0, str(Path(__file__).parent))
from indicators import add_indicators
from strategy import generate_signals as signals_a
from strategy_c import generate_signals as signals_c
from strategy_d import generate_signals as signals_d
from utils.risk import calc_position_size

load_dotenv()

INITIAL_CAPITAL = 1_000_000
MAX_POSITIONS   = 5
BACKTEST_START  = "2024-04-29"
BACKTEST_END    = "2026-04-29"


def _get_client() -> jquantsapi.ClientV2:
    api_key = os.getenv("JQUANTS_REFRESH_TOKEN")
    if not api_key:
        print("エラー: JQUANTS_REFRESH_TOKEN が未設定です。")
        sys.exit(1)
    return jquantsapi.ClientV2(api_key=api_key)


def load_candidate_codes() -> set[str]:
    """各戦略の上位候補銘柄を読み込み、重複排除"""
    codes = set()

    # 戦略A（順張り）上位10銘柄
    if Path("logs/candidates.csv").exists():
        df_a = pd.read_csv("logs/candidates.csv", nrows=10)
        codes.update(df_a["code"].astype(str))

    # 戦略C（逆張り）上位10銘柄
    if Path("logs/strategy_c_candidates.csv").exists():
        df_c = pd.read_csv("logs/strategy_c_candidates.csv", nrows=10)
        codes.update(df_c["Code"].astype(str))

    # 戦略D（ギャップアップ）上位10銘柄
    if Path("logs/strategy_d_candidates.csv").exists():
        df_d = pd.read_csv("logs/strategy_d_candidates.csv", nrows=10)
        codes.update(df_d["Code"].astype(str))

    return codes


def fetch_candidate_data(codes: set[str]) -> dict[str, pd.DataFrame]:
    """候補銘柄のOHLCV取得"""
    client = _get_client()

    # キャッシュから読み込み
    try:
        cached = pd.read_parquet("data/raw/prices_10y.parquet")
        cached["Date"] = pd.to_datetime(cached["Date"])

        result = {}
        for code4, group in cached.groupby("Code4"):
            if str(code4) in codes:
                df = group.drop(columns=["Code4"]).sort_values("Date").reset_index(drop=True)
                if len(df) >= 260:
                    result[str(code4)] = df

        print(f"キャッシュから {len(result)} 銘柄を読み込み")
        return result
    except Exception as e:
        print(f"エラー: {e}")
        return {}


def run_combined_backtest(price_data: dict[str, pd.DataFrame]) -> dict:
    """複合戦略バックテスト"""

    # 全銘柄のシグナルを統合
    all_signals = {}
    for code4, df in price_data.items():
        df = add_indicators(df)

        # 3戦略のシグナル
        df_a = signals_a(df.copy())
        df_c = signals_c(df.copy())
        df_d = signals_d(df.copy())

        # 複合シグナル：いずれかの戦略でシグナルが出たら対象
        df["combined_signal"] = (
            (df_a["signal"] == 1) | (df_c["signal"] == 1) | (df_d["signal"] == 1)
        ).astype(int)

        all_signals[code4] = df

    # ポートフォリオシミュレーション
    start_dt = pd.Timestamp(BACKTEST_START)
    end_dt = pd.Timestamp(BACKTEST_END)

    capital = float(INITIAL_CAPITAL)
    equity_curve = [capital]
    dates = [start_dt]
    positions = {}  # {code4: {entry_date, entry_price, shares, stop_loss, take_profit}}
    trades = []

    # 共通の日付列を生成（全銘柄の日付をマージ）
    all_dates = set()
    for df in all_signals.values():
        all_dates.update(df["Date"].unique())
    all_dates = sorted(all_dates)
    all_dates = [d for d in all_dates if start_dt < d <= end_dt]

    for current_date in all_dates:
        # ポジション評価・エグジット
        codes_to_exit = []
        for code4, pos in positions.items():
            df = all_signals[code4]
            row = df[df["Date"] == current_date]
            if row.empty:
                continue

            row = row.iloc[0]
            hi, lo, cl = row["High"], row["Low"], row["Close"]

            # エグジット判定
            exit_price = exit_reason = None
            if lo <= pos["stop_loss"]:
                exit_price, exit_reason = pos["stop_loss"], "損切り"
            elif hi >= pos["take_profit"]:
                exit_price, exit_reason = pos["take_profit"], "利確"
            elif (current_date - pos["entry_date"]).days >= 10:
                exit_price, exit_reason = cl, "期間満了"

            if exit_price is not None:
                pnl = (exit_price - pos["entry_price"]) * pos["shares"]
                capital += pnl
                trades.append({
                    "code": code4, "entry_date": pos["entry_date"],
                    "exit_date": current_date, "entry_price": pos["entry_price"],
                    "exit_price": exit_price, "shares": pos["shares"],
                    "pnl": pnl, "reason": exit_reason,
                })
                codes_to_exit.append(code4)

        for code4 in codes_to_exit:
            del positions[code4]

        # 新規エントリー
        if len(positions) < MAX_POSITIONS:
            for code4, df in all_signals.items():
                if code4 in positions:
                    continue

                row = df[df["Date"] == current_date]
                if row.empty:
                    continue

                prev_row = df[df["Date"] < current_date]
                if prev_row.empty:
                    continue
                prev_row = prev_row.iloc[-1]

                # シグナル判定
                if prev_row["combined_signal"] == 1 and pd.notna(prev_row["ATR14"]):
                    entry_price = row.iloc[0]["Open"]
                    atr = prev_row["ATR14"]
                    shares, _ = calc_position_size(capital / MAX_POSITIONS, atr, entry_price)

                    if shares > 0:
                        positions[code4] = {
                            "entry_date": current_date,
                            "entry_price": entry_price,
                            "shares": shares,
                            "stop_loss": entry_price - atr * 2,
                            "take_profit": entry_price + atr * 2.5,
                        }
                        capital -= entry_price * shares

        # 日次評価
        pos_value = 0
        for code4, pos in positions.items():
            df = all_signals[code4]
            row = df[df["Date"] == current_date]
            if not row.empty:
                pos_value += pos["shares"] * row.iloc[0]["Close"]

        total_equity = capital + pos_value
        equity_curve.append(total_equity)
        dates.append(current_date)

    # 結果集計
    total_trades = len(trades)
    wins = [t for t in trades if t["pnl"] > 0]
    win_rate = len(wins) / total_trades * 100 if total_trades > 0 else 0
    avg_pnl = sum(t["pnl"] for t in trades) / total_trades if total_trades > 0 else 0
    final_pnl = equity_curve[-1] - INITIAL_CAPITAL

    eq_s = pd.Series(equity_curve)
    max_dd = ((eq_s - eq_s.cummax()) / eq_s.cummax() * 100).min()

    return {
        "trades": trades,
        "equity_curve": equity_curve,
        "dates": dates,
        "total_trades": total_trades,
        "win_rate": win_rate,
        "avg_pnl": avg_pnl,
        "final_pnl": final_pnl,
        "final_capital": equity_curve[-1],
        "max_dd": max_dd,
    }


def main():
    print("=" * 60)
    print("  複合運用シミュレーション（戦略A・C・D統合）")
    print(f"  期間: {BACKTEST_START} ～ {BACKTEST_END}")
    print("=" * 60)

    # 候補銘柄読み込み
    codes = load_candidate_codes()
    print(f"\n対象銘柄: {len(codes)} (A・C・D重複排除後)")

    # データ取得
    price_data = fetch_candidate_data(codes)
    print(f"バックテスト対象: {len(price_data)} 銘柄\n")

    if not price_data:
        print("エラー: データを取得できません")
        return

    # バックテスト実行
    print("複合運用シミュレーション実行中...")
    result = run_combined_backtest(price_data)

    # 結果表示
    print("\n" + "=" * 60)
    print("  結果サマリー")
    print("=" * 60)
    print(f"総取引数: {result['total_trades']}")
    print(f"勝率: {result['win_rate']:.1f}%")
    print(f"平均益: {result['avg_pnl']:.0f}円")
    print(f"最終損益: {result['final_pnl']:+.0f}円")
    print(f"最終資産: {result['final_capital']:,.0f}円")
    print(f"最大DD: {result['max_dd']:.1f}%")
    print(f"年利: {(result['final_pnl'] / INITIAL_CAPITAL / 2 * 100):.1f}%")

    # 上位10取引
    if result['trades']:
        trades_df = pd.DataFrame(result['trades']).sort_values('pnl', ascending=False)
        print("\n上位10取引:")
        print(trades_df[['code', 'entry_date', 'exit_date', 'pnl', 'reason']].head(10).to_string(index=False))

    # グラフ保存
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))

    # 損益曲線
    ax1.plot(result['dates'], result['equity_curve'], linewidth=1.5, color='blue')
    ax1.axhline(INITIAL_CAPITAL, color='red', linestyle='--', alpha=0.5, label='初期資本')
    ax1.set_title('複合運用シミュレーション 損益曲線', fontsize=12)
    ax1.set_ylabel('資産（円）')
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x/1e6):.1f}M'))
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # ドローダウン
    eq_s = pd.Series(result['equity_curve'])
    dd = (eq_s - eq_s.cummax()) / eq_s.cummax() * 100
    ax2.fill_between(range(len(dd)), dd, 0, color='red', alpha=0.3)
    ax2.set_title('ドローダウン', fontsize=12)
    ax2.set_xlabel('日付')
    ax2.set_ylabel('ドローダウン（%）')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    Path("logs").mkdir(parents=True, exist_ok=True)
    plt.savefig("logs/portfolio_combined.png", dpi=150)
    plt.close()
    print("\nグラフを保存: logs/portfolio_combined.png")


if __name__ == "__main__":
    main()
