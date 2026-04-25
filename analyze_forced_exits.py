"""
analyze_forced_exits.py
=======================
「保有10日強制終了」ルールの見直し検討用分析スクリプト
※ oos_backtest.py の実装に完全準拠

【GitHubコード確認済み仕様】
  - 終了理由カラム : reason（"損切り" / "利確" / "期間満了"）
  - 指標ライブラリ : pandas_ta
  - 指標カラム名   : SMA20, SMA50, SMA200, RSI14, ATR14, VOL_MA20
  - ADX           : pandas_ta.adx() → ADX_14カラム
  - TP倍率        : ATR_TP_MULT = 4（設計書確定版）
  - SL倍率        : ATR_STOP_MULT = 2
  - 最大損切り     : -5%

使い方：
    python analyze_forced_exits.py

前提：
    - data/raw/prices_10y.parquet が存在すること
    - logs/final_candidates.csv が存在すること
    - pip install pandas numpy matplotlib pandas_ta pyarrow

出力：
    - コンソール：強制終了トレードの統計・保有上限別比較表・判定
    - logs/forced_exit_analysis.png：グラフ4枚
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pandas_ta as ta
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")

# 日本語フォント（Windows）
plt.rcParams["font.family"] = "MS Gothic"
plt.rcParams["axes.unicode_minus"] = False

# ─────────────────────────────────────────
# 設定（oos_backtest.py と統一）
# ─────────────────────────────────────────
OOS_START   = pd.Timestamp("2023-01-01")
OOS_END     = pd.Timestamp("2026-04-24")
OOS_WARMUP  = pd.Timestamp("2022-01-01")   # 指標ウォームアップ用

MAX_POSITIONS    = 5
ATR_TP_MULT      = 4             # 設計書確定版（×4）
ATR_STOP_MULT    = 2
MAX_HOLD_DAYS    = 10            # 現行ルール
MAX_SL_PCT       = 0.05          # 最大損切り -5%

# 比較する保有上限の候補
ANALYSIS_DAYS = [5, 7, 10, 12, 15, 20]

DATA_PATH       = Path("data/raw/prices_10y.parquet")
CANDIDATES_PATH = Path("logs/final_candidates.csv")
OUTPUT_PATH     = Path("logs/forced_exit_analysis.png")


# ─────────────────────────────────────────
# 指標計算（indicators.py と同一）
# ─────────────────────────────────────────
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["SMA20"]    = ta.sma(df["Close"], length=20)
    df["SMA50"]    = ta.sma(df["Close"], length=50)
    df["SMA200"]   = ta.sma(df["Close"], length=200)
    df["RSI14"]    = ta.rsi(df["Close"], length=14)
    df["ATR14"]    = ta.atr(df["High"], df["Low"], df["Close"], length=14)
    df["VOL_MA20"] = ta.sma(df["Volume"], length=20)
    adx_df = ta.adx(df["High"], df["Low"], df["Close"], length=14)
    if adx_df is not None and "ADX_14" in adx_df.columns:
        df["ADX14"] = adx_df["ADX_14"]
    else:
        df["ADX14"] = np.nan
    return df


# ─────────────────────────────────────────
# シグナル生成（strategy.py の条件を再現）
# ─────────────────────────────────────────
def generate_signals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    c = df["Close"]
    cond = (
        df["SMA20"].gt(df["SMA50"]) &
        df["RSI14"].between(45, 75) &
        df["Volume"].gt(df["VOL_MA20"] * 1.2) &
        c.gt(df["SMA200"]) &
        df["RSI14"].gt(df["RSI14"].shift(3)) &
        (c / df["Open"] - 1).ge(-0.015) &
        c.gt(df["Open"]) &
        df["ADX14"].gt(15)
    )
    df["signal"] = cond.astype(int)
    return df


# ─────────────────────────────────────────
# バックテスト本体（max_hold を引数で変更可能）
# ─────────────────────────────────────────
def run_backtest(oos_data: dict, dates: list, max_hold: int) -> pd.DataFrame:
    """
    oos_backtest.py の構造に準拠したポートフォリオバックテスト。
    oos_data: {ticker: DataFrame（Date indexed, 指標・シグナル計算済み）}
    dates: OOS期間の日付リスト
    戻り値: トレード一覧 DataFrame
    """
    trades = []
    positions = {}  # {ticker: {entry_date, entry_price, take_profit, stop_loss, hold_days}}

    for today in dates:
        # ── エグジット ──────────────────────────
        closed = []
        for ticker, pos in positions.items():
            df = oos_data.get(ticker)
            if df is None or today not in df.index:
                continue
            row = df.loc[today]
            pos["hold_days"] += 1
            hi, lo, cl = row["High"], row["Low"], row["Close"]

            exit_price = exit_reason = None
            if lo <= pos["stop_loss"]:
                exit_price, exit_reason = pos["stop_loss"], "損切り"
            elif hi >= pos["take_profit"]:
                exit_price, exit_reason = pos["take_profit"], "利確"
            elif pos["hold_days"] >= max_hold:
                exit_price, exit_reason = cl, "期間満了"

            if exit_price is not None:
                pnl_pct = (exit_price / pos["entry_price"] - 1) * 100
                trades.append({
                    "ticker":      ticker,
                    "entry_date":  pos["entry_date"],
                    "exit_date":   today,
                    "hold_days":   pos["hold_days"],
                    "entry_price": pos["entry_price"],
                    "exit_price":  exit_price,
                    "pnl_pct":     pnl_pct,
                    "reason":      exit_reason,
                })
                closed.append(ticker)

        for t in closed:
            del positions[t]

        # ── エントリー ──────────────────────────
        if len(positions) >= MAX_POSITIONS:
            continue

        candidates = []
        for ticker, df in oos_data.items():
            if ticker in positions:
                continue
            if today not in df.index:
                continue
            prev_dates = df.index[df.index < today]
            if len(prev_dates) == 0:
                continue
            prev = df.loc[prev_dates[-1]]
            if prev.get("signal", 0) != 1:
                continue
            if pd.isna(prev.get("ATR14")):
                continue
            candidates.append((ticker, float(prev["RSI14"]), prev))

        candidates.sort(key=lambda x: x[1], reverse=True)
        slots = MAX_POSITIONS - len(positions)
        for ticker, _, prev in candidates[:slots]:
            df = oos_data[ticker]
            ep  = float(df.loc[today]["Open"]) if today in df.index else float(prev["Close"])
            atr = float(prev["ATR14"])
            tp  = ep + atr * ATR_TP_MULT
            sl  = ep - atr * ATR_STOP_MULT
            sl  = max(sl, ep * (1 - MAX_SL_PCT))
            positions[ticker] = {
                "entry_date":  today,
                "entry_price": ep,
                "take_profit": tp,
                "stop_loss":   sl,
                "hold_days":   0,
            }

    return pd.DataFrame(trades)


# ─────────────────────────────────────────
# メイン
# ─────────────────────────────────────────
def main():
    print("=" * 60)
    print("強制終了ルール見直し分析")
    print(f"OOS期間: {OOS_START.date()} ～ {OOS_END.date()}")
    print("=" * 60)

    # ── データ読み込み ────────────────────
    print("\n[1/4] データ読み込み・指標計算中...")
    if not DATA_PATH.exists():
        print(f"❌ {DATA_PATH} が見つかりません。")
        return
    if not CANDIDATES_PATH.exists():
        print(f"❌ {CANDIDATES_PATH} が見つかりません。")
        return

    all_prices = pd.read_parquet(DATA_PATH)
    all_prices["Date"] = pd.to_datetime(all_prices["Date"])

    cands = pd.read_csv(CANDIDATES_PATH)
    tickers = cands["code"].astype(str).str.zfill(4).tolist()
    available = set(all_prices["Code4"].astype(str).str.zfill(4).unique())
    tickers = [t for t in tickers if t in available]
    print(f"   対象銘柄数: {len(tickers)}")

    # 指標計算（全銘柄・ウォームアップ含む）
    oos_data = {}
    for i, ticker in enumerate(tickers, 1):
        if i % 10 == 0:
            print(f"   指標計算中... {i}/{len(tickers)}", end="\r")
        df = all_prices[
            (all_prices["Code4"].astype(str).str.zfill(4) == ticker) &
            (all_prices["Date"] >= OOS_WARMUP)
        ].copy().sort_values("Date")
        if len(df) < 60:
            continue
        df = add_indicators(df)
        df = generate_signals(df)
        df = df.set_index("Date")
        oos_data[ticker] = df
    print(f"   指標計算完了: {len(oos_data)}銘柄          ")

    # OOS期間の日付リスト
    dates = sorted(
        all_prices[
            (all_prices["Date"] >= OOS_START) &
            (all_prices["Date"] <= OOS_END)
        ]["Date"].unique()
    )
    print(f"   OOS取引日数: {len(dates)}日")

    # ── 現行（10日）でバックテスト ────────
    print(f"\n[2/4] 現行ルール（{MAX_HOLD_DAYS}日）でバックテスト実行中...")
    trades_base = run_backtest(oos_data, dates, max_hold=MAX_HOLD_DAYS)

    if trades_base.empty:
        print("❌ トレードが0件でした。データや設定を確認してください。")
        return

    forced = trades_base[trades_base["reason"] == "期間満了"]
    tp_tr  = trades_base[trades_base["reason"] == "利確"]
    sl_tr  = trades_base[trades_base["reason"] == "損切り"]

    print(f"\n{'─'*54}")
    print(f"  総トレード数:          {len(trades_base):>5} 件")
    print(f"  利確:                  {len(tp_tr):>5} 件  ({len(tp_tr)/len(trades_base)*100:.1f}%)")
    print(f"  損切り:                {len(sl_tr):>5} 件  ({len(sl_tr)/len(trades_base)*100:.1f}%)")
    print(f"  強制終了（期間満了）:  {len(forced):>5} 件  ({len(forced)/len(trades_base)*100:.1f}%)")
    print(f"{'─'*54}")

    if len(forced) > 0:
        print(f"\n■ 強制終了トレードの損益分布")
        print(f"  平均損益:     {forced['pnl_pct'].mean():+.2f}%")
        print(f"  中央値:       {forced['pnl_pct'].median():+.2f}%")
        print(f"  最大（利益）: {forced['pnl_pct'].max():+.2f}%")
        print(f"  最小（損失）: {forced['pnl_pct'].min():+.2f}%")
        print(f"  プラス件数:   {(forced['pnl_pct'] > 0).sum()} 件 ({(forced['pnl_pct'] > 0).mean()*100:.1f}%)")
        print(f"  マイナス件数: {(forced['pnl_pct'] < 0).sum()} 件 ({(forced['pnl_pct'] < 0).mean()*100:.1f}%)")
        print(f"\n  参考 → 利確平均: {tp_tr['pnl_pct'].mean():+.2f}%  "
              f"/ 損切り平均: {sl_tr['pnl_pct'].mean():+.2f}%")
    else:
        print("\n  強制終了トレードが0件 → 10日ルールの見直しは不要です。")

    # ── 複数の保有上限で比較 ──────────────
    print("\n[3/4] 保有上限を変えて比較中...")
    results = []
    for max_hold in ANALYSIS_DAYS:
        print(f"   {max_hold}日...", end=" ", flush=True)
        tr = run_backtest(oos_data, dates, max_hold=max_hold)
        if tr.empty:
            print("スキップ（0件）")
            continue
        results.append({
            "max_hold":   max_hold,
            "trades":     len(tr),
            "winrate":    (tr["pnl_pct"] > 0).mean() * 100,
            "avg_pnl":    tr["pnl_pct"].mean(),
            "forced_pct": (tr["reason"] == "期間満了").mean() * 100,
            "total_ret":  tr["pnl_pct"].sum(),
        })
        print(f"完了（{len(tr)}件）")

    df_res = pd.DataFrame(results)

    print(f"\n{'─'*68}")
    print(f"{'上限':>4} | {'件数':>5} | {'勝率':>6} | {'平均損益':>8} | {'強制%':>6} | {'累計損益%':>9}")
    print(f"{'─'*68}")
    for _, r in df_res.iterrows():
        marker = " ◀ 現行" if r["max_hold"] == MAX_HOLD_DAYS else ""
        print(f"{int(r['max_hold']):>3}日 | {int(r['trades']):>5} | "
              f"{r['winrate']:>5.1f}% | {r['avg_pnl']:>+7.2f}% | "
              f"{r['forced_pct']:>5.1f}% | {r['total_ret']:>+8.1f}%{marker}")
    print(f"{'─'*68}")

    # ── グラフ ────────────────────────────
    print("\n[4/4] グラフ出力中...")
    fig = plt.figure(figsize=(14, 9))
    fig.suptitle(f"強制終了ルール見直し分析（OOS: {OOS_START.date()}〜{OOS_END.date()}）",
                 fontsize=14, fontweight="bold")
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

    ax1 = fig.add_subplot(gs[0, 0])
    if len(forced) > 0:
        ax1.hist(forced["pnl_pct"], bins=20, color="#4A90D9", edgecolor="white", alpha=0.85)
        ax1.axvline(0, color="red", linestyle="--", linewidth=1.2, label="損益ゼロ")
        ax1.axvline(forced["pnl_pct"].mean(), color="orange", linestyle="-",
                    linewidth=1.5, label=f"平均 {forced['pnl_pct'].mean():+.2f}%")
        ax1.legend(fontsize=8)
    else:
        ax1.text(0.5, 0.5, "強制終了トレードなし", ha="center", va="center",
                 transform=ax1.transAxes, fontsize=12)
    ax1.set_title(f"強制終了トレードの損益分布（{MAX_HOLD_DAYS}日）", fontsize=10)
    ax1.set_xlabel("損益（%）")
    ax1.set_ylabel("件数")

    ax2 = fig.add_subplot(gs[0, 1])
    hold_pnl = trades_base.groupby("hold_days")["pnl_pct"].mean()
    colors = ["#4A90D9" if v >= 0 else "#E05A5A" for v in hold_pnl.values]
    ax2.bar(hold_pnl.index, hold_pnl.values, color=colors, edgecolor="white", alpha=0.85)
    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.set_title("保有日数別・平均損益（全トレード）", fontsize=10)
    ax2.set_xlabel("保有日数")
    ax2.set_ylabel("平均損益（%）")

    ax3 = fig.add_subplot(gs[1, 0])
    bar_colors = ["#E8A838" if r["max_hold"] == MAX_HOLD_DAYS else "#4A90D9"
                  for _, r in df_res.iterrows()]
    bars = ax3.bar(df_res["max_hold"].astype(str) + "日",
                   df_res["total_ret"], color=bar_colors, edgecolor="white", alpha=0.9)
    ax3.axhline(0, color="black", linewidth=0.8)
    ax3.set_title("保有上限別・累計損益%（OOS期間合計）", fontsize=10)
    ax3.set_xlabel("保有上限日数")
    ax3.set_ylabel("累計損益（%、概算）")
    max_bar = df_res["total_ret"].abs().max()
    for bar, (_, r) in zip(bars, df_res.iterrows()):
        if r["max_hold"] == MAX_HOLD_DAYS:
            ax3.text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + max_bar * 0.03,
                     "現行", ha="center", fontsize=8, color="darkorange")

    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(df_res["max_hold"], df_res["winrate"],
             marker="o", color="#4A90D9", linewidth=2, markersize=7)
    ax4.axvline(MAX_HOLD_DAYS, color="orange", linestyle="--",
                linewidth=1.5, label=f"現行（{MAX_HOLD_DAYS}日）")
    ax4.axhline(55, color="green", linestyle=":", linewidth=1.2, label="目標勝率55%")
    ax4.set_title("保有上限別・勝率", fontsize=10)
    ax4.set_xlabel("保有上限日数")
    ax4.set_ylabel("勝率（%）")
    ax4.legend(fontsize=8)

    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   → {OUTPUT_PATH} に保存しました")

    # ── 判定サマリー ─────────────────────
    print(f"\n{'='*60}")
    print("■ 判定サマリー")
    print(f"{'='*60}")

    best    = df_res.loc[df_res["total_ret"].idxmax()]
    current = df_res[df_res["max_hold"] == MAX_HOLD_DAYS].iloc[0]
    diff    = best["total_ret"] - current["total_ret"]

    print(f"  現行（{MAX_HOLD_DAYS}日）累計損益: {current['total_ret']:+.1f}%")
    print(f"  最良（{int(best['max_hold'])}日）累計損益: {best['total_ret']:+.1f}%")
    print(f"  差分:               {diff:+.1f}%")

    if diff > 3.0 and best["max_hold"] != MAX_HOLD_DAYS:
        print(f"\n  → 【見直し推奨】{int(best['max_hold'])}日への変更で")
        print(f"     累計損益が {diff:+.1f}% 改善する可能性があります。")
        print(f"     Phase 4 ペーパートレードで検証後に正式採用してください。")
    elif diff > 1.0 and best["max_hold"] != MAX_HOLD_DAYS:
        print(f"\n  → 【要検討】{int(best['max_hold'])}日への変更でわずかな")
        print(f"     改善（{diff:+.1f}%）の可能性がありますが統計的有意性が低いため、")
        print(f"     現行維持が安全です。")
    else:
        print(f"\n  → 【現行維持】10日ルールを変えても有意な改善はありません。")
        print(f"     現行のまま Phase 4 へ進むことを推奨します。")

    print(f"\n  ※ 概算バックテストです（スリッページ・税金無視）。")
    print(f"     精密な検証は portfolio_backtest.py で実施してください。")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
