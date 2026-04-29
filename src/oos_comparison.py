"""
src/oos_comparison.py
OOSテスト: 東証プライム全銘柄 vs 最終30銘柄の選定バイアス検証

- OOS期間: 2023-01-01 ~ 2026-04-24
- 指標計算: 全データ(2016-04-)を使用（ウォームアップ確保）
- バックテスト: 個別株ベース（初期資金 1,000,000円/銘柄）
"""
import sys
import os
sys.stdout.reconfigure(encoding="utf-8")

from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from indicators import add_indicators
from strategy import generate_signals
from utils.risk import calc_position_size

# ── 定数 ──────────────────────────────────────────────────────
OOS_START       = pd.Timestamp("2023-01-01")
OOS_END         = pd.Timestamp("2026-04-24")
INITIAL_CAPITAL = 1_000_000
ATR_TP_MULT     = 3
ATR_STOP_MULT   = 2
MAX_HOLD_DAYS   = 10
YEARS           = (OOS_END - OOS_START).days / 365   # ≈3.32年
MIN_TRADES      = 3   # 最低取引数（統計的意味のある閾値）

ALL_PRIME_PARQUET  = Path("data/raw/prices_all_prime.parquet")
FINAL_30_CSV       = Path("logs/final_candidates.csv")


def run_stock_backtest(code4: str, df: pd.DataFrame) -> dict | None:
    """
    単一銘柄のOOSバックテストを実行し、統計を返す。
    指標計算は全期間データで行い、取引はOOS_START以降のみ実施。
    """
    if len(df) < 260:
        return None

    df = add_indicators(df)
    df = generate_signals(df)
    df = df.sort_values("Date").reset_index(drop=True)

    capital   = float(INITIAL_CAPITAL)
    in_trade  = False
    trades    = []
    equity    = [capital]

    for i in range(1, len(df)):
        row  = df.iloc[i]
        prev = df.iloc[i - 1]

        if row["Date"] > OOS_END:
            break
        if row["Date"] < OOS_START:
            continue

        # ── 保有中ポジション決済判定 ──
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
                pnl     = (exit_price - entry_price) * shares
                capital += pnl
                trades.append({"pnl": pnl, "win": pnl > 0})
                equity.append(capital)
                in_trade = False

        # ── 新規エントリー判定（前日シグナル→本日始値）──
        if (not in_trade
                and prev.get("signal", 0) == 1
                and pd.notna(prev.get("ATR14"))):
            entry_price = row["Open"]
            if not pd.notna(entry_price) or entry_price <= 0:
                continue
            gap_pct = (entry_price - prev["Close"]) / prev["Close"]
            if gap_pct < -0.015:
                continue
            atr = float(prev["ATR14"])
            shares, stop_loss = calc_position_size(capital, atr, entry_price)
            take_profit = entry_price + atr * ATR_TP_MULT

            if shares > 0 and entry_price * shares <= capital:
                in_trade  = True
                hold_days = 0

    total = len(trades)
    if total == 0:
        return {"code4": code4, "total": 0, "win_rate": 0.0,
                "annual_ret": 0.0, "max_dd": 0.0}

    wins     = sum(1 for t in trades if t["win"])
    win_rate = wins / total * 100

    final_cap  = capital
    annual_ret = ((final_cap / INITIAL_CAPITAL) ** (1 / YEARS) - 1) * 100

    eq_s   = pd.Series(equity)
    max_dd = ((eq_s - eq_s.cummax()) / eq_s.cummax() * 100).min()

    return {
        "code4": code4, "total": total, "win_rate": win_rate,
        "annual_ret": annual_ret, "max_dd": max_dd,
        "final_pnl": final_cap - INITIAL_CAPITAL,
    }


def main() -> None:
    print("=" * 64)
    print("  OOSテスト: 東証プライム全銘柄 vs 最終30銘柄")
    print(f"  OOS期間: {OOS_START.date()} ～ {OOS_END.date()} ({YEARS:.2f}年)")
    print("=" * 64)

    # ── データ読み込み ──
    print("\nprices_all_prime.parquet 読み込み中...")
    raw = pd.read_parquet(ALL_PRIME_PARQUET)
    raw["Date"] = pd.to_datetime(raw["Date"])

    # 不要な生列を除去（調整済列 Open/High/Low/Close/Volume を使用）
    drop_cols = [c for c in ["Code", "O", "H", "L", "C", "Vo", "Va",
                              "UL", "LL", "AdjFactor"]
                 if c in raw.columns]
    raw = raw.drop(columns=drop_cols)
    raw = raw.rename(columns={"Code4": "Code4"})   # 念のため

    all_codes = sorted(raw["Code4"].unique())
    print(f"  総銘柄数: {len(all_codes)}")

    # 30銘柄コード
    final_30 = pd.read_csv(FINAL_30_CSV, encoding="utf-8-sig")
    codes_30 = set(final_30["code"].astype(str).str.zfill(4))
    print(f"  最終30銘柄: {len(codes_30)}")

    # ── 全銘柄バックテスト ──
    print(f"\n全銘柄OOSバックテスト実行中...")
    results = []
    done    = 0
    total   = len(all_codes)

    grouped = raw.groupby("Code4")

    for code4 in all_codes:
        done += 1
        try:
            df = grouped.get_group(code4).copy()
        except KeyError:
            continue

        res = run_stock_backtest(code4, df)
        if res is not None:
            results.append(res)

        if done % 200 == 0 or done == total:
            print(f"  [{done:4d}/{total}] 完了")

    df_res = pd.DataFrame(results)
    print(f"\n結果取得: {len(df_res)} 銘柄")

    # ── 最低取引数フィルター ──
    df_valid = df_res[df_res["total"] >= MIN_TRADES].copy()
    df_30    = df_res[df_res["code4"].isin(codes_30) & (df_res["total"] >= MIN_TRADES)].copy()

    print(f"  有効銘柄（取引{MIN_TRADES}回以上）: {len(df_valid)} 銘柄")
    print(f"  うち30銘柄中の有効: {len(df_30)} 銘柄")

    # ── 統計比較 ──
    def med(s):
        return s.median()

    def pct(s, p):
        return s.quantile(p / 100)

    print(f"\n{'=' * 64}")
    print(f"  OOSテスト結果比較（中央値）")
    print(f"  期間: {OOS_START.date()} ～ {OOS_END.date()}")
    print(f"{'=' * 64}")
    print(f"  {'指標':<18}  {'全銘柄universe':>14}  {'最終30銘柄':>12}  {'差':>8}")
    print(f"  {'-'*60}")

    metrics = [
        ("年利換算(%)",  "annual_ret"),
        ("勝率(%)",      "win_rate"),
        ("最大DD(%)",    "max_dd"),
        ("取引回数",     "total"),
    ]
    diffs = {}
    for label, col in metrics:
        u_med = med(df_valid[col])
        s_med = med(df_30[col])
        diff  = s_med - u_med
        diffs[label] = diff
        print(f"  {label:<18}  {u_med:>14.2f}  {s_med:>12.2f}  {diff:>+8.2f}")

    print(f"{'=' * 64}")

    # ── 分布比較（パーセンタイル）──
    print(f"\n【年利換算(%) 分布】")
    print(f"  {'パーセンタイル':<16}  {'全銘柄':>10}  {'30銘柄':>10}")
    for p in [10, 25, 50, 75, 90]:
        u = pct(df_valid["annual_ret"], p)
        s = pct(df_30["annual_ret"],   p) if len(df_30) > 0 else float("nan")
        print(f"  {p:3d}th              {u:>10.2f}  {s:>10.2f}")

    # ── バイアス定量化 ──
    print(f"\n【バイアス定量化】")
    u_pos = (df_valid["annual_ret"] > 0).mean() * 100
    s_pos = (df_30["annual_ret"]    > 0).mean() * 100 if len(df_30) > 0 else float("nan")
    print(f"  全銘柄 年利プラス比率: {u_pos:.1f}%")
    print(f"  30銘柄 年利プラス比率: {s_pos:.1f}%")

    u_above5 = (df_valid["annual_ret"] > 5).mean() * 100
    s_above5 = (df_30["annual_ret"]    > 5).mean() * 100 if len(df_30) > 0 else float("nan")
    print(f"  全銘柄 年利5%超比率:   {u_above5:.1f}%")
    print(f"  30銘柄 年利5%超比率:   {s_above5:.1f}%")

    # ── 30銘柄個別結果 ──
    print(f"\n【最終30銘柄 OOS個別結果】")
    print(f"  {'コード':<6}  {'取引':>4}  {'勝率':>7}  {'年利':>8}  {'最大DD':>8}")
    print(f"  {'-' * 42}")
    df_30_sorted = df_30.sort_values("annual_ret", ascending=False)
    for _, r in df_30_sorted.iterrows():
        print(f"  {r['code4']:<6}  {int(r['total']):>4}回  {r['win_rate']:>6.1f}%  "
              f"{r['annual_ret']:>+7.2f}%  {r['max_dd']:>7.2f}%")

    no_data = codes_30 - set(df_30["code4"])
    if no_data:
        print(f"\n  ※ 取引{MIN_TRADES}回未満 or データなし: {sorted(no_data)}")

    # 結果CSV保存
    out = Path("logs/oos_comparison_result.csv")
    df_res.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"\n結果保存: {out}")


if __name__ == "__main__":
    main()
