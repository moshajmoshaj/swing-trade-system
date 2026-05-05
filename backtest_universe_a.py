"""
ユニバース型 戦略A バックテスト
目的：IS期間の銘柄固定選定をやめ、全プライム銘柄に戦略A条件を
      毎日動的スキャンした場合のOOS成績を検証する

比較対象：
  固定選定 A v2  : CAGR +8.86%  取引262回  勝率53.8%  MaxDD-10.15%
  ユニバース型 A : 今回検証

OOS期間  : 2023-01-01 ～ 2026-04-24
資金     : 100万円・最大5ポジション・1銘柄上限20万円
"""
import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd

# ── 期間設定 ──────────────────────────────────────────────────
OOS_START = pd.Timestamp("2023-01-01")
OOS_END   = pd.Timestamp("2026-04-24")
WARMUP    = pd.Timestamp("2022-01-01")

# ── 戦略A パラメータ ──────────────────────────────────────────
ATR_TP_MULT   = 6
ATR_SL_MULT   = 2
MAX_HOLD_DAYS = 10
ADX_THRESHOLD = 15

# ── ポートフォリオ設定 ────────────────────────────────────────
INITIAL_CAPITAL = 1_000_000
MAX_POSITIONS   = 5
MAX_POS_RATIO   = 0.20
COMMISSION      = 0.00055
SLIPPAGE        = 0.00050
COST_PER_LEG    = COMMISSION + SLIPPAGE

DATA_PATH = Path("data/raw/prices_20y.parquet")


# ── 全銘柄一括指標計算（ベクトル化）────────────────────────────
def compute_indicators_vectorized(df: pd.DataFrame) -> pd.DataFrame:
    """groupby.transform で全銘柄の指標を一括計算"""
    print("  [1/3] 指標計算中（全銘柄一括）...")
    t0 = time.time()
    df = df.reset_index(drop=True)
    g  = df.groupby("Code", sort=False)

    df["SMA20"]    = g["Close"].transform(lambda x: x.rolling(20).mean())
    df["SMA50"]    = g["Close"].transform(lambda x: x.rolling(50).mean())
    df["SMA200"]   = g["Close"].transform(lambda x: x.rolling(200).mean())
    df["VOL_MA20"] = g["Volume"].transform(lambda x: x.rolling(20).mean())

    delta = g["Close"].diff()
    gain  = delta.clip(lower=0).groupby(df["Code"]).transform(
            lambda x: x.ewm(alpha=1/14, adjust=False).mean())
    loss  = (-delta).clip(lower=0).groupby(df["Code"]).transform(
            lambda x: x.ewm(alpha=1/14, adjust=False).mean())
    df["RSI14"] = 100 - 100 / (1 + gain / loss.replace(0, 1e-9))

    prev_c = g["Close"].shift(1)
    tr = pd.concat([df["High"]-df["Low"],
                    (df["High"]-prev_c).abs(),
                    (df["Low"] -prev_c).abs()], axis=1).max(axis=1)
    df["ATR14"] = tr.groupby(df["Code"]).transform(
                  lambda x: x.ewm(alpha=1/14, adjust=False).mean())

    ph = g["High"].shift(1); pl = g["Low"].shift(1)
    dm_p = np.where((df["High"]-ph > 0) & (df["High"]-ph > pl-df["Low"]),
                    df["High"]-ph, 0.0)
    dm_m = np.where((pl-df["Low"]  > 0) & (pl-df["Low"] > df["High"]-ph),
                    pl-df["Low"], 0.0)
    atr_s = df["ATR14"].replace(0, np.nan)
    di_p  = 100 * pd.Series(dm_p, index=df.index).groupby(df["Code"]).transform(
            lambda x: x.ewm(alpha=1/14, adjust=False).mean()) / atr_s
    di_m  = 100 * pd.Series(dm_m, index=df.index).groupby(df["Code"]).transform(
            lambda x: x.ewm(alpha=1/14, adjust=False).mean()) / atr_s
    dx    = 100*(di_p-di_m).abs()/(di_p+di_m).replace(0, np.nan)
    df["ADX14"] = dx.groupby(df["Code"]).transform(
                  lambda x: x.ewm(alpha=1/14, adjust=False).mean())

    print(f"      完了: {time.time()-t0:.1f}秒")
    return df


# ── 戦略Aシグナル一括生成 ─────────────────────────────────────
def compute_signals_vectorized(df: pd.DataFrame) -> pd.DataFrame:
    print("  [2/3] シグナル生成中...")
    t0 = time.time()
    g = df.groupby("Code", sort=False)
    rsi3 = g["RSI14"].shift(3)
    df["signal"] = (
        (df["SMA20"]  > df["SMA50"]) &
        (df["Close"]  > df["SMA200"]) &
        (df["RSI14"] >= 45) & (df["RSI14"] <= 75) &
        (df["RSI14"]  > rsi3) &
        (df["Volume"] >= df["VOL_MA20"] * 1.2) &
        (df["Close"]  > df["Open"]) &
        (df["ADX14"]  > ADX_THRESHOLD)
    ).astype(int)
    total_sigs = df[df["Date"] >= OOS_START]["signal"].sum()
    print(f"      完了: {time.time()-t0:.1f}秒  OOS期間シグナル総数: {int(total_sigs):,}")
    return df


# ── ポートフォリオシミュレーション ───────────────────────────
@dataclass
class Position:
    code:        str
    entry_date:  pd.Timestamp
    entry_price: float
    shares:      int
    stop_loss:   float
    take_profit: float
    hold_days:   int = 0
    rsi:         float = 0.0

@dataclass
class Trade:
    code:        str
    entry_date:  pd.Timestamp
    exit_date:   pd.Timestamp
    entry_price: float
    exit_price:  float
    shares:      int
    pnl:         float
    reason:      str


def run_portfolio(df: pd.DataFrame) -> tuple:
    print("  [3/3] ポートフォリオシミュレーション中...")
    t0 = time.time()

    # OOS期間のデータのみ（ウォームアップ含む全データから指標計算済み）
    df_oos = df[df["Date"] >= OOS_START].copy()

    # 日付リスト
    all_dates = sorted(df_oos["Date"].unique())

    # 高速ルックアップ: code → {date → row_dict}
    # シグナルを持つ銘柄のみに絞る
    sig_codes = set(df_oos[df_oos["signal"] == 1]["Code"].unique())
    all_codes = set(df_oos["Code"].unique())
    print(f"      OOS期間シグナル発生銘柄: {len(sig_codes):,} / {len(all_codes):,} 銘柄")

    # ルックアップテーブル構築（シグナル銘柄 + 前日情報が必要な銘柄）
    # 前日シグナルの翌日エントリーのため、ウォームアップ含む全期間が必要
    lookup: dict = {}
    for code, grp in df.groupby("Code"):
        lookup[code] = grp.drop_duplicates("Date").set_index("Date").to_dict("index")

    capital   = float(INITIAL_CAPITAL)
    positions: list[Position] = []
    trades:    list[Trade]    = []
    equity    = {}

    cur_month = None; month_start = capital; month_stopped = False; stop_cnt = 0

    for today in all_dates:
        ym = (today.year, today.month)
        if ym != cur_month:
            cur_month = ym; month_start = capital; month_stopped = False

        # 決済判定
        next_pos = []
        for pos in positions:
            row = lookup.get(pos.code, {}).get(today)
            if row is None:
                next_pos.append(pos); continue
            pos.hold_days += 1
            hi, lo, cl = row["High"], row["Low"], row["Close"]
            ep = er = None
            if lo <= pos.stop_loss:
                ep, er = pos.stop_loss, "損切り"
            elif hi >= pos.take_profit:
                ep, er = pos.take_profit, "利確"
            elif pos.hold_days >= MAX_HOLD_DAYS:
                ep, er = cl, "期間満了"
            if ep is not None:
                cost = (pos.entry_price + ep) * pos.shares * COST_PER_LEG
                pnl  = (ep - pos.entry_price) * pos.shares - cost
                capital += pnl
                trades.append(Trade(pos.code, pos.entry_date, today,
                                    pos.entry_price, ep, pos.shares, pnl, er))
            else:
                next_pos.append(pos)
        positions = next_pos

        if not month_stopped and month_start > 0:
            if (capital - month_start) / month_start * 100 <= -10.0:
                month_stopped = True; stop_cnt += 1

        # エントリー
        slots = MAX_POSITIONS - len(positions)
        if slots > 0 and not month_stopped:
            holding = {p.code for p in positions}

            # 前日シグナル銘柄を取得（高速化：シグナル銘柄のみ検索）
            prev_dates_map: dict = {}
            for code in sig_codes:
                code_rows = lookup.get(code, {})
                prev_dates = [d for d in code_rows if d < today]
                if not prev_dates:
                    continue
                prev_date = max(prev_dates)
                prev = code_rows[prev_date]
                if prev.get("signal", 0) != 1:
                    continue
                atr = prev.get("ATR14", 0)
                if not atr or pd.isna(atr):
                    continue
                today_row = code_rows.get(today)
                if today_row is None:
                    continue
                ep  = today_row["Open"]
                if ep <= 0:
                    continue
                gap = (ep - prev["Close"]) / prev["Close"]
                if gap < -0.015:
                    continue
                rsi = prev.get("RSI14", 0)
                prev_dates_map[code] = (rsi, ep, atr, prev["Close"])

            # RSI降順でソート
            candidates = sorted(
                [(rsi, code, ep, atr) for code, (rsi, ep, atr, _) in prev_dates_map.items()
                 if code not in holding],
                key=lambda x: x[0], reverse=True
            )

            for rsi, code, ep, atr in candidates[:slots]:
                sl = ep - atr * ATR_SL_MULT
                tp = ep + atr * ATR_TP_MULT
                sh = min(int(capital * MAX_POS_RATIO / ep),
                         int(INITIAL_CAPITAL * 0.02 / (atr * ATR_SL_MULT)))
                sh = max(sh, 0)
                if sh > 0 and ep * sh <= capital:
                    positions.append(Position(code, today, ep, sh, sl, tp, rsi=rsi))
                    slots -= 1
                    if slots <= 0:
                        break

        equity[today] = capital

    eq_s = pd.Series(equity).sort_index()
    print(f"      完了: {time.time()-t0:.1f}秒  月次ストップ: {stop_cnt}回")
    return trades, eq_s


# ── 結果集計 ──────────────────────────────────────────────────
def compute_stats(trades, equity):
    total    = len(trades)
    wins     = sum(1 for t in trades if t.pnl > 0)
    win_rate = wins / total * 100 if total else 0
    years    = (OOS_END - OOS_START).days / 365
    cagr     = ((equity.iloc[-1] / INITIAL_CAPITAL)**(1/years) - 1) * 100
    peak     = equity.cummax()
    max_dd   = ((equity - peak) / peak * 100).min()
    total_cost = sum((t.entry_price+t.exit_price)*t.shares*COST_PER_LEG for t in trades)
    return {"total": total, "wins": wins, "win_rate": win_rate,
            "cagr": cagr, "max_dd": max_dd, "final": equity.iloc[-1],
            "total_cost": total_cost}


# ── メイン ────────────────────────────────────────────────────
def main():
    print("=" * 68)
    print("  ユニバース型 戦略A バックテスト（全プライム・IS選定なし）")
    print(f"  OOS期間: {OOS_START.date()} ～ {OOS_END.date()}")
    print("=" * 68)

    # データ読み込み
    print("\nデータ読み込み中...")
    t0 = time.time()
    df_all = pd.read_parquet(DATA_PATH)
    df_all["Date"] = pd.to_datetime(df_all["Date"])
    df_all["Code"] = df_all["Code4"].astype(str).str.zfill(4) + "0"
    df_all = df_all[df_all["Date"] >= WARMUP].copy()
    df_all = df_all.sort_values(["Code", "Date"]).reset_index(drop=True)
    n_stocks = df_all["Code"].nunique()
    print(f"  読み込み完了: {len(df_all):,}行  {n_stocks:,}銘柄  ({time.time()-t0:.1f}秒)")

    # 不要列削除
    drop_cols = [c for c in ["Code4","O","H","L","C","Vo","Va","UL","LL","AdjFactor"]
                 if c in df_all.columns]
    df_all.drop(columns=drop_cols, inplace=True)

    # 指標・シグナル計算
    print("\n処理開始...")
    df_all = compute_indicators_vectorized(df_all)
    df_all = compute_signals_vectorized(df_all)

    # ポートフォリオシミュレーション
    trades, equity = run_portfolio(df_all)

    stats = compute_stats(trades, equity)

    # ── 結果表示 ──────────────────────────────────────────────
    print()
    print("=" * 68)
    print("  ★ ユニバース型 戦略A OOS結果 ★")
    print("=" * 68)
    print(f"\n  初期資金   : {INITIAL_CAPITAL:>12,} 円")
    print(f"  最終資産   : {stats['final']:>12,.0f} 円")
    print(f"  総損益     : {stats['final']-INITIAL_CAPITAL:>+12,.0f} 円")
    print(f"  年利(CAGR) : {stats['cagr']:>+11.2f} %")
    print(f"  最大DD     : {stats['max_dd']:>11.2f} %")
    print(f"  総取引数   : {stats['total']:>12} 回")
    print(f"  勝率       : {stats['win_rate']:>11.1f} %")
    print(f"  総コスト   : {stats['total_cost']:>12,.0f} 円")

    print(f"\n【比較】")
    print(f"  固定選定 A v2  : CAGR +8.86%  取引262回  勝率53.8%  MaxDD-10.15%")
    print(f"  ユニバース型 A : CAGR {stats['cagr']:+.2f}%  "
          f"取引{stats['total']}回  "
          f"勝率{stats['win_rate']:.1f}%  "
          f"MaxDD{stats['max_dd']:.2f}%")

    print(f"\n【月別損益（OOS期間）】")
    print(f"  {'年月':8}  {'損益':>12}  {'累計資産':>13}")
    print("  " + "-" * 38)
    eq_m = equity.resample("ME").last()
    prev = INITIAL_CAPITAL
    for ym, cap in eq_m.items():
        if ym < OOS_START:
            continue
        pm   = cap - prev
        sign = "▲" if pm < 0 else "+"
        mark = " ◀" if pm < 0 else ""
        print(f"  {ym.strftime('%Y-%m'):8}  {sign}{abs(pm):>10,.0f}円  {cap:>12,.0f}円{mark}")
        prev = cap

    print()
    verdict = "✅ 固定選定より優位" if stats['cagr'] > 8.86 else "❌ 固定選定に劣る"
    print(f"  判定: {verdict}  (固定選定 +8.86% vs ユニバース {stats['cagr']:+.2f}%)")
    print("=" * 68)


if __name__ == "__main__":
    main()
