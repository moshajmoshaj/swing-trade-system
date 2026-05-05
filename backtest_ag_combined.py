"""
戦略A v2 + 戦略G（EPS加速）合算バックテスト（メモリ効率版）

メモリ設計：
  - G銘柄(1,496銘柄)の指標計算は100銘柄ずつバッチ処理
  - バッチごとにエントリーシグナル情報のみ抽出→価格データを解放
  - 決済用価格データはシグナル発生銘柄のみロード（大幅削減）
  - 想定メモリ使用量：2GB以内

OOS期間: 2023-01-01 ～ 2026-04-24
資金    : 100万円・最大5ポジション・1銘柄上限20万円
"""
import gc
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd

# ── 期間設定 ──────────────────────────────────────────────────
PERIODS = {
    "Pre-IS": (pd.Timestamp("2008-01-01"), pd.Timestamp("2015-12-31")),
    "IS"    : (pd.Timestamp("2016-01-01"), pd.Timestamp("2020-12-31")),
    "Gap"   : (pd.Timestamp("2021-01-01"), pd.Timestamp("2022-12-31")),
    "OOS"   : (pd.Timestamp("2023-01-01"), pd.Timestamp("2026-04-24")),
}
WARMUP = pd.Timestamp("2007-01-01")

# ── 戦略パラメータ ────────────────────────────────────────────
CFG = {
    "A": {"tp": 6, "sl": 2, "hold": 10},
    "G": {"tp": 6, "sl": 2, "hold": 15},
}
ADX_THRESHOLD   = 15
ACCEL_THRESHOLD = 0.10
ENTRY_WINDOW    = 30

INITIAL_CAPITAL = 1_000_000
MAX_POSITIONS   = 5
MAX_POS_RATIO   = 0.20
COST_PER_LEG    = 0.00055 + 0.00050
BATCH_SIZE      = 100    # メモリ節約：バッチ処理サイズ

DATA_PATH   = Path("data/raw/prices_20y.parquet")
FINS_PATH   = Path("data/raw/fins_summary.parquet")
CAND_A_PATH = Path("logs/final_candidates_v2.csv")
INDEX_PATH  = Path("data/raw/indices.parquet")
TOPIX_CODE  = "0000"    # J-Quants indices.parquet の TOPIX コード


# ── 指標計算（単一銘柄） ──────────────────────────────────────
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().sort_values("Date").reset_index(drop=True)
    c = df["Close"]
    df["SMA20"]    = c.rolling(20).mean()
    df["SMA200"]   = c.rolling(200).mean()
    df["VOL_MA20"] = df["Volume"].rolling(20).mean()
    delta = c.diff()
    gain  = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
    loss  = (-delta).clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
    df["RSI14"] = 100 - 100 / (1 + gain / loss.replace(0, 1e-9))
    prev_c = c.shift(1)
    tr = pd.concat([df["High"]-df["Low"],
                    (df["High"]-prev_c).abs(),
                    (df["Low"] -prev_c).abs()], axis=1).max(axis=1)
    df["ATR14"] = tr.ewm(alpha=1/14, adjust=False).mean()
    ph = df["High"].shift(1); pl = df["Low"].shift(1)
    dm_p = np.where((df["High"]-ph > 0) & (df["High"]-ph > pl-df["Low"]), df["High"]-ph, 0.0)
    dm_m = np.where((pl-df["Low"]  > 0) & (pl-df["Low"] > df["High"]-ph), pl-df["Low"],  0.0)
    atr_s = df["ATR14"].replace(0, np.nan)
    di_p  = 100 * pd.Series(dm_p).ewm(alpha=1/14, adjust=False).mean() / atr_s
    di_m  = 100 * pd.Series(dm_m).ewm(alpha=1/14, adjust=False).mean() / atr_s
    dx    = 100*(di_p-di_m).abs()/(di_p+di_m).replace(0, np.nan)
    df["ADX14"] = dx.ewm(alpha=1/14, adjust=False).mean()
    return df


# ── 戦略Aシグナル ─────────────────────────────────────────────
def sig_a(df):
    rsi3 = df["RSI14"].shift(3)
    df["signal"] = (
        (df["SMA20"]  > df.get("SMA50", df["SMA200"])) &
        (df["Close"]  > df["SMA200"]) &
        (df["RSI14"] >= 45) & (df["RSI14"] <= 75) &
        (df["RSI14"]  > rsi3) &
        (df["Volume"] >= df["VOL_MA20"] * 1.2) &
        (df["Close"]  > df["Open"]) &
        (df["ADX14"]  > ADX_THRESHOLD)
    ).astype(int)
    return df

def add_sma50(df):
    df["SMA50"] = df["Close"].rolling(50).mean()
    return df


# ── EPS加速イベント構築 ───────────────────────────────────────
def build_accel_events(fins_path):
    fins = pd.read_parquet(fins_path)
    fins["DiscDate"] = pd.to_datetime(fins["DiscDate"], errors="coerce")
    fy = fins[fins["CurPerType"] == "FY"].copy()
    for col in ["EPS","NP"]:
        fy[col] = pd.to_numeric(fy[col], errors="coerce")
    fy["CurFYEn"] = pd.to_datetime(fy["CurFYEn"], errors="coerce")
    fy = fy.sort_values("DiscDate").drop_duplicates(["Code","CurFYEn"], keep="last")
    fy = fy.sort_values(["Code","DiscDate"]).reset_index(drop=True)
    fy["eps_prev"] = fy.groupby("Code")["EPS"].shift(1)
    valid = (fy["EPS"] > 0) & (fy["eps_prev"] > 0)
    fy["eps_growth"] = np.nan
    fy.loc[valid, "eps_growth"] = (fy.loc[valid,"EPS"]-fy.loc[valid,"eps_prev"])/fy.loc[valid,"eps_prev"]
    fy["growth_prev"] = fy.groupby("Code")["eps_growth"].shift(1)
    fy["accel"] = fy["eps_growth"] - fy["growth_prev"]
    events = fy[(fy["accel"] >= ACCEL_THRESHOLD) & (fy["EPS"] > 0) & (fy["NP"] > 0)].copy()
    events["Code5"] = events["Code"].astype(str).str.strip().str[:4].str.zfill(4) + "0"
    result = defaultdict(list)
    for _, row in events.iterrows():
        result[row["Code5"]].append(row["DiscDate"])
    return dict(result)


# ── エントリーインデックス構築（バッチ処理でメモリ節約） ──────
def build_entry_index_batch(df_all: pd.DataFrame,
                             strategy: str,
                             codes: list,
                             events_by_code: dict = None) -> tuple:
    """
    バッチ処理でエントリーシグナルを抽出。
    戻り値:
      entry_index: {date: [(rsi, code, ep, atr, strat)]}
      signal_codes: シグナルが発生した銘柄セット（決済用に後でロード）
    """
    entry_index = defaultdict(list)
    signal_codes = set()
    n = len(codes)

    for batch_start in range(0, n, BATCH_SIZE):
        batch = codes[batch_start:batch_start + BATCH_SIZE]
        df_batch = df_all[df_all["Code"].isin(batch)].copy()

        for code in batch:
            df_s = df_batch[df_batch["Code"] == code].copy()
            if len(df_s) < 260:
                continue

            df_s = add_indicators(df_s)

            if strategy == "A":
                df_s = add_sma50(df_s)
                df_s = sig_a(df_s)
                sig_col = "signal"
            else:  # G
                df_s["g_flag"] = False
                for disc in events_by_code.get(code, []):
                    mask = (df_s["Date"] > disc) & (df_s["Date"] <= disc + pd.Timedelta(days=ENTRY_WINDOW))
                    df_s.loc[mask, "g_flag"] = True
                df_s["signal"] = (
                    df_s["g_flag"] &
                    (df_s["Close"] > df_s["SMA200"]) &
                    (df_s["RSI14"] >= 45) & (df_s["RSI14"] <= 70) &
                    (df_s["Volume"] >= df_s["VOL_MA20"] * 1.2) &
                    (df_s["Close"] > df_s["Open"])
                ).astype(int)
                sig_col = "signal"

            # シグナル行から翌日エントリー情報を抽出
            sig_rows = df_s[df_s[sig_col] == 1].reset_index(drop=True)
            for _, sig in sig_rows.iterrows():
                nxt = df_s[df_s["Date"] > sig["Date"]]
                if nxt.empty:
                    continue
                nxt_row = nxt.iloc[0]
                ep  = nxt_row["Open"]
                atr = sig["ATR14"]
                if ep <= 0 or not atr or pd.isna(atr):
                    continue
                if (ep - sig["Close"]) / sig["Close"] < -0.015:
                    continue
                entry_index[nxt_row["Date"]].append(
                    (float(sig["RSI14"]), code, float(ep), float(atr), strategy)
                )
                signal_codes.add(code)

        del df_batch
        gc.collect()

        done = min(batch_start + BATCH_SIZE, n)
        if done % 200 == 0 or done == n:
            print(f"    {strategy}: {done}/{n}銘柄処理済")

    # RSI降順ソート
    for d in entry_index:
        entry_index[d].sort(key=lambda x: x[0], reverse=True)

    return dict(entry_index), signal_codes


# ── ポートフォリオシミュレーション ───────────────────────────
@dataclass
class Position:
    code: str; strategy: str; entry_date: pd.Timestamp
    entry_price: float; shares: int; stop_loss: float
    take_profit: float; max_hold: int; hold_days: int = 0

@dataclass
class Trade:
    code: str; strategy: str; entry_date: pd.Timestamp
    exit_date: pd.Timestamp; pnl: float; reason: str


def run_portfolio(entry_index: dict, exit_lookup: dict,
                  period_start, period_end,
                  regime_s: pd.Series = None) -> tuple:
    all_dates = sorted({d for d in exit_lookup.get(
        next(iter(exit_lookup), ""), {}).keys()
        if period_start <= d <= period_end} |
        {d for d in entry_index if period_start <= d <= period_end})

    # 全営業日リストを価格データから構築
    all_dates = sorted({
        d for lk in exit_lookup.values()
        for d in lk if period_start <= d <= period_end
    } | {d for d in entry_index if period_start <= d <= period_end})

    capital = float(INITIAL_CAPITAL)
    positions: list = []
    trades:    list = []
    equity:    dict = {}
    cur_month = None; month_start = capital; month_stopped = False; stop_cnt = 0

    for today in all_dates:
        ym = (today.year, today.month)
        if ym != cur_month:
            cur_month = ym; month_start = capital; month_stopped = False

        next_pos = []
        for pos in positions:
            row = exit_lookup.get(pos.code, {}).get(today)
            if row is None:
                next_pos.append(pos); continue
            pos.hold_days += 1
            hi, lo, cl = row["High"], row["Low"], row["Close"]
            ep = er = None
            if lo <= pos.stop_loss:             ep, er = pos.stop_loss, "損切り"
            elif hi >= pos.take_profit:          ep, er = pos.take_profit, "利確"
            elif pos.hold_days >= pos.max_hold:  ep, er = cl, "期間満了"
            if ep is not None:
                cost = (pos.entry_price + ep) * pos.shares * COST_PER_LEG
                pnl  = (ep - pos.entry_price) * pos.shares - cost
                capital += pnl
                trades.append(Trade(pos.code, pos.strategy, pos.entry_date,
                                    today, pnl, er))
            else:
                next_pos.append(pos)
        positions = next_pos

        if not month_stopped and month_start > 0:
            if (capital - month_start) / month_start * 100 <= -10.0:
                month_stopped = True; stop_cnt += 1

        slots = MAX_POSITIONS - len(positions)
        regime = regime_s.get(today, "BULL") if regime_s is not None else "BULL"
        if slots > 0 and not month_stopped and today in entry_index and regime != "BEAR":
            holding = {p.code for p in positions}
            for rsi, code, ep, atr, strat in entry_index[today]:
                if slots <= 0: break
                if code in holding: continue
                # レジームフィルター：NEUTRAL時はG(ファンダ系)のみ有効
                if regime == "NEUTRAL" and strat == "A":
                    continue
                cfg = CFG[strat]
                sl = ep - atr * cfg["sl"]
                tp = ep + atr * cfg["tp"]
                sh = min(int(capital * MAX_POS_RATIO / ep),
                         int(INITIAL_CAPITAL * 0.02 / (atr * cfg["sl"])))
                if sh > 0 and ep * sh <= capital:
                    positions.append(Position(code, strat, today, ep, sh,
                                              sl, tp, cfg["hold"]))
                    holding.add(code)
                    slots -= 1

        equity[today] = capital

    eq_s = pd.Series(equity).sort_index()
    print(f"    月次ストップ: {stop_cnt}回")
    return trades, eq_s


# ── 市場レジーム系列構築 ──────────────────────────────────────
def build_regime(index_path: Path, code: str = "0000") -> pd.Series:
    """
    TOPIX の SMA50/200 に基づき日次レジームを返す。
    BULL: Close > SMA50 / NEUTRAL: SMA200 < Close ≤ SMA50 / BEAR: Close ≤ SMA200
    """
    idx = pd.read_parquet(index_path)
    idx["Date"] = pd.to_datetime(idx["Date"])
    topix = idx[idx["Code"] == code].sort_values("Date").copy()
    topix = topix.rename(columns={"C": "Close"})
    topix["SMA50"]  = topix["Close"].rolling(50).mean()
    topix["SMA200"] = topix["Close"].rolling(200).mean()
    topix["regime"] = "BEAR"
    topix.loc[topix["Close"] > topix["SMA200"], "regime"] = "NEUTRAL"
    topix.loc[topix["Close"] > topix["SMA50"],  "regime"] = "BULL"
    return topix.set_index("Date")["regime"]


def stats(trades, equity, ps, pe):
    if not trades or equity.empty: return None
    total = len(trades)
    wins  = sum(1 for t in trades if t.pnl > 0)
    years = (pe - ps).days / 365
    cagr  = ((equity.iloc[-1]/INITIAL_CAPITAL)**(1/years)-1)*100 if years>0 else 0
    peak  = equity.cummax()
    mdd   = ((equity-peak)/peak*100).min()
    by_s  = {}
    for t in trades:
        by_s.setdefault(t.strategy, []).append(t.pnl)
    return {"total":total, "wr":wins/total*100, "cagr":cagr, "mdd":mdd,
            "by_s":{s: (len(p), sum(p)) for s,p in by_s.items()}}


# ── メイン ────────────────────────────────────────────────────
def main():
    print("=" * 68)
    print("  合算バックテスト：戦略A v2 + 戦略G（メモリ効率版）")
    print(f"  RAM節約：{BATCH_SIZE}銘柄ずつバッチ処理")
    print("=" * 68)

    # A v2 候補
    df_cand = pd.read_csv(CAND_A_PATH)
    codes_a = sorted((df_cand["code"].astype(str).str.zfill(4) + "0").tolist())
    print(f"\n戦略A v2候補: {len(codes_a)}銘柄")

    # EPS加速イベント
    print("EPS加速イベント構築中...")
    events_by_code = build_accel_events(FINS_PATH)
    codes_g = sorted(events_by_code.keys())
    print(f"戦略G対象: {len(codes_g)}銘柄")

    # 価格データ読み込み（A+G合算 ただし逐次処理のため全銘柄を保持）
    all_codes = sorted(set(codes_a) | set(codes_g))
    print(f"\n価格データ読み込み中（{len(all_codes)}銘柄）...")
    t0 = time.time()
    df_all = pd.read_parquet(DATA_PATH)
    df_all["Date"] = pd.to_datetime(df_all["Date"])
    df_all["Code"] = df_all["Code4"].astype(str).str.zfill(4) + "0"
    df_all = df_all[df_all["Code"].isin(all_codes) & (df_all["Date"] >= WARMUP)].copy()
    drop_cols = [c for c in ["Code4","O","H","L","C","Vo","Va","UL","LL","AdjFactor"]
                 if c in df_all.columns]
    df_all.drop(columns=drop_cols, inplace=True)
    df_all = df_all.sort_values(["Code","Date"]).reset_index(drop=True)
    print(f"  完了: {len(df_all):,}行  {df_all['Code'].nunique()}銘柄  ({time.time()-t0:.1f}秒)")

    # エントリーインデックス構築（バッチ処理）
    print("\n【戦略A v2 エントリーインデックス構築】")
    entry_a, sig_codes_a = build_entry_index_batch(df_all, "A", codes_a)
    print(f"  Aシグナル発生銘柄: {len(sig_codes_a)}")

    print("\n【戦略G エントリーインデックス構築（バッチ）】")
    entry_g, sig_codes_g = build_entry_index_batch(df_all, "G", codes_g, events_by_code)
    print(f"  Gシグナル発生銘柄: {len(sig_codes_g)}")

    # 合算エントリーインデックス
    combined_entry = defaultdict(list)
    for d, cands in entry_a.items():
        combined_entry[d].extend(cands)
    for d, cands in entry_g.items():
        combined_entry[d].extend(cands)
    for d in combined_entry:
        combined_entry[d].sort(key=lambda x: x[0], reverse=True)

    # 決済用価格ルックアップ（シグナル発生銘柄のみ）
    exit_codes = sig_codes_a | sig_codes_g
    print(f"\n決済用価格ルックアップ構築（{len(exit_codes)}銘柄）...")
    exit_lookup = {}
    for code in exit_codes:
        df_s = df_all[df_all["Code"] == code].drop_duplicates("Date")
        if not df_s.empty:
            exit_lookup[code] = df_s.set_index("Date")[["High","Low","Close"]].to_dict("index")

    # 価格データ本体を解放
    del df_all
    gc.collect()
    print("  価格データ解放完了（メモリ節約）")

    # レジーム系列構築
    print("\n市場レジーム構築中（TOPIX SMA50/200）...")
    regime_s = build_regime(INDEX_PATH, TOPIX_CODE)
    bull  = (regime_s == "BULL").sum()
    ntrl  = (regime_s == "NEUTRAL").sum()
    bear  = (regime_s == "BEAR").sum()
    print(f"  全期間: BULL={bull}日  NEUTRAL={ntrl}日  BEAR={bear}日")

    # 4時代バックテスト
    print("\n【4時代バックテスト実行（レジームフィルター付き）】")
    print("  ルール: BULL→A+G有効 / NEUTRAL→Gのみ / BEAR→全停止")
    period_results = {}
    for pname, (ps, pe) in PERIODS.items():
        print(f"\n  {pname} ({ps.date()} ～ {pe.date()})")
        era_entry = {d: v for d, v in combined_entry.items() if ps <= d <= pe}
        trades, equity = run_portfolio(era_entry, exit_lookup, ps, pe, regime_s)
        s = stats(trades, equity, ps, pe)
        period_results[pname] = s
        if s:
            print(f"    取引{s['total']}回  勝率{s['wr']:.1f}%  CAGR{s['cagr']:+.2f}%  MaxDD{s['mdd']:.2f}%")
            for strat, (cnt, pnl) in s["by_s"].items():
                ann = pnl / INITIAL_CAPITAL / ((pe-ps).days/365) * 100
                print(f"    　{strat}: {cnt}回  {pnl:+,.0f}円  年利貢献{ann:+.2f}%")

    # 結果サマリー
    print()
    print("=" * 68)
    print("  ★ A v2 + G 合算結果 ★")
    print("=" * 68)
    print(f"  {'時代':10}  {'取引':>5}  {'勝率':>6}  {'CAGR':>7}  {'MaxDD':>7}  判定")
    print("  " + "-" * 52)
    ref = {"Pre-IS":8.86, "IS":8.86, "Gap":8.86, "OOS":8.86}
    for pname, s in period_results.items():
        if s is None:
            print(f"  {pname:10}  ---"); continue
        ok = "✅" if s["cagr"] > 0 else "❌"
        beat = " ▲A単独" if s["cagr"] > ref.get(pname, 8.86) else " ▼A単独"
        print(f"  {pname:10}  {s['total']:>5}  {s['wr']:>5.1f}%  "
              f"{s['cagr']:>+6.2f}%  {s['mdd']:>+6.2f}%  {ok}{beat}")

    print(f"\n  【参考：A v2単独 OOS +8.86%  MaxDD -10.15%】")
    oos = period_results.get("OOS")
    if oos:
        verdict = "✅ A単独より優位" if oos["cagr"] > 8.86 else "❌ A単独に劣る"
        print(f"\n  判定: {verdict}  (A単独+8.86% vs A+G {oos['cagr']:+.2f}%)")
    print("=" * 68)


if __name__ == "__main__":
    main()
