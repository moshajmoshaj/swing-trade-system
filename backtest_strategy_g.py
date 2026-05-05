"""
戦略G：EPS加速モメンタム バックテスト

設計思想：
  EPS成長率が「加速している」銘柄は機関投資家の注目が集まり
  株価モメンタムが持続しやすい。戦略F（EPS成長20%以上）とは異なり
  「成長率の変化（加速）」をシグナルとする。

EPS加速の定義：
  - 当期EPS成長率（YoY） > 前期EPS成長率（YoY）
  - 成長率の改善幅（加速度）≥ 閾値（パラメータ検証）
  - 当期・前期ともにEPS > 0（黒字）

エントリー条件：
  1. EPS加速開示日から30日以内
  2. 終値 > SMA200（長期上昇トレンド）
  3. RSI 45〜70
  4. 出来高 ≥ 20日平均 × 1.2
  5. 陽線（終値 > 始値）

エグジット：TP=ATR×6 / SL=ATR×2 / 強制15日

4時代検証：Pre-IS(2008-2015) / IS(2016-2020) / Gap(2021-2022) / OOS(2023-2026)
"""
import sys
import time
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

# ── 戦略Gパラメータ ───────────────────────────────────────────
ATR_TP_MULT     = 6
ATR_SL_MULT     = 2
MAX_HOLD_DAYS   = 15
ENTRY_WINDOW    = 30    # 決算開示後の有効エントリー日数（カレンダー日）
ACCEL_THRESHOLD = 0.10  # EPS成長率の改善幅下限（10pp）

# ── ポートフォリオ設定 ────────────────────────────────────────
INITIAL_CAPITAL = 1_000_000
MAX_POSITIONS   = 5
MAX_POS_RATIO   = 0.20
COST_PER_LEG    = 0.00055 + 0.00050

DATA_PATH = Path("data/raw/prices_20y.parquet")
FINS_PATH = Path("data/raw/fins_summary.parquet")


# ── EPS加速イベント構築 ───────────────────────────────────────
def build_accel_events(fins_path: Path, accel_thr: float) -> pd.DataFrame:
    """
    EPS成長率が加速した通期決算開示イベントを返す。
    columns: Code5, DiscDate, eps_growth, accel, Sales
    """
    fins = pd.read_parquet(fins_path)
    fins["DiscDate"] = pd.to_datetime(fins["DiscDate"], errors="coerce")

    # 通期決算のみ・数値変換
    fy = fins[fins["CurPerType"] == "FY"].copy()
    for col in ["EPS", "NP", "Sales"]:
        fy[col] = pd.to_numeric(fy[col], errors="coerce")

    # 重複排除：同一銘柄×決算期末で最新のDiscDateを保持
    fy["CurFYEn"] = pd.to_datetime(fy["CurFYEn"], errors="coerce")
    fy = (fy.sort_values("DiscDate")
            .drop_duplicates(subset=["Code", "CurFYEn"], keep="last")
            .copy())

    # 銘柄×DiscDate順にソートしてYoY EPS成長率を計算
    fy = fy.sort_values(["Code", "DiscDate"]).reset_index(drop=True)
    fy["eps_prev"] = fy.groupby("Code")["EPS"].shift(1)

    # 成長率計算（前期・当期ともに正の場合のみ）
    valid = (fy["EPS"] > 0) & (fy["eps_prev"] > 0)
    fy["eps_growth"] = np.nan
    fy.loc[valid, "eps_growth"] = (
        (fy.loc[valid, "EPS"] - fy.loc[valid, "eps_prev"])
        / fy.loc[valid, "eps_prev"]
    )

    # 成長率の加速度（当期成長率 - 前期成長率）
    fy["growth_prev"] = fy.groupby("Code")["eps_growth"].shift(1)
    fy["accel"] = fy["eps_growth"] - fy["growth_prev"]

    # フィルター：加速度 ≥ 閾値 かつ 黒字
    events = fy[
        (fy["accel"] >= accel_thr) &
        (fy["EPS"] > 0) &
        (fy["NP"] > 0) &
        fy["DiscDate"].notna()
    ][["Code", "DiscDate", "eps_growth", "accel", "Sales"]].copy()

    # 5桁コード変換
    events["Code5"] = events["Code"].astype(str).str.strip().str[:4].str.zfill(4) + "0"

    print(f"  EPS加速イベント: {len(events):,}件  "
          f"対象銘柄: {events['Code5'].nunique():,}銘柄  "
          f"期間: {events['DiscDate'].min().date()} ～ {events['DiscDate'].max().date()}")
    return events.reset_index(drop=True)


# ── 株価データ・指標計算（各銘柄個別） ──────────────────────
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
    return df


# ── エントリーウィンドウフラグ付与 ───────────────────────────
def add_g_flag(df: pd.DataFrame, disc_dates: list, window: int) -> pd.DataFrame:
    df = df.copy()
    df["g_flag"] = False
    for disc in disc_dates:
        mask = (df["Date"] > disc) & (df["Date"] <= disc + pd.Timedelta(days=window))
        df.loc[mask, "g_flag"] = True
    return df


# ── 戦略Gシグナル生成 ─────────────────────────────────────────
def signal_g(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "g_flag" not in df.columns:
        df["signal_G"] = 0
        return df
    df["signal_G"] = (
        df["g_flag"] &
        (df["Close"] > df["SMA200"]) &
        (df["RSI14"] >= 45) & (df["RSI14"] <= 70) &
        (df["Volume"] >= df["VOL_MA20"] * 1.2) &
        (df["Close"] > df["Open"])
    ).astype(int)
    return df


# ── ポートフォリオシミュレーション（高速版）────────────────
@dataclass
class Position:
    code: str; entry_date: pd.Timestamp; entry_price: float
    shares: int; stop_loss: float; take_profit: float
    hold_days: int = 0; rsi: float = 0.0

@dataclass
class Trade:
    code: str; entry_date: pd.Timestamp; exit_date: pd.Timestamp
    entry_price: float; exit_price: float; shares: int
    pnl: float; reason: str


def build_entry_index(stock_signals: dict,
                      period_start: pd.Timestamp,
                      period_end: pd.Timestamp) -> dict:
    """日付→エントリー候補リスト(rsi, code, ep, atr)を事前計算"""
    index: dict = {}
    for code, df in stock_signals.items():
        df_s = df.sort_values("Date").reset_index(drop=True)
        sig_rows = df_s[
            (df_s["signal_G"] == 1) &
            (df_s["Date"] >= period_start) &
            (df_s["Date"] < period_end)
        ]
        for _, sig in sig_rows.iterrows():
            # 翌営業日を取得
            nxt = df_s[df_s["Date"] > sig["Date"]]
            if nxt.empty:
                continue
            nxt_row = nxt.iloc[0]
            if nxt_row["Date"] > period_end:
                continue
            ep  = nxt_row["Open"]
            atr = sig["ATR14"]
            if ep <= 0 or not atr or pd.isna(atr):
                continue
            gap = (ep - sig["Close"]) / sig["Close"]
            if gap < -0.015:
                continue
            entry_date = nxt_row["Date"]
            index.setdefault(entry_date, []).append(
                (float(sig["RSI14"]), code, float(ep), float(atr))
            )
    # 各日付でRSI降順にソート済みにしておく
    for d in index:
        index[d].sort(key=lambda x: x[0], reverse=True)
    return index


def run_portfolio(stock_signals: dict, period_start: pd.Timestamp,
                  period_end: pd.Timestamp) -> tuple:
    all_dates = sorted({d for df in stock_signals.values()
                        for d in df["Date"]
                        if period_start <= d <= period_end})
    if not all_dates:
        return [], pd.Series(dtype=float)

    # 事前計算：エントリー候補インデックス（高速化の核心）
    entry_index = build_entry_index(stock_signals, period_start, period_end)

    # 決済用ルックアップ（保有銘柄のみ必要）
    lookup = {code: df.drop_duplicates("Date").set_index("Date").to_dict("index")
              for code, df in stock_signals.items()}

    capital = float(INITIAL_CAPITAL)
    positions: list[Position] = []
    trades:    list[Trade]    = []
    equity:    dict           = {}
    cur_month = None; month_start = capital; month_stopped = False; stop_cnt = 0

    for today in all_dates:
        ym = (today.year, today.month)
        if ym != cur_month:
            cur_month = ym; month_start = capital; month_stopped = False

        # 決済判定（最大5ポジションのみ＝高速）
        next_pos = []
        for pos in positions:
            row = lookup.get(pos.code, {}).get(today)
            if row is None:
                next_pos.append(pos); continue
            pos.hold_days += 1
            hi, lo, cl = row["High"], row["Low"], row["Close"]
            ep = er = None
            if lo <= pos.stop_loss:             ep, er = pos.stop_loss, "損切り"
            elif hi >= pos.take_profit:          ep, er = pos.take_profit, "利確"
            elif pos.hold_days >= MAX_HOLD_DAYS: ep, er = cl, "期間満了"
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

        # エントリー（事前計算済みインデックスを使用）
        slots = MAX_POSITIONS - len(positions)
        if slots > 0 and not month_stopped and today in entry_index:
            holding = {p.code for p in positions}
            for rsi, code, ep, atr in entry_index[today]:
                if slots <= 0:
                    break
                if code in holding:
                    continue
                sl = ep - atr * ATR_SL_MULT
                tp = ep + atr * ATR_TP_MULT
                sh = min(int(capital * MAX_POS_RATIO / ep),
                         int(INITIAL_CAPITAL * 0.02 / (atr * ATR_SL_MULT)))
                if sh > 0 and ep * sh <= capital:
                    positions.append(Position(code, today, ep, sh, sl, tp, rsi=rsi))
                    holding.add(code)
                    slots -= 1

        equity[today] = capital

    eq_s = pd.Series(equity).sort_index()
    print(f"      月次ストップ: {stop_cnt}回")
    return trades, eq_s


def compute_stats(trades, equity, period_start, period_end):
    if not trades or equity.empty:
        return None
    total = len(trades)
    wins  = sum(1 for t in trades if t.pnl > 0)
    years = (period_end - period_start).days / 365
    cagr  = ((equity.iloc[-1] / INITIAL_CAPITAL)**(1/years) - 1) * 100 if years > 0 else 0
    peak  = equity.cummax()
    mdd   = ((equity - peak) / peak * 100).min()
    wr    = wins / total * 100 if total else 0
    return {"total": total, "wr": wr, "cagr": cagr, "mdd": mdd}


# ── メイン ────────────────────────────────────────────────────
def main():
    print("=" * 68)
    print("  戦略G：EPS加速モメンタム バックテスト（4時代）")
    print(f"  加速閾値: {ACCEL_THRESHOLD*100:.0f}pp以上  "
          f"エントリー窓: {ENTRY_WINDOW}日  "
          f"TP=ATR×{ATR_TP_MULT}  SL=ATR×{ATR_SL_MULT}  強制={MAX_HOLD_DAYS}日")
    print("=" * 68)

    # ── EPS加速イベント構築 ──────────────────────────────────
    print("\nEPS加速イベント構築中...")
    events = build_accel_events(FINS_PATH, ACCEL_THRESHOLD)
    # コード別にイベント日リストを整理
    events_by_code: dict = {}
    for _, row in events.iterrows():
        c = row["Code5"]
        events_by_code.setdefault(c, []).append(row["DiscDate"])

    # ── 価格データ読み込み ──────────────────────────────────
    print("\n価格データ読み込み中...")
    t0 = time.time()
    df_all = pd.read_parquet(DATA_PATH)
    df_all["Date"] = pd.to_datetime(df_all["Date"])
    df_all["Code"] = df_all["Code4"].astype(str).str.zfill(4) + "0"
    target_codes   = set(events_by_code.keys()) & set(df_all["Code"].unique())
    df_all = df_all[df_all["Code"].isin(target_codes) &
                    (df_all["Date"] >= WARMUP)].copy()
    drop_cols = [c for c in ["Code4","O","H","L","C","Vo","Va","UL","LL","AdjFactor"]
                 if c in df_all.columns]
    df_all.drop(columns=drop_cols, inplace=True)
    df_all = df_all.sort_values(["Code","Date"]).reset_index(drop=True)
    print(f"  完了: {len(df_all):,}行  {df_all['Code'].nunique()}銘柄  ({time.time()-t0:.1f}秒)")

    # ── 銘柄別指標・シグナル計算 ─────────────────────────────
    print("\n指標・シグナル計算中...")
    t1 = time.time()
    stock_signals: dict = {}
    for i, code in enumerate(sorted(target_codes)):
        df_s = df_all[df_all["Code"] == code].copy()
        if len(df_s) < 260:
            continue
        df_s = add_indicators(df_s)
        df_s = add_g_flag(df_s, events_by_code.get(code, []), ENTRY_WINDOW)
        df_s = signal_g(df_s)
        stock_signals[code] = df_s
        if (i+1) % 200 == 0:
            print(f"  {i+1}/{len(target_codes)}銘柄 ({time.time()-t1:.0f}秒)")
    print(f"  完了: {len(stock_signals)}銘柄  ({time.time()-t1:.1f}秒)")

    # シグナル総数
    for pname, (ps, pe) in PERIODS.items():
        sigs = sum(
            int(df[(df["Date"] >= ps) & (df["Date"] <= pe)]["signal_G"].sum())
            for df in stock_signals.values()
        )
        print(f"  シグナル数 {pname}: {sigs:,}件")

    # ── 4時代バックテスト ─────────────────────────────────────
    print()
    period_results = {}
    for pname, (ps, pe) in PERIODS.items():
        print(f"【{pname} 期間】{ps.date()} ～ {pe.date()}")
        trades, equity = run_portfolio(stock_signals, ps, pe)
        stats = compute_stats(trades, equity, ps, pe)
        period_results[pname] = stats
        if stats:
            print(f"      取引{stats['total']}回  勝率{stats['wr']:.1f}%  "
                  f"CAGR{stats['cagr']:+.2f}%  MaxDD{stats['mdd']:.2f}%")
        else:
            print("      取引なし")

    # ── 結果サマリー ──────────────────────────────────────────
    print()
    print("=" * 68)
    print("  ★ 戦略G EPS加速モメンタム 4時代結果 ★")
    print("=" * 68)
    print(f"\n  {'時代':12}  {'取引':>5}  {'勝率':>6}  {'CAGR':>7}  {'MaxDD':>7}  判定")
    print("  " + "-" * 55)
    all_positive = True
    for pname, stats in period_results.items():
        if stats is None:
            print(f"  {pname:12}  {'---':>5}  {'---':>6}  {'---':>7}  {'---':>7}  —")
            all_positive = False
            continue
        ok = "✅" if stats["cagr"] > 0 else "❌"
        if stats["cagr"] <= 0:
            all_positive = False
        print(f"  {pname:12}  {stats['total']:>5}  {stats['wr']:>5.1f}%  "
              f"{stats['cagr']:>+6.2f}%  {stats['mdd']:>+6.2f}%  {ok}")

    # 戦略F比較
    print(f"\n【戦略F（PEAD・参考）との比較】")
    print(f"  {'時代':12}  {'戦略F CAGR':>10}  {'戦略G CAGR':>10}")
    print("  " + "-" * 38)
    f_cagr = {"Pre-IS": +3.0, "IS": +6.5, "Gap": +3.6, "OOS": +6.7}
    for pname in PERIODS:
        fc = f_cagr.get(pname, 0)
        gc = period_results[pname]["cagr"] if period_results[pname] else float("nan")
        mark = " ✅" if gc > fc else " ❌"
        print(f"  {pname:12}  {fc:>+9.1f}%  {gc:>+9.2f}%{mark}")

    print()
    oos = period_results.get("OOS")
    if oos:
        verdict = "✅ 全期間プラス・有望" if all_positive else \
                  "⚠️  一部期間マイナス・要改善" if oos["cagr"] > 0 else \
                  "❌ OOS期間マイナス・不採用"
        print(f"  判定: {verdict}")
        print(f"  OOS CAGR: {oos['cagr']:+.2f}%  "
              f"MaxDD: {oos['mdd']:.2f}%  "
              f"取引: {oos['total']}回")
    print("=" * 68)


if __name__ == "__main__":
    main()
