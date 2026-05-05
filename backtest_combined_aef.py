"""
合算バックテスト：戦略A v2 + 戦略E v2 + 戦略F
目的：年利15%超えの可能性を数値で検証する

戦略構成：
  A v2  : final_candidates_v2.csv（29銘柄）TP=ATR×6 SL=ATR×2 保有10日
  E v2  : strategy_e_candidates_v2.csv（30銘柄）TP=ATR×6 SL=ATR×2 保有10日
  F     : EPS成長20%以上・全プライム TP=ATR×5 SL=ATR×2 保有15日

OOS期間  : 2023-01-01 ～ 2026-04-24
資金     : 100万円・最大5ポジション・1銘柄上限20万円
レジーム : 1306(TOPIX ETF) SMA50/200 でBULL/NEUTRAL/BEAR判定
           BULL → A+E+F有効 / NEUTRAL → F のみ / BEAR → 全停止
"""
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd

# ── 期間 ────────────────────────────────────────────────────
OOS_START  = pd.Timestamp("2023-01-01")
OOS_END    = pd.Timestamp("2026-04-24")
WARMUP     = pd.Timestamp("2022-01-01")   # 指標ウォームアップ用

# ── 戦略別パラメータ ─────────────────────────────────────────
STRATEGY_CFG = {
    "A": {"tp": 6, "sl": 2, "hold": 10},
    "E": {"tp": 6, "sl": 2, "hold": 10},
    "F": {"tp": 5, "sl": 2, "hold": 15},
}

# ── ポートフォリオ設定 ────────────────────────────────────────
INITIAL_CAPITAL = 1_000_000
MAX_POSITIONS   = 5
MAX_POS_RATIO   = 0.20      # 1銘柄上限20%
COMMISSION      = 0.00055
SLIPPAGE        = 0.00050
COST_PER_LEG    = COMMISSION + SLIPPAGE

# ── ファイルパス ──────────────────────────────────────────────
DATA_PATH    = Path("data/raw/prices_20y.parquet")
FINS_PATH    = Path("data/raw/fins_summary.parquet")
CAND_A_PATH  = Path("logs/final_candidates_v2.csv")
CAND_E_PATH  = Path("logs/strategy_e_candidates_v2.csv")
REGIME_CODE  = "13060"   # 1306 TOPIX連動ETF（5桁）


# ── 指標計算 ─────────────────────────────────────────────────
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().sort_values("Date").reset_index(drop=True)
    c = df["Close"]
    df["SMA20"]    = c.rolling(20).mean()
    df["SMA50"]    = c.rolling(50).mean()
    df["SMA200"]   = c.rolling(200).mean()
    df["VOL_MA20"] = df["Volume"].rolling(20).mean()

    delta  = c.diff()
    gain   = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
    loss   = (-delta).clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
    df["RSI14"] = 100 - 100 / (1 + gain / loss.replace(0, 1e-9))

    prev_c = c.shift(1)
    tr = pd.concat([df["High"]-df["Low"],
                    (df["High"]-prev_c).abs(),
                    (df["Low"] -prev_c).abs()], axis=1).max(axis=1)
    df["ATR14"] = tr.ewm(alpha=1/14, adjust=False).mean()

    ph = df["High"].shift(1); pl = df["Low"].shift(1)
    dm_p = np.where((df["High"]-ph > 0) & (df["High"]-ph > pl-df["Low"]),
                    df["High"]-ph, 0.0)
    dm_m = np.where((pl-df["Low"]  > 0) & (pl-df["Low"] > df["High"]-ph),
                    pl-df["Low"], 0.0)
    atr_s = df["ATR14"].replace(0, np.nan)
    di_p  = 100 * pd.Series(dm_p).ewm(alpha=1/14, adjust=False).mean() / atr_s
    di_m  = 100 * pd.Series(dm_m).ewm(alpha=1/14, adjust=False).mean() / atr_s
    dx    = 100*(di_p-di_m).abs()/(di_p+di_m).replace(0, np.nan)
    df["ADX14"] = dx.ewm(alpha=1/14, adjust=False).mean()
    return df


# ── 戦略Aシグナル生成 ────────────────────────────────────────
def signal_a(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    rsi3 = df["RSI14"].shift(3)
    df["signal_A"] = (
        (df["SMA20"] > df["SMA50"]) &
        (df["Close"] > df["SMA200"]) &
        (df["RSI14"] >= 45) & (df["RSI14"] <= 75) &
        (df["RSI14"] > rsi3) &
        (df["Volume"] >= df["VOL_MA20"] * 1.2) &
        (df["Close"] > df["Open"]) &
        (df["ADX14"] > 15)
    ).astype(int)
    return df


# ── 戦略Eシグナル生成 ────────────────────────────────────────
def signal_e(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    high_52w = df["Close"].shift(1).rolling(252, min_periods=200).max()
    df["signal_E"] = (
        (df["Close"] > high_52w) &
        (df["Volume"] >= df["VOL_MA20"] * 1.2) &
        (df["Close"] > df["SMA200"]) &
        (df["RSI14"] >= 50) & (df["RSI14"] <= 80) &
        (df["Close"] > df["Open"])
    ).astype(int)
    return df


# ── 戦略Fシグナル生成 ────────────────────────────────────────
def build_f_events(fins_path: Path) -> dict:
    """EPS成長20%以上の通期決算開示日を返す {code5: [DiscDate]}"""
    fins = pd.read_parquet(fins_path)
    fins["DiscDate"] = pd.to_datetime(fins["DiscDate"], errors="coerce")
    ann = fins[fins["CurPerType"] == "FY"].copy()
    for col in ["EPS", "NP"]:
        ann[col] = pd.to_numeric(ann[col], errors="coerce")
    ann = ann.sort_values(["Code", "DiscDate"]).reset_index(drop=True)
    ann["eps_prev"] = ann.groupby("Code")["EPS"].shift(1)
    valid = (ann["EPS"] > 0) & (ann["eps_prev"] > 0)
    ann["eps_growth"] = 0.0
    ann.loc[valid, "eps_growth"] = (ann.loc[valid,"EPS"]-ann.loc[valid,"eps_prev"])/ann.loc[valid,"eps_prev"]
    events = ann[(ann["eps_growth"]>=0.20)&(ann["NP"]>0)].copy()
    events["Code5"] = events["Code"].astype(str).str.strip().str[:4] + "0"
    result: dict = {}
    for _, row in events.iterrows():
        c = row["Code5"]
        result.setdefault(c, []).append(row["DiscDate"])
    return result


def add_f_flag(df: pd.DataFrame, events_for_code: list, window: int = 7) -> pd.DataFrame:
    df = df.copy()
    df["earnings_flag"] = False
    for disc in events_for_code:
        mask = (df["Date"] > disc) & (df["Date"] <= disc + pd.Timedelta(days=window))
        df.loc[mask, "earnings_flag"] = True
    return df


def signal_f(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "earnings_flag" not in df.columns:
        df["signal_F"] = 0
        return df
    df["signal_F"] = (
        (df["earnings_flag"]) &
        (df["Close"] > df["SMA200"]) &
        (df["RSI14"] >= 45) & (df["RSI14"] <= 70) &
        (df["Volume"] >= df["VOL_MA20"] * 1.2) &
        (df["Close"] > df["Open"])
    ).astype(int)
    return df


# ── 市場レジーム判定 ──────────────────────────────────────────
def build_regime(df_regime: pd.DataFrame) -> pd.DataFrame:
    df = df_regime.copy().sort_values("Date").reset_index(drop=True)
    df["SMA50"]  = df["Close"].rolling(50).mean()
    df["SMA200"] = df["Close"].rolling(200).mean()
    df["regime"] = "BEAR"
    df.loc[df["Close"] > df["SMA200"], "regime"] = "NEUTRAL"
    df.loc[df["Close"] > df["SMA50"],  "regime"] = "BULL"
    return df.set_index("Date")["regime"]


# ── ポジション管理 ────────────────────────────────────────────
@dataclass
class Position:
    code:        str
    strategy:    str
    entry_date:  pd.Timestamp
    entry_price: float
    shares:      int
    stop_loss:   float
    take_profit: float
    max_hold:    int
    hold_days:   int = 0
    rsi:         float = 0.0


@dataclass
class Trade:
    code:        str
    strategy:    str
    entry_date:  pd.Timestamp
    exit_date:   pd.Timestamp
    entry_price: float
    exit_price:  float
    shares:      int
    pnl:         float
    reason:      str


# ── ポートフォリオシミュレーション ───────────────────────────
def run_portfolio(stock_signals: dict, regime_s: pd.Series) -> tuple:
    """
    stock_signals: {code5: DataFrame with Date, Open, High, Low, Close,
                            signal_A, signal_E, signal_F, RSI14, ATR14}
    """
    all_dates = sorted({d for df in stock_signals.values()
                        for d in df["Date"] if OOS_START <= d <= OOS_END})

    lookup = {code: df.drop_duplicates("Date").set_index("Date").to_dict("index")
              for code, df in stock_signals.items()}

    capital   = float(INITIAL_CAPITAL)
    positions: list[Position] = []
    trades:    list[Trade]    = []
    equity    = {OOS_START: capital}

    cur_month = None; month_start = capital; month_stopped = False; stop_cnt = 0

    for today in all_dates:
        regime = regime_s.get(today, "NEUTRAL")

        # 月次リセット
        ym = (today.year, today.month)
        if ym != cur_month:
            cur_month = ym; month_start = capital; month_stopped = False

        # 決済判定
        next_pos = []
        for pos in positions:
            row = lookup[pos.code].get(today)
            if row is None:
                next_pos.append(pos); continue
            pos.hold_days += 1
            hi, lo, cl = row["High"], row["Low"], row["Close"]
            ep = er = None
            if lo <= pos.stop_loss:
                ep, er = pos.stop_loss, "損切り"
            elif hi >= pos.take_profit:
                ep, er = pos.take_profit, "利確"
            elif pos.hold_days >= pos.max_hold:
                ep, er = cl, "期間満了"
            if ep is not None:
                cost = (pos.entry_price + ep) * pos.shares * COST_PER_LEG
                pnl  = (ep - pos.entry_price) * pos.shares - cost
                capital += pnl
                trades.append(Trade(pos.code, pos.strategy, pos.entry_date, today,
                                    pos.entry_price, ep, pos.shares, pnl, er))
            else:
                next_pos.append(pos)
        positions = next_pos

        # 月次ストップ
        if not month_stopped and month_start > 0:
            if (capital - month_start) / month_start * 100 <= -10.0:
                month_stopped = True; stop_cnt += 1

        # エントリー
        slots = MAX_POSITIONS - len(positions)
        if slots > 0 and not month_stopped and regime != "BEAR":
            holding = {p.code for p in positions}
            signals = []
            for code, df in stock_signals.items():
                if code in holding:
                    continue
                rows = df[df["Date"] < today]
                if rows.empty:
                    continue
                prev = lookup[code].get(rows["Date"].iloc[-1])
                if prev is None:
                    continue
                today_row = lookup[code].get(today)
                if today_row is None:
                    continue

                ep  = today_row["Open"]
                gap = (ep - prev["Close"]) / prev["Close"] if prev["Close"] else 0
                if gap < -0.015:
                    continue
                atr = prev.get("ATR14", 0)
                if not atr or pd.isna(atr):
                    continue
                rsi = prev.get("RSI14", 0)

                # 戦略別シグナル確認・レジーム適用
                for strat in ["A", "E", "F"]:
                    if regime == "NEUTRAL" and strat != "F":
                        continue
                    sig_col = f"signal_{strat}"
                    if prev.get(sig_col, 0) == 1:
                        cfg = STRATEGY_CFG[strat]
                        tp  = ep + atr * cfg["tp"]
                        sl  = ep - atr * cfg["sl"]
                        sh  = min(int(capital * MAX_POS_RATIO / ep),
                                  int(INITIAL_CAPITAL * 0.02 / (atr * cfg["sl"])))
                        sh  = max(sh, 0)
                        if sh > 0 and ep * sh <= capital:
                            signals.append((rsi, code, strat, ep, sl, tp, sh, cfg["hold"]))

            signals.sort(key=lambda x: x[0], reverse=True)
            seen = set()
            for rsi, code, strat, ep, sl, tp, sh, hold in signals[:slots*3]:
                if code in seen or len(positions) + len([p for p in positions]) >= MAX_POSITIONS:
                    continue
                if len([p for p in positions if p.code == code]) > 0:
                    continue
                seen.add(code)
                positions.append(Position(code, strat, today, ep, sh, sl, tp, hold, rsi=rsi))
                slots -= 1
                if slots <= 0:
                    break

        equity[today] = capital

    eq_s = pd.Series(equity).sort_index()
    print(f"  月次ストップ発動: {stop_cnt}回")
    return trades, eq_s


# ── 結果集計 ──────────────────────────────────────────────────
def compute_stats(trades: list, equity: pd.Series) -> dict:
    total    = len(trades)
    wins     = sum(1 for t in trades if t.pnl > 0)
    win_rate = wins / total * 100 if total else 0
    years    = (OOS_END - OOS_START).days / 365
    cagr     = ((equity.iloc[-1] / INITIAL_CAPITAL)**(1/years) - 1) * 100 if years > 0 else 0
    peak     = equity.cummax()
    max_dd   = ((equity - peak) / peak * 100).min()
    total_cost = sum((t.entry_price+t.exit_price)*t.shares*COST_PER_LEG for t in trades)
    return {"total": total, "wins": wins, "win_rate": win_rate,
            "cagr": cagr, "max_dd": max_dd, "final": equity.iloc[-1],
            "total_cost": total_cost}


# ── メイン ────────────────────────────────────────────────────
def main():
    print("=" * 68)
    print("  合算バックテスト：戦略A v2 + 戦略E v2 + 戦略F")
    print(f"  OOS期間: {OOS_START.date()} ～ {OOS_END.date()}")
    print("=" * 68)

    # 候補銘柄読み込み
    df_a = pd.read_csv(CAND_A_PATH)
    codes_a = set((df_a["code"].astype(str).str.zfill(4) + "0").tolist())
    df_e = pd.read_csv(CAND_E_PATH)
    col_e = "Code" if "Code" in df_e.columns else "code"
    codes_e = set((df_e[col_e].astype(str).str.zfill(4) + "0").tolist())
    print(f"\n候補銘柄: A={len(codes_a)}銘柄  E={len(codes_e)}銘柄")

    # 戦略F イベント読み込み
    print("戦略Fイベント（EPS成長20%以上）読み込み中...")
    f_events = build_f_events(FINS_PATH)
    # OOS期間中に決算イベントがある銘柄のみ
    codes_f = set()
    for code, dates in f_events.items():
        if any(OOS_START <= d <= OOS_END + pd.Timedelta(days=30) for d in dates):
            codes_f.add(code)
    print(f"  戦略F対象: {len(codes_f)}銘柄（OOS期間中にEPSイベントあり）")

    # 全対象銘柄
    all_codes = codes_a | codes_e | codes_f | {REGIME_CODE}
    print(f"  合計ユニーク銘柄: {len(all_codes)}銘柄（レジーム用1306含む）")

    # 価格データ読み込み
    print("\n価格データ読み込み中...")
    t0 = time.time()
    df_all = pd.read_parquet(DATA_PATH)
    df_all["Date"] = pd.to_datetime(df_all["Date"])
    df_all["Code"] = df_all["Code4"].astype(str).str.zfill(4) + "0"
    df_all = df_all[df_all["Code"].isin(all_codes)].copy()
    df_all = df_all.sort_values(["Code", "Date"]).reset_index(drop=True)
    print(f"  読み込み完了: {len(df_all):,}行  {df_all['Code'].nunique()}銘柄  ({time.time()-t0:.1f}秒)")

    # レジーム系列構築
    print("\n市場レジーム構築中...")
    df_reg = df_all[df_all["Code"] == REGIME_CODE].copy()
    if df_reg.empty:
        print("  警告: 1306データなし → 全期間BULLとして処理")
        all_biz = pd.bdate_range(WARMUP, OOS_END)
        regime_s = pd.Series("BULL", index=all_biz)
    else:
        regime_s = build_regime(df_reg)
    oos_regime = regime_s[(regime_s.index >= OOS_START) & (regime_s.index <= OOS_END)]
    bull_cnt    = (oos_regime == "BULL").sum()
    neutral_cnt = (oos_regime == "NEUTRAL").sum()
    bear_cnt    = (oos_regime == "BEAR").sum()
    print(f"  OOS期間レジーム: BULL={bull_cnt}日 NEUTRAL={neutral_cnt}日 BEAR={bear_cnt}日")

    # 銘柄別シグナル生成
    print("\n指標・シグナル計算中...")
    target_codes = codes_a | codes_e | codes_f
    stock_signals: dict = {}
    t1 = time.time()
    for i, code in enumerate(sorted(target_codes)):
        df_s = df_all[(df_all["Code"] == code) & (df_all["Date"] >= WARMUP)].copy()
        if len(df_s) < 260:
            continue
        df_s = add_indicators(df_s)

        # 各戦略のシグナル列を初期化
        df_s["signal_A"] = 0
        df_s["signal_E"] = 0
        df_s["signal_F"] = 0

        if code in codes_a:
            df_s = signal_a(df_s)
        if code in codes_e:
            df_s = signal_e(df_s)
        if code in codes_f:
            df_s = add_f_flag(df_s, f_events.get(code, []))
            df_s = signal_f(df_s)

        stock_signals[code] = df_s

        if (i+1) % 50 == 0:
            print(f"  {i+1}/{len(target_codes)}銘柄処理済 ({time.time()-t1:.1f}秒)")

    print(f"  完了: {len(stock_signals)}銘柄  ({time.time()-t1:.1f}秒)")

    # シグナル数確認
    sig_a = sum(df["signal_A"].sum() for df in stock_signals.values())
    sig_e = sum(df["signal_E"].sum() for df in stock_signals.values())
    sig_f = sum(df["signal_F"].sum() for df in stock_signals.values())
    print(f"\nシグナル数（全期間）: A={int(sig_a)}  E={int(sig_e)}  F={int(sig_f)}")

    # ポートフォリオシミュレーション
    print(f"\nOOSポートフォリオシミュレーション実行中...")
    t2 = time.time()
    trades, equity = run_portfolio(stock_signals, regime_s)
    print(f"  完了: {time.time()-t2:.1f}秒")

    # 結果集計
    stats = compute_stats(trades, equity)

    # 戦略別内訳
    by_strat: dict = {}
    for t in trades:
        by_strat.setdefault(t.strategy, []).append(t)

    # ── 結果表示 ──────────────────────────────────────────────
    print()
    print("=" * 68)
    print("  ★ 合算OOSバックテスト結果 ★")
    print("=" * 68)
    print(f"\n  初期資金   : {INITIAL_CAPITAL:>12,} 円")
    print(f"  最終資産   : {stats['final']:>12,.0f} 円")
    print(f"  総損益     : {stats['final']-INITIAL_CAPITAL:>+12,.0f} 円")
    print(f"  年利(CAGR) : {stats['cagr']:>+11.2f} %  ← 目標15%")
    print(f"  最大DD     : {stats['max_dd']:>11.2f} %  ← 上限-15%")
    print(f"  総取引数   : {stats['total']:>12} 回")
    print(f"  勝率       : {stats['win_rate']:>11.1f} %")
    print(f"  総コスト   : {stats['total_cost']:>12,.0f} 円")

    print(f"\n【戦略別内訳】")
    print(f"  {'戦略':4}  {'取引':>5}  {'勝率':>6}  {'損益':>12}  {'年利目安':>8}")
    print("  " + "-" * 45)
    years = (OOS_END - OOS_START).days / 365
    for strat in ["A", "E", "F"]:
        ts = by_strat.get(strat, [])
        if not ts:
            print(f"  {strat:4}  {'0':>5}  {'---':>6}  {'---':>12}  {'---':>8}")
            continue
        wins_s = sum(1 for t in ts if t.pnl > 0)
        wr_s   = wins_s / len(ts) * 100
        pnl_s  = sum(t.pnl for t in ts)
        ann_s  = pnl_s / INITIAL_CAPITAL / years * 100
        print(f"  {strat:4}  {len(ts):>5}  {wr_s:>5.1f}%  {pnl_s:>+12,.0f}円  {ann_s:>+7.2f}%")

    print(f"\n【月別損益（OOS期間）】")
    print(f"  {'年月':8}  {'損益':>12}  {'累計資産':>13}  レジーム")
    print("  " + "-" * 50)
    eq_m = equity.resample("ME").last()
    prev_cap = INITIAL_CAPITAL
    for ym, cap in eq_m.items():
        if ym < OOS_START:
            continue
        pm   = cap - prev_cap
        reg  = regime_s.get(ym, "?")
        sign = "▲" if pm < 0 else "+"
        mark = " ◀" if pm < 0 else ""
        print(f"  {ym.strftime('%Y-%m'):8}  {sign}{abs(pm):>10,.0f}円  {cap:>12,.0f}円  {reg}{mark}")
        prev_cap = cap

    print()
    print("=" * 68)
    verdict = "✅ 目標達成圏内" if stats['cagr'] >= 15 else \
              "⚠️  目標未達（追加研究が必要）" if stats['cagr'] >= 10 else \
              "❌ 目標に大きく届かない"
    print(f"  判定: {verdict}  (CAGR {stats['cagr']:+.2f}%)")
    print("=" * 68)


if __name__ == "__main__":
    main()
