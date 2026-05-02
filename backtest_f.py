"""
backtest_f.py
戦略F（決算モメンタム・PEAD）の20年IS/OOSバックテスト

コンセプト：
  EPS成長20%以上の通期決算開示後にPEAD（Post-Earnings Announcement Drift）を捉える
  イベント駆動型で戦略A/Eとは独立したエッジ

設計：
  シグナル : EPS成長20%超 × SMA200+ × RSI45-70 × 出来高 × 陽線
  エグジット: TP=ATR×5, SL=ATR×2, 強制15日
  IS期間   : 2016-04-01 ~ 2020-12-31
  OOS期間  : 2023-01-01 ~ 2026-05-01

出力:
  logs/strategy_f_candidates.csv  IS選定済み候補（Phase 5用）
"""
import sys, time
sys.stdout.reconfigure(encoding="utf-8")

from pathlib import Path
from scipy.stats import binom
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent / "src"))
from indicators  import add_indicators
from strategy_f  import build_earnings_events, add_earnings_flag, generate_signals

# ── パス ────────────────────────────────────────────────────
DATA_PATH  = Path("data/raw/prices_20y.parquet")
FINS_PATH  = Path("data/raw/fins_summary.parquet")
OUT_CSV    = Path("logs/strategy_f_candidates.csv")

# ── 期間 ─────────────────────────────────────────────────────
IS_START   = pd.Timestamp("2016-04-01")
IS_END     = pd.Timestamp("2020-12-31")
OOS_WARMUP = pd.Timestamp("2022-01-01")
OOS_START  = pd.Timestamp("2023-01-01")
OOS_END    = pd.Timestamp("2026-05-01")

ERAS = {
    "Pre-IS (2008-2015)": ("2008-05-07", "2015-12-31"),
    "IS     (2016-2020)": ("2016-04-01", "2020-12-31"),
    "Gap    (2021-2022)": ("2021-01-01", "2022-12-31"),
    "OOS    (2023-2026)": ("2023-01-01", "2026-05-01"),
}

# ── IS選定基準 ────────────────────────────────────────────────
MIN_TRADES   = 3       # 決算年1回 → IS5年で最大5回
MIN_WIN_RATE = 55.0
MAX_DD       = -10.0
MIN_PNL      = 10_000
P_VALUE_MAX  = 0.20    # 少サンプルのため緩和
TARGET_N     = 30

# ── バックテスト設定 ─────────────────────────────────────────
INITIAL_CAPITAL = 1_000_000
MAX_POSITIONS   = 5
MAX_POS_SIZE    = 200_000
COST_LEG        = 0.00055 + 0.00050
TP_MULT         = 5.0
SL_MULT         = 2.0
MAX_HOLD        = 15


def run_single_bt(df: pd.DataFrame, bt_start: pd.Timestamp,
                  bt_end: pd.Timestamp) -> dict:
    df = df[df["Date"] <= bt_end].copy()
    capital  = float(INITIAL_CAPITAL)
    trades   = []
    in_trade = False

    for i in range(1, len(df)):
        row  = df.iloc[i]
        prev = df.iloc[i - 1]
        if row["Date"] <= bt_start:
            continue

        if in_trade:
            hold_days += 1
            hi, lo, cl = row["High"], row["Low"], row["Close"]
            ep = reason = None
            if lo <= stop_loss:
                ep, reason = stop_loss, "SL"
            elif hi >= take_profit:
                ep, reason = take_profit, "TP"
            elif hold_days >= MAX_HOLD:
                ep, reason = cl, "HOLD"
            if reason:
                cost = (entry_price + ep) * shares * COST_LEG
                pnl  = (ep - entry_price) * shares - cost
                capital += pnl
                trades.append({"pnl": pnl, "win": pnl > 0})
                in_trade = False

        if not in_trade and prev.get("signal", 0) == 1 and pd.notna(prev.get("ATR14")):
            ep = row["Open"]
            if ep <= 0:
                continue
            atr    = float(prev["ATR14"])
            shares = min(int(MAX_POS_SIZE / ep / 100) * 100, 100)
            if shares <= 0:
                continue
            entry_price = ep
            take_profit = ep + atr * TP_MULT
            stop_loss   = ep - atr * SL_MULT
            hold_days   = 0
            in_trade    = True

    if not trades:
        return {}

    wins   = sum(1 for t in trades if t["win"])
    n      = len(trades)
    pnl    = sum(t["pnl"] for t in trades)
    p_val  = binom.sf(wins - 1, n, 0.5)
    equity = INITIAL_CAPITAL
    peak   = INITIAL_CAPITAL
    max_dd = 0.0
    for t in trades:
        equity += t["pnl"]
        peak    = max(peak, equity)
        max_dd  = min(max_dd, (equity - peak) / peak * 100)

    return {
        "trades": n, "wins": wins,
        "win_pct": wins / n * 100,
        "pnl": pnl, "max_dd": max_dd, "p_val": p_val,
    }


def run_portfolio(stock_data: dict, era_start: str, era_end: str) -> dict:
    t_start = pd.Timestamp(era_start)
    t_end   = pd.Timestamp(era_end)

    all_dates = sorted(set(
        d for df in stock_data.values()
        for d in df.loc[(df["Date"] >= t_start) & (df["Date"] <= t_end), "Date"]
    ))
    if not all_dates:
        return {"trades": 0, "wins": 0, "cagr": 0.0, "max_dd": 0.0}

    lookup = {c: {r["Date"]: r for _, r in df.iterrows()}
              for c, df in stock_data.items()}

    capital   = float(INITIAL_CAPITAL)
    positions = {}
    trades    = []
    equity    = [capital]

    for date in all_dates:
        to_close = []
        for code, pos in positions.items():
            row = lookup[code].get(date)
            if row is None:
                continue
            pos["hold"] += 1
            hi, lo, cl = row["High"], row["Low"], row["Close"]
            ep = reason = None
            if lo <= pos["sl"]:
                ep, reason = pos["sl"], "SL"
            elif hi >= pos["tp"]:
                ep, reason = pos["tp"], "TP"
            elif pos["hold"] >= MAX_HOLD:
                ep, reason = cl, "HOLD"
            if reason:
                cost = (pos["ep"] + ep) * pos["sh"] * COST_LEG
                pnl  = (ep - pos["ep"]) * pos["sh"] - cost
                capital += pnl
                trades.append({"pnl": pnl, "win": pnl > 0})
                to_close.append(code)
        for c in to_close:
            del positions[c]

        if len(positions) < MAX_POSITIONS:
            cands = []
            for code, lk in lookup.items():
                if code in positions:
                    continue
                row = lk.get(date)
                if row is None or row.get("signal", 0) != 1:
                    continue
                atr = row.get("ATR14")
                rsi = row.get("RSI14", 50)
                if pd.isna(atr) or atr <= 0:
                    continue
                cands.append((code, row, rsi if not pd.isna(rsi) else 50))
            cands.sort(key=lambda x: x[2], reverse=True)
            for code, row, _ in cands[:MAX_POSITIONS - len(positions)]:
                ep = row["Open"]
                if ep <= 0:
                    continue
                sh  = min(int(MAX_POS_SIZE / ep / 100) * 100, 100)
                if sh <= 0:
                    continue
                atr = float(row["ATR14"])
                positions[code] = {
                    "ep": ep,
                    "tp": ep + atr * TP_MULT,
                    "sl": ep - atr * SL_MULT,
                    "sh": sh, "hold": 0,
                }
        equity.append(capital)

    if not trades:
        return {"trades": 0, "wins": 0, "cagr": 0.0, "max_dd": 0.0}

    eq    = np.array(equity)
    years = (t_end - t_start).days / 365.25
    try:
        cagr = ((capital / INITIAL_CAPITAL) ** (1 / years) - 1) * 100
    except Exception:
        cagr = 0.0
    peak  = np.maximum.accumulate(eq)
    dd    = float(((eq - peak) / peak).min()) * 100
    wins  = sum(1 for t in trades if t["win"])

    return {
        "trades": len(trades), "wins": wins,
        "win_pct": wins / len(trades) * 100 if trades else 0,
        "cagr": cagr, "max_dd": dd,
    }


def fmt(v):
    return f"{'+' if v >= 0 else ''}{v:.1f}%"


def main():
    print("=" * 65)
    print("  戦略F（決算モメンタム・PEAD）20年IS/OOSバックテスト")
    print("=" * 65)
    t0 = time.time()

    # ── 決算イベント構築 ─────────────────────────────────────
    print("\n決算イベント構築中（EPS成長20%以上・通期決算）...")
    events = build_earnings_events(str(FINS_PATH))
    print(f"  ポジティブ決算イベント: {len(events):,}件  "
          f"({events['Code'].nunique():,}銘柄)")
    print(f"  期間: {events['DiscDate'].min().date()} ~ "
          f"{events['DiscDate'].max().date()}")

    # ── 価格データ読み込み ───────────────────────────────────
    print("\n価格データ読み込み中...")
    prices_all = pd.read_parquet(DATA_PATH)
    prices_all["Date"] = pd.to_datetime(prices_all["Date"])
    prices_all["Code"] = prices_all["Code"].astype(str).str.strip()
    if "Code4" in prices_all.columns:
        mask = prices_all["Code"].str.len() == 4
        prices_all.loc[mask, "Code"] += "0"
    all_codes = prices_all["Code"].unique().tolist()
    print(f"  {len(all_codes):,}銘柄")

    # ── 全銘柄でIS単体バックテスト（候補選定） ───────────────
    print(f"\nIS選定中（{IS_START.date()} ~ {IS_END.date()}）...")
    print(f"  基準: 取引≥{MIN_TRADES} 勝率≥{MIN_WIN_RATE}% "
          f"DD≥{MAX_DD}% PnL≥{MIN_PNL:,} p値≤{P_VALUE_MAX}")

    results = []
    for i, code in enumerate(all_codes):
        sub = prices_all[prices_all["Code"] == code].copy()
        if len(sub) < 300:
            continue
        sub = sub.sort_values("Date").reset_index(drop=True)
        sub = add_indicators(sub)
        sub = add_earnings_flag(sub, events)
        sub = generate_signals(sub)

        r = run_single_bt(sub, IS_START, IS_END)
        if r:
            r["code"] = code
            results.append(r)

        if (i + 1) % 200 == 0:
            elapsed = time.time() - t0
            print(f"  [{i+1}/{len(all_codes)}]  {elapsed:.0f}秒")

    print(f"  IS結果あり: {len(results):,}銘柄")

    if not results:
        print("  ⚠ 結果なし")
        return

    df_res = pd.DataFrame(results)
    selected = df_res[
        (df_res["trades"] >= MIN_TRADES) &
        (df_res["win_pct"] >= MIN_WIN_RATE) &
        (df_res["max_dd"] >= MAX_DD) &
        (df_res["pnl"] >= MIN_PNL) &
        (df_res["p_val"] <= P_VALUE_MAX)
    ].sort_values("pnl", ascending=False).head(TARGET_N)

    print(f"  選定完了: {len(selected)}銘柄")

    if selected.empty:
        print("  ⚠ 候補なし")
        return

    selected.to_csv(OUT_CSV, index=False)
    print(f"  保存: {OUT_CSV}")

    # ── 選定候補で20年時代別ポートフォリオ検証 ──────────────
    print("\n選定候補で時代別ポートフォリオ検証中...")
    sel_codes = selected["code"].tolist()
    stock_data = {}
    for code in sel_codes:
        sub = prices_all[prices_all["Code"] == code].copy()
        if len(sub) < 300:
            continue
        sub = sub.sort_values("Date").reset_index(drop=True)
        sub = add_indicators(sub)
        sub = add_earnings_flag(sub, events)
        sub = generate_signals(sub)
        stock_data[code] = sub

    print(f"  {len(stock_data)}銘柄 準備完了\n")
    era_results = {}
    for era, (s, e) in ERAS.items():
        r = run_portfolio(stock_data, s, e)
        era_results[era] = r
        wp = r["win_pct"] if r["trades"] > 0 else 0
        print(f"  {era}  取引:{r['trades']:4d}  勝率:{wp:5.1f}%  "
              f"CAGR:{fmt(r['cagr'])}  MaxDD:{fmt(r['max_dd'])}")

    # ── 戦略A との比較サマリー ────────────────────────────────
    print(f"\n{'='*65}")
    print("  ▼ 戦略A(既存) vs 戦略F(新規) CAGR比較")
    print(f"{'='*65}")

    # 戦略AのOOSは20年バックテストで +15.3% / +11.4%(Pre-IS) の実績
    ref = {
        "Pre-IS (2008-2015)": +11.4,
        "IS     (2016-2020)": None,
        "Gap    (2021-2022)": +11.2,
        "OOS    (2023-2026)": +15.3,
    }
    print(f"  {'時代':<22} {'戦略A(参考)':>12} {'戦略F(新規)':>12}")
    print("  " + "─" * 48)
    for era in ERAS:
        a_val = ref.get(era)
        f_val = era_results[era]["cagr"]
        a_str = f"{'+' if a_val and a_val >= 0 else ''}{a_val:.1f}%" if a_val else "  (nan)"
        print(f"  {era:<22} {a_str:>12} {fmt(f_val):>12}")

    print(f"\n  総所要時間: {time.time()-t0:.0f}秒")
    print(f"  Phase 4には影響しません。候補: {OUT_CSV}")


if __name__ == "__main__":
    main()
