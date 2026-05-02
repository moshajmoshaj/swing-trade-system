"""
backtest_c_v2.py
戦略C v2（底打ち確認型）の20年IS/OOSバックテスト

比較：
  現行C  : RSI 30-40帯エントリー・TP=ATR×2.5・SL=ATR×1.5・7日
  v2     : RSI 35上抜けエントリー・TP=SMA20・SL=ATR×2.0・10日

出力:
  - 時代別成績比較（現行 vs v2）
  - OOS期間の詳細メトリクス
  - logs/strategy_c_v2_candidates.csv（IS選定済み候補）
"""
import sys, time
sys.stdout.reconfigure(encoding="utf-8")

from pathlib import Path
from scipy.stats import binom
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent / "src"))
from indicators     import add_indicators
from strategy_c     import generate_signals as gen_c_orig
from strategy_c_v2  import generate_signals as gen_c_v2

# ── 設定 ────────────────────────────────────────────────────
DATA_PATH  = Path("data/raw/prices_20y.parquet")
OUT_CSV    = Path("logs/strategy_c_v2_candidates.csv")

IS_START   = pd.Timestamp("2016-04-01")
IS_END     = pd.Timestamp("2020-12-31")
OOS_WARMUP = pd.Timestamp("2022-01-01")
OOS_START  = pd.Timestamp("2023-01-01")
OOS_END    = pd.Timestamp("2026-05-01")

# IS選定基準
MIN_TRADES   = 5        # 逆張りはシグナル頻度が低い → 基準を緩和
MIN_WIN_RATE = 55.0
MAX_DD       = -5.0
MIN_PNL      = 10_000
P_VALUE_MAX  = 0.15     # 逆張りは有意水準を少し緩和
TARGET_N     = 30

# バックテスト設定
INITIAL_CAPITAL = 1_000_000
MAX_POSITIONS   = 5
MAX_POS_SIZE    = 200_000
COMMISSION      = 0.00055
SLIPPAGE        = 0.00050
COST_LEG        = COMMISSION + SLIPPAGE

ERAS = {
    "Pre-IS (2008-2015)": ("2008-05-07", "2015-12-31"),
    "IS     (2016-2020)": ("2016-04-01", "2020-12-31"),
    "Gap    (2021-2022)": ("2021-01-01", "2022-12-31"),
    "OOS    (2023-2026)": ("2023-01-01", "2026-05-01"),
}


def run_single_bt(df: pd.DataFrame, bt_start: pd.Timestamp, bt_end: pd.Timestamp,
                  tp_mult: float, sl_mult: float, max_hold: int,
                  use_sma20_tp: bool = False) -> dict:
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
            elif hold_days >= max_hold:
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
            stop_loss   = ep - atr * sl_mult
            if use_sma20_tp and pd.notna(prev.get("SMA20")) and prev["SMA20"] > ep * 1.005:
                take_profit = float(prev["SMA20"])
            else:
                take_profit = ep + atr * tp_mult
            hold_days = 0
            in_trade  = True

    if not trades:
        return {}

    wins    = sum(1 for t in trades if t["win"])
    n       = len(trades)
    pnl     = sum(t["pnl"] for t in trades)
    p_val   = binom.sf(wins - 1, n, 0.5)
    equity  = INITIAL_CAPITAL
    peak    = INITIAL_CAPITAL
    max_dd  = 0.0
    for t in trades:
        equity += t["pnl"]
        peak    = max(peak, equity)
        dd      = (equity - peak) / peak * 100
        max_dd  = min(max_dd, dd)

    return {
        "trades": n, "wins": wins,
        "win_pct": wins / n * 100,
        "pnl": pnl, "max_dd": max_dd, "p_val": p_val,
    }


def run_portfolio(stock_data: dict, era_start: str, era_end: str,
                  tp_mult: float, sl_mult: float, max_hold: int,
                  use_sma20_tp: bool = False) -> dict:
    t_start = pd.Timestamp(era_start)
    t_end   = pd.Timestamp(era_end)

    all_dates = sorted(set(
        d for df in stock_data.values()
        for d in df.loc[(df["Date"] >= t_start) & (df["Date"] <= t_end), "Date"].tolist()
    ))
    if not all_dates:
        return {"trades": 0, "wins": 0, "cagr": 0.0, "max_dd": 0.0}

    lookup = {code: {r["Date"]: r for _, r in df.iterrows()}
              for code, df in stock_data.items()}

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
            elif pos["hold"] >= max_hold:
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
                rsi = row.get("RSI14")
                if pd.isna(atr) or atr <= 0:
                    continue
                cands.append((code, row, rsi if not pd.isna(rsi) else 0))
            cands.sort(key=lambda x: x[2])  # 逆張りはRSI低い順
            for code, row, _ in cands[:MAX_POSITIONS - len(positions)]:
                ep = row["Open"]
                if ep <= 0:
                    continue
                sh  = min(int(MAX_POS_SIZE / ep / 100) * 100, 100)
                if sh <= 0:
                    continue
                atr = float(row["ATR14"])
                if use_sma20_tp and pd.notna(row.get("SMA20")) and row["SMA20"] > ep * 1.005:
                    tp = float(row["SMA20"])
                else:
                    tp = ep + atr * tp_mult
                positions[code] = {
                    "ep": ep, "tp": tp,
                    "sl": ep - atr * sl_mult,
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

    return {"trades": len(trades), "wins": wins,
            "cagr": cagr, "max_dd": dd,
            "win_pct": wins / len(trades) * 100 if trades else 0}


def prepare(codes: list, prices_all: pd.DataFrame, gen_fn) -> dict:
    out = {}
    for code in codes:
        code5 = code + "0" if len(code) == 4 else code
        sub = prices_all[prices_all["Code"] == code5].copy()
        if len(sub) < 300:
            continue
        sub = sub.sort_values("Date").reset_index(drop=True)
        sub = add_indicators(sub)
        sub = gen_fn(sub)
        out[code5] = sub
    return out


def select_candidates(all_codes: list, prices_all: pd.DataFrame,
                      gen_fn, tp_mult: float, sl_mult: float,
                      max_hold: int, use_sma20_tp: bool) -> pd.DataFrame:
    results = []
    t0 = time.time()
    for i, code in enumerate(all_codes):
        code5 = code + "0" if len(code) == 4 else code
        sub = prices_all[prices_all["Code"] == code5].copy()
        if len(sub) < 300:
            continue
        sub = sub.sort_values("Date").reset_index(drop=True)
        sub = add_indicators(sub)
        sub = gen_fn(sub)
        r = run_single_bt(sub, IS_START, IS_END, tp_mult, sl_mult,
                          max_hold, use_sma20_tp)
        if r:
            r["code"] = code5
            results.append(r)
        if (i + 1) % 200 == 0:
            print(f"    [{i+1}/{len(all_codes)}]  {time.time()-t0:.0f}秒")

    if not results:
        return pd.DataFrame()

    df = pd.DataFrame(results)
    df = df[
        (df["trades"] >= MIN_TRADES) &
        (df["win_pct"] >= MIN_WIN_RATE) &
        (df["max_dd"] >= MAX_DD) &
        (df["pnl"] >= MIN_PNL) &
        (df["p_val"] <= P_VALUE_MAX)
    ]
    return df.sort_values("pnl", ascending=False).head(TARGET_N)


def fmt(v, is_pct=True):
    if is_pct:
        return f"{'+' if v >= 0 else ''}{v:.1f}%"
    return f"{v:.0f}"


def main():
    print("=" * 65)
    print("  戦略C v2 バックテスト（現行C vs 底打ち確認型 v2）")
    print("=" * 65)
    t0 = time.time()

    print("\n価格データ読み込み中...")
    prices_all = pd.read_parquet(DATA_PATH)
    prices_all["Date"] = pd.to_datetime(prices_all["Date"])
    prices_all["Code"] = prices_all["Code"].astype(str).str.strip()
    if "Code4" in prices_all.columns:
        mask = prices_all["Code"].str.len() == 4
        prices_all.loc[mask, "Code"] += "0"
    all_codes = prices_all["Code"].unique().tolist()
    print(f"  {len(all_codes):,}銘柄")

    # ── 既存候補で現行C の成績確認 ────────────────────────────
    print("\n【現行C】既存35候補で時代別ポートフォリオ検証中...")
    orig_csv = pd.read_csv("logs/strategy_c_candidates.csv", dtype=str)
    orig_col = next(c for c in orig_csv.columns if "code" in c.lower())
    orig_codes = [c.zfill(4) + "0" for c in orig_csv[orig_col]]
    orig_data = prepare(orig_codes, prices_all, gen_c_orig)

    print("\n  時代別成績（現行C）:")
    orig_era = {}
    for era, (s, e) in ERAS.items():
        r = run_portfolio(orig_data, s, e, 2.5, 1.5, 7, use_sma20_tp=True)
        orig_era[era] = r
        wp = r["wins"] / r["trades"] * 100 if r["trades"] > 0 else 0
        print(f"  {era}  取引:{r['trades']:4d}  勝率:{wp:5.1f}%  "
              f"CAGR:{fmt(r['cagr'])}  MaxDD:{fmt(r['max_dd'])}")

    # ── v2 IS選定 ─────────────────────────────────────────────
    print(f"\n【v2】全{len(all_codes):,}銘柄でIS選定中...")
    print(f"  基準: 取引≥{MIN_TRADES} 勝率≥{MIN_WIN_RATE}% DD≥{MAX_DD}% "
          f"PnL≥{MIN_PNL:,} p値≤{P_VALUE_MAX}")
    v2_selected = select_candidates(
        all_codes, prices_all, gen_c_v2,
        tp_mult=3.0, sl_mult=2.0, max_hold=10, use_sma20_tp=True
    )
    print(f"  選定完了: {len(v2_selected)}銘柄")

    if v2_selected.empty:
        print("  ⚠ 候補なし。基準を見直してください。")
        return

    v2_selected.to_csv(OUT_CSV, index=False)
    print(f"  保存: {OUT_CSV}")

    # ── v2 全銘柄で時代別ポートフォリオ検証 ────────────────────
    print("\n【v2】選定候補で時代別ポートフォリオ検証中...")
    v2_codes = v2_selected["code"].tolist()
    v2_data  = prepare(v2_codes, prices_all, gen_c_v2)

    print("\n  時代別成績（v2）:")
    v2_era = {}
    for era, (s, e) in ERAS.items():
        r = run_portfolio(v2_data, s, e, 3.0, 2.0, 10, use_sma20_tp=True)
        v2_era[era] = r
        wp = r["wins"] / r["trades"] * 100 if r["trades"] > 0 else 0
        print(f"  {era}  取引:{r['trades']:4d}  勝率:{wp:5.1f}%  "
              f"CAGR:{fmt(r['cagr'])}  MaxDD:{fmt(r['max_dd'])}")

    # ── 比較サマリー ─────────────────────────────────────────
    print(f"\n{'='*65}")
    print("  ▼ 比較サマリー（CAGR）")
    print(f"{'='*65}")
    print(f"  {'時代':<22} {'現行C':>10} {'v2':>10} {'改善':>8}")
    print("  " + "─" * 52)
    for era in ERAS:
        c_orig = orig_era[era]["cagr"]
        c_v2   = v2_era[era]["cagr"]
        delta  = c_v2 - c_orig
        sign   = "↑" if delta > 0 else "↓" if delta < 0 else "─"
        print(f"  {era:<22} {fmt(c_orig):>10} {fmt(c_v2):>10} "
              f"{sign}{abs(delta):>5.1f}pt")

    print(f"\n  総所要時間: {time.time()-t0:.0f}秒")
    print(f"  v2候補: {OUT_CSV}  Phase 4には影響しません。")


if __name__ == "__main__":
    main()
