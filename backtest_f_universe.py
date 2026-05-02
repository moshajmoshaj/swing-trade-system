"""
backtest_f_universe.py
戦略F ユニバース版：IS銘柄選定なしで全プライム株に適用

PAEDは特定銘柄ではなく「条件を満たす全銘柄」に統計的エッジがある。
IS銘柄選定は過適合リスクが高いため、ユニバース全体で検証する。

出力:
  時代別 CAGR / MaxDD / 勝率 / 取引数
  戦略A（参考値）との比較
"""
import sys, time
sys.stdout.reconfigure(encoding="utf-8")

from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent / "src"))
from indicators import add_indicators
from strategy_f import build_earnings_events, add_earnings_flag, generate_signals

DATA_PATH = Path("data/raw/prices_20y.parquet")
FINS_PATH = Path("data/raw/fins_summary.parquet")

ERAS = {
    "Pre-IS (2008-2015)": ("2008-05-07", "2015-12-31"),
    "IS     (2016-2020)": ("2016-04-01", "2020-12-31"),
    "Gap    (2021-2022)": ("2021-01-01", "2022-12-31"),
    "OOS    (2023-2026)": ("2023-01-01", "2026-05-01"),
}

INITIAL_CAPITAL = 1_000_000
MAX_POSITIONS   = 5
MAX_POS_SIZE    = 200_000
COST_LEG        = 0.00055 + 0.00050
TP_MULT         = 5.0
SL_MULT         = 2.0
MAX_HOLD        = 15


def run_portfolio_universe(stock_data: dict, era_start: str, era_end: str) -> dict:
    t_start = pd.Timestamp(era_start)
    t_end   = pd.Timestamp(era_end)

    # 全銘柄の日付統合
    all_dates = sorted(set(
        d for df in stock_data.values()
        for d in df.loc[(df["Date"] >= t_start) & (df["Date"] <= t_end), "Date"]
    ))
    if not all_dates:
        return {"trades": 0, "wins": 0, "cagr": 0.0, "max_dd": 0.0, "win_pct": 0.0}

    lookup = {c: {r["Date"]: r for _, r in df.iterrows()}
              for c, df in stock_data.items()}

    capital   = float(INITIAL_CAPITAL)
    positions = {}
    trades    = []
    equity    = [capital]

    for date in all_dates:
        # 決済
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

        # エントリー（RSI高い順・決算モメンタム銘柄を優先）
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
        return {"trades": 0, "wins": 0, "cagr": 0.0, "max_dd": 0.0, "win_pct": 0.0}

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
    print("  戦略F ユニバース版（IS選定なし・全プライム株適用）")
    print("=" * 65)
    t0 = time.time()

    print("\n決算イベント構築中...")
    events = build_earnings_events(str(FINS_PATH))
    qualifying_codes = set(events["Code"].tolist())
    print(f"  イベント: {len(events):,}件  対象銘柄: {len(qualifying_codes):,}")

    print("\n価格データ読み込み中...")
    prices_all = pd.read_parquet(DATA_PATH)
    prices_all["Date"] = pd.to_datetime(prices_all["Date"])
    prices_all["Code"] = prices_all["Code"].astype(str).str.strip()
    if "Code4" in prices_all.columns:
        mask = prices_all["Code"].str.len() == 4
        prices_all.loc[mask, "Code"] += "0"

    # 決算イベントがある銘柄のみに絞る（計算コスト削減）
    target_codes = [c for c in prices_all["Code"].unique() if c in qualifying_codes]
    print(f"  対象: {len(target_codes):,}銘柄（決算イベントあり）")

    print("\n指標・シグナル計算中（数分かかります）...")
    stock_data = {}
    for i, code in enumerate(target_codes):
        sub = prices_all[prices_all["Code"] == code].copy()
        if len(sub) < 300:
            continue
        sub = sub.sort_values("Date").reset_index(drop=True)
        sub = add_indicators(sub)
        sub = add_earnings_flag(sub, events)
        sub = generate_signals(sub)

        # シグナルが1件以上ある銘柄のみ保持（メモリ節約）
        if sub["signal"].sum() > 0:
            stock_data[code] = sub

        if (i + 1) % 200 == 0:
            print(f"  [{i+1}/{len(target_codes)}]  "
                  f"シグナルあり: {len(stock_data)}銘柄  {time.time()-t0:.0f}秒")

    print(f"  シグナル保有銘柄: {len(stock_data):,}銘柄")

    # シグナル発生回数の統計
    total_signals = sum(df["signal"].sum() for df in stock_data.values())
    print(f"  総シグナル数: {total_signals:,}件")

    print("\n時代別ポートフォリオ検証中...")
    era_results = {}
    for era, (s, e) in ERAS.items():
        r = run_portfolio_universe(stock_data, s, e)
        era_results[era] = r
        wp = r["win_pct"]
        print(f"  {era}  取引:{r['trades']:5d}  勝率:{wp:5.1f}%  "
              f"CAGR:{fmt(r['cagr'])}  MaxDD:{fmt(r['max_dd'])}")

    # 戦略A/E との比較
    ref_a = {
        "Pre-IS (2008-2015)": +11.4,
        "IS     (2016-2020)": None,
        "Gap    (2021-2022)": +11.2,
        "OOS    (2023-2026)": +15.3,
    }
    ref_e = {
        "Pre-IS (2008-2015)": +6.4,
        "IS     (2016-2020)": +10.8,
        "Gap    (2021-2022)": +2.3,
        "OOS    (2023-2026)": +5.3,
    }

    print(f"\n{'='*65}")
    print("  ▼ CAGR比較（戦略A・E・F）")
    print(f"{'='*65}")
    print(f"  {'時代':<22} {'戦略A':>10} {'戦略E':>10} {'戦略F':>10}")
    print("  " + "─" * 54)
    for era in ERAS:
        a = ref_a.get(era)
        e = ref_e.get(era)
        f = era_results[era]["cagr"]
        a_s = f"{'+' if a and a >= 0 else ''}{a:.1f}%" if a else "  (nan)"
        e_s = f"{'+' if e and e >= 0 else ''}{e:.1f}%"
        print(f"  {era:<22} {a_s:>10} {e_s:>10} {fmt(f):>10}")

    print(f"\n  取引数・勝率:")
    for era in ERAS:
        r = era_results[era]
        print(f"  {era}  取引:{r['trades']:5d}  勝率:{r['win_pct']:5.1f}%")

    print(f"\n  総所要時間: {time.time()-t0:.0f}秒")
    print("  Phase 4には影響しません。")


if __name__ == "__main__":
    main()
