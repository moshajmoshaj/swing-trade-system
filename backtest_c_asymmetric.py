"""
backtest_c_asymmetric.py
戦略C：非対称TP/SL設計の検証

仮説:
  TP=SL=1.5（現行）→ 勝率50%必要 → 実測21-44%で機能しない
  TP×4 / SL×1.5   → 勝率27%で損益分岐 → 実測で黒字化の可能性

テスト設計:
  既存35候補 × {現行: TP2.5/SL1.5/7日, 非対称: TP4.0/SL1.5/10日} を比較
  全時代（2008-2026）で検証
"""
import sys, time
sys.stdout.reconfigure(encoding="utf-8")

from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent / "src"))
from indicators  import add_indicators
from strategy_c  import generate_signals as gen_c

DATA_PATH = Path("data/raw/prices_20y.parquet")

INITIAL_CAPITAL = 1_000_000
MAX_POSITIONS   = 5
MAX_POS_SIZE    = 200_000
COST_LEG        = 0.00055 + 0.00050

ERAS = {
    "Pre-IS (2008-2015)": ("2008-05-07", "2015-12-31"),
    "IS     (2016-2020)": ("2016-04-01", "2020-12-31"),
    "Gap    (2021-2022)": ("2021-01-01", "2022-12-31"),
    "OOS    (2023-2026)": ("2023-01-01", "2026-05-01"),
}

VARIANTS = {
    "現行 (TP×2.5/SL×1.5/7日)":   {"tp": 2.5, "sl": 1.5, "hold": 7},
    "非対称(TP×4.0/SL×1.5/10日)": {"tp": 4.0, "sl": 1.5, "hold": 10},
    "非対称(TP×6.0/SL×1.5/10日)": {"tp": 6.0, "sl": 1.5, "hold": 10},
}


def run_portfolio(stock_data, era_start, era_end, tp, sl, hold):
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
            elif pos["hold"] >= hold:
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
            cands.sort(key=lambda x: x[2])  # RSI低い順
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
                    "tp": ep + atr * tp,
                    "sl": ep - atr * sl,
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
    peak   = np.maximum.accumulate(eq)
    dd     = float(((eq - peak) / peak).min()) * 100
    wins   = sum(1 for t in trades if t["win"])
    return {
        "trades": len(trades), "wins": wins,
        "win_pct": wins / len(trades) * 100,
        "cagr": cagr, "max_dd": dd,
    }


def fmt(v):
    return f"{'+' if v >= 0 else ''}{v:.1f}%"


def main():
    print("=" * 70)
    print("  戦略C 非対称TP/SL検証（既存35候補・20年）")
    print("=" * 70)
    t0 = time.time()

    print("\n価格データ読み込み中...")
    prices_all = pd.read_parquet(DATA_PATH)
    prices_all["Date"] = pd.to_datetime(prices_all["Date"])
    prices_all["Code"] = prices_all["Code"].astype(str).str.strip()
    if "Code4" in prices_all.columns:
        mask = prices_all["Code"].str.len() == 4
        prices_all.loc[mask, "Code"] += "0"

    # 既存候補読み込み
    orig_csv = pd.read_csv("logs/strategy_c_candidates.csv", dtype=str)
    orig_col = next(c for c in orig_csv.columns if "code" in c.lower())
    codes    = [c.zfill(4) + "0" for c in orig_csv[orig_col]]
    print(f"  候補: {len(codes)}銘柄")

    # 指標・シグナル計算
    print("  指標・シグナル計算中...")
    stock_data = {}
    for code in codes:
        sub = prices_all[prices_all["Code"] == code].copy()
        if len(sub) < 300:
            continue
        sub = sub.sort_values("Date").reset_index(drop=True)
        sub = add_indicators(sub)
        sub = gen_c(sub)
        stock_data[code] = sub
    print(f"  {len(stock_data)}銘柄 準備完了")

    # 損益分岐点の理論確認
    print("\n  損益分岐点（勝率）:")
    for vname, vcfg in VARIANTS.items():
        r = vcfg["tp"] / (vcfg["tp"] + vcfg["sl"])
        print(f"  {vname:<30} 勝率≥{r*100:.1f}%で黒字")

    # 各バリアント × 各時代でポートフォリオ検証
    all_results = {}
    for vname, vcfg in VARIANTS.items():
        print(f"\n  【{vname}】")
        era_results = {}
        for era, (s, e) in ERAS.items():
            r = run_portfolio(stock_data, s, e,
                              vcfg["tp"], vcfg["sl"], vcfg["hold"])
            era_results[era] = r
            wp = r["win_pct"] if r["trades"] > 0 else 0
            print(f"  {era}  取引:{r['trades']:4d}  勝率:{wp:5.1f}%  "
                  f"CAGR:{fmt(r['cagr'])}  MaxDD:{fmt(r['max_dd'])}")
        all_results[vname] = era_results

    # サマリーテーブル
    print(f"\n{'='*70}")
    print("  ▼ CAGR比較サマリー")
    print(f"{'='*70}")
    vnames = list(VARIANTS.keys())
    header = f"  {'時代':<22}" + "".join(f"{n[:14]:>16}" for n in vnames)
    print(header)
    print("  " + "─" * (22 + 16 * len(vnames)))
    for era in ERAS:
        row = f"  {era:<22}"
        for vname in vnames:
            r = all_results[vname].get(era, {})
            row += f"{fmt(r.get('cagr', 0)):>16}"
        print(row)

    print(f"\n  MaxDD比較サマリー")
    print(f"  {'時代':<22}" + "".join(f"{n[:14]:>16}" for n in vnames))
    print("  " + "─" * (22 + 16 * len(vnames)))
    for era in ERAS:
        row = f"  {era:<22}"
        for vname in vnames:
            r = all_results[vname].get(era, {})
            row += f"{fmt(r.get('max_dd', 0)):>16}"
        print(row)

    print(f"\n  総所要時間: {time.time()-t0:.0f}秒")


if __name__ == "__main__":
    main()
