"""
optimize_hold_days_a.py
戦略A 保有日数最適化（20年データ）

目的:
  PDCAログ記録「保有日数を10日→7日に短縮したかった」を統計的に検証する。
  概算バックテストでは7日が最良だったが、20年ポートフォリオ検証で確認する。

テスト:
  保有上限: 5 / 7 / 10（現行） / 12 / 15 日
  期間: 全4時代（Pre-IS / IS / Gap / OOS）
  候補: 既存 final_candidates.csv（30銘柄・凍結設定）

注意:
  Phase 4は凍結中。本スクリプトの結果は Phase 4 終了後の判断材料とする。
"""
import sys, time
sys.stdout.reconfigure(encoding="utf-8")

from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent / "src"))
from indicators import add_indicators
from strategy   import generate_signals as gen_a

DATA_PATH = Path("data/raw/prices_20y.parquet")
CSV_PATH  = Path("logs/final_candidates.csv")

ERAS = {
    "Pre-IS (2008-2015)": ("2008-05-07", "2015-12-31"),
    "IS     (2016-2020)": ("2016-04-01", "2020-12-31"),
    "Gap    (2021-2022)": ("2021-01-01", "2022-12-31"),
    "OOS    (2023-2026)": ("2023-01-01", "2026-05-01"),
}

HOLD_DAYS_LIST = [5, 7, 10, 12, 15]  # 現行は10日

INITIAL_CAPITAL = 1_000_000
MAX_POSITIONS   = 5
MAX_POS_SIZE    = 200_000
COST_LEG        = 0.00055 + 0.00050
TP_MULT         = 6.0   # 設計書準拠（凍結値）
SL_MULT         = 2.0   # 設計書準拠（凍結値）


def run_portfolio(stock_data: dict, era_start: str, era_end: str,
                  max_hold: int) -> dict:
    t_start = pd.Timestamp(era_start)
    t_end   = pd.Timestamp(era_end)

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
                trades.append({"pnl": pnl, "win": pnl > 0,
                                "reason": reason})
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
        return {"trades": 0, "wins": 0, "cagr": 0.0, "max_dd": 0.0,
                "win_pct": 0.0, "hold_exit_pct": 0.0}

    eq    = np.array(equity)
    years = (t_end - t_start).days / 365.25
    try:
        cagr = ((capital / INITIAL_CAPITAL) ** (1 / years) - 1) * 100
    except Exception:
        cagr = 0.0
    peak  = np.maximum.accumulate(eq)
    dd    = float(((eq - peak) / peak).min()) * 100
    wins  = sum(1 for t in trades if t["win"])
    holds = sum(1 for t in trades if t["reason"] == "HOLD")

    return {
        "trades":        len(trades),
        "wins":          wins,
        "win_pct":       wins / len(trades) * 100,
        "cagr":          cagr,
        "max_dd":        dd,
        "hold_exit_pct": holds / len(trades) * 100,
    }


def fmt(v, unit="%"):
    if unit == "%":
        return f"{'+' if v >= 0 else ''}{v:.1f}%"
    return f"{v:.0f}"


def main():
    print("=" * 70)
    print("  戦略A 保有日数最適化（20年データ・ポートフォリオ検証）")
    print(f"  テスト: {HOLD_DAYS_LIST} 日  現行: 10日")
    print("=" * 70)
    t0 = time.time()

    # 価格データ読み込み
    print("\n価格データ読み込み中...")
    prices_all = pd.read_parquet(DATA_PATH)
    prices_all["Date"] = pd.to_datetime(prices_all["Date"])
    prices_all["Code"] = prices_all["Code"].astype(str).str.strip()
    if "Code4" in prices_all.columns:
        mask = prices_all["Code"].str.len() == 4
        prices_all.loc[mask, "Code"] += "0"

    # 候補銘柄読み込み
    cands_df = pd.read_csv(CSV_PATH, dtype=str)
    col = next(c for c in cands_df.columns if "code" in c.lower())
    codes = [c.zfill(4) + "0" for c in cands_df[col]]
    print(f"  候補: {len(codes)}銘柄")

    # 指標・シグナル計算（一度だけ）
    print("  指標・シグナル計算中...")
    stock_data = {}
    for code in codes:
        sub = prices_all[prices_all["Code"] == code].copy()
        if len(sub) < 300:
            continue
        sub = sub.sort_values("Date").reset_index(drop=True)
        sub = add_indicators(sub)
        sub = gen_a(sub)
        stock_data[code] = sub
    print(f"  {len(stock_data)}銘柄 準備完了")

    # 保有日数 × 時代 の全組み合わせを実行
    results = {}
    for hold in HOLD_DAYS_LIST:
        label = f"{hold}日{'（現行）' if hold == 10 else ''}"
        results[hold] = {}
        for era, (s, e) in ERAS.items():
            r = run_portfolio(stock_data, s, e, hold)
            results[hold][era] = r

    # ── CAGR 比較テーブル ────────────────────────────────────
    print(f"\n{'='*70}")
    print("  ▼ CAGR比較")
    print(f"{'='*70}")
    header = f"  {'時代':<22}" + "".join(f"{str(h)+'日':>10}" for h in HOLD_DAYS_LIST)
    print(header.replace("10日", "10日*"))
    print("  " + "─" * (22 + 10 * len(HOLD_DAYS_LIST)))
    for era in ERAS:
        row = f"  {era:<22}"
        for hold in HOLD_DAYS_LIST:
            v = results[hold][era]["cagr"]
            row += f"{fmt(v):>10}"
        print(row)

    # ── MaxDD 比較テーブル ───────────────────────────────────
    print(f"\n  ▼ MaxDD比較")
    print("  " + "─" * (22 + 10 * len(HOLD_DAYS_LIST)))
    for era in ERAS:
        row = f"  {era:<22}"
        for hold in HOLD_DAYS_LIST:
            v = results[hold][era]["max_dd"]
            row += f"{fmt(v):>10}"
        print(row)

    # ── 取引数・強制終了率 ───────────────────────────────────
    print(f"\n  ▼ OOS取引数・強制終了率（{list(ERAS.keys())[-1]}）")
    print("  " + "─" * (22 + 10 * len(HOLD_DAYS_LIST)))
    for label, key in [("取引数", "trades"), ("強制終了%", "hold_exit_pct")]:
        row = f"  {label:<22}"
        for hold in HOLD_DAYS_LIST:
            v = results[hold]["OOS    (2023-2026)"][key]
            unit = "%" if key == "hold_exit_pct" else ""
            row += f"{v:>9.1f}{unit}"
        print(row)

    # ── 総合評価 ────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("  ▼ OOS CAGR + MaxDD 総合スコア（CAGR - |MaxDD| / 2）")
    print(f"{'='*70}")
    oos_era = "OOS    (2023-2026)"
    scores = {}
    for hold in HOLD_DAYS_LIST:
        r = results[hold][oos_era]
        score = r["cagr"] - abs(r["max_dd"]) / 2
        scores[hold] = score
        current = "← 現行" if hold == 10 else ""
        print(f"  {hold:2d}日: CAGR {fmt(r['cagr']):>7}  MaxDD {fmt(r['max_dd']):>7}"
              f"  スコア {score:+.1f}  {current}")

    best = max(scores, key=scores.get)
    print(f"\n  最良保有日数（OOSスコア）: {best}日")
    if best != 10:
        print(f"  → 現行10日からの変更を Phase 4 終了後に検討推奨")
    else:
        print(f"  → 現行10日が最適。変更不要。")

    print(f"\n  * = 現行設定（Phase 4凍結中・変更不可）")
    print(f"  総所要時間: {time.time()-t0:.0f}秒")


if __name__ == "__main__":
    main()
