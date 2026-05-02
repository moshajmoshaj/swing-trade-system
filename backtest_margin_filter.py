"""
backtest_margin_filter.py
信用倍率フィルターの効果検証（戦略A・E）

仮説:
  信用倍率が異常に高い銘柄（>20倍）はその後下落リスクが高い。
  戦略A/Eから高信用倍率銘柄を除外すると成績が改善するか？

設計:
  信用倍率（制度信用 IssType=2: LongVol/ShrtVol）をフィルターとして適用
  ベースライン vs フィルターあり の20年OOS比較

比較設定:
  baseline:  フィルターなし（現行設定）
  filter10:  信用倍率 > 10 でエントリーを抑制
  filter20:  信用倍率 > 20 でエントリーを抑制（主要テスト）
  filter50:  信用倍率 > 50 でエントリーを抑制

出力:
  - 時代別 CAGR / MaxDD / 取引数の比較
"""
import sys, time
sys.stdout.reconfigure(encoding="utf-8")

from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent / "src"))
from indicators import add_indicators
from strategy   import generate_signals as gen_a
from strategy_e import generate_signals as gen_e

DATA_PATH   = Path("data/raw/prices_20y.parquet")
MARGIN_PATH = Path("data/raw/margin_interest.parquet")

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

# テストする信用倍率上限（Noneはフィルターなし）
THRESHOLDS = {
    "baseline(フィルタなし)": None,
    "倍率>10を除外":          10.0,
    "倍率>20を除外":          20.0,
    "倍率>50を除外":          50.0,
}

STRATEGY_CONFIGS = {
    "A": {"csv": "logs/final_candidates.csv",     "gen": gen_a, "tp": 6.0, "sl": 2.0, "hold": 10},
    "E": {"csv": "logs/strategy_e_candidates.csv","gen": gen_e, "tp": 6.0, "sl": 2.0, "hold": 10},
}


def build_margin_ratio_lookup(codes: list[str]) -> dict[str, pd.DataFrame]:
    """
    信用倍率（制度信用 IssType=2）を銘柄別・日付別に返す。
    週次データを日次に前方補完する。
    """
    margin_all = pd.read_parquet(MARGIN_PATH)
    margin_all["Date"] = pd.to_datetime(margin_all["Date"], errors="coerce")
    margin_all["Code"] = margin_all["Code"].astype(str).str.strip()

    # 制度信用のみ・数値変換
    m = margin_all[margin_all["IssType"].astype(str) == "2"].copy()
    m["LongVol"] = pd.to_numeric(m["LongVol"], errors="coerce")
    m["ShrtVol"] = pd.to_numeric(m["ShrtVol"], errors="coerce")
    m["ratio"]   = m["LongVol"] / m["ShrtVol"].replace(0, np.nan)

    lookup = {}
    for code in codes:
        sub = m[m["Code"] == code][["Date", "ratio"]].copy()
        if sub.empty:
            continue
        sub = sub.sort_values("Date").set_index("Date")
        # 日次インデックスに前方補完
        full_idx = pd.date_range(sub.index.min(), sub.index.max(), freq="D")
        sub = sub.reindex(full_idx).ffill()
        lookup[code] = sub
    return lookup


def get_margin_ratio(lookup: dict, code: str, date: pd.Timestamp) -> float | None:
    sub = lookup.get(code)
    if sub is None or sub.empty:
        return None
    try:
        return float(sub.loc[date, "ratio"]) if date in sub.index else float(sub.asof(date))
    except Exception:
        return None


def run_portfolio(stock_data: dict, margin_lookup: dict,
                  era_start: str, era_end: str,
                  tp: float, sl: float, hold: int,
                  max_ratio: float | None) -> dict:
    t_start = pd.Timestamp(era_start)
    t_end   = pd.Timestamp(era_end)

    all_dates = sorted(set(
        d for df in stock_data.values()
        for d in df.loc[(df["Date"] >= t_start) & (df["Date"] <= t_end), "Date"]
    ))
    if not all_dates:
        return {"trades": 0, "wins": 0, "cagr": 0.0, "max_dd": 0.0, "suppressed": 0}

    lookup = {c: {r["Date"]: r for _, r in df.iterrows()}
              for c, df in stock_data.items()}

    capital   = float(INITIAL_CAPITAL)
    positions = {}
    trades    = []
    equity    = [capital]
    suppressed = 0

    for date in all_dates:
        to_close = []
        for code, pos in positions.items():
            row = lookup[code].get(date)
            if row is None:
                continue
            pos["hold"] += 1
            hi, lo, cl = row["High"], row["Low"], row["Close"]
            if pd.isna(hi) or pd.isna(lo) or pd.isna(cl):
                continue
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
                # 信用倍率フィルター
                if max_ratio is not None:
                    ratio = get_margin_ratio(margin_lookup, code, date)
                    if ratio is not None and ratio > max_ratio:
                        suppressed += 1
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
                    "tp": ep + atr * tp,
                    "sl": ep - atr * sl,
                    "sh": sh, "hold": 0,
                }
        equity.append(capital)

    if not trades:
        return {"trades": 0, "wins": 0, "cagr": 0.0, "max_dd": 0.0,
                "suppressed": suppressed}

    eq    = np.array(equity, dtype=float)
    years = (t_end - t_start).days / 365.25
    try:
        ratio = capital / INITIAL_CAPITAL
        cagr  = ((ratio) ** (1 / years) - 1) * 100 if years > 0 and ratio > 0 else float("nan")
    except Exception:
        cagr = float("nan")
    peak  = np.maximum.accumulate(eq)
    with np.errstate(invalid="ignore", divide="ignore"):
        dd = np.where(peak > 0, (eq - peak) / peak, 0.0)
    max_dd = float(np.nanmin(dd)) * 100
    wins   = sum(1 for t in trades if t["win"])

    return {
        "trades":     len(trades),
        "wins":       wins,
        "win_pct":    wins / len(trades) * 100,
        "cagr":       cagr,
        "max_dd":     max_dd,
        "suppressed": suppressed,
    }


def fmt(v):
    return f"{'+' if v >= 0 else ''}{v:.1f}%"


def main():
    print("=" * 70)
    print("  信用倍率フィルター効果検証（戦略A・E × 20年データ）")
    print("=" * 70)
    t0 = time.time()

    print("\n価格データ読み込み中...")
    prices_all = pd.read_parquet(DATA_PATH)
    prices_all["Date"] = pd.to_datetime(prices_all["Date"])
    prices_all["Code"] = prices_all["Code"].astype(str).str.strip()
    if "Code4" in prices_all.columns:
        mask = prices_all["Code"].str.len() == 4
        prices_all.loc[mask, "Code"] += "0"

    for strat_id, cfg in STRATEGY_CONFIGS.items():
        print(f"\n{'─'*70}")
        print(f"  戦略{strat_id}")
        print(f"{'─'*70}")

        cands_df = pd.read_csv(cfg["csv"], dtype=str)
        col = next(c for c in cands_df.columns if "code" in c.lower())
        codes = [c.zfill(4) + "0" for c in cands_df[col]]

        print(f"  指標・シグナル計算中（{len(codes)}銘柄）...")
        stock_data = {}
        for code in codes:
            sub = prices_all[prices_all["Code"] == code].copy()
            if len(sub) < 300:
                continue
            sub = sub.sort_values("Date").reset_index(drop=True)
            sub = add_indicators(sub)
            sub = cfg["gen"](sub)
            stock_data[code] = sub

        print(f"  信用倍率データ構築中...")
        margin_lookup = build_margin_ratio_lookup(codes)
        print(f"  信用倍率あり: {len(margin_lookup)}銘柄")

        results_by_thresh = {}
        for thresh_name, max_ratio in THRESHOLDS.items():
            era_results = {}
            for era, (s, e) in ERAS.items():
                r = run_portfolio(stock_data, margin_lookup, s, e,
                                  cfg["tp"], cfg["sl"], cfg["hold"], max_ratio)
                era_results[era] = r
            results_by_thresh[thresh_name] = era_results

        # OOS 比較テーブル
        oos_era = "OOS    (2023-2026)"
        print(f"\n  ▼ OOS CAGR比較（{oos_era}）")
        print(f"  {'設定':<22} {'CAGR':>8} {'MaxDD':>8} {'取引数':>6} {'抑制数':>6}")
        print(f"  {'─'*22} {'─'*8} {'─'*8} {'─'*6} {'─'*6}")
        for thresh_name, era_results in results_by_thresh.items():
            r = era_results[oos_era]
            mark = " ←現行" if thresh_name == "baseline(フィルタなし)" else ""
            print(f"  {thresh_name:<22} {fmt(r['cagr']):>8} "
                  f"{fmt(r['max_dd']):>8} {r['trades']:>6} "
                  f"{r['suppressed']:>6}{mark}")

        # 全時代比較（CAGR）
        print(f"\n  ▼ 全時代 CAGR比較")
        thresh_names = list(THRESHOLDS.keys())
        header = f"  {'時代':<22}" + "".join(f"{n[:14]:>16}" for n in thresh_names)
        print(header)
        print("  " + "─" * (22 + 16 * len(thresh_names)))
        for era in ERAS:
            row = f"  {era:<22}"
            for tn in thresh_names:
                v = results_by_thresh[tn][era]["cagr"]
                row += f"{fmt(v):>16}"
            print(row)

    print(f"\n  総所要時間: {time.time()-t0:.0f}秒")


if __name__ == "__main__":
    main()
