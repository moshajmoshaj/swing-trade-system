"""
select_v2_fundamental.py
財務品質フィルター付き候補銘柄選定（Phase 5用・v2候補リスト）

目的:
  既存IS選定（技術指標のみ）に財務品質フィルターを追加し
  Phase 5以降で使用する強化版候補リストを作成する

財務品質フィルター（IS開始時点で適用 → 先読みバイアス排除）:
  - ROE > 10%（収益性）
  - EqAR > 0.30（自己資本比率30%超・安全性）
  - CFO > 0（営業CF黒字・実態利益確認）

出力:
  logs/strategy_a_v2_candidates.csv  戦略A v2候補（Phase 5用）
  logs/strategy_e_v2_candidates.csv  戦略E v2候補（Phase 5用）

注意:
  Phase 4の候補リスト（final_candidates.csv 等）は一切変更しない
"""
import sys, time
sys.stdout.reconfigure(encoding="utf-8")

from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import binom

sys.path.insert(0, str(Path(__file__).parent / "src"))
from indicators import add_indicators
from strategy   import generate_signals as gen_a
from strategy_e import generate_signals as gen_e

# ── パス ────────────────────────────────────────────────────
DATA_PATH  = Path("data/raw/prices_20y.parquet")
FINS_PATH  = Path("data/raw/fins_summary.parquet")

# ── 期間設定（既存OOSバックテストと同一） ─────────────────────
IS_START   = pd.Timestamp("2016-04-01")
IS_END     = pd.Timestamp("2020-12-31")
OOS_START  = pd.Timestamp("2023-01-01")
OOS_END    = pd.Timestamp("2026-05-01")
OOS_WARMUP = pd.Timestamp("2022-01-01")

# ── IS選定基準（既存と同一） ────────────────────────────────
MIN_TRADES   = 10
MIN_WIN_RATE = 60.0
MAX_DD       = -3.0
MIN_PNL      = 30_000
TARGET_N     = 30
P_VALUE_MAX  = 0.10

# ── バックテスト設定 ─────────────────────────────────────────
INITIAL_CAPITAL = 1_000_000
MAX_POSITIONS   = 5
MAX_POS_RATIO   = 0.20
COMMISSION      = 0.00055
SLIPPAGE        = 0.00050
COST_LEG        = COMMISSION + SLIPPAGE

# ── 財務フィルター基準 ────────────────────────────────────────
MIN_ROE     = 10.0   # %
MIN_EQAR    = 0.30   # 小数（30%）
REQUIRE_CFO = True   # CFO > 0


# ────────────────────────────────────────────────────────────
# 財務品質フィルター（ポイントインタイム）
# ────────────────────────────────────────────────────────────
def build_quality_universe(fins_path: Path, as_of_date: pd.Timestamp) -> set[str]:
    """as_of_date 時点で財務品質基準を満たす銘柄コードセットを返す。"""
    fins = pd.read_parquet(fins_path)
    fins["DiscDate"] = pd.to_datetime(fins["DiscDate"], errors="coerce")

    # 年次決算のみ（通期）
    fins = fins[fins["CurPerType"] == "FY"].copy()

    # 数値変換
    for col in ["NP", "Eq", "EqAR", "CFO"]:
        fins[col] = pd.to_numeric(fins[col], errors="coerce")

    # as_of_date 以前に開示された最新年次決算を各銘柄から1件取得
    fins = fins[fins["DiscDate"] <= as_of_date]
    latest = (
        fins.sort_values("DiscDate")
            .groupby("Code")
            .last()
            .reset_index()
    )

    # ROE計算
    latest["ROE"] = latest["NP"] / latest["Eq"] * 100

    # フィルター適用
    mask = (
        (latest["ROE"] > MIN_ROE) &
        (latest["EqAR"] > MIN_EQAR)
    )
    if REQUIRE_CFO:
        mask &= (latest["CFO"] > 0)

    quality_codes = set(latest.loc[mask, "Code"].astype(str).tolist())
    print(f"  財務品質フィルター: {len(latest):,}銘柄 → {len(quality_codes):,}銘柄通過"
          f" (ROE>{MIN_ROE}% & EqAR>{MIN_EQAR*100:.0f}% & CFO>0)")
    return quality_codes


# ────────────────────────────────────────────────────────────
# 単銘柄バックテスト（IS用）
# ────────────────────────────────────────────────────────────
def run_single_bt(df: pd.DataFrame, bt_start: pd.Timestamp, bt_end: pd.Timestamp,
                  tp_mult: float, sl_mult: float, max_hold: int,
                  gen_fn) -> dict:
    df = add_indicators(df)
    df = gen_fn(df)
    df = df[df["Date"] <= bt_end].copy()

    capital = float(INITIAL_CAPITAL)
    trades  = []
    in_trade = False

    for i in range(1, len(df)):
        row  = df.iloc[i]
        prev = df.iloc[i - 1]
        if row["Date"] <= bt_start:
            continue

        if in_trade:
            hold_days += 1
            hi, lo, cl = row["High"], row["Low"], row["Close"]
            exit_price = exit_reason = None
            if lo <= stop_loss:
                exit_price, exit_reason = stop_loss, "SL"
            elif hi >= take_profit:
                exit_price, exit_reason = take_profit, "TP"
            elif hold_days >= max_hold:
                exit_price, exit_reason = cl, "HOLD"
            if exit_price:
                cost = (entry_price + exit_price) * shares * COST_LEG
                pnl  = (exit_price - entry_price) * shares - cost
                capital += pnl
                trades.append({"pnl": pnl, "win": pnl > 0})
                in_trade = False

        if not in_trade and prev.get("signal", 0) == 1 and pd.notna(prev.get("ATR14")):
            ep  = row["Open"]
            gap = (ep - prev["Close"]) / prev["Close"]
            if gap >= -0.015:
                atr    = float(prev["ATR14"])
                shares = min(int(INITIAL_CAPITAL * MAX_POS_RATIO / ep / 100) * 100, 100)
                if shares > 0 and ep * shares <= INITIAL_CAPITAL:
                    entry_price = ep
                    take_profit = ep + atr * tp_mult
                    stop_loss   = ep - atr * sl_mult
                    hold_days   = 0
                    in_trade    = True

    if not trades:
        return {}

    wins    = sum(1 for t in trades if t["win"])
    pnl     = sum(t["pnl"] for t in trades)
    n       = len(trades)
    win_pct = wins / n * 100

    # 二項検定
    p_val = binom.sf(wins - 1, n, 0.5)

    # 最大ドローダウン（簡易）
    equity = INITIAL_CAPITAL
    peak   = INITIAL_CAPITAL
    max_dd = 0.0
    for t in trades:
        equity += t["pnl"]
        if equity > peak:
            peak = equity
        dd = (equity - peak) / peak * 100
        if dd < max_dd:
            max_dd = dd

    return {
        "trades": n, "wins": wins, "win_pct": win_pct,
        "pnl": pnl, "max_dd": max_dd, "p_val": p_val,
    }


def select_candidates(codes: list[str], prices_all: pd.DataFrame,
                      gen_fn, tp_mult: float, sl_mult: float, max_hold: int,
                      label: str) -> pd.DataFrame:
    """IS期間でバックテストして候補銘柄を選定する。"""
    print(f"\n  {label}: {len(codes)}銘柄でIS選定中...")
    t0 = time.time()
    results = []

    for i, code in enumerate(codes):
        code5 = code + "0" if len(code) == 4 else code
        sub = prices_all[prices_all["Code"] == code5].copy()
        if len(sub) < 300:
            continue
        sub = sub.sort_values("Date").reset_index(drop=True)
        r = run_single_bt(sub, IS_START, IS_END, tp_mult, sl_mult, max_hold, gen_fn)
        if r:
            r["code"] = code5
            results.append(r)

        if (i + 1) % 100 == 0:
            print(f"    [{i+1}/{len(codes)}]  経過: {time.time()-t0:.0f}秒")

    if not results:
        return pd.DataFrame()

    df = pd.DataFrame(results)

    # IS選定フィルター（既存基準と同一）
    df = df[
        (df["trades"] >= MIN_TRADES) &
        (df["win_pct"] >= MIN_WIN_RATE) &
        (df["max_dd"] >= MAX_DD) &
        (df["pnl"] >= MIN_PNL) &
        (df["p_val"] <= P_VALUE_MAX)
    ].copy()

    df = df.sort_values("pnl", ascending=False).head(TARGET_N)
    print(f"    選定完了: {len(df)}銘柄  ({time.time()-t0:.0f}秒)")
    return df


# ────────────────────────────────────────────────────────────
# ポートフォリオバックテスト（OOS用）
# ────────────────────────────────────────────────────────────
def run_portfolio_oos(codes: list[str], prices_all: pd.DataFrame, gen_fn,
                      tp_mult: float, sl_mult: float, max_hold: int) -> dict:
    t_start = OOS_WARMUP
    t_end   = OOS_END
    oos_s   = OOS_START

    stock_data = {}
    for code in codes:
        code5 = code + "0" if len(code) == 4 else code
        sub = prices_all[prices_all["Code"] == code5].copy()
        if len(sub) < 300:
            continue
        sub = sub.sort_values("Date").reset_index(drop=True)
        sub = add_indicators(sub)
        sub = gen_fn(sub)
        sub = sub[(sub["Date"] >= t_start) & (sub["Date"] <= t_end)]
        if not sub.empty:
            stock_data[code5] = sub

    if not stock_data:
        return {}

    all_dates = sorted(set(
        d for df in stock_data.values() for d in df["Date"].tolist()
    ))

    lookup = {code: {row["Date"]: row for _, row in df.iterrows()}
              for code, df in stock_data.items()}

    capital   = float(INITIAL_CAPITAL)
    positions = {}
    trades    = []
    equity    = []

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
            elif pos["hold"] >= max_hold:
                ep, reason = cl, "HOLD"
            if reason:
                cost = (pos["ep"] + ep) * pos["sh"] * COST_LEG
                pnl  = (ep - pos["ep"]) * pos["sh"] - cost
                capital += pnl
                if date >= oos_s:
                    trades.append({"pnl": pnl, "win": pnl > 0})
                to_close.append(code)
        for c in to_close:
            del positions[c]

        # エントリー
        if len(positions) < MAX_POSITIONS:
            slots = MAX_POSITIONS - len(positions)
            cands = []
            for code, lk in lookup.items():
                if code in positions:
                    continue
                row = lk.get(date)
                if row is None or row.get("signal", 0) != 1:
                    continue
                atr = row.get("ATR14")
                rsi = row.get("RSI14")
                if pd.isna(atr) or atr <= 0 or pd.isna(rsi):
                    continue
                cands.append((code, row, rsi))
            cands.sort(key=lambda x: x[2], reverse=True)
            for code, row, _ in cands[:slots]:
                ep = row["Open"]
                if ep <= 0:
                    continue
                sh = min(int(INITIAL_CAPITAL * MAX_POS_RATIO / ep / 100) * 100, 100)
                if sh <= 0:
                    continue
                atr = float(row["ATR14"])
                positions[code] = {
                    "ep": ep,
                    "tp": ep + atr * tp_mult,
                    "sl": ep - atr * sl_mult,
                    "sh": sh, "hold": 0,
                }

        if date >= oos_s:
            equity.append(capital)

    if not trades or not equity:
        return {}

    eq    = np.array(equity)
    years = (OOS_END - OOS_START).days / 365.25
    cagr  = ((capital / INITIAL_CAPITAL) ** (1 / years) - 1) * 100
    peak  = np.maximum.accumulate(eq)
    dd    = float(((eq - peak) / peak).min()) * 100
    wins  = sum(1 for t in trades if t["win"])

    return {
        "trades": len(trades), "wins": wins,
        "win_pct": wins / len(trades) * 100,
        "cagr": cagr, "max_dd": dd,
    }


# ────────────────────────────────────────────────────────────
# メイン
# ────────────────────────────────────────────────────────────
def main() -> None:
    print("=" * 65)
    print("  財務品質フィルター付き候補銘柄選定（Phase 5用 v2）")
    print("=" * 65)

    t0 = time.time()

    # ── 価格データ読み込み ───────────────────────────────────
    print("\n価格データ読み込み中...")
    prices_all = pd.read_parquet(DATA_PATH)
    prices_all["Date"] = pd.to_datetime(prices_all["Date"])
    prices_all["Code"] = prices_all["Code"].astype(str).str.strip()
    if "Code4" in prices_all.columns:
        mask = prices_all["Code"].str.len() == 4
        prices_all.loc[mask, "Code"] += "0"

    # IS開始時点の銘柄マスタ（プライム銘柄）
    prime_codes_all = prices_all["Code"].unique().tolist()
    print(f"  価格データ: {len(prices_all):,}行  {len(prime_codes_all):,}銘柄")

    # ── 財務品質フィルター（IS開始時点） ─────────────────────
    print("\n財務品質フィルター適用中（IS開始時点 = 2016-04-01）...")
    quality_codes = build_quality_universe(FINS_PATH, IS_START)

    # コード形式を合わせて絞り込み
    filtered_codes = [c for c in prime_codes_all if c in quality_codes]
    print(f"  品質フィルター後: {len(filtered_codes):,}銘柄")

    # ── 戦略別IS選定 ─────────────────────────────────────────
    strategies = [
        {
            "id": "A", "label": "戦略A v2（順張りモメンタム+財務）",
            "gen_fn": gen_a, "tp": 6.0, "sl": 2.0, "hold": 10,
            "out": "logs/strategy_a_v2_candidates.csv",
            "existing": "logs/final_candidates.csv",
        },
        {
            "id": "E", "label": "戦略E v2（52週高値ブレイク+財務）",
            "gen_fn": gen_e, "tp": 6.0, "sl": 2.0, "hold": 10,
            "out": "logs/strategy_e_v2_candidates.csv",
            "existing": "logs/strategy_e_candidates.csv",
        },
    ]

    comparison = []

    for cfg in strategies:
        print(f"\n{'─'*65}")
        print(f"  {cfg['label']}")
        print(f"{'─'*65}")

        # IS選定（財務フィルター済みユニバース）
        selected = select_candidates(
            filtered_codes, prices_all,
            cfg["gen_fn"], cfg["tp"], cfg["sl"], cfg["hold"],
            cfg["label"]
        )

        if selected.empty:
            print("  候補なし")
            continue

        # 保存
        selected.to_csv(cfg["out"], index=False)
        print(f"  保存: {cfg['out']}  ({len(selected)}銘柄)")

        # 既存候補との比較
        try:
            existing = pd.read_csv(cfg["existing"], dtype=str)
            ex_col   = next(c for c in existing.columns if "code" in c.lower())
            ex_codes = set(existing[ex_col].str.zfill(4) + "0")
            new_codes = set(selected["code"].astype(str))
            overlap  = ex_codes & new_codes
            print(f"  既存({len(ex_codes)}) vs v2({len(new_codes)})  重複: {len(overlap)}銘柄")
            added   = new_codes - ex_codes
            removed = ex_codes - new_codes
            if added:
                print(f"  新規追加: {sorted(added)}")
            if removed:
                print(f"  除外: {sorted(removed)}")
        except Exception as e:
            print(f"  既存比較スキップ: {e}")

        # OOSバックテスト
        print(f"\n  OOSバックテスト中（{OOS_START.date()} ~ {OOS_END.date()}）...")
        new_c = selected["code"].tolist()
        r_new = run_portfolio_oos(new_c, prices_all, cfg["gen_fn"],
                                  cfg["tp"], cfg["sl"], cfg["hold"])

        # 既存候補でのOOS
        try:
            ex_list = list(ex_codes)
            r_ex  = run_portfolio_oos(ex_list, prices_all, cfg["gen_fn"],
                                      cfg["tp"], cfg["sl"], cfg["hold"])
        except Exception:
            r_ex = {}

        print(f"\n  {'':20} {'既存':>10} {'v2(財務+)':>10}")
        print(f"  {'─'*42}")
        for key, label in [("trades","取引数"), ("win_pct","勝率"),
                            ("cagr","OOS年利"), ("max_dd","最大DD")]:
            v_ex  = r_ex.get(key, 0)
            v_new = r_new.get(key, 0)
            fmt   = ".1f"
            unit  = "%" if key in ("win_pct","cagr","max_dd") else ""
            sign  = "+" if key in ("cagr","win_pct") and v_new > 0 else ""
            print(f"  {label:20} {v_ex:>10.{fmt[1]}f}{unit} {sign}{v_new:>9.{fmt[1]}f}{unit}")

        comparison.append({
            "strategy": cfg["id"],
            "ex_cagr": r_ex.get("cagr", 0),
            "v2_cagr": r_new.get("cagr", 0),
            "ex_dd":   r_ex.get("max_dd", 0),
            "v2_dd":   r_new.get("max_dd", 0),
        })

    # ── 最終サマリー ─────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  完了サマリー  （総所要時間: {time.time()-t0:.0f}秒）")
    print(f"{'='*65}")
    for c in comparison:
        delta = c["v2_cagr"] - c["ex_cagr"]
        sign  = "+" if delta >= 0 else ""
        print(f"  戦略{c['strategy']}: 既存 {c['ex_cagr']:+.1f}% → v2 {c['v2_cagr']:+.1f}%"
              f"  (差分: {sign}{delta:.1f}pt)  MaxDD: {c['ex_dd']:.1f}% → {c['v2_dd']:.1f}%")

    print("\n  Phase 5用v2候補リストを保存しました。Phase 4には影響しません。")


if __name__ == "__main__":
    main()
