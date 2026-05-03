"""
backtest_20y.py
既存4戦略を20年データで時代別に検証する

目的:
  Phase 4候補銘柄（IS 2016-2020で選定済み）が
  選定期間外（2008-2015）でも機能するか確認する

時代区分:
  Pre-IS : 2008-05-07 ~ 2015-12-31  GFC・東日本大震災・アベノミクス
  IS     : 2016-01-01 ~ 2020-12-31  銘柄選定期間（参考）
  Gap    : 2021-01-01 ~ 2022-12-31  ギャップ期間
  OOS    : 2023-01-01 ~ 2026-05-01  OOS検証期間（Phase 4根拠）

注意:
  候補銘柄・パラメータはPhase 4凍結値を使用。変更しない。
"""
import sys
import time
sys.stdout.reconfigure(encoding="utf-8")

from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent / "src"))
from indicators import add_indicators
from strategy   import generate_signals as gen_a
from strategy_c import generate_signals as gen_c
from strategy_d import generate_signals as gen_d
from strategy_e import generate_signals as gen_e

# ── データパス ───────────────────────────────────────────────
DATA_PATH = Path("data/raw/prices_20y.parquet")

# ── 時代定義 ─────────────────────────────────────────────────
ERAS = {
    "Pre-IS (2008-2015)": ("2008-05-07", "2015-12-31"),
    "IS     (2016-2020)": ("2016-01-01", "2020-12-31"),
    "Gap    (2021-2022)": ("2021-01-01", "2022-12-31"),
    "OOS    (2023-2026)": ("2023-01-01", "2026-05-01"),
}

# ── 戦略定義 ─────────────────────────────────────────────────
STRATEGIES = {
    "A": {
        "csv":       "logs/final_candidates.csv",
        "gen_fn":    gen_a,
        "tp_mult":   6.0,
        "sl_mult":   2.0,
        "max_hold":  10,
        "label":     "順張りモメンタム",
    },
    "C": {
        "csv":       "logs/strategy_c_candidates.csv",
        "gen_fn":    gen_c,
        "tp_mult":   2.5,
        "sl_mult":   1.5,
        "max_hold":  7,
        "label":     "平均回帰・逆張り",
    },
    "D": {
        "csv":       "logs/strategy_d_candidates.csv",
        "gen_fn":    gen_d,
        "tp_mult":   4.5,
        "sl_mult":   1.5,
        "max_hold":  5,
        "label":     "ギャップアップ翌日",
    },
    "E": {
        "csv":       "logs/strategy_e_candidates.csv",
        "gen_fn":    gen_e,
        "tp_mult":   6.0,
        "sl_mult":   2.0,
        "max_hold":  10,
        "label":     "52週高値ブレイク",
    },
}

# ── ポートフォリオ設定 ────────────────────────────────────────
INITIAL_CAPITAL = 1_000_000
MAX_POSITIONS   = 5
MAX_POS_SIZE    = 200_000
COMMISSION      = 0.00055
SLIPPAGE        = 0.00050
COST_LEG        = COMMISSION + SLIPPAGE


def load_candidates(csv_path: str) -> list[str]:
    df = pd.read_csv(csv_path, dtype=str)
    col = next((c for c in df.columns if "code" in c.lower()), df.columns[0])
    return df[col].str.zfill(4).tolist()


def prepare_stock_data(codes: list[str], prices_all: pd.DataFrame, gen_fn) -> dict[str, pd.DataFrame]:
    stock_data = {}
    for code in codes:
        # prices_20yのコードは5桁（末尾0付き）
        code5 = code + "0" if len(code) == 4 else code
        sub = prices_all[prices_all["Code"] == code5].copy()
        if len(sub) < 250:
            continue
        sub = sub.sort_values("Date").reset_index(drop=True)
        sub = add_indicators(sub)
        sub = gen_fn(sub)
        stock_data[code5] = sub
    return stock_data


def run_portfolio(stock_data: dict, start: str, end: str,
                  tp_mult: float, sl_mult: float, max_hold: int) -> dict:
    t_start = pd.Timestamp(start)
    t_end   = pd.Timestamp(end)

    # 全銘柄の日付を統合してカレンダーを作成
    all_dates = sorted(set(
        d for df in stock_data.values()
        for d in df.loc[(df["Date"] >= t_start) & (df["Date"] <= t_end), "Date"].tolist()
    ))
    if not all_dates:
        return {"trades": 0, "wins": 0, "cagr": 0.0, "max_dd": 0.0, "pnl": 0.0}

    # 各銘柄をDate→行のdictに変換（高速ルックアップ用）
    lookup: dict[str, dict] = {}
    for code, df in stock_data.items():
        lookup[code] = {row["Date"]: row for _, row in df.iterrows()}

    capital   = float(INITIAL_CAPITAL)
    equity    = [capital]
    positions = {}   # code5 → {entry_price, tp, sl, shares, hold_days, entry_rsi}
    trades    = []

    for date in all_dates:
        # ── 既存ポジション決済チェック ──
        to_close = []
        for code, pos in positions.items():
            row = lookup[code].get(date)
            if row is None:
                continue
            pos["hold_days"] += 1
            hi, lo, cl = row["High"], row["Low"], row["Close"]
            # NaNを含む場合はスキップ（20年データの欠損対策）
            if pd.isna(hi) or pd.isna(lo) or pd.isna(cl):
                continue
            reason = None
            exit_p = None
            if lo <= pos["sl"]:
                exit_p, reason = pos["sl"], "SL"
            elif hi >= pos["tp"]:
                exit_p, reason = pos["tp"], "TP"
            elif pos["hold_days"] >= max_hold:
                exit_p, reason = cl, "HOLD"
            if reason:
                cost = (pos["entry_price"] + exit_p) * pos["shares"] * COST_LEG
                pnl  = (exit_p - pos["entry_price"]) * pos["shares"] - cost
                if not np.isfinite(pnl):
                    pnl = 0.0  # 欠損・異常値ガード
                capital += pnl
                trades.append({"pnl": pnl, "win": pnl > 0})
                to_close.append(code)
        for code in to_close:
            del positions[code]

        # ── 新規エントリー ──
        if len(positions) < MAX_POSITIONS:
            slots = MAX_POSITIONS - len(positions)
            candidates = []
            for code, df_lookup in lookup.items():
                if code in positions:
                    continue
                row = df_lookup.get(date)
                if row is None or row.get("signal", 0) != 1:
                    continue
                atr = row.get("ATR14")
                rsi = row.get("RSI14")
                if pd.isna(atr) or atr <= 0 or pd.isna(rsi):
                    continue
                candidates.append((code, row, rsi))

            # RSI降順でエントリー優先
            candidates.sort(key=lambda x: x[2], reverse=True)
            for code, prev_row, _ in candidates[:slots]:
                # 翌日オープンでエントリー → 同日Openで近似
                ep  = prev_row["Open"]
                if ep <= 0:
                    continue
                shares = min(int(MAX_POS_SIZE / ep / 100) * 100, 100)
                if shares <= 0:
                    continue
                atr = float(prev_row["ATR14"])
                tp  = ep + atr * tp_mult
                sl  = ep - atr * sl_mult
                positions[code] = {
                    "entry_price": ep,
                    "tp": tp, "sl": sl,
                    "shares": shares,
                    "hold_days": 0,
                }

        equity.append(capital)

    if not trades:
        return {"trades": 0, "wins": 0, "cagr": 0.0, "max_dd": 0.0, "pnl": 0.0}

    # ── メトリクス計算 ──
    eq = np.array(equity, dtype=float)
    # NaN混入ガード（価格欠損等で capital が NaN になった場合）
    if not np.isfinite(capital):
        capital = float(equity[-2]) if len(equity) > 1 else float(INITIAL_CAPITAL)
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
    wins  = sum(1 for t in trades if t["win"])
    pnl   = sum(t["pnl"] for t in trades)

    return {
        "trades": len(trades),
        "wins":   wins,
        "cagr":   cagr,
        "max_dd": max_dd,
        "pnl":    pnl,
    }


def fmt(v: float, unit: str = "%") -> str:
    if unit == "%":
        sign = "+" if v >= 0 else ""
        return f"{sign}{v:.1f}%"
    return f"¥{v:,.0f}"


def main() -> None:
    print("=" * 70)
    print("  20年データ 時代別バックテスト（既存候補銘柄・凍結パラメータ）")
    print("=" * 70)

    t0 = time.time()
    print("\n価格データ読み込み中...")
    prices_all = pd.read_parquet(DATA_PATH)
    prices_all["Date"] = pd.to_datetime(prices_all["Date"])
    # Code列をoos_backtest.pyと同じ方式で5桁に正規化（Code4列を使用）
    prices_all["Code"] = prices_all["Code4"].astype(str).str.zfill(4) + "0"
    # Open/High/Low/Close/Volumeは prices_20y に既存（リネーム不要）
    drop_c = [c for c in ["O","H","L","C","Vo","Va","UL","LL","AdjFactor",
                           "MO","MH","ML","MC","MUL","MLL","MVo","MVa",
                           "AO","AH","AL","AC","AUL","ALL","AVo","AVa","Code4"]
              if c in prices_all.columns]
    prices_all = prices_all.drop(columns=drop_c)
    print(f"  {len(prices_all):,}行  {prices_all['Code'].nunique():,}銘柄  "
          f"{prices_all['Date'].min().date()} ~ {prices_all['Date'].max().date()}")

    results = {}

    for strat_id, cfg in STRATEGIES.items():
        print(f"\n{'─'*70}")
        print(f"  戦略{strat_id}：{cfg['label']}  [{cfg['csv']}]")
        print(f"{'─'*70}")

        try:
            codes = load_candidates(cfg["csv"])
        except FileNotFoundError:
            print(f"  候補CSVが見つかりません: {cfg['csv']}  スキップ")
            continue
        print(f"  候補銘柄: {len(codes)}銘柄")

        print("  指標・シグナル計算中...")
        stock_data = prepare_stock_data(codes, prices_all, cfg["gen_fn"])
        print(f"  データ取得済み: {len(stock_data)}銘柄")

        era_results = {}
        for era_name, (era_start, era_end) in ERAS.items():
            r = run_portfolio(
                stock_data, era_start, era_end,
                cfg["tp_mult"], cfg["sl_mult"], cfg["max_hold"]
            )
            era_results[era_name] = r
            win_pct = (r["wins"] / r["trades"] * 100) if r["trades"] > 0 else 0
            print(f"  {era_name}  取引:{r['trades']:4d}  勝率:{win_pct:5.1f}%  "
                  f"CAGR:{fmt(r['cagr'])}  MaxDD:{fmt(r['max_dd'])}")

        results[strat_id] = era_results

    # ── サマリーテーブル ──────────────────────────────────────────
    print(f"\n{'='*70}")
    print("  ▼ サマリー（CAGR）")
    print(f"{'='*70}")
    header = f"{'時代':<22}" + "".join(f"{'戦略'+s:>12}" for s in results)
    print(header)
    print("─" * len(header))
    for era_name in ERAS:
        row = f"{era_name:<22}"
        for strat_id, era_results in results.items():
            r = era_results.get(era_name, {})
            row += f"{fmt(r.get('cagr', 0)):>12}"
        print(row)

    print(f"\n{'='*70}")
    print("  ▼ サマリー（最大DD）")
    print(f"{'='*70}")
    print(header)
    print("─" * len(header))
    for era_name in ERAS:
        row = f"{era_name:<22}"
        for strat_id, era_results in results.items():
            r = era_results.get(era_name, {})
            row += f"{fmt(r.get('max_dd', 0)):>12}"
        print(row)

    print(f"\n  総所要時間: {time.time()-t0:.0f}秒")


if __name__ == "__main__":
    main()
