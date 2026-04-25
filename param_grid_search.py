"""
パラメータグリッドサーチ（TP倍率 × RSI範囲）
ADX>15 固定・月次ストップ・決算除外フィルター維持

パターン:
  ベースライン : TP=ATR×3  RSI 45-75
  A: TP×4     : TP=ATR×4  RSI 45-75
  B: RSI拡大   : TP=ATR×3  RSI 40-80
  C: A+B       : TP=ATR×4  RSI 40-80

指標計算（~17秒）は1回だけ実行し、シグナル生成以降をパターン数だけループ
"""
import os, sys, time
sys.stdout.reconfigure(encoding="utf-8")
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
import jquantsapi

sys.path.insert(0, str(Path(__file__).parent / "src"))
from indicators import add_indicators
from utils.risk import calc_position_size

sys.path.insert(0, str(Path(__file__).parent))
from adx_grid_search  import compute_indicators_once, select_stocks
from oos_backtest     import (
    fetch_earnings_exclusion, run_portfolio_backtest, compute_stats,
    IS_BT_START, IS_BT_END, OOS_START, OOS_END, OOS_WARMUP,
    MIN_TRADES, MIN_WIN_RATE, MAX_DD, MIN_PNL,
    TARGET_N, MAX_PER_SEC33, INITIAL_CAPITAL,
    ATR_STOP_MULT, MAX_HOLD_DAYS,
)

load_dotenv()

ADX_THRESH = 15
DATA_PATH  = Path("data/raw/prices_10y.parquet")

PATTERNS = [
    {"label": "ベースライン (TP×3, RSI45-75)", "atr_tp": 3, "rsi_lo": 45, "rsi_hi": 75},
    {"label": "A: TP×4         (RSI45-75)",   "atr_tp": 4, "rsi_lo": 45, "rsi_hi": 75},
    {"label": "B: RSI拡大       (TP×3)",       "atr_tp": 3, "rsi_lo": 40, "rsi_hi": 80},
    {"label": "C: A+B (TP×4, RSI40-80)",       "atr_tp": 4, "rsi_lo": 40, "rsi_hi": 80},
]


def run_is_with_params(df: pd.DataFrame, adx: float, atr_tp: float,
                       rsi_lo: int, rsi_hi: int,
                       earnings_excl: dict | None) -> dict:
    """指標計算済みDFにパラメータを適用してISバックテストを実行。"""
    g = df.groupby("Code", sort=False)
    N = len(df)

    prev_rsi = g["RSI14"].shift(3)
    df["signal"] = (
        (df["SMA20"] > df["SMA50"]) &
        (df["Close"] > df["SMA200"]) &
        (df["RSI14"] >= rsi_lo) & (df["RSI14"] <= rsi_hi) &
        (df["RSI14"] > prev_rsi) &
        (df["Volume"] >= df["VOL_MA20"] * 1.2) &
        (df["Close"] > df["Open"]) &
        (df["ADX14"] > adx)
    ).astype(int)

    sig_mask   = (
        (df["Date"] >= IS_BT_START) & (df["Date"] <= IS_BT_END) &
        (df["signal"] == 1) & df["ATR14"].notna()
    )
    sig_pos    = df.index[sig_mask].to_numpy()
    opens_a    = df["Open"].to_numpy()
    highs_a    = df["High"].to_numpy()
    lows_a     = df["Low"].to_numpy()
    closes_a   = df["Close"].to_numpy()
    atr_a      = df["ATR14"].to_numpy()
    code_int   = pd.factorize(df["Code"])[0]

    nxt        = sig_pos + 1
    same       = (nxt < N) & (code_int[nxt] == code_int[sig_pos])
    entry_open = np.where(same, opens_a[np.clip(nxt, 0, N-1)], np.nan)
    sig_close  = closes_a[sig_pos]
    gap_ok     = same & ((entry_open - sig_close) / sig_close >= -0.015)
    sig_pos    = sig_pos[gap_ok];  entry_open = entry_open[gap_ok]

    if earnings_excl:
        dates_a   = df["Date"].to_numpy()
        codes_a   = df["Code"].to_numpy()
        sig_dates = dates_a[sig_pos]
        ent_dates = dates_a[np.clip(sig_pos + 1, 0, N - 1)]
        keep = np.array([
            sig_dates[i] not in earnings_excl.get(codes_a[sig_pos[i]], set()) and
            ent_dates[i] not in earnings_excl.get(codes_a[sig_pos[i]], set())
            for i in range(len(sig_pos))
        ])
        sig_pos = sig_pos[keep];  entry_open = entry_open[keep]

    atr_v  = atr_a[sig_pos]
    TP     = entry_open + atr_v * atr_tp          # ← パラメータ化
    SL     = entry_open - atr_v * ATR_STOP_MULT
    stop_w = atr_v * 2
    by_r   = (INITIAL_CAPITAL * 0.02 / np.where(stop_w > 0, stop_w, np.inf)).astype(int)
    by_m   = (200_000 / np.where(entry_open > 0, entry_open, np.inf)).astype(int)
    shares = np.minimum(np.minimum(by_r, by_m), 100).clip(min=0)
    ok     = shares > 0
    sig_pos, entry_open, TP, SL, shares = (
        sig_pos[ok], entry_open[ok], TP[ok], SL[ok], shares[ok])

    exit_px = np.full(len(sig_pos), np.nan)
    for k in range(1, MAX_HOLD_DAYS + 1):
        fut     = sig_pos + k
        sc      = (fut < N) & (code_int[np.clip(fut, 0, N-1)] == code_int[sig_pos])
        H       = np.where(sc, highs_a[np.clip(fut, 0, N-1)],  np.nan)
        L       = np.where(sc, lows_a[np.clip(fut, 0, N-1)],   np.nan)
        C       = np.where(sc, closes_a[np.clip(fut, 0, N-1)], np.nan)
        ne      = np.isnan(exit_px)
        sl_hit  = ne & (L <= SL) & ~np.isnan(L)
        tp_hit  = ne & ~sl_hit & (H >= TP) & ~np.isnan(H)
        exit_px = np.where(sl_hit, SL, exit_px)
        exit_px = np.where(tp_hit, TP, exit_px)
        if k == MAX_HOLD_DAYS:
            to = ne & ~sl_hit & ~tp_hit & ~np.isnan(C)
            exit_px = np.where(to, C, exit_px)

    valid   = ~np.isnan(exit_px)
    pnl     = (exit_px[valid] - entry_open[valid]) * shares[valid]
    codes_v = df["Code"].to_numpy()[sig_pos[valid]]

    results: dict = {}
    for code5, grp in pd.DataFrame({"Code": codes_v, "pnl": pnl}).groupby("Code"):
        p = grp["pnl"].to_numpy()
        if len(p) < MIN_TRADES:
            continue
        wr  = float((p > 0).sum() / len(p) * 100)
        fp  = float(p.sum())
        eq  = np.concatenate([[INITIAL_CAPITAL], INITIAL_CAPITAL + np.cumsum(p)])
        pk  = np.maximum.accumulate(eq)
        dd  = float(((eq - pk) / pk * 100).min())
        results[code5] = {"total": len(p), "win_rate": wr,
                          "max_dd": dd, "final_pnl": fp, "trades": []}
    return results


def prepare_oos_stocks(df_all: pd.DataFrame, selected5: list, drop_cols: list,
                       adx: float, rsi_lo: int, rsi_hi: int) -> dict:
    """OOS用株データ: indicators + ADX14 + シグナル再計算（RSI範囲・ADX閾値適用）"""
    stock_data = {}
    for code5 in selected5:
        df_s = df_all[(df_all["Code"] == code5) & (df_all["Date"] >= OOS_WARMUP)].copy()
        df_s = df_s.drop(columns=drop_cols, errors="ignore").reset_index(drop=True)
        if len(df_s) < 260:
            continue
        df_s = add_indicators(df_s)

        # ADX14
        ph  = df_s["High"].shift(1);  pl = df_s["Low"].shift(1)
        dr  = (df_s["High"] - ph).to_numpy()
        dmr = (pl - df_s["Low"]).to_numpy()
        dpv = np.where((dr > 0) & (dr > dmr), dr, 0.0)
        dmv = np.where((dmr > 0) & (dmr > dr), dmr, 0.0)
        as_ = df_s["ATR14"].replace(0, np.nan)
        sdp = pd.Series(dpv, index=df_s.index).ewm(alpha=1/14, adjust=False).mean()
        sdm = pd.Series(dmv, index=df_s.index).ewm(alpha=1/14, adjust=False).mean()
        dip = 100 * sdp / as_;  dim = 100 * sdm / as_
        dxs = 100 * (dip - dim).abs() / (dip + dim).replace(0, np.nan)
        df_s["ADX14"] = dxs.ewm(alpha=1/14, adjust=False).mean()

        # シグナル再計算（RSI範囲・ADX閾値をパターンに合わせて上書き）
        prev_rsi = df_s["RSI14"].shift(3)
        df_s["signal"] = (
            (df_s["SMA20"] > df_s["SMA50"]) &
            (df_s["Close"] > df_s["SMA200"]) &
            (df_s["RSI14"] >= rsi_lo) & (df_s["RSI14"] <= rsi_hi) &
            (df_s["RSI14"] > prev_rsi) &
            (df_s["Volume"] >= df_s["VOL_MA20"] * 1.2) &
            (df_s["Close"] > df_s["Open"]) &
            (df_s["ADX14"] > adx)
        ).astype(int)
        stock_data[code5] = df_s
    return stock_data


def main() -> None:
    print("=" * 68)
    print("  パラメータグリッドサーチ（TP倍率 × RSI範囲）")
    print(f"  ADX固定: > {ADX_THRESH}  月次ストップ: 10%  決算除外: 有")
    print(f"  IS: {IS_BT_START.date()} ～ {IS_BT_END.date()}")
    print(f"  OOS: {OOS_START.date()} ～ {OOS_END.date()}")
    print("=" * 68)

    # ── 1回だけ読み込む ─────────────────────────────────────
    print("\nデータ・マスタ・決算日を読み込み中...")
    df_all = pd.read_parquet(DATA_PATH)
    df_all["Date"] = pd.to_datetime(df_all["Date"])
    df_all = df_all.sort_values(["Code", "Date"]).reset_index(drop=True)

    client = jquantsapi.ClientV2(api_key=os.getenv("JQUANTS_REFRESH_TOKEN"))
    master = client.get_eq_master()
    name_map   = dict(zip(master["Code"], master["CoName"]))
    sector_map = dict(zip(master["Code"], master["S33Nm"]))
    earnings_excl = fetch_earnings_exclusion(client)

    drop_cols = [c for c in ["Code4", "O", "H", "L", "C", "Vo", "Va", "UL", "LL", "AdjFactor"]
                 if c in df_all.columns]

    # ── 指標計算（全パターン共通・1回のみ） ────────────────
    print("\n指標を一括計算中（全パターン共通）...")
    t_indic = time.time()
    df_clean = df_all.drop(columns=drop_cols).sort_values(["Code", "Date"]).reset_index(drop=True)
    df_indic = compute_indicators_once(df_clean.copy())
    print(f"  完了: {time.time()-t_indic:.1f}秒")

    # ── パターンループ ──────────────────────────────────────
    grid = []
    for pat in PATTERNS:
        lbl    = pat["label"]
        atr_tp = pat["atr_tp"]
        rsi_lo = pat["rsi_lo"]
        rsi_hi = pat["rsi_hi"]

        print(f"\n{'─'*60}")
        print(f"  {lbl}")
        t_loop = time.time()

        # IS バックテスト
        is_res = run_is_with_params(
            df_indic.copy(), ADX_THRESH, atr_tp, rsi_lo, rsi_hi, earnings_excl)
        selected5 = select_stocks(is_res, sector_map)
        print(f"  IS完了: {len(is_res)}銘柄  選定: {len(selected5)}銘柄")
        if not selected5:
            print("  ❌ 選定銘柄なし → スキップ")
            continue

        # OOS 指標・シグナル計算
        stock_data_oos = prepare_oos_stocks(
            df_all, selected5, drop_cols, ADX_THRESH, rsi_lo, rsi_hi)
        names_oos = {c: name_map.get(c, c) for c in stock_data_oos}

        # OOS ポートフォリオバックテスト
        trades, equity = run_portfolio_backtest(
            stock_data_oos, names_oos,
            earnings_excl=earnings_excl,
            adx_threshold=ADX_THRESH,
            atr_tp_mult=atr_tp,
        )
        stats   = compute_stats(trades, equity)
        elapsed = time.time() - t_loop

        grid.append({
            "label":    lbl,
            "atr_tp":   atr_tp,
            "rsi":      f"{rsi_lo}-{rsi_hi}",
            "annual":   stats["annual"],
            "win_rate": stats["win_rate"],
            "max_dd":   stats["max_dd"],
            "trades":   stats["total"],
            "selected": len(selected5),
            "elapsed":  elapsed,
        })
        print(f"  年利: {stats['annual']:+.2f}%  勝率: {stats['win_rate']:.1f}%  "
              f"最大DD: {stats['max_dd']:.2f}%  取引: {stats['total']}回  ({elapsed:.0f}秒)")

    # ── 結果サマリー ────────────────────────────────────────
    print(f"\n{'='*68}")
    print("  比較結果サマリー")
    print(f"  OOS期間: {OOS_START.date()} ～ {OOS_END.date()}")
    print(f"  目標: 年利10%以上 かつ 最大DD -10%以内")
    print(f"{'='*68}")
    hdr = f"  {'パターン':<32}  {'TP':>4}  {'RSI':>7}  {'年利':>8}  {'勝率':>6}  {'最大DD':>8}  {'取引数':>5}  判定"
    print(hdr)
    print(f"  {'─'*78}")

    best_score = -999
    best_row   = None
    for r in grid:
        g_an = r["annual"] >= 10.0
        g_dd = r["max_dd"] >= -10.0
        if g_an and g_dd:
            icon = "✅ 達成"
            score = r["annual"] - abs(r["max_dd"])
            if score > best_score:
                best_score, best_row = score, r
        elif g_an:
            icon = "⚠️ DD超過"
        elif g_dd:
            icon = "⚠️ 年利不足"
        else:
            icon = "❌"
        print(f"  {r['label']:<32}  ×{r['atr_tp']:>1}  {r['rsi']:>7}  "
              f"{r['annual']:>+7.2f}%  {r['win_rate']:>5.1f}%  "
              f"{r['max_dd']:>7.2f}%  {r['trades']:>5}回  {icon}")

    print(f"  {'─'*78}")
    if best_row:
        print(f"\n  ★ 目標達成パターン: {best_row['label']}")
        print(f"     年利 {best_row['annual']:+.2f}%  勝率 {best_row['win_rate']:.1f}%  "
              f"最大DD {best_row['max_dd']:.2f}%  取引 {best_row['trades']}回")
    else:
        best_ann = max(grid, key=lambda x: x["annual"])
        best_dd  = max(grid, key=lambda x: x["max_dd"])
        print(f"\n  目標（年利10%+ かつ DD -10%以内）達成パターンなし")
        print(f"  年利最良: {best_ann['label']}  ({best_ann['annual']:+.2f}%)")
        print(f"  DD最良 : {best_dd['label']}  ({best_dd['max_dd']:.2f}%)")
    print(f"{'='*68}")


if __name__ == "__main__":
    main()
