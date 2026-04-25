"""
ADX閾値グリッドサーチ
- テスト値: 15 / 20 / 25 / 30
- 指標計算（最重処理）は1回だけ実行し、ADX閾値ループはシグナル生成以降のみ
- 月次ストップ・決算除外フィルターは全パターン共通で維持
- 目標: 年利10%以上 かつ 最大DD -10%以内
"""
import os
import sys
import time
sys.stdout.reconfigure(encoding="utf-8")
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
import jquantsapi

sys.path.insert(0, str(Path(__file__).parent / "src"))
from indicators import add_indicators
from strategy   import generate_signals
from utils.risk import calc_position_size

# oos_backtest.py から共通関数をインポート
sys.path.insert(0, str(Path(__file__).parent))
from oos_backtest import (
    fetch_earnings_exclusion,
    run_portfolio_backtest,
    compute_stats,
    IS_BT_START, IS_BT_END,
    OOS_START, OOS_END, OOS_WARMUP,
    MIN_TRADES, MIN_WIN_RATE, MAX_DD, MIN_PNL,
    TARGET_N, MAX_PER_SEC33,
    INITIAL_CAPITAL, ATR_TP_MULT, ATR_STOP_MULT, MAX_HOLD_DAYS,
)

load_dotenv()

ADX_THRESHOLDS = [15, 20, 25, 30]
DATA_PATH = Path("data/raw/prices_10y.parquet")


def compute_indicators_once(df: pd.DataFrame) -> pd.DataFrame:
    """指標をすべて一括計算（ADX含む）。グリッドサーチ前に1回だけ呼ぶ。"""
    print(f"  指標計算中（{len(df):,}行・全銘柄）...")
    g = df.groupby("Code", sort=False)

    df["SMA20"]    = g["Close"].transform(lambda x: x.rolling(20).mean())
    df["SMA50"]    = g["Close"].transform(lambda x: x.rolling(50).mean())
    df["SMA200"]   = g["Close"].transform(lambda x: x.rolling(200).mean())
    df["VOL_MA20"] = g["Volume"].transform(lambda x: x.rolling(20).mean())

    delta = g["Close"].diff()
    avg_g = delta.clip(lower=0).groupby(df["Code"]).transform(
        lambda x: x.ewm(alpha=1/14, adjust=False).mean())
    avg_l = (-delta).clip(lower=0).groupby(df["Code"]).transform(
        lambda x: x.ewm(alpha=1/14, adjust=False).mean())
    df["RSI14"] = 100 - 100 / (1 + avg_g / avg_l.replace(0, 1e-9))

    prev_cl = g["Close"].shift(1)
    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - prev_cl).abs(),
        (df["Low"]  - prev_cl).abs(),
    ], axis=1).max(axis=1)
    df["ATR14"] = tr.groupby(df["Code"]).transform(
        lambda x: x.ewm(alpha=1/14, adjust=False).mean())

    prev_high = g["High"].shift(1)
    prev_low  = g["Low"].shift(1)
    dm_p_r = (df["High"] - prev_high).to_numpy()
    dm_m_r = (prev_low  - df["Low"]).to_numpy()
    dm_p = np.where((dm_p_r > 0) & (dm_p_r > dm_m_r), dm_p_r, 0.0)
    dm_m = np.where((dm_m_r > 0) & (dm_m_r > dm_p_r), dm_m_r, 0.0)
    df["_dp"] = dm_p;  df["_dm"] = dm_m
    sm_dp = df.groupby("Code")["_dp"].transform(lambda x: x.ewm(alpha=1/14, adjust=False).mean())
    sm_dm = df.groupby("Code")["_dm"].transform(lambda x: x.ewm(alpha=1/14, adjust=False).mean())
    atr_s = df["ATR14"].replace(0, np.nan)
    di_p  = 100 * sm_dp / atr_s
    di_m  = 100 * sm_dm / atr_s
    dx    = 100 * (di_p - di_m).abs() / (di_p + di_m).replace(0, np.nan)
    df["ADX14"] = dx.groupby(df["Code"]).transform(lambda x: x.ewm(alpha=1/14, adjust=False).mean())
    df.drop(columns=["_dp", "_dm"], inplace=True)

    return df


def run_is_with_adx(df: pd.DataFrame, adx_thresh: float,
                    earnings_excl: dict | None) -> dict:
    """指標計算済みDataFrameにADX閾値を適用してISバックテスト実行。"""
    g = df.groupby("Code", sort=False)
    N  = len(df)

    # シグナル生成
    prev_rsi = g["RSI14"].shift(3)
    df["signal"] = (
        (df["SMA20"] > df["SMA50"]) &
        (df["Close"] > df["SMA200"]) &
        (df["RSI14"] >= 45) & (df["RSI14"] <= 75) &
        (df["RSI14"] > prev_rsi) &
        (df["Volume"] >= df["VOL_MA20"] * 1.2) &
        (df["Close"] > df["Open"]) &
        (df["ADX14"] > adx_thresh)
    ).astype(int)

    # IS期間シグナル抽出
    sig_mask = (
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

    # 決算除外
    if earnings_excl:
        dates_a    = df["Date"].to_numpy()
        codes_a    = df["Code"].to_numpy()
        sig_dates  = dates_a[sig_pos]
        ent_dates  = dates_a[np.clip(sig_pos + 1, 0, N - 1)]
        keep = np.array([
            sig_dates[i]  not in earnings_excl.get(codes_a[sig_pos[i]], set()) and
            ent_dates[i]  not in earnings_excl.get(codes_a[sig_pos[i]], set())
            for i in range(len(sig_pos))
        ])
        sig_pos = sig_pos[keep];  entry_open = entry_open[keep]

    atr_v  = atr_a[sig_pos]
    TP     = entry_open + atr_v * ATR_TP_MULT
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
        fut    = sig_pos + k
        sc     = (fut < N) & (code_int[np.clip(fut, 0, N-1)] == code_int[sig_pos])
        H      = np.where(sc, highs_a[np.clip(fut, 0, N-1)],  np.nan)
        L      = np.where(sc, lows_a[np.clip(fut, 0, N-1)],   np.nan)
        C      = np.where(sc, closes_a[np.clip(fut, 0, N-1)], np.nan)
        ne     = np.isnan(exit_px)
        sl_hit = ne & (L <= SL) & ~np.isnan(L)
        tp_hit = ne & ~sl_hit & (H >= TP) & ~np.isnan(H)
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


def select_stocks(is_results: dict, sector_map: dict) -> list:
    """選定基準フィルタ→業種分散で30銘柄選定。"""
    cands = {c: r for c, r in is_results.items()
             if r["win_rate"] >= MIN_WIN_RATE and r["max_dd"] >= MAX_DD
             and r["final_pnl"] >= MIN_PNL}
    if len(cands) < TARGET_N:
        return list(cands.keys())
    scored = sorted(cands.items(),
                    key=lambda x: x[1]["win_rate"] * (1 - abs(x[1]["max_dd"]) / 100),
                    reverse=True)
    selected, cnt = [], {}
    for code5, _ in scored:
        sec = sector_map.get(code5, "その他")
        if cnt.get(sec, 0) < MAX_PER_SEC33:
            selected.append(code5)
            cnt[sec] = cnt.get(sec, 0) + 1
        if len(selected) >= TARGET_N:
            break
    return selected


def prepare_oos_stocks(df_all: pd.DataFrame, selected5: list,
                       drop_cols: list) -> dict:
    """選定銘柄のOOS用DataFrameを作成（add_indicators + generate_signals + ADX）。"""
    stock_data = {}
    for code5 in selected5:
        df_s = df_all[(df_all["Code"] == code5) & (df_all["Date"] >= OOS_WARMUP)].copy()
        df_s = df_s.drop(columns=drop_cols, errors="ignore").reset_index(drop=True)
        if len(df_s) < 260:
            continue
        df_s = add_indicators(df_s)
        df_s = generate_signals(df_s)
        # ADX14 for OOS
        ph   = df_s["High"].shift(1);  pl = df_s["Low"].shift(1)
        dr   = (df_s["High"] - ph).to_numpy()
        dmr  = (pl - df_s["Low"]).to_numpy()
        dpv  = np.where((dr > 0) & (dr > dmr), dr, 0.0)
        dmv  = np.where((dmr > 0) & (dmr > dr), dmr, 0.0)
        as_  = df_s["ATR14"].replace(0, np.nan)
        sdp  = pd.Series(dpv, index=df_s.index).ewm(alpha=1/14, adjust=False).mean()
        sdm  = pd.Series(dmv, index=df_s.index).ewm(alpha=1/14, adjust=False).mean()
        dip  = 100 * sdp / as_;  dim = 100 * sdm / as_
        dxs  = 100 * (dip - dim).abs() / (dip + dim).replace(0, np.nan)
        df_s["ADX14"] = dxs.ewm(alpha=1/14, adjust=False).mean()
        stock_data[code5] = df_s
    return stock_data


def main() -> None:
    print("=" * 68)
    print("  ADX閾値グリッドサーチ")
    print(f"  テスト値: {ADX_THRESHOLDS}")
    print(f"  IS選定期間 : {IS_BT_START.date()} ～ {IS_BT_END.date()}")
    print(f"  OOSテスト期間: {OOS_START.date()} ～ {OOS_END.date()}")
    print("=" * 68)

    # ── データ・マスタ・決算日（1回だけ読み込む） ───────────────
    print("\nデータ読み込み中...")
    df_all = pd.read_parquet(DATA_PATH)
    df_all["Date"] = pd.to_datetime(df_all["Date"])
    df_all = df_all.sort_values(["Code", "Date"]).reset_index(drop=True)
    print(f"  {len(df_all):,}行  {df_all['Code'].nunique():,}銘柄")

    api_key = os.getenv("JQUANTS_REFRESH_TOKEN")
    client  = jquantsapi.ClientV2(api_key=api_key)
    master  = client.get_eq_master()
    name_map   = dict(zip(master["Code"], master["CoName"]))
    sector_map = dict(zip(master["Code"], master["S33Nm"]))

    earnings_excl = fetch_earnings_exclusion(client)

    # ── 指標計算（1回だけ：最重処理） ─────────────────────────
    drop_cols = [c for c in ["Code4", "O", "H", "L", "C", "Vo", "Va", "UL", "LL", "AdjFactor"]
                 if c in df_all.columns]
    df_clean = df_all.drop(columns=drop_cols).sort_values(["Code", "Date"]).reset_index(drop=True)

    t_indic = time.time()
    print("\n指標を事前一括計算中（全閾値共通）...")
    df_indic = compute_indicators_once(df_clean.copy())
    print(f"  完了: {time.time()-t_indic:.1f}秒")

    # ── ADX閾値グリッドサーチ ────────────────────────────────
    grid_results = []

    for adx_thresh in ADX_THRESHOLDS:
        print(f"\n{'─'*60}")
        print(f"  ADX > {adx_thresh} でテスト中...")
        t_loop = time.time()

        # IS期間バックテスト（シグナル生成以降のみ）
        is_res = run_is_with_adx(df_indic.copy(), adx_thresh, earnings_excl)
        print(f"  IS完了: {len(is_res)}銘柄が取引あり")

        # 30銘柄選定
        selected5 = select_stocks(is_res, sector_map)
        print(f"  選定: {len(selected5)}銘柄")
        if not selected5:
            print("  ❌ 選定銘柄なし → スキップ")
            continue

        # OOS指標計算
        stock_data_oos = prepare_oos_stocks(df_all, selected5, drop_cols)
        names_oos      = {c: name_map.get(c, c) for c in stock_data_oos}

        # OOSポートフォリオバックテスト
        trades, equity = run_portfolio_backtest(
            stock_data_oos, names_oos,
            earnings_excl=earnings_excl,
            adx_threshold=adx_thresh,
        )
        stats = compute_stats(trades, equity)
        elapsed = time.time() - t_loop

        grid_results.append({
            "adx":      adx_thresh,
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
    print("  ADX閾値グリッドサーチ結果サマリー")
    print(f"  OOS期間: {OOS_START.date()} ～ {OOS_END.date()}")
    print(f"  目標: 年利10%以上 かつ 最大DD -10%以内")
    print(f"{'='*68}")
    print(f"  {'ADX閾値':>6}  {'年利':>8}  {'勝率':>6}  {'最大DD':>8}  {'取引数':>6}  {'選定数':>6}  目標達成")
    print(f"  {'─'*62}")

    for r in grid_results:
        goal_annual = r["annual"] >= 10.0
        goal_dd     = r["max_dd"] >= -10.0
        achieved    = "✅ 達成" if (goal_annual and goal_dd) else (
                      "⚠️ DD未達" if goal_annual else
                      "⚠️ 年利未達" if goal_dd else "❌")
        mark = " ◀ 最適" if (goal_annual and goal_dd) else ""
        print(f"  ADX>{r['adx']:>2}  {r['annual']:>+7.2f}%  {r['win_rate']:>5.1f}%  "
              f"{r['max_dd']:>7.2f}%  {r['trades']:>6}回  {r['selected']:>6}銘柄  "
              f"{achieved}{mark}")

    print(f"{'─'*68}")
    best = max(grid_results, key=lambda x: x["annual"] if x["max_dd"] >= -10.0 else -999,
               default=None)
    if best and best["max_dd"] >= -10.0:
        print(f"\n  ★ 最適値: ADX > {best['adx']}")
        print(f"     年利 {best['annual']:+.2f}%  勝率 {best['win_rate']:.1f}%  "
              f"最大DD {best['max_dd']:.2f}%  取引 {best['trades']}回")
    else:
        print(f"\n  目標（年利10%以上 かつ DD -10%以内）を達成したパターンはありません。")
        best_dd = max(grid_results, key=lambda x: x["max_dd"])
        best_ret = max(grid_results, key=lambda x: x["annual"])
        print(f"  最大DD最良: ADX>{best_dd['adx']}  ({best_dd['max_dd']:.2f}%)")
        print(f"  年利最良  : ADX>{best_ret['adx']}  ({best_ret['annual']:+.2f}%)")
    print(f"{'='*68}")


if __name__ == "__main__":
    main()
