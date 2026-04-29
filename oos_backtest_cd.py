"""
戦略C/D OOSバックテスト（取引コスト込み）
IS期間 : 2016-04-01 ～ 2020-12-31（候補選定）
OOS期間: 2023-01-01 ～ 2026-04-24（検証）
"""
import os
import sys
import time
sys.stdout.reconfigure(encoding="utf-8")

from pathlib import Path
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from scipy.stats import binom
import jquantsapi

sys.path.insert(0, str(Path(__file__).parent / "src"))
from indicators import add_indicators
from strategy_c import generate_signals as gen_c
from strategy_d import generate_signals as gen_d
from strategy_e import generate_signals as gen_e

load_dotenv()

# ── 期間設定 ────────────────────────────────────────────────
IS_START  = pd.Timestamp("2016-04-01")
IS_END    = pd.Timestamp("2020-12-31")
OOS_START = pd.Timestamp("2023-01-01")
OOS_END   = pd.Timestamp("2026-04-24")
OOS_WARMUP= pd.Timestamp("2022-01-01")

INITIAL_CAPITAL = 1_000_000
MAX_POSITIONS   = 5
MAX_POS_RATIO   = 0.20
DATA_PATH       = Path("data/raw/prices_10y.parquet")

# ── 取引コスト（戦略Aと同一） ────────────────────────────────
COMMISSION_RATE = 0.00055
SLIPPAGE_RATE   = 0.00050
COST_PER_LEG    = COMMISSION_RATE + SLIPPAGE_RATE

# ── 戦略別パラメータ ─────────────────────────────────────────
P_VALUE_MAX = 0.10  # 二項検定p値上限（IS過適合対策）

PARAMS = {
    "C": dict(atr_tp=2.5, atr_sl=1.5, max_hold=7,
              min_trades=5, min_win_rate=40.0, max_dd=-5.0, min_pnl=10_000, target_n=35),
    "D": dict(atr_tp=4.5, atr_sl=1.5, max_hold=5,
              min_trades=3, min_win_rate=50.0, max_dd=-5.0, min_pnl=10_000, target_n=50),
    "E": dict(atr_tp=6.0, atr_sl=2.0, max_hold=10,      # 52週高値ブレイクアウト（戦略A同等設定）
              min_trades=10, min_win_rate=60.0, max_dd=-3.0, min_pnl=30_000, target_n=30),
}


def run_is_backtest(df_all: pd.DataFrame, strategy: str) -> dict:
    """IS期間 全銘柄バックテスト（ベクトル化・コスト込み）"""
    p    = PARAMS[strategy]
    gen  = gen_c if strategy == "C" else (gen_d if strategy == "D" else gen_e)
    t0   = time.time()

    print(f"  [1/3] 指標・シグナル計算中（{len(df_all):,}行）...")
    df = df_all.reset_index(drop=True)
    g  = df.groupby("Code", sort=False)

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
    tr = pd.concat([df["High"]-df["Low"],
                    (df["High"]-prev_cl).abs(),
                    (df["Low"]-prev_cl).abs()], axis=1).max(axis=1)
    df["ATR14"] = tr.groupby(df["Code"]).transform(
        lambda x: x.ewm(alpha=1/14, adjust=False).mean())

    # 銘柄ごとにシグナル生成（groupbyで銘柄分離）
    sig_list = []
    for code, grp in df.groupby("Code", sort=False):
        s = gen(grp.copy())
        sig_list.append(s["signal"])
    df["signal"] = pd.concat(sig_list).reindex(df.index).fillna(0).astype(int)
    print(f"      完了: {time.time()-t0:.1f}秒")

    print(f"  [2/3] トレード計算中...")
    t1   = time.time()
    N    = len(df)
    code_int = pd.factorize(df["Code"])[0]

    sig_mask = (
        (df["Date"] >= IS_START) & (df["Date"] <= IS_END) &
        (df["signal"] == 1) & df["ATR14"].notna()
    )
    sig_pos = df.index[sig_mask].to_numpy()

    opens_a  = df["Open"].to_numpy()
    highs_a  = df["High"].to_numpy()
    lows_a   = df["Low"].to_numpy()
    closes_a = df["Close"].to_numpy()
    atr_a    = df["ATR14"].to_numpy()

    nxt  = sig_pos + 1
    same = (nxt < N) & (code_int[np.clip(nxt,0,N-1)] == code_int[sig_pos])
    entry_open = np.where(same, opens_a[np.clip(nxt,0,N-1)], np.nan)

    valid = same & ~np.isnan(entry_open)
    sig_pos    = sig_pos[valid]
    entry_open = entry_open[valid]

    atr_v = atr_a[sig_pos]
    TP    = entry_open + atr_v * p["atr_tp"]
    SL    = entry_open - atr_v * p["atr_sl"]

    by_max = (200_000 / np.where(entry_open > 0, entry_open, np.inf)).astype(int)
    shares = np.minimum(by_max, 100).clip(min=1)

    n_sigs  = len(sig_pos)
    exit_px = np.full(n_sigs, np.nan)
    for k in range(1, p["max_hold"] + 1):
        fut = sig_pos + k
        ok  = (fut < N) & (code_int[np.clip(fut,0,N-1)] == code_int[sig_pos])
        H   = np.where(ok, highs_a[np.clip(fut,0,N-1)],  np.nan)
        L   = np.where(ok, lows_a[np.clip(fut,0,N-1)],   np.nan)
        C   = np.where(ok, closes_a[np.clip(fut,0,N-1)], np.nan)
        not_ex = np.isnan(exit_px)
        sl_hit = not_ex & (L <= SL) & ~np.isnan(L)
        tp_hit = not_ex & ~sl_hit & (H >= TP) & ~np.isnan(H)
        exit_px = np.where(sl_hit, SL, exit_px)
        exit_px = np.where(tp_hit, TP, exit_px)
        if k == p["max_hold"]:
            timeout = not_ex & ~sl_hit & ~tp_hit & ~np.isnan(C)
            exit_px = np.where(timeout, C, exit_px)

    valid2     = ~np.isnan(exit_px)
    entry_open = entry_open[valid2]
    exit_px    = exit_px[valid2]
    shares     = shares[valid2]
    sig_pos    = sig_pos[valid2]
    cost       = (entry_open + exit_px) * shares * COST_PER_LEG
    pnl        = (exit_px - entry_open) * shares - cost
    codes_v    = df["Code"].to_numpy()[sig_pos]

    print(f"      完了: {time.time()-t1:.1f}秒  有効トレード: {len(pnl):,}件")

    print(f"  [3/3] 銘柄別集計中...")
    trade_df = pd.DataFrame({"Code": codes_v, "pnl": pnl})
    results  = {}
    for code, grp in trade_df.groupby("Code"):
        p_ = grp["pnl"].to_numpy()
        total = len(p_)
        if total < PARAMS[strategy]["min_trades"]:
            continue
        wins      = int((p_ > 0).sum())
        win_rate  = float(wins / total * 100)
        final_pnl = float(p_.sum())
        eq        = np.concatenate([[INITIAL_CAPITAL], INITIAL_CAPITAL + np.cumsum(p_)])
        peak      = np.maximum.accumulate(eq)
        max_dd    = float(((eq - peak) / peak * 100).min())
        p_value   = float(1 - binom.cdf(wins - 1, total, 0.5))
        results[code] = {"total": total, "win_rate": win_rate,
                         "max_dd": max_dd, "final_pnl": final_pnl, "p_value": p_value}

    print(f"      有効銘柄: {len(results)}  総経過: {time.time()-t0:.1f}秒")
    return results


def run_oos_portfolio(stock_data: dict, names: dict, strategy: str) -> tuple:
    """OOSポートフォリオバックテスト（コスト込み）"""
    p   = PARAMS[strategy]
    gen = gen_c if strategy == "C" else (gen_d if strategy == "D" else gen_e)

    all_dates = sorted({d for df in stock_data.values()
                        for d in df["Date"] if OOS_START <= d <= OOS_END})
    lookup = {}
    for code, df in stock_data.items():
        df2 = add_indicators(df.copy())
        df2 = gen(df2)
        df2 = df2.drop_duplicates(subset=["Date"], keep="last").reset_index(drop=True)
        lookup[code] = df2.set_index("Date").to_dict("index")

    capital   = float(INITIAL_CAPITAL)
    positions = []
    trades    = []
    equity    = {OOS_START: capital}

    cur_month = None; month_cap = capital; monthly_stopped = False; stop_cnt = 0

    for today in all_dates:
        ym = (today.year, today.month)
        if ym != cur_month:
            cur_month = ym; month_cap = capital; monthly_stopped = False

        # 決済
        next_pos = []
        for pos in positions:
            row = lookup[pos["code"]].get(today)
            if row is None:
                next_pos.append(pos); continue
            pos["days"] += 1
            hi, lo, cl = row["High"], row["Low"], row["Close"]
            ep = er = None
            if lo <= pos["sl"]:
                ep, er = pos["sl"], "損切り"
            elif hi >= pos["tp"]:
                ep, er = pos["tp"], "利確"
            elif pos["days"] >= p["max_hold"]:
                ep, er = cl, "強制終了"
            if ep is not None:
                cost = (pos["entry"] + ep) * pos["shares"] * COST_PER_LEG
                pnl  = (ep - pos["entry"]) * pos["shares"] - cost
                capital += pnl
                trades.append({"code": pos["code"], "pnl": pnl, "reason": er, "date": today})
            else:
                next_pos.append(pos)
        positions = next_pos

        if not monthly_stopped and month_cap > 0:
            if (capital - month_cap) / month_cap * 100 <= -10.0:
                monthly_stopped = True; stop_cnt += 1

        slots   = MAX_POSITIONS - len(positions)
        holding = {pos["code"] for pos in positions}
        if slots > 0 and not monthly_stopped:
            candidates = []
            for code, rows in lookup.items():
                if code in holding: continue
                prev_dates = [d for d in rows if d < today]
                if not prev_dates: continue
                prev_row = rows[max(prev_dates)]
                if prev_row.get("signal", 0) != 1: continue
                if pd.isna(prev_row.get("ATR14")): continue
                today_row = rows.get(today)
                if today_row is None: continue
                ep  = today_row["Open"]
                atr = float(prev_row["ATR14"])
                candidates.append((float(prev_row.get("RSI14", 0)), code, ep, atr))

            candidates.sort(reverse=True)
            for rsi, code, ep, atr in candidates[:slots]:
                sh = max(1, int(200_000 / ep))
                tp = ep + atr * p["atr_tp"]
                sl = ep - atr * p["atr_sl"]
                if ep * sh <= capital:
                    positions.append({"code": code, "entry": ep, "tp": tp,
                                      "sl": sl, "shares": sh, "days": 0})
        equity[today] = capital

    equity_s = pd.Series(equity).sort_index()
    print(f"  月次ストップ発動: {stop_cnt}回")
    return trades, equity_s


def compute_stats(trades: list, equity: pd.Series) -> dict:
    total     = len(trades)
    wins      = [t for t in trades if t["pnl"] > 0]
    win_rate  = len(wins) / total * 100 if total else 0
    final_pnl = equity.iloc[-1] - INITIAL_CAPITAL
    years     = (OOS_END - OOS_START).days / 365
    annual    = ((equity.iloc[-1] / INITIAL_CAPITAL) ** (1 / years) - 1) * 100
    peak      = equity.cummax()
    max_dd    = ((equity - peak) / peak * 100).min()
    total_cost= sum((200_000 / max(t.get("entry_price", 200_000), 1))
                    * COST_PER_LEG for t in trades) if trades else 0
    return {"total": total, "win_rate": win_rate, "final_pnl": final_pnl,
            "annual": annual, "max_dd": max_dd, "final_cap": equity.iloc[-1]}


def run_strategy(strategy: str, df_all: pd.DataFrame, name_map: dict, sector_map: dict) -> None:
    p = PARAMS[strategy]
    print(f"\n{'='*68}")
    print(f"  戦略{strategy} OOSバックテスト（コスト込み）")
    print(f"{'='*68}")

    # IS選定
    print(f"\n【STEP1】IS期間バックテスト: {IS_START.date()} ～ {IS_END.date()}")
    drop_cols = [c for c in ["Code4","O","H","L","C","Vo","Va","UL","LL","AdjFactor"]
                 if c in df_all.columns]
    df_clean  = df_all.drop(columns=drop_cols).sort_values(["Code","Date"]).reset_index(drop=True)
    is_results = run_is_backtest(df_clean, strategy)

    # フィルタリング
    print(f"\n【STEP2】選定基準フィルタリング")
    candidates = {c: r for c, r in is_results.items()
                  if r["win_rate"]           >= p["min_win_rate"]
                  and r["max_dd"]            >= p["max_dd"]
                  and r["final_pnl"]         >= p["min_pnl"]
                  and r.get("p_value", 1.0)  <= P_VALUE_MAX}
    print(f"  基準通過: {len(candidates)} 銘柄（p値{P_VALUE_MAX}以下フィルター適用）")

    scored = sorted(candidates.items(),
                    key=lambda x: x[1]["win_rate"] * (1 - abs(x[1]["max_dd"]) / 100),
                    reverse=True)
    selected = [c for c, _ in scored[:p["target_n"]]]
    print(f"  選定: {len(selected)} 銘柄")

    # OOS指標計算
    print(f"\n【STEP3】OOS指標計算中（ウォームアップ: {OOS_WARMUP.date()}〜）")
    stock_data = {}
    for code in selected:
        df_s = df_all[(df_all["Code"] == code) & (df_all["Date"] >= OOS_WARMUP)].copy()
        df_s = df_s.drop(columns=drop_cols, errors="ignore").reset_index(drop=True)
        if len(df_s) >= 260:
            stock_data[code] = df_s
    print(f"  OOS対象: {len(stock_data)} 銘柄")

    names = {c: name_map.get(c, c) for c in stock_data}

    # OOSポートフォリオバックテスト
    print(f"\n【STEP4】OOSポートフォリオバックテスト実行中...")
    trades, equity = run_oos_portfolio(stock_data, names, strategy)
    stats = compute_stats(trades, equity)

    # 結果表示
    print(f"\n{'='*68}")
    print(f"  ★ 戦略{strategy} OOSテスト結果 ★")
    print(f"{'='*68}")
    print(f"  初期資金      : {INITIAL_CAPITAL:>12,} 円")
    print(f"  最終資産      : {stats['final_cap']:>12,.0f} 円")
    print(f"  総損益        : {stats['final_pnl']:>+12,.0f} 円")
    print(f"  年利換算(CAGR): {stats['annual']:>+11.2f} %")
    print(f"  総取引数      : {stats['total']:>12} 回")
    print(f"  勝率          : {stats['win_rate']:>11.1f} %")
    print(f"  最大DD        : {stats['max_dd']:>11.2f} %")
    print(f"  ── 取引コスト（片道{COST_PER_LEG*100:.3f}%） ──────────────────────")

    # 選定銘柄リスト表示
    print(f"\n【選定{len(selected)}銘柄（IS成績順）】")
    print(f"  {'コード':6}  {'銘柄名':18}  {'勝率':>6}  {'DD':>6}  {'損益':>10}  {'取引':>4}  {'p値':>6}")
    print("  " + "-" * 66)
    for code in selected[:20]:
        r   = candidates[code]
        nm  = name_map.get(code, code)[:16]
        print(f"  {code:6}  {nm:18}  {r['win_rate']:5.1f}%  "
              f"{r['max_dd']:5.1f}%  {r['final_pnl']:>10,.0f}  {r['total']:>4}  {r.get('p_value',1.0):5.3f}")
    if len(selected) > 20:
        print(f"  ... 他{len(selected)-20}銘柄")


def main() -> None:
    print("=" * 68)
    print("  戦略C/D OOSバックテスト（取引コスト込み）")
    print(f"  IS: {IS_START.date()} ～ {IS_END.date()}")
    print(f"  OOS: {OOS_START.date()} ～ {OOS_END.date()}")
    print("=" * 68)

    print("\nデータ読み込み中...")
    df_all = pd.read_parquet(DATA_PATH)
    df_all["Date"] = pd.to_datetime(df_all["Date"])
    df_all["Code"] = df_all["Code4"].astype(str) + "0"
    df_all = df_all.sort_values(["Code", "Date"]).reset_index(drop=True)
    print(f"  {len(df_all):,} 行  {df_all['Code'].nunique():,} 銘柄")

    print("銘柄マスタ取得中...")
    api_key    = os.getenv("JQUANTS_REFRESH_TOKEN")
    client     = jquantsapi.ClientV2(api_key=api_key)
    master     = client.get_eq_master()
    name_map   = dict(zip(master["Code"], master["CoName"]))
    sector_map = dict(zip(master["Code"], master["S33Nm"]))

    run_strategy("C", df_all, name_map, sector_map)
    run_strategy("D", df_all, name_map, sector_map)
    run_strategy("E", df_all, name_map, sector_map)


if __name__ == "__main__":
    main()
