"""
STEP 3: 戦略A + 戦略B 統合バックテスト
  IS選定期間  : 2021-01-01 ～ 2022-12-31（戦略A銘柄選定）
  OOS期間     : 2023-01-01 ～ 2026-04-24
  比較パターン: 戦略A単独 / 戦略B単独 / A+B統合（B優先）
"""
import os, sys, time, bisect
sys.stdout.reconfigure(encoding="utf-8")
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
import jquantsapi

sys.path.insert(0, str(Path(__file__).parent / "src"))
from indicators import add_indicators
from strategy   import generate_signals
from strategy_b import (add_indicators_b, generate_signals_b,
                         calc_stop_loss_b, calc_take_profit_b)
from utils.risk import calc_position_size

import oos_backtest as _ob
_ob.IS_BT_START = pd.Timestamp("2016-04-01")
_ob.IS_BT_END   = pd.Timestamp("2020-12-31")

load_dotenv()

# ── 定数 ────────────────────────────────────────────────────
IS_BT_START  = pd.Timestamp("2016-04-01")
IS_BT_END    = pd.Timestamp("2020-12-31")
OOS_START    = pd.Timestamp("2023-01-01")
OOS_END      = pd.Timestamp("2026-04-24")
OOS_WARMUP_A = pd.Timestamp("2022-01-01")   # 戦略A指標ウォームアップ
OOS_WARMUP_B = pd.Timestamp("2021-07-01")   # 戦略B 52週高値に260日必要

INITIAL_CAPITAL = 1_000_000
MAX_POSITIONS   = 5
MAX_POS_RATIO   = 0.20
MAX_HOLD_DAYS   = 10
ATR_TP_MULT     = 3
ATR_STOP_MULT   = 2
ADX_THRESHOLD   = 25.0

MIN_TRADES   = 10
MIN_WIN_RATE = 60.0
MAX_DD_IS    = -3.0
MIN_PNL      = 30_000
TARGET_N     = 30
MAX_PER_SEC33 = 3

DATA_PATH = Path("data/raw/prices_10y.parquet")


# ── データクラス ─────────────────────────────────────────────
@dataclass
class Position:
    code:        str
    name:        str
    entry_date:  pd.Timestamp
    entry_price: float
    shares:      int
    stop_loss:   float
    take_profit: float
    hold_days:   int = 0
    strategy:    str = "A"

@dataclass
class Trade:
    code:        str
    name:        str
    entry_date:  pd.Timestamp
    exit_date:   pd.Timestamp
    entry_price: float
    exit_price:  float
    shares:      int
    pnl:         float
    reason:      str
    strategy:    str = "A"


# ── 戦略A銘柄選定（IS 2021-2022 ベクトル化） ───────────────
def select_a_stocks(df_all, sector_map, earnings_excl):
    drop_cols = [c for c in ["Code4","O","H","L","C","Vo","Va","UL","LL","AdjFactor"]
                 if c in df_all.columns]
    df_clean = df_all.drop(columns=drop_cols).sort_values(["Code","Date"]).reset_index(drop=True)

    print(f"  IS期間バックテスト中（{IS_BT_START.date()} ～ {IS_BT_END.date()}）...")
    is_results = _ob.run_is_backtest_vectorized(df_clean, earnings_excl=earnings_excl)
    print(f"  IS完了: {len(is_results)} 銘柄")

    candidates = {c: r for c, r in is_results.items()
                  if (r["win_rate"]  >= MIN_WIN_RATE and
                      r["max_dd"]    >= MAX_DD_IS and
                      r["final_pnl"] >= MIN_PNL)}
    print(f"  フィルター通過: {len(candidates)} 銘柄")

    if len(candidates) <= TARGET_N:
        selected = list(candidates.keys())
    else:
        scored = sorted(candidates.items(),
                        key=lambda x: x[1]["win_rate"] * (1 - abs(x[1]["max_dd"])/100),
                        reverse=True)
        selected, sec_cnt = [], {}
        for code, _ in scored:
            sec = sector_map.get(code, "その他")
            if sec_cnt.get(sec, 0) < MAX_PER_SEC33:
                selected.append(code)
                sec_cnt[sec] = sec_cnt.get(sec, 0) + 1
            if len(selected) >= TARGET_N:
                break

    print(f"  選定銘柄数: {len(selected)}")
    return selected, candidates, is_results


# ── 戦略A OOSデータ準備 ─────────────────────────────────────
def prepare_oos_a(df_all, selected_codes):
    drop_cols = [c for c in ["Code4","O","H","L","C","Vo","Va","UL","LL","AdjFactor"]
                 if c in df_all.columns]
    stock_data = {}
    for code in selected_codes:
        df_s = df_all[(df_all["Code"] == code) &
                      (df_all["Date"] >= OOS_WARMUP_A)].copy()
        df_s = df_s.drop(columns=drop_cols, errors="ignore").reset_index(drop=True)
        if len(df_s) < 260:
            continue
        df_s = add_indicators(df_s)
        df_s = generate_signals(df_s)

        # ADX14計算
        prev_hi  = df_s["High"].shift(1)
        prev_lo  = df_s["Low"].shift(1)
        dm_p_r   = (df_s["High"] - prev_hi).to_numpy()
        dm_m_r   = (prev_lo - df_s["Low"]).to_numpy()
        dm_p_v   = np.where((dm_p_r > 0) & (dm_p_r > dm_m_r), dm_p_r, 0.0)
        dm_m_v   = np.where((dm_m_r > 0) & (dm_m_r > dm_p_r), dm_m_r, 0.0)
        atr_safe = df_s["ATR14"].replace(0, np.nan)
        sm_dp    = pd.Series(dm_p_v, index=df_s.index).ewm(alpha=1/14, adjust=False).mean()
        sm_dm    = pd.Series(dm_m_v, index=df_s.index).ewm(alpha=1/14, adjust=False).mean()
        di_p     = 100 * sm_dp / atr_safe
        di_m     = 100 * sm_dm / atr_safe
        dx       = 100 * (di_p - di_m).abs() / (di_p + di_m).replace(0, np.nan)
        df_s["ADX14"] = dx.ewm(alpha=1/14, adjust=False).mean()

        df_s.rename(columns={"signal": "signal_a"}, inplace=True)
        stock_data[code] = df_s
    return stock_data


# ── 戦略B OOSデータ準備（全銘柄ベクトル化） ──────────────────
def prepare_oos_b(df_all, eligible_codes):
    t0 = time.time()
    df = df_all[(df_all["Code"].isin(eligible_codes)) &
                (df_all["Date"] >= OOS_WARMUP_B)].copy()
    df = df.sort_values(["Code","Date"]).reset_index(drop=True)
    print(f"  B対象データ: {len(df):,}行  {df['Code'].nunique():,}銘柄")

    g = df.groupby("Code", sort=False)

    df["High52W_prev"] = g["High"].transform(
        lambda x: x.rolling(260, min_periods=260).max().shift(1))
    df["VOL_MA20_B"]   = g["Volume"].transform(
        lambda x: x.rolling(20, min_periods=20).mean())
    df["Low10D"]       = g["Low"].transform(
        lambda x: x.rolling(10, min_periods=1).min())
    if "Va" in df.columns:
        df["Va_MA20"]  = g["Va"].transform(
            lambda x: x.rolling(20, min_periods=20).mean())
        mcap_ok = df["Va_MA20"] >= 500_000_000
    else:
        mcap_ok = pd.Series(True, index=df.index)

    has_hist  = df["High52W_prev"].notna() & df["VOL_MA20_B"].notna()
    new_high  = df["Close"] >= df["High52W_prev"]
    vol_surge = df["Volume"] >= df["VOL_MA20_B"] * 2.0
    price_ok  = (df["Close"] >= 500) & (df["Close"] <= 3_000)
    bullish   = df["Close"] > df["Open"]

    df["signal_b"] = (has_hist & new_high & vol_surge & price_ok & bullish & mcap_ok).astype(int)

    # OOS期間にシグナルのある銘柄だけ残す
    oos_mask     = (df["Date"] >= OOS_START) & (df["Date"] <= OOS_END)
    active_codes = (df[oos_mask].groupby("Code")["signal_b"].sum() > 0)
    active_codes = active_codes[active_codes].index.tolist()
    print(f"  OOSシグナルあり: {len(active_codes)} 銘柄  ({time.time()-t0:.1f}秒)")

    stock_data = {}
    for code, grp in df[df["Code"].isin(active_codes)].groupby("Code"):
        stock_data[code] = grp.reset_index(drop=True)
    return stock_data


# ── シグナルイベント事前計算 ────────────────────────────────
def precompute_signals(stock_data_a: dict, stock_data_b: dict) -> tuple[dict, dict]:
    """各日付のシグナルを事前集計（シミュレーション高速化）
    Returns: (a_events, b_events)  各 {date: [(code, row_dict), ...]}
    """
    a_events: dict = {}
    for code, df in stock_data_a.items():
        oos = df[(df["Date"] >= OOS_START) & (df["Date"] <= OOS_END) &
                 (df["signal_a"] == 1) & df["ATR14"].notna() &
                 (df["ADX14"] > ADX_THRESHOLD)]
        for _, row in oos.iterrows():
            d = row["Date"]
            a_events.setdefault(d, []).append((code, row.to_dict()))

    b_events: dict = {}
    for code, df in stock_data_b.items():
        oos = df[(df["Date"] >= OOS_START) & (df["Date"] <= OOS_END) &
                 (df["signal_b"] == 1)]
        for _, row in oos.iterrows():
            d = row["Date"]
            b_events.setdefault(d, []).append((code, row.to_dict()))

    return a_events, b_events


# ── OOSポートフォリオシミュレーション ──────────────────────
def run_portfolio(stock_data_a: dict, stock_data_b: dict, names: dict,
                  a_events: dict, b_events: dict,
                  earnings_excl: dict, mode: str) -> tuple:
    """
    mode: "A"  → 戦略A単独
          "B"  → 戦略B単独
          "AB" → A+B統合（B優先）
    """
    # 統合ルックアップ（exit用 OHLC）
    all_stock_data = {}
    if mode in ("A", "AB"):
        all_stock_data.update(stock_data_a)
    if mode in ("B", "AB"):
        all_stock_data.update(stock_data_b)

    # ルックアップテーブル {code: {date: row_dict}}
    lookup: dict = {}
    for code, df in all_stock_data.items():
        lookup[code] = df.set_index("Date").to_dict("index")

    # ソート済み日付（バイナリサーチ用）
    sorted_dates: dict = {c: sorted(r.keys()) for c, r in lookup.items()}

    all_dates = sorted({d for rows in lookup.values()
                        for d in rows if OOS_START <= d <= OOS_END})

    capital          = float(INITIAL_CAPITAL)
    positions: list  = []
    trades:    list  = []
    equity           = {OOS_START: capital}

    cur_month        = None
    month_start_cap  = capital
    monthly_stopped  = False
    monthly_stop_cnt = 0

    for i, today in enumerate(all_dates):
        ym = (today.year, today.month)
        if ym != cur_month:
            cur_month       = ym
            month_start_cap = capital
            monthly_stopped = False

        # ── 決済判定 ───────────────────────────────────────
        next_pos = []
        for pos in positions:
            row = lookup[pos.code].get(today)
            if row is None:
                next_pos.append(pos)
                continue
            pos.hold_days += 1
            hi, lo, cl = row["High"], row["Low"], row["Close"]
            ep = er = None
            if lo <= pos.stop_loss:
                ep, er = pos.stop_loss, "損切り"
            elif hi >= pos.take_profit:
                ep, er = pos.take_profit, "利確"
            elif pos.hold_days >= MAX_HOLD_DAYS:
                ep, er = cl, "期間満了"
            if ep is not None:
                pnl = (ep - pos.entry_price) * pos.shares
                capital += pnl
                trades.append(Trade(pos.code, names.get(pos.code, pos.code),
                                    pos.entry_date, today,
                                    pos.entry_price, ep, pos.shares, pnl, er,
                                    pos.strategy))
            else:
                next_pos.append(pos)
        positions = next_pos

        # 月次損失チェック
        if not monthly_stopped and month_start_cap > 0:
            if (capital - month_start_cap) / month_start_cap * 100 <= -10.0:
                monthly_stopped = True
                monthly_stop_cnt += 1

        # ── エントリー判定 ─────────────────────────────────
        slots = MAX_POSITIONS - len(positions)
        if slots <= 0 or monthly_stopped:
            equity[today] = capital
            continue

        holding  = {p.code for p in positions}
        prev_day = all_dates[i-1] if i > 0 else None
        if prev_day is None:
            equity[today] = capital
            continue

        # シグナル収集（B優先）
        candidates: dict = {}   # code → ("A"/"B", prev_row_dict)
        if mode in ("A", "AB"):
            for code, prev_row in a_events.get(prev_day, []):
                if code in holding:
                    continue
                candidates[code] = ("A", prev_row)
        if mode in ("B", "AB"):
            for code, prev_row in b_events.get(prev_day, []):
                if code in holding:
                    continue
                candidates[code] = ("B", prev_row)   # B が A を上書き

        if not candidates:
            equity[today] = capital
            continue

        # エントリー計算
        valid_entries = []
        for code, (strat, prev_row) in candidates.items():
            today_row = lookup.get(code, {}).get(today)
            if today_row is None:
                continue
            ep  = today_row["Open"]
            gap = (ep - prev_row["Close"]) / prev_row["Close"]
            if gap < -0.015:
                continue
            if earnings_excl and today in earnings_excl.get(code, set()):
                continue

            if strat == "A":
                atr = float(prev_row.get("ATR14", 0))
                if atr <= 0:
                    continue
                sh, sl = calc_position_size(capital, atr, ep)
                sh     = min(sh, int(capital * MAX_POS_RATIO / ep))
                tp     = ep + atr * ATR_TP_MULT
            else:  # B
                low10d    = float(prev_row.get("Low10D", ep * 0.97))
                sl        = calc_stop_loss_b(ep, low10d)
                tp        = calc_take_profit_b(ep)
                stop_w    = ep - sl
                if stop_w <= 0:
                    continue
                by_risk   = int(capital * 0.02 / stop_w)
                by_max    = int(capital * MAX_POS_RATIO / ep)
                sh        = min(by_risk, by_max, 100)

            if sh <= 0 or ep * sh > capital:
                continue

            rsi = float(prev_row.get("RSI14", 50) or 50)
            valid_entries.append((rsi, code, ep, sh, sl, tp, strat))

        valid_entries.sort(key=lambda x: x[0], reverse=True)
        for rsi, code, ep, sh, sl, tp, strat in valid_entries[:slots]:
            positions.append(Position(code, names.get(code, code),
                                      today, ep, sh, sl, tp,
                                      strategy=strat))

        equity[today] = capital

    eq_s = pd.Series(equity).sort_index()
    print(f"    月次ストップ発動: {monthly_stop_cnt}回")
    return trades, eq_s


# ── 統計計算 ────────────────────────────────────────────────
def compute_stats(trades: list, equity: pd.Series) -> dict:
    total     = len(trades)
    wins      = [t for t in trades if t.pnl > 0]
    win_rate  = len(wins) / total * 100 if total else 0
    final_pnl = equity.iloc[-1] - INITIAL_CAPITAL
    years     = (OOS_END - OOS_START).days / 365
    annual    = ((equity.iloc[-1] / INITIAL_CAPITAL) ** (1 / years) - 1) * 100
    peak      = equity.cummax()
    max_dd    = ((equity - peak) / peak * 100).min()
    monthly   = equity.resample("ME").last().diff()
    monthly.iloc[0] = equity.resample("ME").last().iloc[0] - INITIAL_CAPITAL
    a_cnt = sum(1 for t in trades if t.strategy == "A")
    b_cnt = sum(1 for t in trades if t.strategy == "B")
    return {"total": total, "win_rate": win_rate, "final_pnl": final_pnl,
            "annual": annual, "max_dd": max_dd, "monthly": monthly,
            "final_cap": equity.iloc[-1], "a_cnt": a_cnt, "b_cnt": b_cnt}


# ── 市場区分から時価総額500億相当コードを取得 ───────────────
def get_large_cap_codes(client: jquantsapi.ClientV2) -> set:
    """ScaleCategory で大型株・中型株を500億以上の近似フィルターとして使用"""
    try:
        master = client.get_eq_master()
        if "ScaleCategory" in master.columns:
            large = master[master["ScaleCategory"].isin(["大型株", "中型株", "TOPIX Large70",
                                                          "TOPIX Core30", "TOPIX Mid400"])]
            codes = set(large["Code"].astype(str))
            print(f"  市場区分フィルター（500億相当）: {len(codes)} 銘柄")
            return codes
        else:
            print(f"  ScaleCategory列なし → 全銘柄を対象（Va代替フィルター適用）")
            return set(master["Code"].astype(str))
    except Exception as e:
        print(f"  警告: 市場区分取得失敗 ({e}) → 全銘柄対象")
        return set()


# ── メイン ──────────────────────────────────────────────────
def main():
    print("=" * 72)
    print("  戦略A + 戦略B 統合バックテスト")
    print(f"  IS選定期間: {IS_BT_START.date()} ～ {IS_BT_END.date()}")
    print(f"  OOS期間   : {OOS_START.date()} ～ {OOS_END.date()}")
    print("=" * 72)

    # ── データ読み込み ────────────────────────────────────
    print("\nデータ読み込み中...")
    df_all = pd.read_parquet(DATA_PATH)
    df_all["Date"] = pd.to_datetime(df_all["Date"])
    df_all = df_all.sort_values(["Code","Date"]).reset_index(drop=True)
    print(f"  {len(df_all):,}行  {df_all['Code'].nunique():,}銘柄")

    # ── 銘柄マスタ・API ───────────────────────────────────
    print("銘柄マスタ取得中...")
    api_key = os.getenv("JQUANTS_REFRESH_TOKEN")
    client  = jquantsapi.ClientV2(api_key=api_key)
    master  = client.get_eq_master()
    name_map   = dict(zip(master["Code"].astype(str), master["CoName"]))
    sector_map = dict(zip(master["Code"].astype(str), master["S33Nm"]))

    # 時価総額500億相当コード（戦略B用）
    large_cap_codes = get_large_cap_codes(client)

    # 決算除外データ
    print("決算除外データ取得中...")
    try:
        earnings_excl = _ob.fetch_earnings_exclusion(client)
    except Exception as e:
        print(f"  警告: 決算データ取得失敗 ({e}) → 除外なし")
        earnings_excl = {}

    # ────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  【戦略A銘柄選定】IS期間バックテスト")
    print("=" * 72)
    selected_a, candidates, is_results = select_a_stocks(df_all, sector_map, earnings_excl)

    # ────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  【OOS指標計算】戦略A・戦略B")
    print("=" * 72)

    print("\n戦略A OOSデータ準備中...")
    t0 = time.time()
    stock_data_a = prepare_oos_a(df_all, selected_a)
    print(f"  戦略A: {len(stock_data_a)}銘柄  ({time.time()-t0:.1f}秒)")

    print("\n戦略B OOSデータ準備中...")
    # large_cap_codes が空なら全銘柄対象（Va フィルターが働く）
    b_target = large_cap_codes if large_cap_codes else set(df_all["Code"].astype(str))
    stock_data_b = prepare_oos_b(df_all, b_target)

    # ────────────────────────────────────────────────────
    print("\nシグナルイベント事前計算中...")
    a_events, b_events = precompute_signals(stock_data_a, stock_data_b)
    total_a_sigs = sum(len(v) for v in a_events.values())
    total_b_sigs = sum(len(v) for v in b_events.values())
    print(f"  戦略A シグナル: {total_a_sigs:,}件  /  戦略B シグナル: {total_b_sigs:,}件")

    # 全銘柄名dict（A+B共通）
    names = {c: name_map.get(c, c) for c in set(stock_data_a) | set(stock_data_b)}

    # ────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  【OOSバックテスト実行】3パターン")
    print("=" * 72)

    results = {}
    for mode, label in [("A", "戦略A単独"), ("B", "戦略B単独"), ("AB", "A+B統合")]:
        print(f"\n  {label} 実行中...")
        t0 = time.time()
        trades, equity = run_portfolio(stock_data_a, stock_data_b, names,
                                       a_events, b_events, earnings_excl, mode)
        stats = compute_stats(trades, equity)
        results[mode] = (label, trades, equity, stats)
        elapsed = time.time() - t0
        print(f"    完了: {elapsed:.1f}秒  取引: {stats['total']}件  年利: {stats['annual']:+.2f}%")

    # ────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  ★ 比較結果 ★")
    print(f"  OOS期間: {OOS_START.date()} ～ {OOS_END.date()}")
    print("=" * 72)

    hdr = f"  {'戦略':10}  {'年利':>8}  {'勝率':>6}  {'最大DD':>8}  {'取引':>5}  {'判定':10}"
    sep = "  " + "-" * 60
    print(hdr)
    print(sep)

    for mode in ("A", "B", "AB"):
        label, trades, equity, s = results[mode]
        ok_ann = s["annual"] >= 10.0
        ok_dd  = s["max_dd"] >= -10.0
        if ok_ann and ok_dd:
            judge = "✅ 達成"
        elif ok_ann or ok_dd:
            judge = "⚠  一部達成"
        else:
            judge = "❌ 未達成"
        a_b_info = ""
        if mode == "AB":
            a_b_info = f" (A:{s['a_cnt']} B:{s['b_cnt']})"
        print(f"  {label:10}  {s['annual']:>+7.2f}%  {s['win_rate']:>5.1f}%  "
              f"{s['max_dd']:>7.2f}%  {s['total']:>5}{a_b_info}  {judge}")

    print(sep)
    print(f"  ※目標: 年利 +10%以上 かつ 最大DD -10%以内")

    # ── 月別損益（3パターン横並び） ─────────────────────
    print("\n【月別損益比較（OOS期間）】")
    eq_a = results["A"][2].resample("ME").last()
    eq_b = results["B"][2].resample("ME").last()
    eq_ab = results["AB"][2].resample("ME").last()

    mn_a  = results["A"][3]["monthly"]
    mn_b  = results["B"][3]["monthly"]
    mn_ab = results["AB"][3]["monthly"]

    all_months = sorted(set(eq_a.index) | set(eq_b.index) | set(eq_ab.index))
    print(f"  {'年月':8}  {'A損益':>10}  {'B損益':>10}  {'AB損益':>10}")
    print("  " + "-" * 48)
    for ym in all_months:
        if ym < OOS_START:
            continue
        def fmt(mn, ym):
            v = mn.get(ym, 0)
            s = "▲" if v < 0 else "+"
            return f"{s}{abs(v):>8,.0f}"
        print(f"  {ym.strftime('%Y-%m'):8}  {fmt(mn_a,ym)}  {fmt(mn_b,ym)}  {fmt(mn_ab,ym)}")

    print()
    print("=" * 72)
    print("  【A選定銘柄リスト（IS期間成績順）】")
    print(f"  {'コード':6}  {'銘柄名':16}  {'業種':16}  {'勝率':>6}  {'DD':>6}  {'損益':>10}")
    print("  " + "-" * 70)
    for code in selected_a:
        r   = candidates.get(code, is_results.get(code, {}))
        nm  = name_map.get(code, code)[:14]
        sec = sector_map.get(code, "")[:14]
        print(f"  {code:6}  {nm:16}  {sec:16}  {r.get('win_rate',0):5.1f}%  "
              f"{r.get('max_dd',0):5.1f}%  {r.get('final_pnl',0):>10,.0f}")
    print()


if __name__ == "__main__":
    main()
