"""
真のOOSバックテスト
- 選定期間(IS): 2016-04-01 ～ 2020-12-31（銘柄選定）
- ギャップ期間 : 2021-01-01 ～ 2022-12-31（未使用）
- OOSテスト期間: 2023-01-01 ～ 2026-04-24（検証）
- 使用データ   : data/raw/prices_10y.parquet
"""
import os
import sys
import time
sys.stdout.reconfigure(encoding="utf-8")

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from scipy.stats import binom
import jquantsapi

sys.path.insert(0, str(Path(__file__).parent / "src"))
from indicators import add_indicators
from strategy   import generate_signals
from utils.risk import calc_position_size

load_dotenv()

# ── 期間設定 ────────────────────────────────────────────────
IS_BT_START = pd.Timestamp("2016-04-01")
IS_BT_END   = pd.Timestamp("2020-12-31")
OOS_START   = pd.Timestamp("2023-01-01")
OOS_END     = pd.Timestamp("2026-04-24")
OOS_WARMUP  = pd.Timestamp("2022-01-01")   # 指標ウォームアップ用

# ── 銘柄選定基準 ────────────────────────────────────────────
MIN_TRADES    = 10
MIN_WIN_RATE  = 60.0
MAX_DD        = -3.0     # -3%以内
MIN_PNL       = 30_000
TARGET_N      = 30
MAX_PER_SEC33 = 3        # 業種33分類で同一業種の上限
P_VALUE_MAX   = 0.10     # 二項検定p値上限（有意水準10%）← IS過適合対策

# ── バックテスト設定 ────────────────────────────────────────
INITIAL_CAPITAL = 1_000_000
MAX_POSITIONS   = 5
MAX_POS_RATIO   = 0.20
ATR_TP_MULT     = 6   # 改善設定（TP×6）：コスト比改善のため4→6に引き上げ
ATR_STOP_MULT   = 2
MAX_HOLD_DAYS   = 10
ADX_THRESHOLD   = 15  # 確定ベスト設定（ADX>15）

# ── 取引コスト設定 ───────────────────────────────────────────
# 片道コスト = 手数料 + スリッページ
# 手数料: SBI/楽天等スタンダードプラン相当 0.055%
# スリッページ: 東証プライム流動株の推定値 0.05%
COMMISSION_RATE = 0.00055
SLIPPAGE_RATE   = 0.00050
COST_PER_LEG    = COMMISSION_RATE + SLIPPAGE_RATE  # 片道 0.105%

DATA_PATH = Path("data/raw/prices_20y.parquet")


def run_single_backtest(df: pd.DataFrame, bt_start: pd.Timestamp, bt_end: pd.Timestamp) -> dict:
    df = add_indicators(df)
    df = generate_signals(df)
    df = df[df["Date"] <= bt_end].copy()

    capital  = float(INITIAL_CAPITAL)
    trades   = []
    equity   = []
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
                exit_price, exit_reason = stop_loss, "損切り"
            elif hi >= take_profit:
                exit_price, exit_reason = take_profit, "利確"
            elif hold_days >= MAX_HOLD_DAYS:
                exit_price, exit_reason = cl, "期間満了"
            if exit_price is not None:
                cost = (entry_price + exit_price) * shares * COST_PER_LEG
                pnl = (exit_price - entry_price) * shares - cost
                capital += pnl
                trades.append({"entry_date": entry_date, "exit_date": row["Date"],
                               "entry_price": entry_price, "exit_price": exit_price,
                               "shares": shares, "pnl": pnl, "cost": cost, "reason": exit_reason})
                in_trade = False
        equity.append((row["Date"], capital))

        if not in_trade and prev.get("signal", 0) == 1 and pd.notna(prev.get("ATR14")):
            ep = row["Open"]
            gap = (ep - prev["Close"]) / prev["Close"]
            if gap >= -0.015:
                atr = float(prev["ATR14"])
                sh, sl = calc_position_size(capital, atr, ep)
                tp = ep + atr * ATR_TP_MULT
                if sh > 0 and ep * sh <= capital:
                    in_trade    = True
                    entry_price = ep
                    stop_loss   = sl
                    take_profit = tp
                    entry_date  = row["Date"]
                    hold_days   = 0
                    shares      = sh

    total    = len(trades)
    wins     = [t for t in trades if t["pnl"] > 0]
    win_rate = len(wins) / total * 100 if total else 0
    final_pnl = capital - INITIAL_CAPITAL

    eq_s   = pd.Series([e[1] for e in equity]) if equity else pd.Series([INITIAL_CAPITAL])
    max_dd = ((eq_s - eq_s.cummax()) / eq_s.cummax() * 100).min() if len(eq_s) > 1 else 0.0

    return {"total": total, "win_rate": win_rate, "max_dd": float(max_dd),
            "final_pnl": final_pnl, "trades": trades}


def run_is_backtest_vectorized(df_all: pd.DataFrame,
                               earnings_excl: dict | None = None,
                               adx_threshold: float = 25.0) -> dict:
    """全銘柄IS期間バックテストを完全ベクトル化で一括実行。

    処理フロー:
      1. groupby.transform で全銘柄指標を一括計算（pandas_ta 個別呼び出し廃止）
      2. ブール演算でシグナルを一括生成
      3. numpy 配列インデックスで次10日の出口価格を O(n_sigs×10) で計算
      4. groupby.agg で銘柄別メトリクスを集計

    Returns: {code5: {"total", "win_rate", "max_dd", "final_pnl"}}
    """
    t0 = time.time()

    # ── [1/4] 全銘柄指標を一括ベクトル計算 ──────────────────
    print(f"  [1/4] 指標計算中（{len(df_all):,}行・全銘柄一括）...")
    # df_all は [Code, Date] ソート済み・integer index 前提
    df = df_all.reset_index(drop=True)
    g  = df.groupby("Code", sort=False)

    # SMA: rolling mean
    df["SMA20"]    = g["Close"].transform(lambda x: x.rolling(20).mean())
    df["SMA50"]    = g["Close"].transform(lambda x: x.rolling(50).mean())
    df["SMA200"]   = g["Close"].transform(lambda x: x.rolling(200).mean())
    df["VOL_MA20"] = g["Volume"].transform(lambda x: x.rolling(20).mean())

    # RSI14: Wilder's smoothing (EWM alpha=1/14)
    delta = g["Close"].diff()
    avg_g = delta.clip(lower=0).groupby(df["Code"]).transform(
        lambda x: x.ewm(alpha=1/14, adjust=False).mean())
    avg_l = (-delta).clip(lower=0).groupby(df["Code"]).transform(
        lambda x: x.ewm(alpha=1/14, adjust=False).mean())
    df["RSI14"] = 100 - 100 / (1 + avg_g / avg_l.replace(0, 1e-9))

    # ATR14: True Range の Wilder's smoothing
    prev_cl = g["Close"].shift(1)
    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - prev_cl).abs(),
        (df["Low"]  - prev_cl).abs(),
    ], axis=1).max(axis=1)
    df["ATR14"] = tr.groupby(df["Code"]).transform(
        lambda x: x.ewm(alpha=1/14, adjust=False).mean())

    # ADX14: DM+/DM- → DI+/DI- → DX → ADX（Wilder's smoothing）
    prev_high = g["High"].shift(1)
    prev_low  = g["Low"].shift(1)
    dm_p_raw  = (df["High"] - prev_high).to_numpy()
    dm_m_raw  = (prev_low  - df["Low"]).to_numpy()
    dm_p = np.where((dm_p_raw > 0) & (dm_p_raw > dm_m_raw), dm_p_raw, 0.0)
    dm_m = np.where((dm_m_raw > 0) & (dm_m_raw > dm_p_raw), dm_m_raw, 0.0)
    df["_dm_p"] = dm_p
    df["_dm_m"] = dm_m
    sm_dp = df.groupby("Code")["_dm_p"].transform(lambda x: x.ewm(alpha=1/14, adjust=False).mean())
    sm_dm = df.groupby("Code")["_dm_m"].transform(lambda x: x.ewm(alpha=1/14, adjust=False).mean())
    atr_s = df["ATR14"].replace(0, np.nan)
    di_p  = 100 * sm_dp / atr_s
    di_m  = 100 * sm_dm / atr_s
    dx    = 100 * (di_p - di_m).abs() / (di_p + di_m).replace(0, np.nan)
    df["ADX14"] = dx.groupby(df["Code"]).transform(lambda x: x.ewm(alpha=1/14, adjust=False).mean())
    df.drop(columns=["_dm_p", "_dm_m"], inplace=True)

    print(f"      完了: {time.time()-t0:.1f}秒")
    t1 = time.time()

    # ── [2/4] シグナル一括生成（ブール演算） ─────────────────
    print(f"  [2/4] シグナル生成中...")
    prev_rsi = g["RSI14"].shift(3)
    df["signal"] = (
        (df["SMA20"] > df["SMA50"]) &
        (df["Close"] > df["SMA200"]) &
        (df["RSI14"] >= 45) & (df["RSI14"] <= 75) &
        (df["RSI14"] > prev_rsi) &
        (df["Volume"] >= df["VOL_MA20"] * 1.2) &
        (df["Close"] > df["Open"]) &
        (df["ADX14"] > adx_threshold)               # 【改善2】ADXフィルター
    ).astype(int)
    print(f"      完了: {time.time()-t1:.1f}秒")
    t2 = time.time()

    # ── [3/4] トレード計算（numpy 配列インデックス） ──────────
    print(f"  [3/4] トレード計算中...")
    N = len(df)

    # IS期間のシグナル行位置を取得（integer index）
    sig_mask = (
        (df["Date"] >= IS_BT_START) & (df["Date"] <= IS_BT_END) &
        (df["signal"] == 1) & df["ATR14"].notna()
    )
    sig_pos = df.index[sig_mask].to_numpy()   # integer positions

    # numpy 配列に変換（高速 lookup 用）
    opens_a  = df["Open"].to_numpy()
    highs_a  = df["High"].to_numpy()
    lows_a   = df["Low"].to_numpy()
    closes_a = df["Close"].to_numpy()
    atr_a    = df["ATR14"].to_numpy()

    # Code を整数化（文字列比較より高速）
    code_int = pd.factorize(df["Code"])[0]

    # 翌日エントリー価格（同一銘柄の次行）
    nxt = sig_pos + 1
    same = (nxt < N) & (code_int[nxt] == code_int[sig_pos])
    entry_open = np.where(same, opens_a[np.clip(nxt, 0, N-1)], np.nan)

    # ギャップダウン除外（-1.5% 以上）
    sig_close = closes_a[sig_pos]
    gap_ok = same & ((entry_open - sig_close) / sig_close >= -0.015)

    sig_pos    = sig_pos[gap_ok]
    entry_open = entry_open[gap_ok]

    # 【改善3】決算除外フィルター（シグナル日または翌日が除外対象ならスキップ）
    if earnings_excl:
        dates_a    = df["Date"].to_numpy()
        codes_a_e  = df["Code"].to_numpy()
        sig_dates  = dates_a[sig_pos]
        entry_dates = dates_a[np.clip(sig_pos + 1, 0, N - 1)]
        keep = np.array([
            sig_dates[i]   not in earnings_excl.get(codes_a_e[sig_pos[i]], set()) and
            entry_dates[i] not in earnings_excl.get(codes_a_e[sig_pos[i]], set())
            for i in range(len(sig_pos))
        ], dtype=bool)
        sig_pos    = sig_pos[keep]
        entry_open = entry_open[keep]
        print(f"      決算除外後シグナル: {len(sig_pos):,}件")

    atr_v      = atr_a[sig_pos]

    # TP / SL
    TP = entry_open + atr_v * ATR_TP_MULT
    SL = entry_open - atr_v * ATR_STOP_MULT

    # ポジションサイズ: min(by_risk, by_max_pos, 100株)
    stop_w  = atr_v * 2
    by_risk = (INITIAL_CAPITAL * 0.02 / np.where(stop_w > 0, stop_w, np.inf)).astype(int)
    by_max  = (200_000 / np.where(entry_open > 0, entry_open, np.inf)).astype(int)
    shares  = np.minimum(np.minimum(by_risk, by_max), 100).clip(min=0)

    pos_ok  = shares > 0
    sig_pos, entry_open, TP, SL, shares = (
        sig_pos[pos_ok], entry_open[pos_ok],
        TP[pos_ok], SL[pos_ok], shares[pos_ok]
    )
    n_sigs = len(sig_pos)

    # 出口価格：翌日から最大10日間チェック（numpy ベクトル演算）
    exit_px = np.full(n_sigs, np.nan)
    for k in range(1, MAX_HOLD_DAYS + 1):
        fut = sig_pos + k
        ok  = (fut < N) & (code_int[np.clip(fut, 0, N-1)] == code_int[sig_pos])
        H   = np.where(ok, highs_a[np.clip(fut, 0, N-1)],  np.nan)
        L   = np.where(ok, lows_a[np.clip(fut, 0, N-1)],   np.nan)
        C   = np.where(ok, closes_a[np.clip(fut, 0, N-1)], np.nan)
        not_ex  = np.isnan(exit_px)
        sl_hit  = not_ex & (L <= SL) & ~np.isnan(L)
        tp_hit  = not_ex & ~sl_hit & (H >= TP) & ~np.isnan(H)
        exit_px = np.where(sl_hit, SL, exit_px)
        exit_px = np.where(tp_hit, TP, exit_px)
        if k == MAX_HOLD_DAYS:
            timeout = not_ex & ~sl_hit & ~tp_hit & ~np.isnan(C)
            exit_px = np.where(timeout, C, exit_px)

    # 有効トレードのみ保持
    valid    = ~np.isnan(exit_px)
    sig_pos  = sig_pos[valid];  entry_open = entry_open[valid]
    exit_px  = exit_px[valid];  shares     = shares[valid]
    cost     = (entry_open + exit_px) * shares * COST_PER_LEG
    pnl      = (exit_px - entry_open) * shares - cost
    codes_v  = df["Code"].to_numpy()[sig_pos]

    print(f"      完了: {time.time()-t2:.1f}秒  有効トレード: {len(pnl):,}件")
    t3 = time.time()

    # ── [4/4] 銘柄別メトリクス集計 ─────────────────────────
    print(f"  [4/4] 銘柄別集計中...")
    trade_df = pd.DataFrame({"Code": codes_v, "pnl": pnl})
    results: dict = {}
    for code5, grp in trade_df.groupby("Code"):
        p = grp["pnl"].to_numpy()
        total = len(p)
        if total < MIN_TRADES:
            continue
        wins      = int((p > 0).sum())
        win_rate  = float(wins / total * 100)
        final_pnl = float(p.sum())
        eq        = np.concatenate([[INITIAL_CAPITAL], INITIAL_CAPITAL + np.cumsum(p)])
        peak      = np.maximum.accumulate(eq)
        max_dd    = float(((eq - peak) / peak * 100).min())
        # 二項検定：H0=勝率50%（エッジなし）、片側検定
        p_value   = float(1 - binom.cdf(wins - 1, total, 0.5))
        results[code5] = {
            "total": total, "win_rate": win_rate,
            "max_dd": max_dd, "final_pnl": final_pnl,
            "p_value": p_value, "trades": [],
        }

    elapsed = time.time() - t0
    print(f"      完了: {time.time()-t3:.1f}秒  有効銘柄: {len(results)}")
    print(f"\n  ★ ベクトル化IS完了: {elapsed:.1f}秒 ★")
    return results


def _is_worker(args: tuple) -> tuple:
    """IS期間バックテストの並列ワーカー。
    Windows spawn 方式のため module-level で定義が必須。"""
    code5, df_s = args
    res = run_single_backtest(df_s, IS_BT_START, IS_BT_END)
    return code5, (res if res["total"] >= MIN_TRADES else None)


def fetch_earnings_exclusion(client: jquantsapi.ClientV2) -> dict:
    """【改善3】J-Quants Bulk fins/summary から決算発表日±5営業日を除外セットとして取得。
    Returns: {code5: set of pd.Timestamp}
    """
    import io
    import requests as _req
    from jquantsapi.enums import BulkEndpoint

    print("  決算発表日データ取得中（Bulk fins/summary）...")
    try:
        bulk_list = client.get_bulk_list(endpoint=BulkEndpoint.FIN_SUMMARY)
    except Exception as e:
        print(f"  警告: 決算データ取得失敗 ({e}) → 除外フィルタなし")
        return {}

    dfs = []
    for _, row in bulk_list.iterrows():
        try:
            url  = client.get_bulk(key=row["Key"])
            resp = _req.get(url, timeout=60)
            df   = pd.read_csv(io.BytesIO(resp.content), compression="gzip",
                               low_memory=False, usecols=["DiscDate", "Code"])
            dfs.append(df)
        except Exception:
            continue

    if not dfs:
        print("  警告: 決算データが空 → 除外フィルタなし")
        return {}

    earn = pd.concat(dfs, ignore_index=True)
    earn["DiscDate"] = pd.to_datetime(earn["DiscDate"], errors="coerce")
    earn = earn[earn["DiscDate"].notna()].copy()
    earn["Code"] = earn["Code"].astype(str).str.strip()
    print(f"  決算発表日: {len(earn):,}件  コード数: {earn['Code'].nunique():,}")

    # 発表日±5営業日を除外セットに追加（約3取引日に相当）
    exclusion: dict[str, set] = {}
    for code, disc_date in zip(earn["Code"], earn["DiscDate"]):
        offsets = pd.bdate_range(disc_date - pd.Timedelta(days=7),
                                  disc_date + pd.Timedelta(days=7))
        # 実際に±3営業日以内に絞る
        nearby = [d for d in offsets
                  if abs((pd.Timestamp(d) - disc_date).days) <= 5]
        if code not in exclusion:
            exclusion[code] = set()
        for d in nearby:
            exclusion[code].add(pd.Timestamp(d))

    print(f"  除外セット作成完了: {len(exclusion):,}銘柄分")
    return exclusion


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
    signal_rsi:  float = 0.0


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


def run_portfolio_backtest(stock_data: dict, names: dict,
                           earnings_excl: dict | None = None,
                           adx_threshold: float = 25.0,
                           atr_tp_mult: float = ATR_TP_MULT) -> tuple:
    all_dates = sorted({d for df in stock_data.values()
                        for d in df["Date"] if OOS_START <= d <= OOS_END})
    lookup = {code: df.set_index("Date").to_dict("index")
              for code, df in stock_data.items()}

    capital          = float(INITIAL_CAPITAL)
    positions: list  = []
    trades:    list  = []
    equity           = {OOS_START: capital}

    # 【改善1】月次損失10%ストップ用トラッキング
    cur_month        = None
    month_start_cap  = capital
    monthly_stopped  = False
    monthly_stop_cnt = 0

    for today in all_dates:
        # 月次リセット
        ym = (today.year, today.month)
        if ym != cur_month:
            cur_month       = ym
            month_start_cap = capital
            monthly_stopped = False

        # 決済判定
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
                cost = (pos.entry_price + ep) * pos.shares * COST_PER_LEG
                pnl = (ep - pos.entry_price) * pos.shares - cost
                capital += pnl
                trades.append(Trade(pos.code, pos.name, pos.entry_date, today,
                                    pos.entry_price, ep, pos.shares, pnl, er))
            else:
                next_pos.append(pos)
        positions = next_pos

        # 月次損失チェック（決済後に計算）
        if not monthly_stopped and month_start_cap > 0:
            loss_pct = (capital - month_start_cap) / month_start_cap * 100
            if loss_pct <= -10.0:
                monthly_stopped = True
                monthly_stop_cnt += 1

        # エントリー判定
        slots = MAX_POSITIONS - len(positions)
        if slots > 0 and not monthly_stopped:   # 【改善1】月次ストップ中はエントリー禁止
            holding = {p.code for p in positions}
            signals = []
            for code, df in stock_data.items():
                if code in holding:
                    continue
                prev_dates = df.loc[df["Date"] < today, "Date"]
                if prev_dates.empty:
                    continue
                prev_row = lookup[code].get(prev_dates.iloc[-1])
                if prev_row is None or prev_row.get("signal", 0) != 1:
                    continue
                if pd.isna(prev_row.get("ATR14")):
                    continue
                # 【改善2】ADXフィルター
                if prev_row.get("ADX14", 0) <= adx_threshold:
                    continue
                today_row = lookup[code].get(today)
                if today_row is None:
                    continue
                ep = today_row["Open"]
                gap = (ep - prev_row["Close"]) / prev_row["Close"]
                if gap < -0.015:
                    continue
                # 【改善3】決算除外フィルター
                if earnings_excl:
                    excl_set = earnings_excl.get(code, set())
                    if today in excl_set:
                        continue
                signals.append((float(prev_row["RSI14"]), code, prev_row, ep))

            signals.sort(key=lambda x: x[0], reverse=True)
            for rsi, code, prev_row, ep in signals[:slots]:
                atr = float(prev_row["ATR14"])
                sh, sl = calc_position_size(capital, atr, ep)
                sh = min(sh, int(capital * MAX_POS_RATIO / ep))
                tp = ep + atr * atr_tp_mult
                if sh > 0 and ep * sh <= capital:
                    positions.append(Position(code, names.get(code, code),
                                              today, ep, sh, sl, tp, signal_rsi=rsi))
        equity[today] = capital

    equity_s = pd.Series(equity).sort_index()
    print(f"  月次ストップ発動: {monthly_stop_cnt}回")
    return trades, equity_s


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
    total_cost = sum(
        (t.entry_price + t.exit_price) * t.shares * COST_PER_LEG
        for t in trades
    )
    return {"total": total, "win_rate": win_rate, "final_pnl": final_pnl,
            "annual": annual, "max_dd": max_dd, "monthly": monthly,
            "final_cap": equity.iloc[-1], "total_cost": total_cost}


def main() -> None:
    print("=" * 68)
    print("  真のOOSバックテスト")
    print(f"  IS選定期間  : {IS_BT_START.date()} ～ {IS_BT_END.date()}")
    print(f"  ギャップ期間: 2021-01-01 ～ 2022-12-31（未使用）")
    print(f"  OOSテスト期間: {OOS_START.date()} ～ {OOS_END.date()}")
    print("=" * 68)

    # ── データ読み込み ──────────────────────────────────────
    print("\nデータ読み込み中...")
    df_all = pd.read_parquet(DATA_PATH)
    df_all["Date"] = pd.to_datetime(df_all["Date"])
    # parquet の Code 列は壊れている場合があるため Code4（4桁）から5桁コードを再生成
    df_all["Code"] = df_all["Code4"].astype(str) + "0"
    df_all = df_all.sort_values(["Code", "Date"]).reset_index(drop=True)
    print(f"  {len(df_all):,} 行  {df_all['Code'].nunique():,} 銘柄")

    # ── マスタ（銘柄名・業種）──────────────────────────────
    print("銘柄マスタ取得中...")
    api_key = os.getenv("JQUANTS_REFRESH_TOKEN")
    client  = jquantsapi.ClientV2(api_key=api_key)
    master  = client.get_eq_master()
    name_map    = dict(zip(master["Code"], master["CoName"]))
    sector_map  = dict(zip(master["Code"], master["S33Nm"]))

    # ── 【改善3】決算除外日セット取得 ──────────────────────
    earnings_excl = fetch_earnings_exclusion(client)

    # ── IS期間 銘柄別バックテスト（完全ベクトル化） ───────────
    print(f"\n【STEP1】IS期間バックテスト実行中: {IS_BT_START.date()} ～ {IS_BT_END.date()}")
    t0 = time.time()

    # 生列削除（prices_10y.parquet の O/H/L/C 等を除去して指標計算列と衝突回避）
    drop_cols = [c for c in ["Code4", "O", "H", "L", "C", "Vo", "Va", "UL", "LL", "AdjFactor"]
                 if c in df_all.columns]
    df_clean = df_all.drop(columns=drop_cols).sort_values(["Code", "Date"]).reset_index(drop=True)

    is_results = run_is_backtest_vectorized(df_clean, earnings_excl=earnings_excl,
                                            adx_threshold=ADX_THRESHOLD)
    print(f"  IS完了: {len(is_results)} 銘柄が取引あり  総経過: {time.time()-t0:.1f}秒")

    # ── 銘柄フィルタリング ──────────────────────────────────
    print(f"\n【STEP2】選定基準フィルタリング")
    candidates = {}
    for code5, res in is_results.items():
        if (res["win_rate"]  >= MIN_WIN_RATE and
            res["max_dd"]    >= MAX_DD and
            res["final_pnl"] >= MIN_PNL and
            res.get("p_value", 1.0) <= P_VALUE_MAX):
            candidates[code5] = res
    print(f"  基準通過: {len(candidates)} 銘柄")
    print(f"    条件: 取引{MIN_TRADES}回+ / 勝率{MIN_WIN_RATE}%+ / DD{MAX_DD}%以内 / 損益{MIN_PNL:,}円+ / p値{P_VALUE_MAX}以下")

    if len(candidates) < TARGET_N:
        print(f"  警告: 候補が{TARGET_N}銘柄に満たない → 全候補({len(candidates)}銘柄)を使用")
        selected5 = list(candidates.keys())
    else:
        # スコア計算（勝率 × (1 - |DD|/100)）
        scored = sorted(candidates.items(),
                        key=lambda x: x[1]["win_rate"] * (1 - abs(x[1]["max_dd"]) / 100),
                        reverse=True)

        # 業種分散（同一業種33分類で上限MAX_PER_SEC33）
        selected5   = []
        sector_count = {}
        for code5, _ in scored:
            sec = sector_map.get(code5, "その他")
            if sector_count.get(sec, 0) < MAX_PER_SEC33:
                selected5.append(code5)
                sector_count[sec] = sector_count.get(sec, 0) + 1
            if len(selected5) >= TARGET_N:
                break

    print(f"  選定銘柄: {len(selected5)} 銘柄（業種分散適用）")

    # Phase 5 候補リスト保存（_v2: Phase 4候補を上書きしない）
    out_v2 = Path("logs/final_candidates_v2.csv")
    rows = []
    for c in selected5:
        r = candidates.get(c) or is_results.get(c, {})
        rows.append({"code": c[:4], "name": name_map.get(c, c),
                     "trades": r.get("total", 0),
                     "win_rate": round(r.get("win_rate", 0), 1),
                     "final_pnl": round(r.get("final_pnl", 0), 0),
                     "max_dd": round(r.get("max_dd", 0), 2),
                     "p_value": round(r.get("p_value", 1.0), 3)})
    pd.DataFrame(rows).to_csv(out_v2, index=False, encoding="utf-8-sig")
    print(f"  → {out_v2} に保存（Phase 5移行用・prices_20y使用）")

    # ── OOS指標計算（ウォームアップ含む全期間データ使用）─────
    print(f"\n【STEP3】OOS指標計算中（ウォームアップ: {OOS_WARMUP.date()}〜）")
    stock_data_oos = {}
    for code5 in selected5:
        df_s = df_all[(df_all["Code"] == code5) & (df_all["Date"] >= OOS_WARMUP)].copy()
        df_s = df_s.drop(columns=drop_cols, errors="ignore").reset_index(drop=True)
        if len(df_s) < 260:
            continue
        df_s = add_indicators(df_s)
        df_s = generate_signals(df_s)
        # 【改善2】OOS用 ADX14 追加
        g_s       = df_s.groupby(df_s.index // len(df_s))  # dummy; use Series ops directly
        prev_hi   = df_s["High"].shift(1)
        prev_lo   = df_s["Low"].shift(1)
        dm_p_r    = (df_s["High"] - prev_hi).to_numpy()
        dm_m_r    = (prev_lo - df_s["Low"]).to_numpy()
        dm_p_v    = np.where((dm_p_r > 0) & (dm_p_r > dm_m_r), dm_p_r, 0.0)
        dm_m_v    = np.where((dm_m_r > 0) & (dm_m_r > dm_p_r), dm_m_r, 0.0)
        atr_safe  = df_s["ATR14"].replace(0, np.nan)
        sm_dp_s   = pd.Series(dm_p_v, index=df_s.index).ewm(alpha=1/14, adjust=False).mean()
        sm_dm_s   = pd.Series(dm_m_v, index=df_s.index).ewm(alpha=1/14, adjust=False).mean()
        di_p_s    = 100 * sm_dp_s / atr_safe
        di_m_s    = 100 * sm_dm_s / atr_safe
        dx_s      = 100 * (di_p_s - di_m_s).abs() / (di_p_s + di_m_s).replace(0, np.nan)
        df_s["ADX14"] = dx_s.ewm(alpha=1/14, adjust=False).mean()
        stock_data_oos[code5] = df_s
    print(f"  OOS対象: {len(stock_data_oos)} 銘柄")

    names_oos = {c: name_map.get(c, c) for c in stock_data_oos}

    # ── OOSポートフォリオバックテスト ──────────────────────
    print(f"\n【STEP4】OOSポートフォリオバックテスト実行中...")
    trades, equity = run_portfolio_backtest(stock_data_oos, names_oos,
                                            earnings_excl=earnings_excl,
                                            adx_threshold=ADX_THRESHOLD,
                                            atr_tp_mult=ATR_TP_MULT)
    stats = compute_stats(trades, equity)

    # ── 結果表示 ────────────────────────────────────────────
    print()
    print("=" * 68)
    print("  ★ 真のOOSテスト結果 ★")
    print(f"  OOS期間: {OOS_START.date()} ～ {OOS_END.date()}")
    print("=" * 68)

    # 選定30銘柄リスト
    print(f"\n【選定{len(selected5)}銘柄リスト（IS期間成績順）】")
    print(f"  {'コード':6}  {'銘柄名':16}  {'業種':16}  {'勝率':>6}  {'DD':>6}  {'損益':>10}  {'取引':>4}  {'p値':>6}")
    print("  " + "-" * 80)
    for code5 in selected5:
        r   = candidates.get(code5, is_results.get(code5, {}))
        nm  = name_map.get(code5, code5)[:14]
        sec = sector_map.get(code5, "")[:14]
        wr  = r.get("win_rate", 0)
        dd  = r.get("max_dd", 0)
        pnl = r.get("final_pnl", 0)
        tr  = r.get("total", 0)
        pv  = r.get("p_value", 1.0)
        print(f"  {code5:6}  {nm:16}  {sec:16}  {wr:5.1f}%  {dd:5.1f}%  {pnl:>10,.0f}  {tr:>4}  {pv:5.3f}")

    # OOS成績
    print(f"\n【OOS期間パフォーマンス】")
    print(f"  初期資金      : {INITIAL_CAPITAL:>12,} 円")
    print(f"  最終資産      : {stats['final_cap']:>12,.0f} 円")
    print(f"  総損益        : {stats['final_pnl']:>+12,.0f} 円")
    print(f"  年利換算(CAGR): {stats['annual']:>+11.2f} %")
    print(f"  総取引数      : {stats['total']:>12} 回")
    print(f"  勝率          : {stats['win_rate']:>11.1f} %")
    print(f"  最大DD        : {stats['max_dd']:>11.2f} %")
    print(f"  ── 取引コスト内訳 ──────────────────────────────")
    print(f"  総コスト      : {stats['total_cost']:>12,.0f} 円  ← 手数料+スリッページ(片道{COST_PER_LEG*100:.3f}%)")
    print(f"  1取引平均コスト: {stats['total_cost']/max(stats['total'],1):>11,.0f} 円")
    print()

    # 月別損益
    print(f"【月別損益（OOS期間）】")
    print(f"  {'年月':8}  {'損益':>12}  {'累計資産':>13}")
    print("  " + "-" * 38)
    eq_m = equity.resample("ME").last()
    for ym, cap in eq_m.items():
        if ym < OOS_START:
            continue
        pm = stats["monthly"].get(ym, 0)
        sign = "▲" if pm < 0 else "+"
        mark = " ◀赤字" if pm < 0 else ""
        print(f"  {ym.strftime('%Y-%m'):8}  {sign}{abs(pm):>10,.0f}円  {cap:>12,.0f}円{mark}")

    print()
    print("=" * 68)
    print("  ※ IS選定期間(2016-2020)と完全に分離した未来データで検証")
    print("  ※ ギャップ期間(2021-2022)はIS選定にも使用していない")
    print("  ※ これが「真のOOS」＝実運用に最も近い結果")
    print("=" * 68)


if __name__ == "__main__":
    main()
