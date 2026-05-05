"""
ユニバース型 戦略A フィルター最適化
目的：CAGR+23%を維持しつつ最大DD-25%を-15%以内に抑える組み合わせを探す

検証フィルター：
  ADX閾値     : 15（現行）/ 20 / 25
  出来高下限   : なし / 日次平均10万株以上 / 30万株以上
  業種集中上限 : なし / 同業種3銘柄まで

目標：CAGR ≥ 13.2%（退職金条件）かつ MaxDD ≥ -15%
"""
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from itertools import product

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd

OOS_START = pd.Timestamp("2023-01-01")
OOS_END   = pd.Timestamp("2026-04-24")
WARMUP    = pd.Timestamp("2022-01-01")

ATR_TP_MULT   = 6
ATR_SL_MULT   = 2
MAX_HOLD_DAYS = 10

INITIAL_CAPITAL = 1_000_000
MAX_POSITIONS   = 5
MAX_POS_RATIO   = 0.20
COST_PER_LEG    = 0.00055 + 0.00050

DATA_PATH    = Path("data/raw/prices_20y.parquet")
SECTOR_PATH  = None   # 銘柄マスタは別途取得


# ── 全銘柄一括指標計算 ────────────────────────────────────────
def compute_base(df: pd.DataFrame) -> pd.DataFrame:
    print("指標計算中（全銘柄一括）...")
    t0 = time.time()
    df = df.reset_index(drop=True)
    g  = df.groupby("Code", sort=False)

    df["SMA20"]    = g["Close"].transform(lambda x: x.rolling(20).mean())
    df["SMA50"]    = g["Close"].transform(lambda x: x.rolling(50).mean())
    df["SMA200"]   = g["Close"].transform(lambda x: x.rolling(200).mean())
    df["VOL_MA20"] = g["Volume"].transform(lambda x: x.rolling(20).mean())

    delta = g["Close"].diff()
    gain  = delta.clip(lower=0).groupby(df["Code"]).transform(
            lambda x: x.ewm(alpha=1/14, adjust=False).mean())
    loss  = (-delta).clip(lower=0).groupby(df["Code"]).transform(
            lambda x: x.ewm(alpha=1/14, adjust=False).mean())
    df["RSI14"] = 100 - 100 / (1 + gain / loss.replace(0, 1e-9))

    prev_c = g["Close"].shift(1)
    tr = pd.concat([df["High"]-df["Low"],
                    (df["High"]-prev_c).abs(),
                    (df["Low"]-prev_c).abs()], axis=1).max(axis=1)
    df["ATR14"] = tr.groupby(df["Code"]).transform(
                  lambda x: x.ewm(alpha=1/14, adjust=False).mean())

    ph = g["High"].shift(1); pl = g["Low"].shift(1)
    dm_p = np.where((df["High"]-ph > 0) & (df["High"]-ph > pl-df["Low"]),
                    df["High"]-ph, 0.0)
    dm_m = np.where((pl-df["Low"]  > 0) & (pl-df["Low"] > df["High"]-ph),
                    pl-df["Low"], 0.0)
    atr_s = df["ATR14"].replace(0, np.nan)
    di_p  = 100 * pd.Series(dm_p, index=df.index).groupby(df["Code"]).transform(
            lambda x: x.ewm(alpha=1/14, adjust=False).mean()) / atr_s
    di_m  = 100 * pd.Series(dm_m, index=df.index).groupby(df["Code"]).transform(
            lambda x: x.ewm(alpha=1/14, adjust=False).mean()) / atr_s
    dx    = 100*(di_p-di_m).abs()/(di_p+di_m).replace(0, np.nan)
    df["ADX14"] = dx.groupby(df["Code"]).transform(
                  lambda x: x.ewm(alpha=1/14, adjust=False).mean())

    print(f"  完了: {time.time()-t0:.1f}秒")
    return df


# ── フィルター適用後シグナル生成 ──────────────────────────────
def apply_signal(df: pd.DataFrame, adx_thr: float, vol_min: float) -> pd.DataFrame:
    g    = df.groupby("Code", sort=False)
    rsi3 = g["RSI14"].shift(3)
    vol_cond = (df["VOL_MA20"] >= vol_min) if vol_min > 0 else True
    df["signal"] = (
        (df["SMA20"]  > df["SMA50"]) &
        (df["Close"]  > df["SMA200"]) &
        (df["RSI14"] >= 45) & (df["RSI14"] <= 75) &
        (df["RSI14"]  > rsi3) &
        (df["Volume"] >= df["VOL_MA20"] * 1.2) &
        vol_cond &
        (df["Close"]  > df["Open"]) &
        (df["ADX14"]  > adx_thr)
    ).astype(int)
    return df


# ── ポートフォリオシミュレーション ───────────────────────────
@dataclass
class Position:
    code: str; entry_date: pd.Timestamp; entry_price: float
    shares: int; stop_loss: float; take_profit: float
    hold_days: int = 0; rsi: float = 0.0
    sector: str = ""

def run_sim(df: pd.DataFrame, sector_map: dict, max_per_sector: int) -> tuple:
    df_oos = df[df["Date"] >= OOS_START].copy()
    all_dates = sorted(df_oos["Date"].unique())
    sig_codes = set(df_oos[df_oos["signal"] == 1]["Code"].unique())

    lookup: dict = {}
    for code, grp in df.groupby("Code"):
        lookup[code] = grp.drop_duplicates("Date").set_index("Date").to_dict("index")

    capital = float(INITIAL_CAPITAL)
    positions: list = []
    trades:    list = []
    equity:    dict = {}
    cur_month = None; month_start = capital; month_stopped = False; stop_cnt = 0

    for today in all_dates:
        ym = (today.year, today.month)
        if ym != cur_month:
            cur_month = ym; month_start = capital; month_stopped = False

        next_pos = []
        for pos in positions:
            row = lookup.get(pos.code, {}).get(today)
            if row is None:
                next_pos.append(pos); continue
            pos.hold_days += 1
            hi, lo, cl = row["High"], row["Low"], row["Close"]
            ep = er = None
            if lo <= pos.stop_loss:    ep, er = pos.stop_loss, "損切り"
            elif hi >= pos.take_profit: ep, er = pos.take_profit, "利確"
            elif pos.hold_days >= MAX_HOLD_DAYS: ep, er = cl, "期間満了"
            if ep is not None:
                cost = (pos.entry_price + ep) * pos.shares * COST_PER_LEG
                pnl  = (ep - pos.entry_price) * pos.shares - cost
                capital += pnl
                trades.append(pnl)
            else:
                next_pos.append(pos)
        positions = next_pos

        if not month_stopped and month_start > 0:
            if (capital - month_start) / month_start * 100 <= -10.0:
                month_stopped = True; stop_cnt += 1

        slots = MAX_POSITIONS - len(positions)
        if slots > 0 and not month_stopped:
            holding = {p.code for p in positions}
            # 業種別保有数
            sector_cnt: dict = {}
            for p in positions:
                s = sector_map.get(p.code, "other")
                sector_cnt[s] = sector_cnt.get(s, 0) + 1

            cands = []
            for code in sig_codes:
                if code in holding: continue
                code_rows = lookup.get(code, {})
                prev_dates = [d for d in code_rows if d < today]
                if not prev_dates: continue
                prev = code_rows[max(prev_dates)]
                if prev.get("signal", 0) != 1: continue
                atr = prev.get("ATR14", 0)
                if not atr or pd.isna(atr): continue
                today_row = code_rows.get(today)
                if today_row is None: continue
                ep = today_row["Open"]
                if ep <= 0: continue
                if (ep - prev["Close"]) / prev["Close"] < -0.015: continue
                sec = sector_map.get(code, "other")
                if max_per_sector > 0 and sector_cnt.get(sec, 0) >= max_per_sector:
                    continue
                cands.append((prev.get("RSI14", 0), code, ep, atr, sec))

            cands.sort(key=lambda x: x[0], reverse=True)
            for rsi, code, ep, atr, sec in cands[:slots]:
                sl = ep - atr * ATR_SL_MULT
                tp = ep + atr * ATR_TP_MULT
                sh = min(int(capital * MAX_POS_RATIO / ep),
                         int(INITIAL_CAPITAL * 0.02 / (atr * ATR_SL_MULT)))
                if sh > 0 and ep * sh <= capital:
                    positions.append(Position(code, today, ep, sh, sl, tp,
                                              rsi=rsi, sector=sec))
                    sector_cnt[sec] = sector_cnt.get(sec, 0) + 1
                    slots -= 1
                    if slots <= 0: break

        equity[today] = capital

    eq_s  = pd.Series(equity).sort_index()
    total = len(trades)
    wins  = sum(1 for p in trades if p > 0)
    years = (OOS_END - OOS_START).days / 365
    cagr  = ((eq_s.iloc[-1] / INITIAL_CAPITAL)**(1/years) - 1) * 100
    peak  = eq_s.cummax()
    mdd   = ((eq_s - peak) / peak * 100).min()
    wr    = wins / total * 100 if total else 0
    return {"cagr": cagr, "mdd": mdd, "total": total, "wr": wr,
            "stop": stop_cnt, "final": eq_s.iloc[-1]}


# ── メイン ────────────────────────────────────────────────────
def main():
    print("=" * 72)
    print("  ユニバース型 戦略A フィルター最適化")
    print("  目標: CAGR ≥ 13.2%  かつ  MaxDD ≥ -15%")
    print("=" * 72)

    # データ読み込み
    print("\nデータ読み込み中...")
    t0 = time.time()
    df_all = pd.read_parquet(DATA_PATH)
    df_all["Date"] = pd.to_datetime(df_all["Date"])
    df_all["Code"] = df_all["Code4"].astype(str).str.zfill(4) + "0"
    df_all = df_all[df_all["Date"] >= WARMUP].copy()
    drop_cols = [c for c in ["Code4","O","H","L","C","Vo","Va","UL","LL","AdjFactor"]
                 if c in df_all.columns]
    df_all.drop(columns=drop_cols, inplace=True)
    df_all = df_all.sort_values(["Code","Date"]).reset_index(drop=True)
    print(f"  完了: {len(df_all):,}行  {df_all['Code'].nunique()}銘柄  ({time.time()-t0:.1f}秒)")

    # 業種マスタ取得
    print("業種マスタ取得中...")
    sector_map: dict = {}
    try:
        import os; from dotenv import load_dotenv; import jquantsapi
        load_dotenv()
        client = jquantsapi.ClientV2(api_key=os.getenv("JQUANTS_REFRESH_TOKEN"))
        master = client.get_eq_master()
        sector_map = dict(zip(master["Code"], master["S33Nm"]))
        print(f"  取得完了: {len(sector_map)}銘柄")
    except Exception as e:
        print(f"  警告: 業種マスタ取得失敗 ({e}) → 業種フィルタなし")

    # 指標計算（1回のみ）
    df_base = compute_base(df_all.copy())

    # フィルター組み合わせ定義
    # adx_thr, vol_min(株), max_per_sector(0=無制限), label
    combos = [
        (15,  0,       0, "ベースライン（前回結果）"),
        (20,  0,       0, "ADX>20"),
        (25,  0,       0, "ADX>25"),
        (15,  100_000, 0, "ADX>15 + 出来高10万株+"),
        (15,  300_000, 0, "ADX>15 + 出来高30万株+"),
        (20,  100_000, 0, "ADX>20 + 出来高10万株+"),
        (20,  300_000, 0, "ADX>20 + 出来高30万株+"),
        (20,  100_000, 3, "ADX>20 + 出来高10万株+ + 業種3銘柄上限"),
        (25,  100_000, 0, "ADX>25 + 出来高10万株+"),
    ]

    results = []
    print(f"\n{len(combos)}パターン検証開始...\n")

    for i, (adx, vol, sec_lim, label) in enumerate(combos):
        t1 = time.time()
        df_sig = apply_signal(df_base.copy(), adx_thr=adx, vol_min=vol)
        sig_cnt = int(df_sig[df_sig["Date"] >= OOS_START]["signal"].sum())
        r = run_sim(df_sig, sector_map, max_per_sector=sec_lim)
        elapsed = time.time() - t1

        ok = "✅" if r["cagr"] >= 13.2 and r["mdd"] >= -15.0 else \
             "⚠️" if r["cagr"] >= 10.0 and r["mdd"] >= -20.0 else "❌"
        results.append({**r, "label": label, "adx": adx, "vol": vol,
                        "sec": sec_lim, "sigs": sig_cnt, "ok": ok})
        print(f"  [{i+1:2d}/{len(combos)}] {ok} {label}")
        print(f"        CAGR {r['cagr']:+6.2f}%  MaxDD {r['mdd']:6.2f}%  "
              f"取引{r['total']:4d}回  勝率{r['wr']:5.1f}%  "
              f"シグナル{sig_cnt:6,}  ({elapsed:.0f}秒)")

    # ── 結果サマリー ──────────────────────────────────────────
    print()
    print("=" * 72)
    print("  ★ フィルター最適化 結果サマリー ★")
    print("=" * 72)
    print(f"  目標: CAGR ≥ 13.2%  かつ  MaxDD ≥ -15.0%\n")
    print(f"  {'判定':2}  {'CAGR':>7}  {'MaxDD':>7}  {'取引':>5}  {'勝率':>6}  設定")
    print("  " + "-" * 65)

    # 目標達成順にソート
    def sort_key(r):
        ok_score = 2 if r["ok"] == "✅" else (1 if r["ok"] == "⚠️" else 0)
        return (-ok_score, -r["cagr"])
    for r in sorted(results, key=sort_key):
        print(f"  {r['ok']:2}  {r['cagr']:>+6.2f}%  {r['mdd']:>+6.2f}%  "
              f"{r['total']:>5}回  {r['wr']:>5.1f}%  {r['label']}")

    # 目標達成パターンの詳細
    best = [r for r in results if r["ok"] == "✅"]
    if best:
        b = max(best, key=lambda x: x["cagr"])
        print(f"\n  最優秀: {b['label']}")
        print(f"  CAGR {b['cagr']:+.2f}%  MaxDD {b['mdd']:.2f}%  "
              f"取引{b['total']}回  勝率{b['wr']:.1f}%")
    else:
        print("\n  ⚠️  目標（CAGR≥13.2% かつ DD≥-15%）を同時達成するパターンなし")
        print("  → より高いフィルター強度 または 新戦略研究（EPS加速）へ移行")

    print("=" * 72)


if __name__ == "__main__":
    main()
