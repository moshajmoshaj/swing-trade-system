"""
analyze_short_ratio_oos.py
業種別空売り比率 × 戦略A OOS期間シグナル 相関検証

目的:
  IS期間分析（相関+0.055）がOOS期間でも再現するかを検証し、
  Phase 5 での空売り比率フィルター採用可否を判断する。

IS期間: 2016-04-01 ～ 2020-12-31（analyze_short_ratio.py 済み）
OOS期間: 2023-01-01 ～ 2026-04-24

使用データ:
  - data/raw/prices_20y.parquet
  - data/raw/short_ratio.parquet
  - logs/final_candidates.csv: 戦略A Phase4 候補30銘柄

出力:
  - コンソール: IS vs OOS 相関係数比較・バケット別勝率
  - logs/short_ratio_oos_analysis.csv
"""
import sys, os, time
sys.stdout.reconfigure(encoding="utf-8")

from pathlib import Path
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import jquantsapi

load_dotenv()

PRICES_PATH = Path("data/raw/prices_20y.parquet")
SR_PATH     = Path("data/raw/short_ratio.parquet")
CAND_PATH   = Path("logs/final_candidates.csv")
IS_CSV      = Path("logs/short_ratio_analysis.csv")   # IS 期間結果（参照用）
OUT_CSV     = Path("logs/short_ratio_oos_analysis.csv")

OOS_START = pd.Timestamp("2023-01-01")
OOS_END   = pd.Timestamp("2026-04-24")
WARMUP    = pd.Timestamp("2022-01-01")  # SMA200計算用ウォームアップ

ATR_TP   = 6.0
ATR_SL   = 2.0
MAX_HOLD = 10
ADX_MIN  = 15


# ─── Strategy A シグナル計算 ────────────────────────────────────────────────

def compute_signals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().reset_index(drop=True)

    df["SMA20"]    = df["Close"].rolling(20).mean()
    df["SMA50"]    = df["Close"].rolling(50).mean()
    df["SMA200"]   = df["Close"].rolling(200).mean()
    df["VOL_MA20"] = df["Volume"].rolling(20).mean()

    delta = df["Close"].diff()
    gain  = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
    loss  = (-delta).clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
    df["RSI14"]    = 100 - 100 / (1 + gain / loss.replace(0, 1e-9))
    df["RSI_lag3"] = df["RSI14"].shift(3)

    prev_cl = df["Close"].shift(1)
    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - prev_cl).abs(),
        (df["Low"]  - prev_cl).abs(),
    ], axis=1).max(axis=1)
    df["ATR14"] = tr.ewm(alpha=1/14, adjust=False).mean()

    prev_hi = df["High"].shift(1)
    prev_lo = df["Low"].shift(1)
    dm_p = np.where((df["High"] - prev_hi) > (prev_lo - df["Low"]),
                    np.maximum(df["High"] - prev_hi, 0.0), 0.0)
    dm_m = np.where((prev_lo - df["Low"]) > (df["High"] - prev_hi),
                    np.maximum(prev_lo - df["Low"], 0.0), 0.0)
    atr_s = df["ATR14"].replace(0, np.nan)
    sm_dp = pd.Series(dm_p, index=df.index).ewm(alpha=1/14, adjust=False).mean()
    sm_dm = pd.Series(dm_m, index=df.index).ewm(alpha=1/14, adjust=False).mean()
    di_p  = 100 * sm_dp / atr_s
    di_m  = 100 * sm_dm / atr_s
    dx    = 100 * (di_p - di_m).abs() / (di_p + di_m).replace(0, np.nan)
    df["ADX14"] = dx.ewm(alpha=1/14, adjust=False).mean()

    df["signal"] = (
        df["SMA20"].notna() & df["SMA200"].notna() &
        df["RSI14"].notna() & df["ATR14"].notna() & df["ADX14"].notna() &
        (df["SMA20"]  > df["SMA50"]) &
        (df["Close"]  > df["SMA200"]) &
        (df["RSI14"]  >= 45) & (df["RSI14"] <= 75) &
        (df["RSI14"]  > df["RSI_lag3"]) &
        (df["Volume"] >= df["VOL_MA20"] * 1.2) &
        (df["Close"]  > df["Open"]) &
        (df["ADX14"]  > ADX_MIN)
    ).astype(int)

    return df


# ─── OOS トレード抽出 ───────────────────────────────────────────────────────

def extract_oos_trades(df: pd.DataFrame, code5: str) -> list[dict]:
    trades = []
    oos_sigs = df[
        (df["Date"] >= OOS_START) &
        (df["Date"] <= OOS_END) &
        (df["signal"] == 1)
    ]

    for i in oos_sigs.index:
        if i + 1 >= len(df):
            continue
        sig = df.iloc[i]
        nxt = df.iloc[i + 1]
        ep  = nxt["Open"]
        atr = sig["ATR14"]

        if (ep - sig["Close"]) / sig["Close"] < -0.015:
            continue

        tp = ep + atr * ATR_TP
        sl = ep - atr * ATR_SL
        exit_px = reason = None

        for k in range(1, MAX_HOLD + 1):
            if i + 1 + k >= len(df):
                break
            fut = df.iloc[i + 1 + k]
            if fut["Low"] <= sl:
                exit_px, reason = sl, "損切り"
                break
            elif fut["High"] >= tp:
                exit_px, reason = tp, "利確"
                break
            elif k == MAX_HOLD:
                exit_px, reason = fut["Close"], "強制終了"

        if exit_px is None:
            continue

        trades.append({
            "signal_date": sig["Date"],
            "entry_price": ep,
            "exit_price":  exit_px,
            "reason":      reason,
            "pnl_pct":     (exit_px - ep) / ep,
            "win":         exit_px > ep,
            "atr":         atr,
            "rsi":         sig["RSI14"],
            "code":        code5,
        })
    return trades


# ─── バケット分析 ────────────────────────────────────────────────────────────

def print_bucket_analysis(df_w: pd.DataFrame, label: str):
    n_base  = len(df_w)
    wr_base = df_w["win"].mean() * 100
    avg_base = df_w["pnl_pct"].mean() * 100
    corr    = df_w["short_ratio"].corr(df_w["pnl_pct"])

    q25 = df_w["short_ratio"].quantile(0.25)
    q50 = df_w["short_ratio"].quantile(0.50)
    q75 = df_w["short_ratio"].quantile(0.75)

    print(f"\n{'='*65}")
    print(f"  【{label}】")
    print(f"  件数={n_base}  勝率={wr_base:.1f}%  平均損益={avg_base:+.2f}%")
    print(f"  short_ratio × pnl_pct 相関係数: {corr:+.4f}")
    print(f"{'─'*65}")
    print(f"  {'バケット':18} {'件数':>5} {'勝率':>7} {'平均損益':>9} {'損切率':>7}")
    print(f"  {'-'*18} {'-'*5} {'-'*7} {'-'*9} {'-'*7}")

    buckets = [
        (f"低（<{q25:.1%}）",
         df_w[df_w["short_ratio"] < q25]),
        (f"中低（{q25:.1%}-{q50:.1%}）",
         df_w[(df_w["short_ratio"] >= q25) & (df_w["short_ratio"] < q50)]),
        (f"中高（{q50:.1%}-{q75:.1%}）",
         df_w[(df_w["short_ratio"] >= q50) & (df_w["short_ratio"] < q75)]),
        (f"高（≥{q75:.1%}）",
         df_w[df_w["short_ratio"] >= q75]),
    ]
    for lbl, sub in buckets:
        if len(sub) == 0:
            continue
        wr  = sub["win"].mean() * 100
        avg = sub["pnl_pct"].mean() * 100
        sl  = (sub["reason"] == "損切り").mean() * 100 if "reason" in sub.columns else 0
        print(f"  {lbl:18} {len(sub):>5} {wr:>6.1f}% {avg:>+8.2f}% {sl:>6.1f}%")

    return corr, q25, q50, q75, n_base, wr_base


# ─── メイン ──────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("  業種別空売り比率 × 戦略A OOS検証")
    print(f"  OOS期間: {OOS_START.date()} ～ {OOS_END.date()}")
    print("=" * 65)

    # ── 候補銘柄読み込み ─────────────────────────────────
    cand = pd.read_csv(CAND_PATH, dtype=str)
    col  = next(c for c in cand.columns if "code" in c.lower())
    codes4 = [c.zfill(4) for c in cand[col]]
    codes5 = [c + "0"    for c in codes4]
    print(f"\n  候補銘柄: {len(codes4)}銘柄（final_candidates.csv）")

    # ── 株価データ読み込み ───────────────────────────────
    print("\nデータ読み込み中...")
    t0 = time.time()
    prices = pd.read_parquet(PRICES_PATH)
    prices["Date"] = pd.to_datetime(prices["Date"])
    prices["Code5"] = prices["Code4"].astype(str).str.zfill(4) + "0"
    prices = prices[prices["Code5"].isin(set(codes5))].copy()
    prices = prices[prices["Date"] >= WARMUP].copy()
    prices = prices.sort_values(["Code5", "Date"]).reset_index(drop=True)
    print(f"  株価: {len(prices):,}行  ({time.time()-t0:.1f}秒)")

    # ── 空売り比率読み込み ───────────────────────────────
    sr = pd.read_parquet(SR_PATH)
    sr["Date"] = pd.to_datetime(sr["Date"], errors="coerce")
    sr = sr[sr["Date"] >= OOS_START].copy()
    total_sell = sr["SellExShortVa"] + sr["ShrtWithResVa"] + sr["ShrtNoResVa"]
    sr["short_ratio"] = np.where(
        total_sell > 0,
        (sr["ShrtWithResVa"] + sr["ShrtNoResVa"]) / total_sell,
        np.nan
    )
    sr["S33_str"] = sr["S33"].astype(str).str.zfill(4)
    print(f"  空売り比率データ: {len(sr):,}件（OOS期間以降）")

    # ── 銘柄→S33マッピング ──────────────────────────────
    print("\n銘柄マスター取得中...")
    api_key = os.getenv("JQUANTS_REFRESH_TOKEN")
    client  = jquantsapi.ClientV2(api_key=api_key)
    master  = client.get_eq_master()
    code5_to_s33 = {
        str(row["Code"]).strip(): str(row.get("S33", "")).strip().zfill(4)
        for _, row in master.iterrows()
    }
    print(f"  マスター取得完了: {len(master)}銘柄")

    # ── OOS シグナル・トレード抽出 ───────────────────────
    print("\nOOS期間シグナル抽出中...")
    t0 = time.time()
    all_trades = []
    for code5 in codes5:
        df_s = prices[prices["Code5"] == code5].copy().reset_index(drop=True)
        if len(df_s) < 300:
            continue
        df_s = df_s.rename(columns={"Open": "Open", "High": "High",
                                     "Low": "Low", "Close": "Close",
                                     "Volume": "Volume"})
        df_s = compute_signals(df_s)
        trades = extract_oos_trades(df_s, code5)
        all_trades.extend(trades)

    print(f"  抽出完了: {len(all_trades)}トレード  ({time.time()-t0:.1f}秒)")

    if len(all_trades) < 30:
        print(f"  OOSトレード数が少なすぎます（{len(all_trades)}件）")
        return

    tdf = pd.DataFrame(all_trades)

    # ── S33付与 ──────────────────────────────────────────
    tdf["s33"] = tdf["code"].map(lambda c: code5_to_s33.get(c, ""))
    mapped = (tdf["s33"] != "").sum()
    print(f"\n  S33マッピング: {mapped}/{len(tdf)}件")

    # ── 空売り比率ジョイン ───────────────────────────────
    sr_lookup = sr[["Date", "S33_str", "short_ratio"]].rename(
        columns={"Date": "signal_date", "S33_str": "s33"}
    )
    tdf = tdf.merge(sr_lookup, on=["signal_date", "s33"], how="left")
    n_match = tdf["short_ratio"].notna().sum()
    print(f"  空売り比率マッチ: {n_match}/{len(tdf)}件 ({n_match/len(tdf)*100:.0f}%)")

    df_oos = tdf[tdf["short_ratio"].notna()].copy()

    # ── IS期間結果ロード（比較用） ───────────────────────
    is_corr = None
    if IS_CSV.exists():
        is_df = pd.read_csv(IS_CSV, parse_dates=["signal_date"])
        is_df_w = is_df[is_df["short_ratio"].notna()].copy()
        if len(is_df_w) >= 30:
            is_corr = is_df_w["short_ratio"].corr(is_df_w["pnl_pct"])

    # ── 分析出力 ─────────────────────────────────────────
    oos_corr, q25, q50, q75, n_oos, wr_oos = print_bucket_analysis(
        df_oos, f"OOS期間 ({OOS_START.date()}～{OOS_END.date()})"
    )

    # 閾値フィルター効果
    print(f"\n{'='*65}")
    print(f"  【閾値フィルター効果（OOS期間）】")
    print(f"{'─'*65}")
    print(f"  {'閾値':12} {'件数':>6} {'勝率':>7} {'平均損益':>9} {'除外率':>7}")
    print(f"  {'-'*12} {'-'*6} {'-'*7} {'-'*9} {'-'*7}")
    print(f"  (全体)      {n_oos:>6} {wr_oos:>6.1f}%   n/a    0.0%除外")
    for thr in [q25, q50, q75]:
        sub = df_oos[df_oos["short_ratio"] < thr]
        if len(sub) < 10:
            continue
        wr_s  = sub["win"].mean() * 100
        avg_s = sub["pnl_pct"].mean() * 100
        excl  = (1 - len(sub) / n_oos) * 100
        diff  = wr_s - wr_oos
        sign  = "↑" if diff > 0.5 else ("↓" if diff < -0.5 else "→")
        print(f"  <{thr:.1%}   {len(sub):>6} {wr_s:>6.1f}%{sign} {avg_s:>+8.2f}% {excl:>6.1f}%除外")

    # ── IS vs OOS 比較 ────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  【IS vs OOS 相関係数比較】")
    print(f"{'─'*65}")
    if is_corr is not None:
        print(f"  IS期間  (2016-2020): {is_corr:+.4f}")
    print(f"  OOS期間 (2023-2026): {oos_corr:+.4f}")

    # ── 最終結論 ──────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  【Phase 5 採用判断】")
    print(f"{'─'*65}")
    if is_corr is not None:
        same_sign = (is_corr > 0) == (oos_corr > 0)
        print(f"  IS/OOS で同方向: {'✅ YES' if same_sign else '❌ NO'}")

    if abs(oos_corr) < 0.05:
        print(f"  OOS相関 {oos_corr:+.4f} → 実質無相関")
        print(f"  ✅ 不採用推奨（IS同様にエッジなし）")
        verdict = "不採用"
    elif oos_corr > 0.05:
        print(f"  OOS相関 {oos_corr:+.4f} → IS (+0.055) と一致")
        if oos_corr > 0.10:
            print(f"  ✅ Phase 5 フィルター採用を検討してください")
            print(f"     空売り比率が高い業種 → 上昇傾向（ショートスクイーズ）")
            verdict = "採用候補"
        else:
            print(f"  ⚠️  弱い正相関。IS同様のパターンだが効果量が小さい。")
            print(f"     Phase 5 実運用での継続観察を推奨")
            verdict = "観察継続"
    else:
        print(f"  OOS相関 {oos_corr:+.4f} → ISと逆転（過適合の可能性）")
        print(f"  ✅ 不採用推奨")
        verdict = "不採用"

    print(f"\n  → 判定: 【{verdict}】")

    # ── CSV保存 ───────────────────────────────────────────
    tdf.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print(f"\n  → {OUT_CSV} に保存")
    print("=" * 65)


if __name__ == "__main__":
    main()
