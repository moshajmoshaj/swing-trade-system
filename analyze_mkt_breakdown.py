"""
analyze_mkt_breakdown.py
売買内訳（mkt_breakdown）× 戦略A シグナル 相関分析

目的:
  シグナル日の買い圧力 (LongBuyVa / 合計出来高) と
  トレード結果（勝率・期待値）の相関を検証する。
  高い買い圧力 → 勝率向上 であれば Phase 5 フィルター候補となる。

使用データ:
  - data/raw/prices_20y.parquet: IS期間株価
  - data/raw/mkt_breakdown.parquet: 銘柄別売買内訳
  - logs/final_candidates.csv: 戦略A候補30銘柄

出力:
  - コンソール: バケット別勝率・相関分析・採用判断
  - logs/mkt_breakdown_analysis.csv: シグナル日の詳細データ
"""
import sys, time
sys.stdout.reconfigure(encoding="utf-8")

from pathlib import Path
import pandas as pd
import numpy as np

PRICES_PATH = Path("data/raw/prices_20y.parquet")
MKT_PATH    = Path("data/raw/mkt_breakdown.parquet")
CAND_PATH   = Path("logs/final_candidates.csv")
OUT_CSV     = Path("logs/mkt_breakdown_analysis.csv")

IS_START = pd.Timestamp("2016-04-01")
IS_END   = pd.Timestamp("2020-12-31")

ATR_TP   = 6.0
ATR_SL   = 2.0
MAX_HOLD = 10
ADX_MIN  = 15


def compute_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Strategy A の全9条件シグナルを計算（prices_20y 列名前提）"""
    df = df.copy().reset_index(drop=True)

    df["SMA20"]    = df["Close"].rolling(20).mean()
    df["SMA50"]    = df["Close"].rolling(50).mean()
    df["SMA200"]   = df["Close"].rolling(200).mean()
    df["VOL_MA20"] = df["Volume"].rolling(20).mean()

    delta  = df["Close"].diff()
    gain   = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
    loss   = (-delta).clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
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
    atr_s  = df["ATR14"].replace(0, np.nan)
    sm_dp  = pd.Series(dm_p, index=df.index).ewm(alpha=1/14, adjust=False).mean()
    sm_dm  = pd.Series(dm_m, index=df.index).ewm(alpha=1/14, adjust=False).mean()
    di_p   = 100 * sm_dp / atr_s
    di_m   = 100 * sm_dm / atr_s
    dx     = 100 * (di_p - di_m).abs() / (di_p + di_m).replace(0, np.nan)
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


def extract_is_trades(df: pd.DataFrame) -> list[dict]:
    """IS期間のシグナルからトレード結果を抽出"""
    trades = []
    is_sigs = df[(df["Date"] >= IS_START) & (df["Date"] <= IS_END) & (df["signal"] == 1)]

    for i in is_sigs.index:
        if i + 1 >= len(df):
            continue
        sig  = df.iloc[i]
        nxt  = df.iloc[i + 1]
        ep   = nxt["Open"]
        atr  = sig["ATR14"]

        if (ep - sig["Close"]) / sig["Close"] < -0.015:  # ギャップダウン除外
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
        })
    return trades


def main():
    print("=" * 65)
    print("  売買内訳（mkt_breakdown）× 戦略A IS期間シグナル 相関分析")
    print(f"  IS期間: {IS_START.date()} ～ {IS_END.date()}")
    print("=" * 65)

    # ── 候補銘柄読み込み ─────────────────────────────────
    cand = pd.read_csv(CAND_PATH, dtype=str)
    col  = next(c for c in cand.columns if "code" in c.lower())
    codes4 = [c.zfill(4) for c in cand[col]]
    codes5 = [c + "0"    for c in codes4]
    print(f"\n  候補銘柄: {len(codes4)}銘柄")

    # ── 株価データ読み込み（IS+ウォームアップ） ──────────
    # prices_20y.parquet は Open/High/Low/Close/Volume が既存列（リネーム不要）
    print("データ読み込み中...")
    t0 = time.time()
    prices = pd.read_parquet(PRICES_PATH)
    prices["Date"] = pd.to_datetime(prices["Date"])
    prices["Code"] = prices["Code4"].astype(str).str.zfill(4) + "0"
    drop_c = [c for c in ["Code4","O","H","L","C","Vo","Va","UL","LL","AdjFactor",
                           "MO","MH","ML","MC","MUL","MLL","MVo","MVa",
                           "AO","AH","AL","AC","AUL","ALL","AVo","AVa"]
              if c in prices.columns]
    prices = prices.drop(columns=drop_c)
    prices = prices[prices["Code"].isin(set(codes5))].copy()
    prices = prices[prices["Date"] >= pd.Timestamp("2015-01-01")].copy()
    prices = prices.sort_values(["Code", "Date"]).reset_index(drop=True)
    print(f"  株価: {len(prices):,}行  ({time.time()-t0:.1f}秒)")

    # ── 売買内訳読み込み ─────────────────────────────────
    mb = pd.read_parquet(MKT_PATH,
                         columns=["Date", "Code", "LongBuyVa", "LongSellVa",
                                  "MrgnBuyNewVa", "MrgnSellNewVa"])
    mb["Date"] = pd.to_datetime(mb["Date"])
    mb = mb[mb["Code"].isin(set(codes5))].copy()
    mb = mb[mb["Date"] >= IS_START].copy()

    total_va = mb["LongBuyVa"] + mb["LongSellVa"]
    mb["buying_pressure"] = np.where(total_va > 0,
                                     mb["LongBuyVa"] / total_va, np.nan)
    total_mg = mb["MrgnBuyNewVa"] + mb["MrgnSellNewVa"]
    mb["margin_pressure"] = np.where(total_mg > 0,
                                     mb["MrgnBuyNewVa"] / total_mg, np.nan)
    mb = mb.rename(columns={"Code": "code"})
    mb = mb[["Date", "code", "buying_pressure", "margin_pressure"]].copy()
    print(f"  売買内訳: {len(mb):,}行")

    # ── シグナル・トレード抽出 ───────────────────────────
    print("\nIS期間シグナル抽出中...")
    t0 = time.time()
    all_trades = []
    for code5 in codes5:
        df_s = prices[prices["Code"] == code5].copy().reset_index(drop=True)
        if len(df_s) < 210:
            continue
        df_s = compute_signals(df_s)
        trades = extract_is_trades(df_s)
        for t in trades:
            t["code"] = code5
        all_trades.extend(trades)

    print(f"  抽出完了: {len(all_trades)}トレード  ({time.time()-t0:.1f}秒)")

    if not all_trades:
        print("IS期間のトレードが見つかりません")
        return

    tdf = pd.DataFrame(all_trades)

    # ── 売買内訳とジョイン ───────────────────────────────
    tdf = tdf.merge(
        mb.rename(columns={"Date": "signal_date"}),
        on=["signal_date", "code"],
        how="left"
    )
    n_match = tdf["buying_pressure"].notna().sum()
    print(f"  売買内訳マッチ: {n_match}/{len(tdf)}件 "
          f"({n_match/len(tdf)*100:.0f}%)")

    # ── ベースライン ─────────────────────────────────────
    n_all  = len(tdf)
    wr_all = tdf["win"].mean() * 100
    avg_r  = tdf["pnl_pct"].mean() * 100
    print(f"\n{'='*65}")
    print(f"  【ベースライン（全トレード）】")
    print(f"  件数={n_all}  勝率={wr_all:.1f}%  平均損益={avg_r:+.2f}%")

    df_w = tdf[tdf["buying_pressure"].notna()].copy()
    if len(df_w) < 30:
        print("  売買内訳マッチ件数が少なすぎます（<30）- 分析を中止")
        tdf.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
        return

    # ── バケット別分析 ───────────────────────────────────
    bins   = [0, 0.40, 0.45, 0.50, 0.55, 0.60, 1.01]
    labels = ["～40%", "40-45%", "45-50%", "50-55%", "55-60%", "60%～"]
    df_w["bp_bucket"] = pd.cut(df_w["buying_pressure"],
                                bins=bins, labels=labels, right=False)

    print(f"\n{'='*65}")
    print(f"  【buying_pressure バケット別 勝率・期待値】")
    print(f"  (buying_pressure = 現物買金額 / (現物買+現物売))")
    print(f"{'─'*65}")
    print(f"  {'バケット':8} {'件数':>5} {'勝率':>7} {'平均損益':>9} "
          f"{'利確率':>7} {'損切率':>7}")
    print(f"  {'-'*8} {'-'*5} {'-'*7} {'-'*9} {'-'*7} {'-'*7}")
    for label in labels:
        grp = df_w[df_w["bp_bucket"] == label]
        if len(grp) == 0:
            continue
        n   = len(grp)
        wr  = grp["win"].mean() * 100
        avg = grp["pnl_pct"].mean() * 100
        tp  = (grp["reason"] == "利確").mean() * 100
        sl  = (grp["reason"] == "損切り").mean() * 100
        print(f"  {label:8} {n:>5} {wr:>6.1f}% {avg:>+8.2f}% {tp:>6.1f}% {sl:>6.1f}%")

    corr = df_w["buying_pressure"].corr(df_w["pnl_pct"])
    print(f"\n  buying_pressure × pnl_pct 相関係数: {corr:+.4f}")

    # ── 閾値フィルター効果 ───────────────────────────────
    print(f"\n{'='*65}")
    print(f"  【閾値フィルター効果（buying_pressure > threshold のみ採用）】")
    print(f"{'─'*65}")
    print(f"  {'閾値':6} {'件数':>6} {'勝率':>7} {'平均損益':>9} {'除外率':>7}")
    print(f"  {'-'*6} {'-'*6} {'-'*7} {'-'*9} {'-'*7}")
    baseline_wr  = df_w["win"].mean() * 100
    baseline_avg = df_w["pnl_pct"].mean() * 100
    print(f"  (全体)  {len(df_w):>6} {baseline_wr:>6.1f}% {baseline_avg:>+8.2f}%   0.0%除外")
    for thr in [0.45, 0.48, 0.50, 0.52, 0.55]:
        sub = df_w[df_w["buying_pressure"] >= thr]
        if len(sub) < 10:
            continue
        wr_s  = sub["win"].mean() * 100
        avg_s = sub["pnl_pct"].mean() * 100
        excl  = (1 - len(sub) / len(df_w)) * 100
        diff  = wr_s - baseline_wr
        sign  = "↑" if diff > 0.5 else ("↓" if diff < -0.5 else "→")
        print(f"  >{thr:.2f} {len(sub):>6} {wr_s:>6.1f}%{sign} {avg_s:>+8.2f}% {excl:>6.1f}%除外")

    # 信用買い圧力
    has_mp = df_w["margin_pressure"].notna().sum()
    if has_mp >= 30:
        corr_m = df_w["margin_pressure"].corr(df_w["pnl_pct"])
        print(f"\n  margin_pressure × pnl_pct 相関係数: {corr_m:+.4f} (n={has_mp})")

    # ── 結論 ─────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  【クオンツとしての有効性】")
    if abs(corr) < 0.05:
        print(f"  相関係数 {corr:+.4f} → 実質無相関。")
        print("  buying_pressure はシグナル精度に寄与しない可能性が高い。")
        concl = "❌ 不採用推奨（Track A パターン踏襲: 追加フィルター=取引数減少）"
    elif corr > 0.05:
        print(f"  相関係数 {corr:+.4f} → 弱い正相関。高い買い圧力ほど勝率が高い傾向。")
        concl = "⚠️  Phase 5 で OOS 検証後に採用判断"
    else:
        print(f"  相関係数 {corr:+.4f} → 弱い負相関。")
        concl = "❌ 不採用推奨"
    print(f"  → {concl}")

    print(f"\n  【監査官としてのリスク指摘】")
    print(f"  IS期間（2016-2020）のみの分析。OOS確認なしに採用不可。")
    print(f"  mkt_breakdown は東証のみ。デリバティブ・ETF等の影響は未考慮。")

    # ── CSV保存 ──────────────────────────────────────────
    tdf.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print(f"\n  → {OUT_CSV} に保存")


if __name__ == "__main__":
    main()
