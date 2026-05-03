"""
analyze_short_ratio.py
業種別空売り比率 × 戦略A IS期間シグナル 相関分析

目的:
  シグナル日に対象銘柄の業種で空売り比率が高い場合、
  ショートスクイーズによる上昇期待 or 弱気の表れとして
  トレード結果と相関するか検証する。

使用データ:
  - data/raw/short_ratio.parquet: 業種別空売り比率（S33日次）
  - logs/mkt_breakdown_analysis.csv: IS期間シグナル・結果
  - J-Quants master API: 銘柄→業種S33マッピング

出力:
  - コンソール: 空売り比率レベル別の勝率・相関係数
  - logs/short_ratio_analysis.csv
"""
import sys, os
sys.stdout.reconfigure(encoding="utf-8")

from pathlib import Path
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import jquantsapi

load_dotenv()

SR_PATH  = Path("data/raw/short_ratio.parquet")
BASE_CSV = Path("logs/mkt_breakdown_analysis.csv")
OUT_CSV  = Path("logs/short_ratio_analysis.csv")


def main():
    print("=" * 65)
    print("  業種別空売り比率 × 戦略A IS期間シグナル 相関分析")
    print("=" * 65)

    # ── IS期間シグナルデータ読み込み ───────────────────────
    if not BASE_CSV.exists():
        print(f"ERROR: {BASE_CSV} なし。先に analyze_mkt_breakdown.py を実行。")
        return
    tdf = pd.read_csv(BASE_CSV, parse_dates=["signal_date"], dtype={"code": str})
    print(f"\n  シグナル件数: {len(tdf)}件")

    # ── 空売り比率読み込み ───────────────────────────────
    sr = pd.read_parquet(SR_PATH)
    sr["Date"] = pd.to_datetime(sr["Date"], errors="coerce")
    sr = sr[sr["Date"] >= pd.Timestamp("2016-01-01")].copy()

    # 空売り比率計算: 空売り / (通常売 + 空売り)
    total_sell = sr["SellExShortVa"] + sr["ShrtWithResVa"] + sr["ShrtNoResVa"]
    sr["short_ratio"] = np.where(
        total_sell > 0,
        (sr["ShrtWithResVa"] + sr["ShrtNoResVa"]) / total_sell,
        np.nan
    )
    # S33 整数 → 4桁文字列（masterのS33コードと一致させる）
    sr["S33_str"] = sr["S33"].astype(str).str.zfill(4)
    print(f"  空売り比率データ: {len(sr):,}件")

    # ── 銘柄→S33マッピング ──────────────────────────────
    api_key = os.getenv("JQUANTS_REFRESH_TOKEN")
    client  = jquantsapi.ClientV2(api_key=api_key)
    master  = client.get_eq_master()
    # code5 → S33_str
    code5_to_s33 = {}
    for _, row in master.iterrows():
        code5 = str(row["Code"]).strip()
        s33   = str(row.get("S33", "")).strip().zfill(4)
        code5_to_s33[code5] = s33

    # 候補銘柄のS33を付与
    codes5 = tdf["code"].unique().tolist() if "code" in tdf.columns else []
    code_s33_map = {c: code5_to_s33.get(c, "") for c in codes5}
    tdf["s33"] = tdf["code"].map(code_s33_map)
    mapped = (tdf["s33"] != "").sum()
    print(f"  S33マッピング済み: {mapped}/{len(tdf)}件")

    # ── 空売り比率とジョイン ─────────────────────────────
    # シグナル日の業種別空売り比率
    sr_lookup = sr[["Date", "S33_str", "short_ratio"]].copy()
    tdf = tdf.merge(
        sr_lookup.rename(columns={"Date": "signal_date", "S33_str": "s33",
                                   "short_ratio": "short_ratio"}),
        on=["signal_date", "s33"],
        how="left"
    )
    n_match = tdf["short_ratio"].notna().sum()
    print(f"  空売り比率マッチ: {n_match}/{len(tdf)}件 ({n_match/len(tdf)*100:.0f}%)")

    df_w = tdf[tdf["short_ratio"].notna()].copy()
    if len(df_w) < 30:
        print("  マッチ件数が少なすぎます（<30）- 分析中止")
        tdf.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
        return

    # ── 分析 ─────────────────────────────────────────────
    n_base = len(df_w)
    wr_base = df_w["win"].mean() * 100
    avg_base = df_w["pnl_pct"].mean() * 100
    corr = df_w["short_ratio"].corr(df_w["pnl_pct"])

    print(f"\n{'='*65}")
    print(f"  【ベースライン】 件数={n_base}  勝率={wr_base:.1f}%  平均損益={avg_base:+.2f}%")

    print(f"\n{'='*65}")
    print(f"  【空売り比率バケット別 勝率・期待値】")
    print(f"  (short_ratio = 空売り / 全売り金額)")
    print(f"{'─'*65}")
    print(f"  {'バケット':16} {'件数':>5} {'勝率':>7} {'平均損益':>9} {'損切率':>7}")
    print(f"  {'-'*16} {'-'*5} {'-'*7} {'-'*9} {'-'*7}")

    q25 = df_w["short_ratio"].quantile(0.25)
    q50 = df_w["short_ratio"].quantile(0.50)
    q75 = df_w["short_ratio"].quantile(0.75)
    buckets = [
        (f"低（<{q25:.1%}）",  df_w[df_w["short_ratio"] <  q25]),
        (f"中低（{q25:.1%}-{q50:.1%}）", df_w[(df_w["short_ratio"] >= q25) & (df_w["short_ratio"] < q50)]),
        (f"中高（{q50:.1%}-{q75:.1%}）", df_w[(df_w["short_ratio"] >= q50) & (df_w["short_ratio"] < q75)]),
        (f"高（≥{q75:.1%}）",  df_w[df_w["short_ratio"] >= q75]),
    ]
    for label, sub in buckets:
        if len(sub) == 0:
            continue
        wr  = sub["win"].mean() * 100
        avg = sub["pnl_pct"].mean() * 100
        sl  = (sub["reason"] == "損切り").mean() * 100 if "reason" in sub.columns else 0
        print(f"  {label:16} {len(sub):>5} {wr:>6.1f}% {avg:>+8.2f}% {sl:>6.1f}%")

    print(f"\n  short_ratio × pnl_pct 相関係数: {corr:+.4f}")

    # 閾値フィルター効果
    print(f"\n{'='*65}")
    print(f"  【閾値フィルター効果（空売り比率 > threshold を除外）】")
    print(f"{'─'*65}")
    print(f"  {'閾値':10} {'件数':>6} {'勝率':>7} {'平均損益':>9} {'除外率':>7}")
    print(f"  {'-'*10} {'-'*6} {'-'*7} {'-'*9} {'-'*7}")
    print(f"  (全体)    {n_base:>6} {wr_base:>6.1f}% {avg_base:>+8.2f}%   0.0%除外")
    for thr in [q25, q50, q75]:
        sub = df_w[df_w["short_ratio"] < thr]
        if len(sub) < 10:
            continue
        wr_s  = sub["win"].mean() * 100
        avg_s = sub["pnl_pct"].mean() * 100
        excl  = (1 - len(sub) / n_base) * 100
        diff  = wr_s - wr_base
        sign  = "↑" if diff > 0.5 else ("↓" if diff < -0.5 else "→")
        print(f"  <{thr:.1%}  {len(sub):>6} {wr_s:>6.1f}%{sign} {avg_s:>+8.2f}% {excl:>6.1f}%除外")

    # ── 結論 ─────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  【結論】")
    if abs(corr) < 0.05:
        print(f"  相関係数 {corr:+.4f} → 実質無相関。")
        print("  業種別空売り比率はシグナル精度に寄与しない。")
        print("  ✅ 不採用推奨（Track A の法則: 追加フィルター=取引数減少）")
    elif corr > 0.05:
        print(f"  相関係数 {corr:+.4f} → 弱い正相関（高い空売り比率→上昇傾向）。")
        print("  ⚠️  ショートスクイーズ効果の可能性。Phase 5 で OOS検証推奨。")
    else:
        print(f"  相関係数 {corr:+.4f} → 弱い負相関（高い空売り比率→下落傾向）。")
        print("  ✅ 不採用推奨。")

    tdf.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print(f"\n  → {OUT_CSV} に保存")


if __name__ == "__main__":
    main()
