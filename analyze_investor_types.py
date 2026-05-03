"""
analyze_investor_types.py
外国人投資家フロー × 戦略A IS期間シグナル 相関分析

目的:
  週次外国人フロー（FrgnBal）が市場レジームの代替/補完指標として
  機能するか検証する。

使用データ:
  - data/raw/investor_types.parquet: 週次投資家別売買動向（東証プライム）
  - logs/mkt_breakdown_analysis.csv: IS期間シグナル・結果（再利用）

出力:
  - コンソール: フロー方向別勝率・相関分析・採用判断
  - logs/investor_types_analysis.csv
"""
import sys
sys.stdout.reconfigure(encoding="utf-8")

from pathlib import Path
import pandas as pd
import numpy as np

IT_PATH   = Path("data/raw/investor_types.parquet")
BASE_CSV  = Path("logs/mkt_breakdown_analysis.csv")
OUT_CSV   = Path("logs/investor_types_analysis.csv")


def main():
    print("=" * 65)
    print("  外国人投資家フロー × 戦略A IS期間シグナル 相関分析")
    print("=" * 65)

    # ── IS期間シグナルデータ読み込み ───────────────────────
    if not BASE_CSV.exists():
        print(f"ERROR: {BASE_CSV} が見つかりません。先に analyze_mkt_breakdown.py を実行してください。")
        return
    tdf = pd.read_csv(BASE_CSV, parse_dates=["signal_date"], dtype={"code": str})
    print(f"\n  シグナル件数: {len(tdf)}件")

    # ── 投資家別フロー読み込み（東証プライムのみ）─────────
    it = pd.read_parquet(IT_PATH)
    it["PubDate"] = pd.to_datetime(it["PubDate"], errors="coerce")
    it["StDate"]  = pd.to_datetime(it["StDate"],  errors="coerce")
    it["EnDate"]  = pd.to_datetime(it["EnDate"],  errors="coerce")

    # IS期間(2016-2020)はTSE1st（東証一部）、2022-以降はTSEPrime
    # 両セクションを結合して連続したタイムラインを作成
    prime_1st   = it[it["Section"] == "TSE1st"].copy()
    prime_prime = it[it["Section"] == "TSEPrime"].copy()
    prime = pd.concat([prime_1st, prime_prime], ignore_index=True)
    prime = prime.sort_values("StDate").drop_duplicates(subset=["StDate"]).reset_index(drop=True)
    print(f"  東証1部/プライム週次データ: {len(prime)}週分  "
          f"({prime['StDate'].min().date()} ～ {prime['EnDate'].max().date()})")

    # ── 数値型へ変換 ─────────────────────────────────────
    for col in ["FrgnBal", "IndBal", "InvTrBal", "BrkBal"]:
        if col in prime.columns:
            prime[col] = pd.to_numeric(prime[col], errors="coerce")

    # 4週ローリング合計（トレンド判定用）
    prime["FrgnBal_4w"] = prime["FrgnBal"].rolling(4).sum()

    # ── シグナル日を週にマッピング ─────────────────────────
    def find_week(signal_date: pd.Timestamp) -> int | None:
        mask = (prime["StDate"] <= signal_date) & (prime["EnDate"] >= signal_date)
        idx = prime.index[mask]
        return int(idx[0]) if len(idx) > 0 else None

    print("シグナル日と週次データをマッピング中...")
    tdf["week_idx"] = tdf["signal_date"].apply(find_week)
    matched = tdf["week_idx"].notna().sum()
    print(f"  マッチ率: {matched}/{len(tdf)}件 ({matched/len(tdf)*100:.0f}%)")

    tdf = tdf[tdf["week_idx"].notna()].copy()
    tdf["week_idx"] = tdf["week_idx"].astype(int)

    # フロー情報をジョイン
    week_cols = ["FrgnBal", "FrgnBal_4w", "IndBal"]
    available = [c for c in week_cols if c in prime.columns]
    tdf = tdf.merge(
        prime[available].reset_index().rename(columns={"index": "week_idx"}),
        on="week_idx", how="left"
    )

    # ── 分析 ─────────────────────────────────────────────
    n_all = len(tdf)
    wr_all = tdf["win"].mean() * 100
    print(f"\n{'='*65}")
    print(f"  【ベースライン】 件数={n_all}  勝率={wr_all:.1f}%")

    # ─ 週次 FrgnBal 方向別 ─
    print(f"\n{'='*65}")
    print(f"  【当週 FrgnBal（外国人フロー）方向別 勝率・期待値】")
    print(f"{'─'*65}")
    print(f"  {'カテゴリ':20} {'件数':>5} {'勝率':>7} {'平均損益':>9}")
    print(f"  {'-'*20} {'-'*5} {'-'*7} {'-'*9}")

    corr_frg = tdf["FrgnBal"].corr(tdf["pnl_pct"]) if "FrgnBal" in tdf.columns else np.nan

    if "FrgnBal" in tdf.columns:
        def show(label, sub):
            if len(sub) == 0: return
            print(f"  {label:20} {len(sub):>5} {sub['win'].mean()*100:>6.1f}% "
                  f"{sub['pnl_pct'].mean()*100:>+8.2f}%")
        show("外国人 大幅買越 (>2兆)",  tdf[tdf["FrgnBal"] >  2e12])
        show("外国人 買越 (0~2兆)",    tdf[(tdf["FrgnBal"] > 0) & (tdf["FrgnBal"] <= 2e12)])
        show("外国人 売越 (<0)",        tdf[tdf["FrgnBal"] < 0])
        print(f"\n  FrgnBal（当週）× pnl_pct 相関係数: {corr_frg:+.4f}")

    # ─ 4週トレンド別 ─
    print(f"\n{'='*65}")
    print(f"  【FrgnBal 4週合計（トレンド）別 勝率】")
    print(f"{'─'*65}")
    if "FrgnBal_4w" in tdf.columns:
        corr_4w = tdf["FrgnBal_4w"].corr(tdf["pnl_pct"])
        q1 = tdf["FrgnBal_4w"].quantile(0.33)
        q2 = tdf["FrgnBal_4w"].quantile(0.67)
        show("下位1/3（売越傾向）",  tdf[tdf["FrgnBal_4w"] < q1])
        show("中位1/3",              tdf[(tdf["FrgnBal_4w"] >= q1) & (tdf["FrgnBal_4w"] < q2)])
        show("上位1/3（買越傾向）",  tdf[tdf["FrgnBal_4w"] >= q2])
        print(f"\n  FrgnBal_4w × pnl_pct 相関係数: {corr_4w:+.4f}")

    # ─ 市場レジーム代替指標としての評価 ─
    print(f"\n{'='*65}")
    print(f"  【市場レジーム代替指標としての評価】")
    print(f"{'─'*65}")
    if "FrgnBal" in tdf.columns and n_all > 0:
        # 現行TOPIXフィルターの代替: FrgnBal > 0 の週のみ有効
        frg_pos = tdf[tdf["FrgnBal"] > 0]
        frg_neg = tdf[tdf["FrgnBal"] <= 0]
        excl = (1 - len(frg_pos) / n_all) * 100
        print(f"  FrgnBal > 0 の週のみ: 件数={len(frg_pos)} "
              f"勝率={frg_pos['win'].mean()*100:.1f}% "
              f"除外={excl:.1f}%")
        print(f"  FrgnBal ≤ 0 の週:     件数={len(frg_neg)} "
              f"勝率={frg_neg['win'].mean()*100:.1f}%")

    # ── 結論 ─────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  【結論】")

    main_corr = corr_frg if not np.isnan(corr_frg) else (corr_4w if "corr_4w" in dir() else 0)
    if abs(main_corr) < 0.05:
        print(f"  相関係数 {main_corr:+.4f} → 実質無相関。")
        print("  外国人フローはシグナル精度に寄与しない。")
        print("  ✅ 現行 TOPIX SMA フィルターで十分。代替不要。")
    elif main_corr > 0.05:
        print(f"  相関係数 {main_corr:+.4f} → 弱い正相関。")
        print("  ⚠️  TOPIXフィルターの補完として Phase 5 で OOS検証を推奨。")
    else:
        print(f"  相関係数 {main_corr:+.4f} → 弱い負相関。不採用推奨。")

    tdf.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print(f"\n  → {OUT_CSV} に保存")


if __name__ == "__main__":
    main()
