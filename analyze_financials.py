"""
analyze_financials.py
財務諸表データ（fins_summary）× 戦略A/E候補銘柄 プロファイル分析

目的:
  1. IS期間（2016-2020）に選定された候補銘柄の財務プロファイルを把握する
  2. 財務指標（EPS成長・ROE・営業CF）とOOS成績の相関を確認する
  3. 戦略F（PEAD）用EPS成長パターンの検証

使用データ:
  - data/raw/fins_summary.parquet: 財務サマリー（EPS/ROE/CF等）
  - logs/final_candidates.csv: 戦略A候補30銘柄（IS選定）
  - logs/strategy_e_candidates.csv: 戦略E候補30銘柄（IS選定）
  - logs/mkt_breakdown_analysis.csv: IS期間OOS成績（pnl_pct）

出力:
  - コンソール: 候補銘柄財務プロファイル・市場平均との比較・OOS相関
  - logs/financials_analysis.csv
"""
import sys, os
sys.stdout.reconfigure(encoding="utf-8")

from pathlib import Path
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import jquantsapi

load_dotenv()

FS_PATH      = Path("data/raw/fins_summary.parquet")
CAND_A_PATH  = Path("logs/final_candidates.csv")
CAND_E_PATH  = Path("logs/strategy_e_candidates.csv")
BASE_CSV     = Path("logs/mkt_breakdown_analysis.csv")
OUT_CSV      = Path("logs/financials_analysis.csv")

IS_START = pd.Timestamp("2016-01-01")
IS_END   = pd.Timestamp("2020-12-31")


def load_codes(path: Path) -> list[str]:
    df = pd.read_csv(path, dtype=str)
    col = next(c for c in df.columns if "code" in c.lower())
    return [c.zfill(4) + "0" for c in df[col].tolist()]


def compute_roe(fs_row: pd.Series) -> float | None:
    """ROE = NP / Eq"""
    np_val = pd.to_numeric(fs_row.get("NP"), errors="coerce")
    eq_val = pd.to_numeric(fs_row.get("Eq"),  errors="coerce")
    if pd.isna(np_val) or pd.isna(eq_val) or eq_val <= 0:
        return None
    return float(np_val / eq_val * 100)


def compute_op_margin(fs_row: pd.Series) -> float | None:
    """営業利益率 = OP / Sales"""
    op    = pd.to_numeric(fs_row.get("OP"),    errors="coerce")
    sales = pd.to_numeric(fs_row.get("Sales"), errors="coerce")
    if pd.isna(op) or pd.isna(sales) or sales <= 0:
        return None
    return float(op / sales * 100)


def compute_eps_growth(hist: pd.DataFrame) -> float | None:
    """直近2期の EPS 成長率"""
    eps = hist.sort_values("DiscDate")["EPS"].apply(
        lambda x: pd.to_numeric(x, errors="coerce")).dropna()
    if len(eps) < 2 or eps.iloc[-2] <= 0:
        return None
    return float((eps.iloc[-1] - eps.iloc[-2]) / eps.iloc[-2] * 100)


def financial_profile(code5: str, fs: pd.DataFrame) -> dict:
    """IS期間中の最終通期決算（FY）から財務指標を抽出"""
    mask = (
        (fs["Code"] == code5) &
        (fs["CurPerType"] == "FY") &
        (fs["DiscDate"] >= IS_START) &
        (fs["DiscDate"] <= IS_END)
    )
    sub = fs[mask].sort_values("DiscDate")
    if sub.empty:
        return {}

    latest = sub.iloc[-1]
    return {
        "code":         code5,
        "latest_disc":  latest["DiscDate"],
        "roe":          compute_roe(latest),
        "op_margin":    compute_op_margin(latest),
        "eps_growth":   compute_eps_growth(sub),
        "eq_ratio":     pd.to_numeric(latest.get("EqAR"), errors="coerce"),
        "cfo_m":        pd.to_numeric(latest.get("CFO"),  errors="coerce"),
        "eps":          pd.to_numeric(latest.get("EPS"),  errors="coerce"),
    }


def main():
    print("=" * 65)
    print("  財務諸表（fins_summary）× 戦略A/E候補銘柄 プロファイル分析")
    print(f"  IS期間: {IS_START.date()} ～ {IS_END.date()}")
    print("=" * 65)

    # ── データ読み込み ────────────────────────────────────
    fs = pd.read_parquet(FS_PATH)
    fs["DiscDate"] = pd.to_datetime(fs["DiscDate"], errors="coerce")
    fs["Code"]     = fs["Code"].astype(str).str.strip()
    print(f"\n  fins_summary: {len(fs):,}行  {fs['Code'].nunique():,}銘柄")

    codes_a = load_codes(CAND_A_PATH) if CAND_A_PATH.exists() else []
    codes_e = load_codes(CAND_E_PATH) if CAND_E_PATH.exists() else []
    all_cands = list(dict.fromkeys(codes_a + codes_e))
    print(f"  候補銘柄: A={len(codes_a)} E={len(codes_e)} 合計ユニーク={len(all_cands)}")

    # ── 候補銘柄プロファイル ───────────────────────────────
    print("\n候補銘柄の財務プロファイル計算中...")
    profiles = [financial_profile(c, fs) for c in all_cands]
    prof_df  = pd.DataFrame([p for p in profiles if p])
    print(f"  プロファイル取得: {len(prof_df)}/{len(all_cands)}銘柄")

    # ── 全プライム株との比較 ──────────────────────────────
    # IS期間の全銘柄最終通期決算
    fs_all = fs[(fs["CurPerType"] == "FY") &
                (fs["DiscDate"] >= IS_START) &
                (fs["DiscDate"] <= IS_END)].copy()
    fs_all = fs_all.sort_values("DiscDate").groupby("Code").last().reset_index()

    market_roes = fs_all.apply(compute_roe, axis=1).dropna()
    market_ops  = fs_all.apply(compute_op_margin, axis=1).dropna()

    # ── 結果表示 ──────────────────────────────────────────
    metrics = [
        ("ROE (%)",       "roe",       market_roes),
        ("営業利益率 (%)", "op_margin", market_ops),
        ("EPS成長率 (%)",  "eps_growth",None),
        ("自己資本比率(%)", "eq_ratio",  None),
    ]

    print(f"\n{'='*65}")
    print("  【候補銘柄 財務プロファイル vs 市場平均】")
    print(f"{'─'*65}")
    print(f"  {'指標':16} {'候補中央値':>10} {'候補平均':>10} {'市場中央値':>10}")
    print(f"  {'-'*16} {'-'*10} {'-'*10} {'-'*10}")

    for label, col, market_series in metrics:
        cand_vals = prof_df[col].dropna() if col in prof_df.columns else pd.Series(dtype=float)
        cand_vals = pd.to_numeric(cand_vals, errors="coerce").dropna()
        c_med  = cand_vals.median() if len(cand_vals) > 0 else float("nan")
        c_mean = cand_vals.mean()   if len(cand_vals) > 0 else float("nan")
        m_med  = market_series.median() if market_series is not None and len(market_series) > 0 else float("nan")
        print(f"  {label:16} {c_med:>9.1f}% {c_mean:>9.1f}% "
              f"{m_med:>9.1f}%" if not pd.isna(m_med) else
              f"  {label:16} {c_med:>9.1f}% {c_mean:>9.1f}%       N/A")

    # ── OOS成績との相関 ───────────────────────────────────
    if BASE_CSV.exists():
        print(f"\n{'='*65}")
        print("  【IS期間財務指標 × OOS取引成績 相関分析】")
        print(f"{'─'*65}")

        tdf = pd.read_csv(BASE_CSV, dtype={"code": str})
        # 銘柄ごとのOOS平均損益
        oos_by_code = tdf.groupby("code")["pnl_pct"].agg(["mean","count","sum"]).reset_index()
        oos_by_code.columns = ["code", "avg_pnl", "trades", "total_pnl"]
        oos_by_code["win_rate"] = tdf.groupby("code")["win"].mean().values

        merged = prof_df.merge(oos_by_code, on="code", how="inner")
        print(f"  マッチ銘柄数: {len(merged)}")

        print(f"\n  {'財務指標':16} {'OOS平均損益相関':>14} {'OOS勝率相関':>12}")
        print(f"  {'-'*16} {'-'*14} {'-'*12}")
        for col in ["roe", "op_margin", "eps_growth", "eq_ratio"]:
            if col not in merged.columns:
                continue
            vals = pd.to_numeric(merged[col], errors="coerce")
            valid = merged[vals.notna()].copy()
            valid[col] = pd.to_numeric(valid[col], errors="coerce")
            if len(valid) < 5:
                continue
            c_pnl = valid[col].corr(valid["avg_pnl"])
            c_wr  = valid[col].corr(valid["win_rate"])
            label_map = {"roe": "ROE", "op_margin": "営業利益率",
                         "eps_growth": "EPS成長率", "eq_ratio": "自己資本比率"}
            print(f"  {label_map.get(col, col):16} {c_pnl:>+13.4f}  {c_wr:>+11.4f}")

        print(f"\n  → 相関係数が |0.1| 未満の場合: 財務指標とOOS成績は独立（Track A結論と整合）")

    # ── 戦略F研究: EPS成長パターン ────────────────────────
    print(f"\n{'='*65}")
    print("  【戦略F研究: EPS成長率 × PEAD効果（IS期間 通期FY）】")
    print(f"{'─'*65}")

    # EPS成長20%以上の開示後の株価動向を確認
    # fins_summaryの全銘柄でEPS成長率を計算
    fy_all = fs[(fs["CurPerType"] == "FY") &
                (fs["DiscDate"] >= IS_START) &
                (fs["DiscDate"] <= IS_END)].copy()
    fy_all["EPS_num"] = pd.to_numeric(fy_all["EPS"], errors="coerce")

    fy_sorted = fy_all.sort_values(["Code", "DiscDate"])
    fy_sorted["EPS_prev"] = fy_sorted.groupby("Code")["EPS_num"].shift(1)
    fy_sorted["EPS_growth"] = np.where(
        (fy_sorted["EPS_prev"] > 0) & fy_sorted["EPS_num"].notna(),
        (fy_sorted["EPS_num"] - fy_sorted["EPS_prev"]) / fy_sorted["EPS_prev"] * 100,
        np.nan
    )

    growth_dist = fy_sorted["EPS_growth"].dropna()
    print(f"  EPS成長率の分布（全プライム・IS期間）:")
    print(f"    件数: {len(growth_dist):,}  中央値: {growth_dist.median():+.1f}%  "
          f"平均: {growth_dist.mean():+.1f}%")
    for thr in [20, 30, 50]:
        n = (growth_dist >= thr).sum()
        pct = n / len(growth_dist) * 100
        print(f"    EPS成長≥{thr}%: {n:,}件 ({pct:.1f}%)")

    # ── 結論 ──────────────────────────────────────────────
    print(f"\n{'='*65}")
    print("  【クオンツ視点】")
    print("  財務指標（ROE・営業利益率・EPS成長・自己資本比率）と")
    print("  OOS取引成績の相関係数が |0.1| 未満であれば Track A 結論と一致:")
    print("  『モメンタムのエッジは財務品質とは独立して機能する』")
    print()
    print("  【戦略F実装の前提確認】")
    print(f"  IS期間の EPS成長≥20%開示: 毎年 ~{len(growth_dist)//5:,} 件存在")
    print("  → scanner.py の fins_summary から毎日 EPS成長≥20% の通期開示を検出し")
    print("    SMA200+ / RSI45-70 / 出来高 / 陽線 でフィルタリングする戦略Fは現実的。")

    # CSV保存
    if not prof_df.empty:
        prof_df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
        print(f"\n  → {OUT_CSV} に保存")


if __name__ == "__main__":
    main()
