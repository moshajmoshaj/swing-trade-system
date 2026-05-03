"""
analyze_dividends.py
配当落ち日前後の戦略A シグナル品質を検証する。

目的:
  ExDate（配当落ち日）の前後 N 日にエントリーしたシグナルは
  勝率が低いか？配当落ちによる株価下落が損切りを誘発するリスクを評価。
  → Phase 5 で配当除外フィルターを追加するか判断する材料とする。

使用データ:
  - data/raw/dividends.parquet: 配当データ（ExDate含む）
  - logs/mkt_breakdown_analysis.csv: IS期間シグナル・結果（再利用）

出力:
  - コンソール: 配当落ち近接別の勝率・損切率比較
  - logs/dividends_analysis.csv
"""
import sys
sys.stdout.reconfigure(encoding="utf-8")

from pathlib import Path
import pandas as pd
import numpy as np

DIV_PATH  = Path("data/raw/dividends.parquet")
BASE_CSV  = Path("logs/mkt_breakdown_analysis.csv")
OUT_CSV   = Path("logs/dividends_analysis.csv")


def main():
    print("=" * 65)
    print("  配当落ち日前後 × 戦略A IS期間シグナル 品質分析")
    print("=" * 65)

    # ── IS期間シグナルデータ読み込み ───────────────────────
    if not BASE_CSV.exists():
        print(f"ERROR: {BASE_CSV} なし。先に analyze_mkt_breakdown.py を実行。")
        return
    tdf = pd.read_csv(BASE_CSV, parse_dates=["signal_date"], dtype={"code": str})
    print(f"\n  シグナル件数: {len(tdf)}件")

    # ── 配当データ読み込み ───────────────────────────────
    div = pd.read_parquet(DIV_PATH)
    div["ExDate"] = pd.to_datetime(div["ExDate"], errors="coerce")
    div["Code"]   = div["Code"].astype(str).str.strip()

    # IS期間の配当落ち日のみ（ウォームアップ含め前後1ヶ月バッファ）
    is_start = pd.Timestamp("2015-12-01")
    is_end   = pd.Timestamp("2021-01-31")
    div = div[(div["ExDate"] >= is_start) & (div["ExDate"] <= is_end)].copy()
    div = div[div["ExDate"].notna()].copy()

    # 候補銘柄コードのみ（5桁形式）
    codes5 = tdf["code"].unique().tolist() if "code" in tdf.columns else []
    codes_set = set(codes5)
    div = div[div["Code"].isin(codes_set)].copy()
    print(f"  配当落ち日データ: {len(div)}件（候補銘柄 × IS期間）")

    # ── 各シグナルに最短配当落ち距離を計算 ─────────────────
    # (code, signal_date) → 最寄りの ExDate までの日数
    def get_min_exdate_dist(code: str, sig_date: pd.Timestamp) -> int | None:
        rows = div[div["Code"] == code]
        if rows.empty:
            return None
        diffs = (rows["ExDate"] - sig_date).dt.days
        # 前後両方向で最小絶対値
        min_abs = diffs.abs().min()
        # 「前」（シグナルがExDateより前）か「後」かも記録
        nearest_idx = diffs.abs().idxmin()
        return int(diffs[nearest_idx])  # 正 = ExDateがシグナルより未来, 負 = 過去

    print("配当落ち日距離を計算中...")
    tdf["div_days"] = tdf.apply(
        lambda r: get_min_exdate_dist(str(r["code"]), r["signal_date"])
        if "code" in r.index else None, axis=1)

    n_has_div = tdf["div_days"].notna().sum()
    n_no_div  = tdf["div_days"].isna().sum()
    print(f"  配当銘柄シグナル: {n_has_div}件  無配当銘柄: {n_no_div}件")

    # ── 分析 ─────────────────────────────────────────────
    df_w = tdf[tdf["div_days"].notna()].copy()
    df_w["div_days_abs"] = df_w["div_days"].abs()

    def show(label: str, sub: pd.DataFrame) -> None:
        if len(sub) == 0:
            return
        wr  = sub["win"].mean() * 100
        avg = sub["pnl_pct"].mean() * 100
        sl  = (sub["reason"] == "損切り").mean() * 100 if "reason" in sub.columns else 0
        print(f"  {label:28} {len(sub):>5} {wr:>6.1f}% {avg:>+8.2f}% {sl:>6.1f}%損切")

    print(f"\n{'='*65}")
    print(f"  【配当落ち日（ExDate）との距離別 勝率比較】")
    print(f"  正の値 = ExDate が未来（エントリーはExDate前）")
    print(f"  負の値 = ExDate が過去（エントリーはExDate後）")
    print(f"{'─'*65}")
    print(f"  {'カテゴリ':28} {'件数':>5} {'勝率':>7} {'平均損益':>9} {'損切率':>8}")
    print(f"  {'-'*28} {'-'*5} {'-'*7} {'-'*9} {'-'*8}")

    show("全配当銘柄（ベース）",          df_w)
    show("ExDate 3日以内（前）",          df_w[(df_w["div_days"] >= 0) & (df_w["div_days"] <= 3)])
    show("ExDate 4-10日前",               df_w[(df_w["div_days"] > 3) & (df_w["div_days"] <= 10)])
    show("ExDate 11-20日前",              df_w[(df_w["div_days"] > 10) & (df_w["div_days"] <= 20)])
    show("ExDate 3日以内（後）",          df_w[(df_w["div_days"] < 0) & (df_w["div_days"] >= -3)])
    show("ExDate 4-10日後",               df_w[(df_w["div_days"] < -3) & (df_w["div_days"] >= -10)])
    show("ExDate 20日以上離れた",         df_w[df_w["div_days_abs"] > 20])

    # 月別配当集中パターン（日本株は3月・9月）
    print(f"\n{'='*65}")
    print(f"  【シグナル月別 × 配当近接状況】（3・9月の影響確認）")
    print(f"{'─'*65}")
    df_w["signal_month"] = pd.to_datetime(df_w["signal_date"]).dt.month
    for m in [3, 6, 9, 12]:
        sub = df_w[df_w["signal_month"] == m]
        near = sub[sub["div_days_abs"] <= 10]
        if len(sub) == 0:
            continue
        wr_all = sub["win"].mean() * 100
        wr_near = near["win"].mean() * 100 if len(near) > 0 else float("nan")
        print(f"  {m:2d}月: 全{len(sub):3d}件 勝率{wr_all:.1f}%  "
              f"ExDate10日以内{len(near):3d}件 勝率{wr_near:.1f}%")

    # ── 結論 ─────────────────────────────────────────────
    near3  = df_w[(df_w["div_days"] >= 0) & (df_w["div_days"] <= 3)]
    base_wr = df_w["win"].mean() * 100 if len(df_w) > 0 else 0
    near_wr = near3["win"].mean() * 100 if len(near3) > 0 else 0
    diff    = near_wr - base_wr

    print(f"\n{'='*65}")
    print(f"  【結論】")
    print(f"  ExDate 3日以内シグナルの勝率変化: {base_wr:.1f}% → {near_wr:.1f}% ({diff:+.1f}pt)  件数: {len(near3)}")
    if diff < -5 and len(near3) >= 10:
        print("  ❗ 配当落ち直前シグナルの勝率が大幅低下。")
        print("  ⚠️  Phase 5 で ExDate 前 N 日の除外フィルター追加を検討。")
    elif diff < -2 and len(near3) >= 5:
        print("  ⚠️  わずかな勝率低下。件数が少なく統計的信頼性は低い。")
        print("  → Phase 5 で引き続き観察推奨。フィルター追加は保留。")
    else:
        print("  ✅ 配当落ち日前後で勝率に有意な差なし。フィルター不要。")
        print("  → 戦略Aの陽線条件が配当落ちシグナルを自然に排除している可能性。")

    tdf.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print(f"\n  → {OUT_CSV} に保存")


if __name__ == "__main__":
    main()
