"""
analyze_phase4.py
Phase 4 ペーパートレード 実績分析スクリプト

目的:
  - Phase 4の取引記録を戦略別に集計
  - OOSバックテスト予測との比較
  - 月次推移・保有中ポジション確認

実行タイミング:
  - Phase 4期間中（途中経過確認）
  - Phase 4終了時（2026-07-27）の最終評価

使用方法:
  python analyze_phase4.py

データ少なくても正常動作する（0件でもエラーなし）
"""
import sys
sys.stdout.reconfigure(encoding="utf-8")

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
from datetime import datetime, date
import pandas as pd
import numpy as np

TRADE_LOG    = Path("logs/paper_trade_log.xlsx")
SCANNER_LOG  = Path("logs/scanner_log.csv")
PHASE4_START = date(2026, 4, 27)
PHASE4_END   = date(2026, 7, 27)
INITIAL_CAPITAL = 1_000_000

# OOSバックテスト予測値（20年バックテスト・2026-05-02実施）
OOS_PREDICTIONS = {
    "A": {"cagr": 15.3, "win_pct": 64.5, "max_dd": -2.0},
    "C": {"cagr":  0.8, "win_pct": 44.8, "max_dd": -3.3},
    "D": {"cagr":  0.2, "win_pct": 100.0,"max_dd":  0.0},
    "E": {"cagr":  5.3, "win_pct": 71.1, "max_dd": -1.8},
    "F": {"cagr":  6.7, "win_pct": 60.7, "max_dd": -2.5},
}


def load_trades() -> tuple[pd.DataFrame, pd.DataFrame]:
    """取引記録を読み込み（クローズ済み・保有中に分割）"""
    if not TRADE_LOG.exists():
        print(f"⚠️  {TRADE_LOG} が見つかりません")
        return pd.DataFrame(), pd.DataFrame()

    df = pd.read_excel(TRADE_LOG, sheet_name="取引記録")
    df.columns = [
        "entry_date", "code", "entry_price", "stop_loss", "take_profit",
        "atr", "rsi", "adx", "exit_date", "exit_price",
        "pnl_yen", "pnl_pct", "exit_reason", "strategy"
    ]
    df["entry_date"] = pd.to_datetime(df["entry_date"], errors="coerce")
    df["exit_date"]  = pd.to_datetime(df["exit_date"],  errors="coerce")
    df["pnl_yen"]    = pd.to_numeric(df["pnl_yen"],     errors="coerce")
    df["pnl_pct"]    = pd.to_numeric(df["pnl_pct"],     errors="coerce")
    df["strategy"]   = df["strategy"].fillna("A").astype(str).str.strip()
    df = df.dropna(subset=["entry_date"])

    closed = df[df["exit_date"].notna()].copy()
    open_  = df[df["exit_date"].isna()].copy()
    return closed, open_


def calc_strategy_metrics(closed: pd.DataFrame, strategy: str) -> dict:
    """1戦略の主要メトリクスを計算"""
    df = closed[closed["strategy"] == strategy].copy()
    if df.empty:
        return {"trades": 0}

    wins    = (df["pnl_yen"] > 0).sum()
    n       = len(df)
    total   = df["pnl_yen"].sum()
    avg     = df["pnl_yen"].mean()
    win_pct = wins / n * 100

    # 最大ドローダウン（簡易：累積損益の最大下落）
    equity  = INITIAL_CAPITAL + df.sort_values("exit_date")["pnl_yen"].cumsum()
    peak    = equity.cummax()
    dd      = ((equity - peak) / peak * 100).min()

    # 年利換算（Phase 4期間ベース）
    days_elapsed = (date.today() - PHASE4_START).days
    if days_elapsed > 0:
        cagr = ((INITIAL_CAPITAL + total) / INITIAL_CAPITAL) ** (365 / days_elapsed) - 1
        cagr *= 100
    else:
        cagr = 0.0

    return {
        "trades":   n,
        "wins":     wins,
        "win_pct":  win_pct,
        "total":    total,
        "avg":      avg,
        "max_dd":   dd,
        "cagr":     cagr,
    }


def print_header(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def fmt_pct(v: float, pred: float | None = None) -> str:
    sign = "+" if v >= 0 else ""
    s = f"{sign}{v:.1f}%"
    if pred is not None:
        diff = v - pred
        dsign = "+" if diff >= 0 else ""
        s += f"  (予測:{pred:+.1f}%  差:{dsign}{diff:.1f}pt)"
    return s


def main() -> None:
    today = date.today()
    elapsed = (today - PHASE4_START).days
    remaining = (PHASE4_END - today).days

    print_header("Phase 4 ペーパートレード 実績分析")
    print(f"  Phase 4 期間   : {PHASE4_START} 〜 {PHASE4_END}")
    print(f"  経過日数       : {elapsed}日 / 91日")
    print(f"  残り日数       : {remaining}日")
    print(f"  分析実行日時   : {datetime.now().strftime('%Y-%m-%d %H:%M')}")

    closed, open_ = load_trades()

    # ── 保有中ポジション ───────────────────────────────────
    print_header("保有中ポジション")
    if open_.empty:
        print("  現在の保有なし")
    else:
        print(f"  {'戦略':<4} {'コード':<7} {'エントリー日':<12} {'価格':>7} {'SL':>7} {'TP':>7}")
        print(f"  {'-'*4} {'-'*7} {'-'*12} {'-'*7} {'-'*7} {'-'*7}")
        for _, row in open_.iterrows():
            ep   = f"{row['entry_price']:>7,.0f}" if pd.notna(row['entry_price']) else "    ---"
            sl   = f"{row['stop_loss']:>7,.0f}"   if pd.notna(row['stop_loss'])   else "    ---"
            tp   = f"{row['take_profit']:>7,.0f}" if pd.notna(row['take_profit']) else "    ---"
            edt  = row['entry_date'].strftime('%Y-%m-%d') if pd.notna(row['entry_date']) else "---"
            strat = str(row['strategy'])
            print(f"  {strat:<4} {str(row['code']):<7} {edt:<12} {ep} {sl} {tp}")

    # ── 総合サマリー ──────────────────────────────────────
    print_header("総合サマリー（確定取引）")
    if closed.empty:
        print("  確定取引なし（データ蓄積待ち）")
    else:
        total_pnl = closed["pnl_yen"].sum()
        total_win = (closed["pnl_yen"] > 0).sum()
        total_n   = len(closed)
        print(f"  総取引数     : {total_n}件")
        print(f"  総勝率       : {total_win/total_n*100:.1f}%")
        print(f"  累計損益     : {total_pnl:+,.0f}円")
        print(f"  平均損益/取引: {closed['pnl_yen'].mean():+,.0f}円")

    # ── 戦略別集計 ────────────────────────────────────────
    print_header("戦略別集計（OOS予測との比較）")
    strategies = sorted(set(closed["strategy"].unique()) | set(OOS_PREDICTIONS.keys()))

    print(f"  {'戦略':<4} {'取引':>4} {'勝率':>7} {'累計損益':>10} {'推定年利':>10} {'OOS予測':>10}")
    print(f"  {'-'*4} {'-'*4} {'-'*7} {'-'*10} {'-'*10} {'-'*10}")

    for strat in strategies:
        m    = calc_strategy_metrics(closed, strat)
        pred = OOS_PREDICTIONS.get(strat, {})

        if m["trades"] == 0:
            print(f"  {strat:<4} {'0':>4}  {'---':>7}  {'---':>10}  {'---':>10}  {pred.get('cagr', 0):>9.1f}%")
            continue

        win_pct_str = f"{m['win_pct']:.1f}%"
        pnl_str     = f"{m['total']:+,.0f}円"
        cagr_str    = f"{m['cagr']:+.1f}%"
        pred_str    = f"{pred.get('cagr', 0):+.1f}%"
        print(f"  {strat:<4} {m['trades']:>4}  {win_pct_str:>7}  {pnl_str:>10}  {cagr_str:>10}  {pred_str:>10}")

    # ── 詳細比較 ──────────────────────────────────────────
    if not closed.empty:
        print_header("予実比較詳細（取引あり戦略のみ）")
        active_strats = [s for s in strategies
                         if calc_strategy_metrics(closed, s)["trades"] > 0]
        for strat in active_strats:
            m    = calc_strategy_metrics(closed, strat)
            pred = OOS_PREDICTIONS.get(strat, {})
            print(f"\n  【戦略{strat}】 {m['trades']}取引")
            print(f"  勝率   : {m['win_pct']:.1f}%  （OOS予測: {pred.get('win_pct', 0):.1f}%）")
            print(f"  推定年利: {fmt_pct(m['cagr'], pred.get('cagr'))}")
            print(f"  最大DD : {m['max_dd']:.1f}%  （OOS予測: {pred.get('max_dd', 0):.1f}%）")

    # ── 月次推移 ──────────────────────────────────────────
    if not closed.empty and len(closed) >= 2:
        print_header("月次推移")
        closed["ym"] = closed["exit_date"].dt.to_period("M")
        monthly = closed.groupby("ym").agg(
            trades=("pnl_yen", "count"),
            wins=("pnl_yen", lambda x: (x > 0).sum()),
            pnl=("pnl_yen", "sum")
        ).reset_index()
        monthly["win_pct"] = monthly["wins"] / monthly["trades"] * 100
        monthly["cumulative"] = monthly["pnl"].cumsum()

        print(f"  {'年月':<8} {'取引':>4} {'勝率':>7} {'月次損益':>10} {'累計損益':>10}")
        print(f"  {'-'*8} {'-'*4} {'-'*7} {'-'*10} {'-'*10}")
        for _, r in monthly.iterrows():
            print(f"  {str(r['ym']):<8} {r['trades']:>4} {r['win_pct']:>6.1f}% "
                  f"{r['pnl']:>+10,.0f}円 {r['cumulative']:>+10,.0f}円")

    # ── 決済理由分析 ──────────────────────────────────────
    if not closed.empty:
        print_header("決済理由内訳")
        reasons = closed.groupby(["strategy", "exit_reason"])["pnl_yen"].agg(["count", "mean", "sum"])
        reasons.columns = ["件数", "平均損益", "合計損益"]
        for strat in sorted(closed["strategy"].unique()):
            r = reasons.loc[strat] if strat in reasons.index else None
            if r is None or r.empty:
                continue
            print(f"  戦略{strat}:")
            for reason, row in r.iterrows():
                print(f"    {reason:<10} {row['件数']:>3}件  "
                      f"平均{row['平均損益']:>+8,.0f}円  "
                      f"合計{row['合計損益']:>+10,.0f}円")

    # ── Phase 4 合格基準確認 ──────────────────────────────
    print_header("Phase 4 合格基準確認")
    total_pnl_all = closed["pnl_yen"].sum() if not closed.empty else 0.0
    if elapsed > 0:
        annualized = ((INITIAL_CAPITAL + total_pnl_all) / INITIAL_CAPITAL) ** (365 / elapsed) - 1
        annualized *= 100
    else:
        annualized = 0.0

    p4_ok_cagr = annualized >= 8.0
    # 月次損失10%超の発動チェック（簡易: 月次最大損失を確認）
    monthly_stop_fired = False
    if not closed.empty:
        monthly_pnl = closed.groupby(closed["exit_date"].dt.to_period("M"))["pnl_yen"].sum()
        monthly_stop_fired = (monthly_pnl < -INITIAL_CAPITAL * 0.10).any()

    print(f"  基準1: 模擬年利8%以上  → {annualized:+.1f}%  {'✅' if p4_ok_cagr else '⚠️  未達'}")
    print(f"  基準2: 月次損失10%以内  → {'⚠️  発動あり' if monthly_stop_fired else '✅ 発動なし'}")
    print(f"  基準3: 期間3ヶ月        → {elapsed}日経過 / {'✅ 完了' if today >= PHASE4_END else f'残{remaining}日'}")
    print()
    print(f"  累計損益: {total_pnl_all:+,.0f}円")
    print(f"  推定年利: {annualized:+.1f}%")

    if today >= PHASE4_END and p4_ok_cagr and not monthly_stop_fired:
        print("\n  🎉 Phase 4 合格基準達成！Phase 5 への移行を検討してください。")
        print("     docs/phase5_checklist.md を確認してください。")
    elif today >= PHASE4_END:
        print("\n  ⚠️  Phase 4 終了しましたが合格基準未達。Phase 4 延長を検討してください。")
    else:
        print(f"\n  Phase 4 実行中（残り{remaining}日）。引き続きデータを蓄積してください。")

    # ── シャープレシオ ────────────────────────────────────
    if not closed.empty and len(closed) >= 5:
        print_header("シャープレシオ（取引ベース）")
        pnls = closed.sort_values("exit_date")["pnl_yen"]
        mean_r = pnls.mean()
        std_r  = pnls.std()
        sharpe = (mean_r / std_r * np.sqrt(252)) if std_r > 0 else 0.0
        print(f"  平均損益/取引: {mean_r:+,.0f}円")
        print(f"  標準偏差    : {std_r:,.0f}円")
        print(f"  シャープレシオ（年換算）: {sharpe:.2f}")
        print(f"  ※ 取引ベース簡易計算。1以上が目標。")

    # ── シグナル頻度分析 ──────────────────────────────────
    if SCANNER_LOG.exists():
        print_header("シグナル頻度分析（scanner_log.csv）")
        try:
            sc = pd.read_csv(SCANNER_LOG, dtype=str)
            sc["Date"]   = pd.to_datetime(sc["Date"], errors="coerce")
            sc["Signal"] = sc["Signal"].astype(str).str.strip()
            sc = sc[sc["Date"] >= pd.Timestamp(PHASE4_START)]

            scan_days   = sc["Date"].nunique()
            total_sigs  = (sc["Signal"] == "True").sum()
            print(f"  スキャン実行回数: {scan_days}日")
            print(f"  総シグナル数    : {total_sigs}件")
            print(f"  1日平均シグナル : {total_sigs/scan_days:.1f}件" if scan_days > 0 else "")

            if "Strategy" in sc.columns:
                print(f"\n  戦略別シグナル（Signal=True）:")
                by_strat = sc[sc["Signal"] == "True"].groupby("Strategy")["Date"].count()
                for strat, n in by_strat.items():
                    print(f"    {strat}: {n}件 (うち採用率未追跡)")

        except Exception as e:
            print(f"  読み込みエラー: {e}")

    # ── エクイティカーブ（テキスト）────────────────────────
    if not closed.empty and len(closed) >= 3:
        print_header("エクイティカーブ（テキスト）")
        eq_series = closed.sort_values("exit_date").set_index("exit_date")["pnl_yen"].cumsum()
        eq_series = INITIAL_CAPITAL + eq_series
        # 月次でリサンプル
        eq_m = eq_series.resample("ME").last().dropna()
        if len(eq_m) > 0:
            peak_cap = eq_m.max()
            bar_width = 30
            print(f"  {'年月':<8} {'資産':>10}  グラフ（{INITIAL_CAPITAL/10000:.0f}万起点）")
            print(f"  {'-'*8} {'-'*10}  {'-'*bar_width}")
            for ym, cap in eq_m.items():
                diff = cap - INITIAL_CAPITAL
                n_bars = int((cap - INITIAL_CAPITAL * 0.95) /
                             max(peak_cap - INITIAL_CAPITAL * 0.95, 1) * bar_width)
                bar  = "█" * max(0, n_bars)
                mark = " ▲" if diff > 0 else " ▼"
                print(f"  {ym.strftime('%Y-%m'):<8} {cap:>10,.0f}  {bar}{mark}{diff:+,.0f}円")


if __name__ == "__main__":
    main()
