"""
src/auto_report_p4b.py
Phase 4-2 日次レポート通知
"""
import sys
import os
sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent))
from notifier import send_notify, send_error_notify

load_dotenv()

TRADE_LOG  = "logs/paper_trade_log_p4b.csv"
SCHED_LOG  = "logs/scheduler_log_p4b.txt"
INITIAL_CAP = 1_000_000
START_DATE  = "2026-05-07"   # Phase 4-2 開始日


def log(msg):
    ts   = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] [p4b-report] {msg}"
    print(line)
    os.makedirs("logs", exist_ok=True)
    try:
        with open(SCHED_LOG, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except PermissionError:
        pass


def main():
    log("auto_report_p4b.py 開始")
    try:
        if not Path(TRADE_LOG).exists():
            log("INFO: 取引記録なし")
            send_notify("p4b-report", "[P4-2] 取引記録なし（まだエントリーなし）")
            return

        trades = pd.read_csv(TRADE_LOG, dtype={"Code": str})
        closed = trades[trades["Status"] == "CLOSED"].copy()
        open_t = trades[trades["Status"] == "OPEN"].copy()

        closed["PnL"] = pd.to_numeric(closed["PnL"], errors="coerce").fillna(0)

        total_pnl  = closed["PnL"].sum()
        total_cnt  = len(closed)
        wins       = (closed["PnL"] > 0).sum() if total_cnt > 0 else 0
        win_rate   = wins / total_cnt * 100 if total_cnt > 0 else 0

        # 年利換算（経過日数ベース）
        from datetime import date
        start = datetime.strptime(START_DATE, "%Y-%m-%d").date()
        days  = max((date.today() - start).days, 1)
        annual = (total_pnl / INITIAL_CAP) / days * 365 * 100

        # 戦略別損益
        by_strat = {}
        for strat in ["A", "G"]:
            s = closed[closed["Strategy"] == strat]
            by_strat[strat] = (len(s), s["PnL"].sum() if len(s) > 0 else 0)

        # 当月損益
        today     = date.today()
        this_month = today.strftime("%Y-%m")
        month_pnl  = closed[closed["ExitDate"].str.startswith(this_month)]["PnL"].sum() \
                     if not closed.empty else 0

        msg = (
            f"[P4-2 日次レポート] {today}\n"
            f"累計損益: {total_pnl:+,.0f}円\n"
            f"年利推計: {annual:+.1f}%（目標13.2%）\n"
            f"取引数: {total_cnt}回  勝率: {win_rate:.1f}%\n"
            f"当月損益: {month_pnl:+,.0f}円\n"
            f"現保有: {len(open_t)}銘柄\n"
            f"A: {by_strat['A'][0]}回 {by_strat['A'][1]:+,.0f}円  "
            f"G: {by_strat['G'][0]}回 {by_strat['G'][1]:+,.0f}円"
        )
        log(f"INFO: {msg.replace(chr(10), ' ')}")
        send_notify("p4b-report", msg)

        # 月別集計
        if not closed.empty:
            closed["Month"] = closed["ExitDate"].str[:7]
            for month, grp in closed.groupby("Month"):
                cnt = len(grp); w = (grp["PnL"] > 0).sum()
                log(f"  {month}  {cnt}件  勝率{w/cnt*100:.1f}%  {grp['PnL'].sum():+,.0f}円")

    except Exception as e:
        import traceback
        log(f"ERROR: {e}\n{traceback.format_exc()}")
        send_error_notify("auto_report_p4b.py", str(e), traceback.format_exc())


if __name__ == "__main__":
    main()
