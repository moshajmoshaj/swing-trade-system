"""
src/auto_exit_p4b.py
Phase 4-2 決済自動判定（A=10日 / G=15日）
ログ先: logs/paper_trade_log_p4b.csv
"""
import sys
import os
sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from dotenv import load_dotenv
import jpholiday
import jquantsapi

sys.path.insert(0, str(Path(__file__).parent))
from notifier import send_notify, send_error_notify, call_with_retry

load_dotenv()

TRADE_LOG    = "logs/paper_trade_log_p4b.csv"
SCHED_LOG    = "logs/scheduler_log_p4b.txt"
MONTHLY_STOP = "logs/monthly_stop_p4b.txt"
INITIAL_CAP  = 1_000_000

FORCED_EXIT_DAYS = {"A": 10, "G": 15}
COST_RATE = 0.00105   # 片道0.105%（手数料+スリッページ）


def log(msg):
    ts   = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] [p4b-exit] {msg}"
    print(line)
    os.makedirs("logs", exist_ok=True)
    try:
        with open(SCHED_LOG, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except PermissionError:
        pass


def bizdays_between(start_str: str, end_date) -> int:
    start = datetime.strptime(start_str, "%Y-%m-%d").date()
    count = 0
    d = start
    while d < end_date:
        d += timedelta(days=1)
        if d.weekday() < 5 and not jpholiday.is_holiday(d):
            count += 1
    return count


def main():
    log("auto_exit_p4b.py 開始")
    try:
        api_key = os.getenv("JQUANTS_REFRESH_TOKEN")
        client  = jquantsapi.ClientV2(api_key=api_key)
        today   = datetime.now().date()

        if not Path(TRADE_LOG).exists():
            log("INFO: 保有中ポジションなし")
            return

        trades = pd.read_csv(TRADE_LOG, dtype={"Code": str})
        open_t = trades[trades["Status"] == "OPEN"].copy()

        if open_t.empty:
            log("INFO: 保有中ポジションなし")
            return

        log(f"INFO: 保有中 {len(open_t)} 銘柄の決済判定を開始")

        # 価格取得
        codes = (open_t["Code"].str.zfill(4) + "0").tolist()
        start_str = (today - timedelta(days=5)).strftime("%Y-%m-%d")
        end_str   = today.strftime("%Y-%m-%d")
        try:
            prices = call_with_retry(
                client.get_price_range,
                start_date=start_str, end_date=end_str, codes=codes
            )
            prices["Date"] = pd.to_datetime(prices["Date"])
        except Exception as e:
            log(f"ERROR: 価格取得失敗 ({e})")
            prices = pd.DataFrame()

        exited = 0
        for idx, row in open_t.iterrows():
            code5  = row["Code"].zfill(4) + "0"
            strat  = str(row["Strategy"])
            held   = bizdays_between(str(row["EntryDate"]), today)
            max_h  = FORCED_EXIT_DAYS.get(strat, 10)

            ep     = float(row["EntryPrice"])
            sl     = float(row["SL"])
            tp     = float(row["TP"])
            shares = int(row["Shares"])

            if not prices.empty:
                today_px = prices[(prices["Code"] == code5) &
                                   (prices["Date"].dt.date == today)]
            else:
                today_px = pd.DataFrame()

            if today_px.empty:
                log(f"WARN: {row['Code']} 当日価格データ未取得")
                continue

            hi = float(today_px["High"].iloc[-1])
            lo = float(today_px["Low"].iloc[-1])
            cl = float(today_px["AdjClose"].iloc[-1])

            exit_price = exit_reason = None
            if lo <= sl:
                exit_price, exit_reason = sl, "損切り"
            elif hi >= tp:
                exit_price, exit_reason = tp, "利確"
            elif held >= max_h:
                exit_price, exit_reason = cl, "期間満了"

            if exit_price is None:
                log(f"HOLD: [{strat}] {row['Code']} 保有{held}/{max_h}日  "
                    f"H={hi:,.0f}  L={lo:,.0f}  SL={sl:,.0f}  TP={tp:,.0f}")
                continue

            cost = (ep + exit_price) * shares * COST_RATE
            pnl  = (exit_price - ep) * shares - cost
            trades.loc[idx, "ExitDate"]   = today.strftime("%Y-%m-%d")
            trades.loc[idx, "ExitPrice"]  = exit_price
            trades.loc[idx, "ExitReason"] = exit_reason
            trades.loc[idx, "PnL"]        = round(pnl, 0)
            trades.loc[idx, "Status"]     = "CLOSED"
            exited += 1
            log(f"EXIT: [{strat}] {row['Code']}  理由={exit_reason}  "
                f"決済={exit_price:,.0f}  損益={pnl:+,.0f}円 ({pnl/ep/shares*100:+.2f}%)  保有{held}日")

        trades.to_csv(TRADE_LOG, index=False, encoding="utf-8-sig")
        log(f"INFO: {exited} 件決済記録 → {TRADE_LOG}")

        # 月次損益集計
        closed = trades[trades["Status"] == "CLOSED"].copy()
        closed["PnL"] = pd.to_numeric(closed["PnL"], errors="coerce").fillna(0)
        this_month = today.strftime("%Y-%m")
        month_pnl  = closed[closed["ExitDate"].str.startswith(this_month)]["PnL"].sum()
        stop_line  = -INITIAL_CAP * 0.10
        log(f"INFO: 当月損益合計 {month_pnl:+,.0f}円  ストップライン {stop_line:,.0f}円")

        # 月次ストップ判定
        if month_pnl <= stop_line:
            with open(MONTHLY_STOP, "w") as f:
                f.write(this_month)
            log(f"WARN: 月次ストップ発動！ {month_pnl:,.0f}円 ≤ {stop_line:,.0f}円")
            send_notify("p4b-exit", f"[P4-2 月次ストップ発動] {month_pnl:+,.0f}円")

        if exited > 0:
            send_notify("p4b-exit", f"[P4-2 決済] {exited}件  当月損益{month_pnl:+,.0f}円")

    except Exception as e:
        import traceback
        log(f"ERROR: {e}\n{traceback.format_exc()}")
        send_error_notify("auto_exit_p4b.py", str(e), traceback.format_exc())


if __name__ == "__main__":
    main()
