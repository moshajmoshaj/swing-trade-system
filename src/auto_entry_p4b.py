"""
src/auto_entry_p4b.py
Phase 4-2 エントリー自動記録（A v2 + G）
ログ先: logs/paper_trade_log_p4b.csv
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

SCANNER_LOG  = "logs/scanner_log_p4b.csv"
TRADE_LOG    = "logs/paper_trade_log_p4b.csv"
SCHED_LOG    = "logs/scheduler_log_p4b.txt"
MONTHLY_STOP = "logs/monthly_stop_p4b.txt"

MAX_POSITIONS = 5
MAX_PER_STOCK = 200_000
INITIAL_CAP   = 1_000_000


def log(msg):
    ts   = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] [p4b-entry] {msg}"
    print(line)
    os.makedirs("logs", exist_ok=True)
    try:
        with open(SCHED_LOG, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except PermissionError:
        pass


COLS = ["TradeID","Code","Strategy","EntryDate","EntryPrice","Shares",
        "SL","TP","MaxHold","ExitDate","ExitPrice","ExitReason","PnL","Status"]


def load_trade_log():
    if Path(TRADE_LOG).exists():
        return pd.read_csv(TRADE_LOG, dtype={"Code": str})
    return pd.DataFrame(columns=COLS)


def save_trade_log(df):
    df.to_csv(TRADE_LOG, index=False, encoding="utf-8-sig")


def is_monthly_stopped():
    if not Path(MONTHLY_STOP).exists():
        return False
    with open(MONTHLY_STOP) as f:
        return f.read().strip() == datetime.now().strftime("%Y-%m")


def main():
    log("auto_entry_p4b.py 開始")
    try:
        if is_monthly_stopped():
            log("INFO: 月次ストップ中 → エントリーしない")
            return

        # スキャンログ読み込み
        if not Path(SCANNER_LOG).exists():
            log("INFO: scanner_log_p4b.csv なし → スキップ")
            return

        today_str = datetime.now().strftime("%Y-%m-%d")
        try:
            scans = pd.read_csv(SCANNER_LOG, dtype={"Code": str})
        except Exception:
            log("INFO: 本日のシグナル銘柄なし（スキャンログ空）")
            return
        today_sigs = scans[scans["Date"] == today_str].sort_values("RSI", ascending=False)

        if today_sigs.empty:
            log("INFO: 本日のシグナル銘柄なし")
            return

        log(f"INFO: 本日のシグナル {len(today_sigs)} 件")

        # 現保有読み込み
        trades = load_trade_log()
        open_trades = trades[trades["Status"] == "OPEN"] if not trades.empty else pd.DataFrame()
        holding     = set(open_trades["Code"].tolist()) if not open_trades.empty else set()
        n_open      = len(open_trades)
        slots       = MAX_POSITIONS - n_open

        log(f"INFO: 現保有 {n_open} 銘柄 / 上限 {MAX_POSITIONS} / 空き {slots} 枠")

        if slots <= 0:
            log("INFO: 空き枠なし → エントリーしない")
            return

        # 現在の資産評価（簡易：初期資金 + 累計損益）
        closed = trades[trades["Status"] == "CLOSED"] if not trades.empty else pd.DataFrame()
        realized_pnl = closed["PnL"].sum() if not closed.empty else 0
        capital = INITIAL_CAP + realized_pnl

        entered = 0
        new_id  = int(trades["TradeID"].max()) + 1 if not trades.empty and "TradeID" in trades.columns else 1

        for _, sig in today_sigs.iterrows():
            if slots <= 0:
                break
            code = str(sig["Code"]).zfill(4)
            if code in holding:
                log(f"SKIP: {code} は既に保有中")
                continue

            price  = float(sig["Close"])
            shares = min(int(MAX_PER_STOCK / price), int(capital * 0.20 / price))
            shares = max(shares, 0)
            if shares == 0 or price * shares > capital:
                log(f"SKIP: {code} 株数ゼロまたは資金不足")
                continue

            row = {
                "TradeID"   : new_id,
                "Code"      : code,
                "Strategy"  : sig["Strategy"],
                "EntryDate" : today_str,
                "EntryPrice": price,
                "Shares"    : shares,
                "SL"        : float(sig["SL"]),
                "TP"        : float(sig["TP"]),
                "MaxHold"   : int(sig["MaxHold"]),
                "ExitDate"  : "",
                "ExitPrice" : "",
                "ExitReason": "",
                "PnL"       : "",
                "Status"    : "OPEN",
            }
            trades = pd.concat([trades, pd.DataFrame([row])], ignore_index=True)
            holding.add(code)
            slots   -= 1
            new_id  += 1
            entered += 1
            log(f"ENTRY: [{sig['Strategy']}] {code} @ {price:,.0f}円  "
                f"株数={shares}  SL={sig['SL']:,.0f}  TP={sig['TP']:,.0f}  "
                f"RSI={sig['RSI']}  MaxHold={int(sig['MaxHold'])}日")

        save_trade_log(trades)
        log(f"INFO: {entered} 件エントリー記録 → {TRADE_LOG}")

        if entered > 0:
            send_notify("p4b-entry", f"[P4-2 エントリー] {entered}件\n" +
                        "\n".join(f"{r['Strategy']} {r['Code']} @{r['EntryPrice']:,.0f}"
                                  for _, r in trades.tail(entered).iterrows()))

    except Exception as e:
        import traceback
        log(f"ERROR: {e}\n{traceback.format_exc()}")
        send_error_notify("auto_entry_p4b.py", str(e), traceback.format_exc())


if __name__ == "__main__":
    main()
