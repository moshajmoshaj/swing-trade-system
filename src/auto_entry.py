"""
src/auto_entry.py
Phase 4 ペーパートレード - エントリー自動記録
scanner.py 実行後に呼び出す。当日シグナル銘柄を paper_trade_log.xlsx に記録する。
"""
import sys
import os
sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd
from datetime import date, datetime
from pathlib import Path
from dotenv import load_dotenv
from openpyxl import load_workbook
import sys as _sys
_sys.path.insert(0, str(Path(__file__).parent))
from notifier import send_notify, send_error_notify

load_dotenv()

_SCRIPT      = "auto_entry.py"
SCANNER_LOG  = "logs/scanner_log.csv"
TRADE_LOG    = "logs/paper_trade_log.xlsx"
SCHED_LOG    = "logs/scheduler_log.txt"
MONTHLY_STOP = "logs/monthly_stop.txt"

MAX_POSITIONS = 5
MAX_PER_STOCK = 200_000


def log(msg: str) -> None:
    ts   = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] [entry] {msg}"
    print(line)
    os.makedirs("logs", exist_ok=True)
    with open(SCHED_LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def read_config(wb) -> dict:
    ws = wb["設定"]
    config = {}
    for row in ws.iter_rows(min_row=3, max_row=15, values_only=True):
        if row[0] and row[1] is not None:
            config[str(row[0])] = row[1]
    return config


def get_current_holdings(ws_trade) -> list[dict]:
    holdings = []
    for row in ws_trade.iter_rows(min_row=2):
        if row[0].value is None:
            continue
        if str(row[12].value).strip() == "保有中":
            holdings.append({"code": str(row[1].value).strip()})
    return holdings


def get_next_empty_row(ws_trade) -> int:
    for row in ws_trade.iter_rows(min_row=2, max_row=500):
        if row[0].value is None:
            return row[0].row
    return ws_trade.max_row + 1


def load_today_signals() -> pd.DataFrame:
    if not Path(SCANNER_LOG).exists():
        log(f"WARN: {SCANNER_LOG} が見つかりません")
        return pd.DataFrame()

    df = pd.read_csv(SCANNER_LOG, dtype={"Code": str})
    today_str = date.today().isoformat()
    df = df[df["Date"] == today_str]
    df = df.drop_duplicates(subset=["Code"], keep="last")
    df = df[df["Signal"].astype(str).str.strip() == "True"]
    return df


def main() -> None:
    log("auto_entry.py 開始")

    # 月次ストップチェック
    today_ym = date.today().strftime("%Y-%m")
    if Path(MONTHLY_STOP).exists():
        stop_ym = Path(MONTHLY_STOP).read_text(encoding="utf-8").strip()
        if stop_ym == today_ym:
            log(f"SKIP: 月次損失ストップ中 ({today_ym}) - 新規エントリーなし")
            send_notify("【ST】エントリー", f"月次損失ストップ中 - 新規エントリーなし ({today_ym})")
            return

    signals = load_today_signals()
    if signals.empty:
        log("INFO: 本日のシグナル銘柄なし")
        send_notify("【ST】エントリー", "本日のシグナルなし")
        return

    signals = signals.sort_values("RSI14", ascending=False).reset_index(drop=True)
    log(f"INFO: 本日のシグナル {len(signals)} 件")

    wb       = load_workbook(TRADE_LOG)
    ws_trade = wb["取引記録"]
    config   = read_config(wb)

    max_pos  = int(config.get("最大同時保有", MAX_POSITIONS))
    max_stk  = float(config.get("1銘柄上限", MAX_PER_STOCK))

    holdings       = get_current_holdings(ws_trade)
    available      = max_pos - len(holdings)
    held_codes     = {h["code"] for h in holdings}
    log(f"INFO: 現保有 {len(holdings)} 銘柄 / 上限 {max_pos} / 空き {available} 枠")

    if available <= 0:
        log("INFO: 保有上限に達しているためエントリーなし")
        wb.close()
        return

    today        = date.today()
    next_row     = get_next_empty_row(ws_trade)
    entered      = 0
    entries_info = []   # (code, rsi) for LINE notification

    for _, sig in signals.iterrows():
        if entered >= available:
            break

        code        = str(sig["Code"]).zfill(4)
        entry_price = float(sig["Close"])
        stop_loss   = float(sig["StopLoss"])
        take_profit = float(sig["TakeProfit"])
        atr  = float(sig["ATR14"]) if pd.notna(sig.get("ATR14"))  else None
        rsi  = float(sig["RSI14"]) if pd.notna(sig.get("RSI14"))  else None
        adx  = float(sig["ADX14"]) if pd.notna(sig.get("ADX14"))  else None

        if code in held_codes:
            log(f"SKIP: {code} は既に保有中")
            continue

        # 1銘柄上限: 1株でも購入できるか確認
        if entry_price > max_stk:
            log(f"SKIP: {code} 株価 {entry_price:,.0f}円 > 上限 {max_stk:,.0f}円")
            continue

        ws_trade.cell(row=next_row, column=1,  value=today)
        ws_trade.cell(row=next_row, column=2,  value=code)
        ws_trade.cell(row=next_row, column=3,  value=entry_price)
        ws_trade.cell(row=next_row, column=4,  value=stop_loss)
        ws_trade.cell(row=next_row, column=5,  value=take_profit)
        ws_trade.cell(row=next_row, column=6,  value=atr)
        ws_trade.cell(row=next_row, column=7,  value=rsi)
        ws_trade.cell(row=next_row, column=8,  value=adx)
        ws_trade.cell(row=next_row, column=9,  value=None)
        ws_trade.cell(row=next_row, column=10, value=None)
        ws_trade.cell(row=next_row, column=11, value=None)
        ws_trade.cell(row=next_row, column=12, value=None)
        ws_trade.cell(row=next_row, column=13, value="保有中")

        shares = max(1, int(max_stk // entry_price))
        log(f"ENTRY: {code} @ {entry_price:,.0f}円  株数={shares}  "
            f"SL={stop_loss:,.0f}  TP={take_profit:,.0f}  "
            f"RSI={rsi:.1f}  ADX={adx:.1f}")

        entries_info.append((code, rsi))
        held_codes.add(code)
        next_row += 1
        entered  += 1

    wb.save(TRADE_LOG)
    log(f"INFO: {entered} 件エントリー記録 → {TRADE_LOG}")

    # Gmail 通知（エントリー結果サマリー）
    total_held = len(holdings) + entered
    body_lines = [f"新規：{entered}件"]
    for code, rsi in entries_info:
        rsi_str = f"{rsi:.1f}" if rsi is not None else "N/A"
        body_lines.append(f"　{code} RSI {rsi_str}")
    body_lines.append(f"保有中：{total_held}銘柄")
    send_notify("【ST】エントリー記録完了", "\n".join(body_lines))


if __name__ == "__main__":
    try:
        main()
    except PermissionError as e:
        log(f"ERROR: PermissionError: {e}")
        send_error_notify(_SCRIPT, "PermissionError", str(e))
        sys.exit(1)
    except FileNotFoundError as e:
        log(f"ERROR: FileNotFoundError: {e}")
        send_error_notify(_SCRIPT, "FileNotFoundError", str(e))
        sys.exit(1)
    except ValueError as e:
        log(f"ERROR: ValueError: {e}")
        send_error_notify(_SCRIPT, "ValueError", str(e))
        sys.exit(1)
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        log(f"ERROR: {type(e).__name__}: {e}\n{tb}")
        send_error_notify(_SCRIPT, type(e).__name__, str(e))
        sys.exit(1)
