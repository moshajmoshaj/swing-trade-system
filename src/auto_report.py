"""
src/auto_report.py
Phase 4 ペーパートレード - 月次集計自動更新
実行順序: 4番目（最後）
"""
import sys
import os
sys.stdout.reconfigure(encoding="utf-8")

from datetime import datetime
from pathlib import Path
import pandas as pd
from openpyxl import load_workbook
import sys as _sys
_sys.path.insert(0, str(Path(__file__).parent))
from notifier import send_notify, send_error_notify

_SCRIPT   = "auto_report.py"
TRADE_LOG = "logs/paper_trade_log.xlsx"
SCHED_LOG = "logs/scheduler_log.txt"


def log(msg: str) -> None:
    ts   = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] [report] {msg}"
    print(line)
    os.makedirs("logs", exist_ok=True)
    try:
        with open(SCHED_LOG, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except PermissionError:
        pass


def read_config(wb) -> dict:
    ws = wb["設定"]
    config = {}
    for row in ws.iter_rows(min_row=3, max_row=15, values_only=True):
        if row[0] and row[1] is not None:
            config[str(row[0])] = row[1]
    return config


def load_completed_trades(ws_trade) -> pd.DataFrame:
    """決済済み（保有中以外）の全取引をDataFrameで返す"""
    rows = []
    cols = ["日付", "銘柄コード", "エントリー価格", "損切り価格", "利確価格",
            "ATR", "RSI", "ADX", "決済日", "決済価格", "損益円", "損益率", "決済理由"]

    for row in ws_trade.iter_rows(min_row=2, values_only=True):
        if row[0] is None:
            continue
        reason = str(row[12]).strip() if row[12] is not None else ""
        if reason == "保有中" or reason == "":
            continue
        rows.append(row[:13])

    if not rows:
        return pd.DataFrame(columns=cols)

    df = pd.DataFrame(rows, columns=cols)
    df["決済日"]     = pd.to_datetime(df["決済日"], errors="coerce")
    df["損益円"]     = pd.to_numeric(df["損益円"],     errors="coerce")
    df["損益率"]     = pd.to_numeric(df["損益率"],     errors="coerce")
    df["エントリー価格"] = pd.to_numeric(df["エントリー価格"], errors="coerce")
    return df.dropna(subset=["決済日", "損益円"])


def build_monthly_summary(df: pd.DataFrame, capital: float) -> pd.DataFrame:
    df["年月"] = df["決済日"].dt.to_period("M")

    monthly = (
        df.groupby("年月")
        .agg(
            取引数      = ("損益円", "count"),
            勝率        = ("損益円", lambda x: (x > 0).mean()),
            月次損益円  = ("損益円", "sum"),
        )
        .reset_index()
    )

    monthly["月次損益率"]       = monthly["月次損益円"] / capital
    monthly["月次ストップ発動"] = (monthly["月次損益円"] < -capital * 0.10).astype(int)
    monthly["年月"]             = monthly["年月"].astype(str)
    return monthly


def update_summary_sheet(ws_summary, monthly: pd.DataFrame) -> None:
    # データ行をクリア（ヘッダー行1は残す）
    for row in ws_summary.iter_rows(min_row=2, max_row=ws_summary.max_row):
        for cell in row:
            cell.value = None

    for i, r in enumerate(monthly.itertuples(index=False), start=2):
        ws_summary.cell(row=i, column=1, value=r.年月)
        ws_summary.cell(row=i, column=2, value=int(r.取引数))
        ws_summary.cell(row=i, column=3, value=round(float(r.勝率), 4))
        ws_summary.cell(row=i, column=4, value=round(float(r.月次損益円), 0))
        ws_summary.cell(row=i, column=5, value=round(float(r.月次損益率), 6))
        ws_summary.cell(row=i, column=6, value=int(r.月次ストップ発動))


def main() -> None:
    log("auto_report.py 開始")

    if not Path(TRADE_LOG).exists():
        log(f"INFO: {TRADE_LOG} が見つかりません - スキップ")
        return

    wb          = load_workbook(TRADE_LOG)
    ws_trade    = wb["取引記録"]
    ws_summary  = wb["月次集計"]
    config      = read_config(wb)
    capital     = float(config.get("運用資金", 1_000_000))

    df = load_completed_trades(ws_trade)

    # 保有中銘柄数
    holding_count = sum(
        1 for row in ws_trade.iter_rows(min_row=2, values_only=True)
        if row[0] is not None and str(row[12]).strip() == "保有中"
    )

    # 日次レポート（完了取引ゼロでも常に送信）
    total_pnl   = float(df["損益円"].sum()) if not df.empty else 0.0
    current     = capital + total_pnl
    target_pnl  = capital * 0.02
    target_asset = capital + target_pnl
    achievement = (total_pnl / target_pnl * 100) if target_pnl != 0 else 0.0
    daily_body  = (
        f"保有中：{holding_count}銘柄\n"
        f"初期資金：{capital:,.0f}円\n"
        f"累計確定損益：{total_pnl:+,.0f}円\n"
        f"現在資産：{current:,.0f}円\n"
        f"目標（年利8%・3ヶ月）：{target_asset:,.0f}円\n"
        f"達成率：{achievement:.0f}%"
    )
    send_notify("【ST】日次レポート", daily_body)
    log(f"INFO: 日次レポート送信  累計損益 {total_pnl:+,.0f}円  達成率 {achievement:.0f}%")

    if df.empty:
        log("INFO: 集計対象の完了取引なし")
        wb.close()
        return

    monthly = build_monthly_summary(df, capital)
    update_summary_sheet(ws_summary, monthly)
    wb.save(TRADE_LOG)

    # サマリーログ
    total_trades = int(monthly["取引数"].sum())
    log(f"INFO: 集計完了  {len(monthly)} ヶ月 / 計{total_trades}取引 / "
        f"累計損益 {total_pnl:+,.0f}円 → {TRADE_LOG}")

    for _, row in monthly.iterrows():
        stop = "★ストップ発動" if row["月次ストップ発動"] else ""
        log(f"  {row['年月']}  {int(row['取引数'])}件  "
            f"勝率{row['勝率']:.1%}  "
            f"{row['月次損益円']:+,.0f}円 ({row['月次損益率']:+.2%})  {stop}")


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
    finally:
        os.makedirs("logs", exist_ok=True)
        try:
            with open(SCHED_LOG, "a", encoding="utf-8") as f:
                f.write("========================================\n")
        except PermissionError:
            pass
