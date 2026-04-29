"""
src/scanner.py
Phase 4 ペーパートレード用シグナルスキャナー
毎営業日 15:30 に手動実行する

実行方法:
    python src/scanner.py

出力:
    - コンソール: シグナル銘柄一覧
    - logs/scanner_log.csv: 実行履歴
"""

import sys
import os
sys.stdout.reconfigure(encoding="utf-8")

import time
import pandas as pd
import pandas_ta as ta
from datetime import datetime, timedelta
from pathlib import Path
from dotenv import load_dotenv
import jquantsapi
import sys as _sys
_sys.path.insert(0, str(Path(__file__).parent))
from notifier import send_notify, send_error_notify, call_with_retry

load_dotenv()

# ──────────────────────────────────────────
# 設定
# ──────────────────────────────────────────
_SCRIPT          = "scanner.py"
CANDIDATES_CSV   = "logs/final_candidates.csv"          # 戦略A: 30銘柄
CANDIDATES_C_CSV = "logs/strategy_c_candidates.csv"     # 戦略C: 35銘柄
CANDIDATES_D_CSV = "logs/strategy_d_candidates.csv"     # 戦略D: 144銘柄
CANDIDATES_E_CSV = "logs/strategy_e_candidates.csv"     # 戦略E: 30銘柄
SCANNER_LOG_CSV  = "logs/scanner_log.csv"               # スキャン履歴
SCHED_LOG        = "logs/scheduler_log.txt"             # スケジューラーログ
DAYS_TO_FETCH    = 250                                  # 指標計算に必要な日数（SMA200対応）

# 戦略Aパラメータ（設計書準拠）
SMA_SHORT        = 20
SMA_LONG         = 50
SMA_TREND        = 200
RSI_PERIOD       = 14
RSI_MIN          = 45
RSI_MAX          = 75
RSI_LOOKBACK     = 3       # RSI上昇確認日数
ATR_PERIOD       = 14
ADX_PERIOD       = 14
ADX_MIN          = 15
VOL_MA_PERIOD    = 20
VOL_SURGE_RATIO  = 1.2
GAP_DOWN_THRESH  = -0.015  # -1.5%
EARNINGS_WINDOW  = 3       # 決算前後営業日

# 戦略Cパラメータ（平均回帰・逆張り）
RSI_C_MIN        = 30
RSI_C_MAX        = 40
VOL_C_RATIO      = 1.5
ATR_C_TP         = 1.5
ATR_C_SL         = 1.5
MAX_HOLD_C       = 7

# 戦略Dパラメータ（ギャップアップ翌日）
GAP_D_MIN        = 0.03    # +3%以上
VOL_D_RATIO      = 1.5
RSI_D_MIN        = 50
ATR_D_TP         = 3.0
ATR_D_SL         = 1.5
MAX_HOLD_D       = 5

# 戦略Eパラメータ（52週高値ブレイクアウト）
RSI_E_MIN        = 50
RSI_E_MAX        = 80
VOL_E_RATIO      = 1.2
ATR_E_TP         = 6.0
ATR_E_SL         = 2.0
MAX_HOLD_E       = 10


def log(msg: str) -> None:
    ts   = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] [scanner] {msg}"
    print(line)
    os.makedirs("logs", exist_ok=True)
    with open(SCHED_LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


# ──────────────────────────────────────────
# J-Quants クライアント取得
# ──────────────────────────────────────────
def get_client() -> jquantsapi.ClientV2:
    token = os.getenv("JQUANTS_REFRESH_TOKEN")
    if not token:
        raise EnvironmentError(".env に JQUANTS_REFRESH_TOKEN が設定されていません")
    return jquantsapi.ClientV2(api_key=token)


# ──────────────────────────────────────────
# 対象銘柄コード読み込み
# ──────────────────────────────────────────
def load_candidates(path: str = CANDIDATES_CSV) -> list[str]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} が見つかりません")
    df = pd.read_csv(path, dtype=str)
    col = next((c for c in df.columns if "code" in c.lower()), df.columns[0])
    codes = df[col].str.zfill(4).tolist()
    return codes


# ──────────────────────────────────────────
# 株価データ取得
# ──────────────────────────────────────────
def fetch_prices(client: jquantsapi.ClientV2, codes: list[str]) -> pd.DataFrame:
    end_date   = datetime.today()
    start_date = end_date - timedelta(days=DAYS_TO_FETCH * 1.5)  # 営業日換算バッファ

    print(f"株価データ取得中... ({start_date.strftime('%Y-%m-%d')} ～ {end_date.strftime('%Y-%m-%d')})")

    col_rename = {"O": "Open", "H": "High", "L": "Low", "C": "Close", "Vo": "Volume"}

    dfs = []
    for i, code in enumerate(codes):
        try:
            df = client.get_eq_bars_daily(
                code=code,
                from_yyyymmdd=start_date.strftime("%Y%m%d"),
                to_yyyymmdd=end_date.strftime("%Y%m%d"),
            )
            if df is not None and not df.empty:
                df = df.rename(columns=col_rename)
                df["Code"] = code
                dfs.append(df)
        except Exception as e:
            print(f"  警告: {code} の取得失敗 ({e})")
        # APIレート制限対策（120req/分 → 0.52秒間隔で安全マージン確保）
        if (i + 1) % 10 == 0:
            time.sleep(0.5)

    if not dfs:
        raise ConnectionError("全銘柄の株価データ取得に失敗しました")

    all_df = pd.concat(dfs, ignore_index=True)
    all_df["Date"] = pd.to_datetime(all_df["Date"])
    all_df = all_df.sort_values(["Code", "Date"]).reset_index(drop=True)
    print(f"取得完了: {len(all_df)} 件")
    return all_df


# ──────────────────────────────────────────
# テクニカル指標計算
# ──────────────────────────────────────────
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["SMA20"]    = ta.sma(df["Close"], length=SMA_SHORT)
    df["SMA50"]    = ta.sma(df["Close"], length=SMA_LONG)
    df["SMA200"]   = ta.sma(df["Close"], length=SMA_TREND)
    df["RSI14"]    = ta.rsi(df["Close"], length=RSI_PERIOD)
    df["ATR14"]    = ta.atr(df["High"], df["Low"], df["Close"], length=ATR_PERIOD)
    df["VOL_MA20"] = ta.sma(df["Volume"], length=VOL_MA_PERIOD)

    # ADX
    adx_df = ta.adx(df["High"], df["Low"], df["Close"], length=ADX_PERIOD)
    if adx_df is not None and f"ADX_{ADX_PERIOD}" in adx_df.columns:
        df["ADX14"] = adx_df[f"ADX_{ADX_PERIOD}"]
    else:
        df["ADX14"] = None

    # 前日終値（ギャップダウン計算用）
    df["PrevClose"] = df["Close"].shift(1)
    return df


# ──────────────────────────────────────────
# 決算日データ取得（簡易版：APIから取得）
# ──────────────────────────────────────────
def fetch_earnings_dates(client: jquantsapi.ClientV2, codes: list[str]) -> dict[str, list]:
    """決算発表日を取得してコードごとに返す"""
    earnings = {}
    try:
        df = client.get_eq_earnings_cal()  # 決算発表予定日カレンダー
        if df is not None and not df.empty:
            date_col = next((c for c in df.columns if "date" in c.lower()), None)
            code_col = next((c for c in df.columns if "code" in c.lower()), None)
            if date_col and code_col:
                df[date_col] = pd.to_datetime(df[date_col])
                for code in codes:
                    dates = df[df[code_col] == code][date_col].tolist()
                    earnings[code] = dates
    except Exception as e:
        print(f"  警告: 決算日データ取得失敗 ({e}) → 決算除外フィルターをスキップ")
    return earnings


# ──────────────────────────────────────────
# 戦略Aシグナル判定（設計書全9条件）
# ──────────────────────────────────────────
def check_signal(row: pd.Series, today: pd.Timestamp, earnings_dates: list) -> tuple[bool, list[str]]:
    """
    Returns:
        (signal: bool, failed_conditions: list[str])
    """
    failed = []

    # 条件1: SMA20 > SMA50（上昇トレンド）
    if not (pd.notna(row["SMA20"]) and pd.notna(row["SMA50"]) and row["SMA20"] > row["SMA50"]):
        failed.append("SMA20>SMA50")

    # 条件2: RSI 45〜75
    if not (pd.notna(row["RSI14"]) and RSI_MIN <= row["RSI14"] <= RSI_MAX):
        failed.append(f"RSI({row['RSI14']:.1f})" if pd.notna(row["RSI14"]) else "RSI(NaN)")

    # 条件3: 出来高フィルター（20日平均の1.2倍以上）
    if not (pd.notna(row["VOL_MA20"]) and row["Volume"] >= row["VOL_MA20"] * VOL_SURGE_RATIO):
        failed.append("出来高")

    # 条件4: 終値 > SMA200（長期トレンド）
    if not (pd.notna(row["SMA200"]) and row["Close"] > row["SMA200"]):
        failed.append("SMA200")

    # 条件5: RSI14が3日前より上昇中（モメンタム）
    if not (pd.notna(row.get("RSI14_lag3")) and row["RSI14"] > row["RSI14_lag3"]):
        failed.append("RSI上昇")

    # 条件6: ギャップダウン除外（前日比-1.5%以上の窓開け下落を除外）
    if pd.notna(row["PrevClose"]) and row["PrevClose"] > 0:
        gap = (row["Open"] - row["PrevClose"]) / row["PrevClose"]
        if gap <= GAP_DOWN_THRESH:
            failed.append(f"ギャップダウン({gap*100:.1f}%)")

    # 条件7: 陽線（終値 > 始値）
    if not (row["Close"] > row["Open"]):
        failed.append("陰線")

    # 条件8: ADX > 15
    if not (pd.notna(row.get("ADX14")) and row["ADX14"] > ADX_MIN):
        failed.append(f"ADX({row['ADX14']:.1f})" if pd.notna(row.get("ADX14")) else "ADX(NaN)")

    # 条件9: 決算除外（前後3営業日）
    if earnings_dates:
        for edate in earnings_dates:
            delta = abs((today - edate).days)
            if delta <= EARNINGS_WINDOW * 1.5:  # 営業日換算バッファ
                failed.append(f"決算近接({edate.strftime('%m/%d')})")
                break

    return len(failed) == 0, failed


# ──────────────────────────────────────────
# 戦略Cシグナル判定（平均回帰・逆張り）
# ──────────────────────────────────────────
def check_signal_c(df: pd.DataFrame) -> tuple[bool, list[str]]:
    row = df.iloc[-1]
    failed = []

    # 条件1: 終値 > SMA200（長期上昇トレンド内）
    if not (pd.notna(row["SMA200"]) and row["Close"] > row["SMA200"]):
        failed.append("SMA200")

    # 条件2: RSI 30〜40（短期押し目）
    if not (pd.notna(row["RSI14"]) and RSI_C_MIN <= row["RSI14"] <= RSI_C_MAX):
        failed.append(f"RSI({row['RSI14']:.1f})" if pd.notna(row["RSI14"]) else "RSI(NaN)")

    # 条件3: 出来高スパイク（20日平均の1.5倍以上）
    if not (pd.notna(row["VOL_MA20"]) and row["Volume"] >= row["VOL_MA20"] * VOL_C_RATIO):
        failed.append("出来高")

    atr   = row["ATR14"] if pd.notna(row["ATR14"]) else None
    stop  = round(row["Close"] - atr * ATR_C_SL, 1) if atr else None
    tp    = round(row["Close"] + atr * ATR_C_TP, 1) if atr else None
    return len(failed) == 0, failed, stop, tp


# ──────────────────────────────────────────
# 戦略Dシグナル判定（ギャップアップ翌日）
# ──────────────────────────────────────────
def check_signal_d(df: pd.DataFrame) -> tuple[bool, list[str]]:
    if len(df) < 3:
        return False, ["データ不足"], None, None

    today     = df.iloc[-1]
    yesterday = df.iloc[-2]
    failed    = []

    # 条件1: 前日ギャップアップ（前日比+3%以上）
    prev_close = yesterday["PrevClose"]
    if pd.notna(prev_close) and prev_close > 0:
        gap = (yesterday["Open"] - prev_close) / prev_close
        if gap < GAP_D_MIN:
            failed.append(f"ギャップ不足({gap*100:.1f}%)")
    else:
        failed.append("前日終値なし")

    # 条件2: 当日出来高 ≥ 20日平均×1.5
    if not (pd.notna(today["VOL_MA20"]) and today["Volume"] >= today["VOL_MA20"] * VOL_D_RATIO):
        failed.append("出来高")

    # 条件3: RSI ≥ 50
    if not (pd.notna(today["RSI14"]) and today["RSI14"] >= RSI_D_MIN):
        failed.append(f"RSI({today['RSI14']:.1f})" if pd.notna(today["RSI14"]) else "RSI(NaN)")

    # 条件4: 終値 > SMA200
    if not (pd.notna(today["SMA200"]) and today["Close"] > today["SMA200"]):
        failed.append("SMA200")

    atr  = today["ATR14"] if pd.notna(today["ATR14"]) else None
    stop = round(today["Close"] - atr * ATR_D_SL, 1) if atr else None
    tp   = round(today["Close"] + atr * ATR_D_TP, 1) if atr else None
    return len(failed) == 0, failed, stop, tp


# ──────────────────────────────────────────
# 戦略Eシグナル判定（52週高値ブレイクアウト）
# ──────────────────────────────────────────
def check_signal_e(df: pd.DataFrame) -> tuple[bool, list[str], float, float]:
    if len(df) < 200:
        return False, ["データ不足"], None, None

    row    = df.iloc[-1]
    failed = []

    # 条件1: 52週高値更新（前日までの252営業日最高値）
    lookback = min(252, len(df) - 1)
    high_52w = df["Close"].iloc[-lookback - 1:-1].max()
    if not (row["Close"] > high_52w):
        failed.append(f"52週高値未達({high_52w:,.0f})")

    # 条件2: 出来高フィルター（20日平均の1.2倍以上）
    if not (pd.notna(row["VOL_MA20"]) and row["Volume"] >= row["VOL_MA20"] * VOL_E_RATIO):
        failed.append("出来高")

    # 条件3: 終値 > SMA200
    if not (pd.notna(row["SMA200"]) and row["Close"] > row["SMA200"]):
        failed.append("SMA200")

    # 条件4: RSI 50〜80（過熱除外）
    if not (pd.notna(row["RSI14"]) and RSI_E_MIN <= row["RSI14"] <= RSI_E_MAX):
        failed.append(f"RSI({row['RSI14']:.1f})" if pd.notna(row["RSI14"]) else "RSI(NaN)")

    # 条件5: 陽線
    if not (row["Close"] > row["Open"]):
        failed.append("陰線")

    atr  = row["ATR14"] if pd.notna(row["ATR14"]) else None
    stop = round(row["Close"] - atr * ATR_E_SL, 1) if atr else None
    tp   = round(row["Close"] + atr * ATR_E_TP, 1) if atr else None
    return len(failed) == 0, failed, stop, tp


# ──────────────────────────────────────────
# スキャン実行
# ──────────────────────────────────────────
def run_scan(all_df: pd.DataFrame, codes: list[str], earnings_map: dict,
             strategy: str = "A") -> pd.DataFrame:
    today = all_df["Date"].max()
    results = []

    for code in codes:
        df = all_df[all_df["Code"] == code].copy().reset_index(drop=True)
        if len(df) < SMA_TREND + 10:
            continue

        df = add_indicators(df)
        df["RSI14_lag3"] = df["RSI14"].shift(RSI_LOOKBACK)

        latest = df.iloc[-1]
        if latest["Date"] != today:
            continue

        if strategy == "A":
            earnings_dates = earnings_map.get(code, [])
            signal, failed = check_signal(latest, today, earnings_dates)
            atr = latest["ATR14"] if pd.notna(latest["ATR14"]) else None
            stop_loss   = round(latest["Close"] - atr * 2, 1) if atr else None
            take_profit = round(latest["Close"] + atr * 4, 1) if atr else None
        elif strategy == "C":
            signal, failed, stop_loss, take_profit = check_signal_c(df)
            atr = latest["ATR14"] if pd.notna(latest["ATR14"]) else None
        elif strategy == "D":
            signal, failed, stop_loss, take_profit = check_signal_d(df)
            atr = latest["ATR14"] if pd.notna(latest["ATR14"]) else None
        elif strategy == "E":
            signal, failed, stop_loss, take_profit = check_signal_e(df)
            atr = latest["ATR14"] if pd.notna(latest["ATR14"]) else None
        else:
            continue

        results.append({
            "Strategy"    : strategy,
            "Code"        : code,
            "Date"        : today.strftime("%Y-%m-%d"),
            "Signal"      : signal,
            "Close"       : latest["Close"],
            "RSI14"       : round(latest["RSI14"], 1) if pd.notna(latest["RSI14"]) else None,
            "ADX14"       : round(latest["ADX14"], 1) if pd.notna(latest.get("ADX14")) else None,
            "StopLoss"    : stop_loss,
            "TakeProfit"  : take_profit,
            "ATR14"       : round(atr, 1) if atr else None,
            "FailedConds" : ", ".join(failed) if not signal else "",
        })

    cols = ["Strategy", "Code", "Date", "Signal", "Close", "RSI14", "ADX14",
            "StopLoss", "TakeProfit", "ATR14", "FailedConds"]
    if not results:
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(results, columns=cols)


# ──────────────────────────────────────────
# 結果表示・ログ保存
# ──────────────────────────────────────────
def display_and_save(result_df: pd.DataFrame) -> None:
    signals    = result_df[result_df["Signal"] == True].sort_values("RSI14", ascending=False)
    no_signals = result_df[result_df["Signal"] == False]

    print(f"\n{'='*60}")
    print(f"【シグナル銘柄】合計 {len(signals)} 件")
    print(f"{'='*60}")

    if signals.empty:
        print("  本日のシグナルなし")
    else:
        print(f"  {'戦略':<4} {'コード':<6} {'終値':>7} {'RSI':>6} {'損切り':>8} {'利確':>8} {'保有上限':>6}")
        print(f"  {'-'*4} {'-'*6} {'-'*7} {'-'*6} {'-'*8} {'-'*8} {'-'*6}")
        hold_map = {"A": "10日", "C": "7日", "D": "5日", "E": "10日"}
        for _, row in signals.iterrows():
            sl  = f"{row['StopLoss']:>8,.0f}" if pd.notna(row["StopLoss"]) else "     ---"
            tp  = f"{row['TakeProfit']:>8,.0f}" if pd.notna(row["TakeProfit"]) else "     ---"
            rsi = f"{row['RSI14']:>6.1f}" if pd.notna(row["RSI14"]) else "   N/A"
            print(f"  {row['Strategy']:<4} {row['Code']:<6} {row['Close']:>7,.0f} {rsi} {sl} {tp} {hold_map.get(row['Strategy'], '---'):>6}")

    print(f"\n【非シグナル銘柄】{len(no_signals)} 件（条件未達）")

    os.makedirs("logs", exist_ok=True)
    if os.path.exists(SCANNER_LOG_CSV):
        existing = pd.read_csv(SCANNER_LOG_CSV, dtype=str)
        combined = pd.concat([existing, result_df.astype(str)], ignore_index=True)
    else:
        combined = result_df.astype(str)

    combined.to_csv(SCANNER_LOG_CSV, index=False, encoding="utf-8-sig")
    print(f"\nログ保存: {SCANNER_LOG_CSV}")


# ──────────────────────────────────────────
# メイン
# ──────────────────────────────────────────
def main():
    log("scanner.py 開始")
    print("=" * 60)
    print("  スイングトレード シグナルスキャナー（戦略A/C/D）")
    print(f"  実行日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    client = call_with_retry(get_client)

    # 各戦略の候補銘柄を読み込み（重複排除して一括取得）
    codes_a = load_candidates(CANDIDATES_CSV)
    codes_c = load_candidates(CANDIDATES_C_CSV) if os.path.exists(CANDIDATES_C_CSV) else []
    codes_d = load_candidates(CANDIDATES_D_CSV) if os.path.exists(CANDIDATES_D_CSV) else []
    codes_e = load_candidates(CANDIDATES_E_CSV) if os.path.exists(CANDIDATES_E_CSV) else []
    all_codes = list(dict.fromkeys(codes_a + codes_c + codes_d + codes_e))
    print(f"対象銘柄数: A={len(codes_a)} C={len(codes_c)} D={len(codes_d)} E={len(codes_e)} 合計（ユニーク）={len(all_codes)}")

    all_df = fetch_prices(client, all_codes)

    print("決算日データ取得中...")
    earnings_map = fetch_earnings_dates(client, codes_a)

    today = all_df["Date"].max()
    print(f"\nスキャン基準日: {today.strftime('%Y-%m-%d (%a)')}")

    df_a = run_scan(all_df, codes_a, earnings_map, strategy="A")
    df_c = run_scan(all_df, codes_c, {},            strategy="C")
    df_d = run_scan(all_df, codes_d, {},            strategy="D")
    df_e = run_scan(all_df, codes_e, {},            strategy="E")

    result_df = pd.concat([df_a, df_c, df_d, df_e], ignore_index=True)
    display_and_save(result_df)

    print("\n完了。シグナル銘柄をpaper_trade_log.xlsxに記録してください。")
    log("scanner.py 完了")

    signals = result_df[result_df["Signal"] == True] if not result_df.empty else pd.DataFrame()
    if signals.empty:
        notify_body = "本日のシグナルなし"
    else:
        lines = [f"シグナル：{len(signals)}件"]
        for _, row in signals.iterrows():
            lines.append(f"  [{row['Strategy']}] {row['Code']}  RSI={row['RSI14']}")
        notify_body = "\n".join(lines)
    send_notify("【ST】スキャン完了", notify_body)


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
    except ConnectionError as e:
        log(f"ERROR: ConnectionError: {e}")
        send_error_notify(_SCRIPT, "ConnectionError", str(e))
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
