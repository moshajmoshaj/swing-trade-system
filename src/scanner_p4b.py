"""
src/scanner_p4b.py
Phase 4-2 シグナルスキャナー（戦略A v2 + 戦略G）

戦略A v2 : final_candidates_v2.csv（29銘柄）×テクニカル条件
戦略G    : EPS加速銘柄（全プライム・直近30日以内に開示）×テクニカル条件
出力     : logs/scanner_log_p4b.csv
"""
import sys
import os
sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd
import pandas_ta as ta
from datetime import datetime, timedelta
from pathlib import Path
from dotenv import load_dotenv
import jquantsapi

sys.path.insert(0, str(Path(__file__).parent))
from notifier import send_notify, send_error_notify, call_with_retry

load_dotenv()

SCHED_LOG        = "logs/scheduler_log_p4b.txt"
SCANNER_LOG      = "logs/scanner_log_p4b.csv"
CAND_A_V2        = "logs/final_candidates_v2.csv"
FINS_PATH        = "data/raw/fins_summary.parquet"
DAYS_TO_FETCH    = 250
ACCEL_THRESHOLD  = 0.10
G_WINDOW_DAYS    = 30   # 開示後何日以内

# 戦略Aパラメータ
ADX_MIN = 15
RSI_MIN_A, RSI_MAX_A = 45, 75
# 戦略Gパラメータ
RSI_MIN_G, RSI_MAX_G = 45, 70


def log(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] [p4b-scanner] {msg}"
    print(line)
    os.makedirs("logs", exist_ok=True)
    try:
        with open(SCHED_LOG, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except PermissionError:
        pass


def get_g_target_codes(scan_date: datetime.date) -> dict:
    """直近G_WINDOW_DAYS日以内にEPS加速開示があった銘柄 → {code5: disc_date}"""
    if not Path(FINS_PATH).exists():
        log("WARN: fins_summary.parquet なし → G戦略スキップ")
        return {}

    fins = pd.read_parquet(FINS_PATH)
    fins["DiscDate"] = pd.to_datetime(fins["DiscDate"], errors="coerce")
    fy = fins[fins["CurPerType"] == "FY"].copy()
    for col in ["EPS", "NP"]:
        fy[col] = pd.to_numeric(fy[col], errors="coerce")
    fy["CurFYEn"] = pd.to_datetime(fy["CurFYEn"], errors="coerce")
    fy = fy.sort_values("DiscDate").drop_duplicates(["Code","CurFYEn"], keep="last")
    fy = fy.sort_values(["Code","DiscDate"]).reset_index(drop=True)
    fy["eps_prev"] = fy.groupby("Code")["EPS"].shift(1)
    import numpy as np
    valid = (fy["EPS"] > 0) & (fy["eps_prev"] > 0)
    fy["eps_growth"] = np.nan
    fy.loc[valid, "eps_growth"] = (fy.loc[valid,"EPS"]-fy.loc[valid,"eps_prev"])/fy.loc[valid,"eps_prev"]
    fy["growth_prev"] = fy.groupby("Code")["eps_growth"].shift(1)
    fy["accel"] = fy["eps_growth"] - fy["growth_prev"]

    cutoff = pd.Timestamp(scan_date) - pd.Timedelta(days=G_WINDOW_DAYS)
    recent = fy[
        (fy["accel"] >= ACCEL_THRESHOLD) &
        (fy["EPS"] > 0) & (fy["NP"] > 0) &
        (fy["DiscDate"] >= cutoff) &
        (fy["DiscDate"] <= pd.Timestamp(scan_date))
    ].copy()
    recent["Code5"] = recent["Code"].astype(str).str.strip().str[:4].str.zfill(4) + "0"
    result = {}
    for _, row in recent.iterrows():
        c = row["Code5"]
        if c not in result or row["DiscDate"] > result[c]:
            result[c] = row["DiscDate"]
    log(f"G対象（直近{G_WINDOW_DAYS}日・EPS加速）: {len(result)}銘柄")
    return result


def fetch_and_scan(client, codes: list, scan_date: datetime.date,
                   strategy: str, g_disc_dates: dict = None) -> list:
    """指定銘柄リストのシグナルをスキャン"""
    if not codes:
        return []

    end_dt   = scan_date.strftime("%Y-%m-%d")
    start_dt = (scan_date - timedelta(days=DAYS_TO_FETCH + 10)).strftime("%Y-%m-%d")

    try:
        prices = call_with_retry(
            client.get_price_range,
            start_date=start_dt, end_date=end_dt, codes=codes
        )
    except Exception as e:
        log(f"ERROR: 価格取得失敗 ({e})")
        return []

    if prices.empty:
        return []

    prices["Date"] = pd.to_datetime(prices["Date"])
    signals = []

    for code in codes:
        df = prices[prices["Code"] == code].sort_values("Date").copy()
        if len(df) < 60:
            continue

        # 指標計算
        df["SMA20"]    = df["AdjClose"].rolling(20).mean()
        df["SMA50"]    = df["AdjClose"].rolling(50).mean()
        df["SMA200"]   = df["AdjClose"].rolling(200).mean()
        df["VOL_MA20"] = df["Volume"].rolling(20).mean()
        try:
            df["RSI14"] = ta.rsi(df["AdjClose"], length=14)
            df["ATR14"] = ta.atr(df["High"], df["Low"], df["AdjClose"], length=14)
            adx_df      = ta.adx(df["High"], df["Low"], df["AdjClose"], length=14)
            df["ADX14"] = adx_df[f"ADX_14"] if adx_df is not None else 0
        except Exception:
            continue

        today_row = df[df["Date"].dt.date == scan_date]
        if today_row.empty:
            continue
        row = today_row.iloc[-1]

        if pd.isna(row["RSI14"]) or pd.isna(row["ATR14"]) or pd.isna(row["SMA200"]):
            continue

        close = row["AdjClose"]
        vol   = row["Volume"]

        if strategy == "A":
            # 戦略Aシグナル条件
            rsi3 = df["RSI14"].shift(3).iloc[-1] if len(df) > 3 else None
            if (row["SMA20"] > row["SMA50"] and
                close > row["SMA200"] and
                RSI_MIN_A <= row["RSI14"] <= RSI_MAX_A and
                (rsi3 is None or row["RSI14"] > rsi3) and
                vol >= row["VOL_MA20"] * 1.2 and
                close > row["Open"] and
                row["ADX14"] > ADX_MIN):
                signals.append({
                    "Code": code[:4], "Strategy": "A",
                    "Close": close, "RSI": round(row["RSI14"], 1),
                    "ADX": round(row["ADX14"], 1),
                    "ATR": round(row["ATR14"], 1),
                    "SL": round(close - row["ATR14"] * 2, 0),
                    "TP": round(close + row["ATR14"] * 6, 0),
                    "MaxHold": 10
                })

        elif strategy == "G":
            # 戦略Gシグナル条件（EPS加速ウィンドウ内）
            disc = g_disc_dates.get(code)
            if disc is None:
                continue
            in_window = disc < pd.Timestamp(scan_date) <= disc + pd.Timedelta(days=G_WINDOW_DAYS)
            if not in_window:
                continue
            if (close > row["SMA200"] and
                RSI_MIN_G <= row["RSI14"] <= RSI_MAX_G and
                vol >= row["VOL_MA20"] * 1.2 and
                close > row["Open"]):
                signals.append({
                    "Code": code[:4], "Strategy": "G",
                    "Close": close, "RSI": round(row["RSI14"], 1),
                    "ADX": 0,
                    "ATR": round(row["ATR14"], 1),
                    "SL": round(close - row["ATR14"] * 2, 0),
                    "TP": round(close + row["ATR14"] * 6, 0),
                    "MaxHold": 15
                })

    return signals


def main():
    log("scanner_p4b.py 開始")
    try:
        api_key = os.getenv("JQUANTS_REFRESH_TOKEN")
        client  = jquantsapi.ClientV2(api_key=api_key)
        today   = datetime.now().date()
        scan_date = today

        # 戦略A v2 候補
        cand_a = pd.read_csv(CAND_A_V2)
        codes_a = (cand_a["code"].astype(str).str.zfill(4) + "0").tolist()

        # 戦略G 対象銘柄（直近30日のEPS加速開示）
        g_targets = get_g_target_codes(scan_date)
        codes_g   = list(g_targets.keys())

        # スキャン実行
        sigs_a = fetch_and_scan(client, codes_a, scan_date, "A")
        sigs_g = fetch_and_scan(client, codes_g, scan_date, "G", g_targets)
        all_sigs = sigs_a + sigs_g

        # ログ保存
        today_str = scan_date.strftime("%Y-%m-%d")
        rows = [{"Date": today_str, **s} for s in all_sigs]
        df_log = pd.DataFrame(rows)
        log_path = Path(SCANNER_LOG)
        if log_path.exists():
            old = pd.read_csv(log_path)
            old = old[old["Date"] != today_str]
            df_log = pd.concat([old, df_log], ignore_index=True)
        df_log.to_csv(log_path, index=False, encoding="utf-8-sig")

        # 結果表示
        log(f"A: {len(sigs_a)}件  G: {len(sigs_g)}件  合計: {len(all_sigs)}件")
        if all_sigs:
            send_notify("p4b-scanner", f"[P4-2] シグナル {len(all_sigs)}件\n" +
                        "\n".join(f"{s['Strategy']} {s['Code']} RSI={s['RSI']}" for s in all_sigs))
        else:
            send_notify("p4b-scanner", f"[P4-2] {today_str} シグナルなし")

        log("scanner_p4b.py 完了")
    except Exception as e:
        import traceback
        log(f"ERROR: {e}\n{traceback.format_exc()}")
        send_error_notify("scanner_p4b.py", str(e), traceback.format_exc())


if __name__ == "__main__":
    main()
