"""
市場レジーム判定
TOPIX連動ETF(1306)の SMA50/200 で相場環境を3段階に分類する

  BULL   : 終値 > SMA50            → 戦略A/D/E 有効
  NEUTRAL: SMA200 < 終値 ≤ SMA50   → 戦略C のみ有効
  BEAR   : 終値 ≤ SMA200           → 全戦略停止
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
import pandas_ta as ta
from datetime import datetime, timedelta
import jquantsapi

_TOPIX_PROXY = "1306"   # NEXT FUNDS TOPIX連動型上場投信

REGIME_STRATEGIES: dict[str, list[str]] = {
    "BULL":    ["A", "D", "E"],
    "NEUTRAL": ["C"],
    "BEAR":    [],
}

REGIME_LABEL: dict[str, str] = {
    "BULL":    "強気（TOPIX>SMA50）",
    "NEUTRAL": "調整（SMA200<TOPIX≤SMA50）",
    "BEAR":    "弱気（TOPIX≤SMA200）",
}


def fetch_topix_close(client: jquantsapi.ClientV2, days: int = 320) -> pd.Series:
    """1306 ETF の調整済終値(AdjC)を取得してTOPIXの代理とする。
    AdjCは株式分割を調整した一貫した系列。
    """
    end   = datetime.today()
    start = end - timedelta(days=int(days * 1.5))
    df = client.get_eq_bars_daily(
        code=_TOPIX_PROXY,
        from_yyyymmdd=start.strftime("%Y%m%d"),
        to_yyyymmdd=end.strftime("%Y%m%d"),
    )
    if df is None or df.empty:
        raise ValueError(f"{_TOPIX_PROXY} のデータ取得に失敗")
    # AdjCが存在する場合はそちらを使用（分割調整済み）、なければC
    close_col = "AdjC" if "AdjC" in df.columns else "C"
    df = df.rename(columns={close_col: "Close"})
    df["Date"] = pd.to_datetime(df["Date"])
    return df.sort_values("Date").set_index("Date")["Close"]


def get_regime(client: jquantsapi.ClientV2) -> tuple[str, dict]:
    """
    現在の市場レジームを返す。

    Returns:
        regime : "BULL" | "NEUTRAL" | "BEAR"
        info   : topix / sma50 / sma200 の値と有効戦略リスト
    """
    close  = fetch_topix_close(client)
    latest = float(close.iloc[-1])
    sma50  = ta.sma(close, length=50)
    sma200 = ta.sma(close, length=200)

    s50  = float(sma50.iloc[-1])
    s200 = float(sma200.iloc[-1])

    if pd.isna(s50) or pd.isna(s200):
        return "BULL", {
            "topix": round(latest, 1), "sma50": None, "sma200": None,
            "active_strategies": REGIME_STRATEGIES["BULL"],
            "warning": "SMA計算不可・デフォルトBULL適用",
        }

    if latest > s50:
        regime = "BULL"
    elif latest > s200:
        regime = "NEUTRAL"
    else:
        regime = "BEAR"

    return regime, {
        "topix":             round(latest, 1),
        "sma50":             round(s50, 1),
        "sma200":            round(s200, 1),
        "active_strategies": REGIME_STRATEGIES[regime],
    }
