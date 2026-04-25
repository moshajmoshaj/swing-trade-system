"""
戦略B: 出来高ブレイクアウト戦略

エントリー条件:
  1. 52週高値更新（Close >= 前日までの260日High最大値）
  2. 出来高 >= 20日平均出来高 × 2
  3. 株価 500〜3,000円
  4. 時価総額500億円以上（eligible_codesで外部フィルター）

損切り: max(ブレイク起点安値, entry × 0.97) ← 損失が小さい方（価格が高い方）
利確 : entry × 1.08（+8%）
保有 : 最大10日
"""
import numpy as np
import pandas as pd


def add_indicators_b(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["High52W_prev"] = (
        df["High"].rolling(260, min_periods=260).max().shift(1)
    )
    df["VOL_MA20_B"] = df["Volume"].rolling(20, min_periods=20).mean()
    df["Low10D"]     = df["Low"].rolling(10, min_periods=1).min()
    if "Va" in df.columns:
        df["Va_MA20"] = df["Va"].rolling(20, min_periods=20).mean()
    return df


def generate_signals_b(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    has_history   = df["High52W_prev"].notna() & df["VOL_MA20_B"].notna()
    new_52w_high  = df["Close"] >= df["High52W_prev"]
    vol_surge     = df["Volume"] >= df["VOL_MA20_B"] * 2.0
    price_ok      = (df["Close"] >= 500) & (df["Close"] <= 3_000)
    bullish       = df["Close"] > df["Open"]

    if "Va_MA20" in df.columns:
        mcap_ok = df["Va_MA20"] >= 500_000_000
    else:
        mcap_ok = pd.Series(True, index=df.index)

    df["signal_b"] = (
        has_history & new_52w_high & vol_surge & price_ok & bullish & mcap_ok
    ).astype(int)

    return df


def calc_stop_loss_b(entry_price: float, low10d: float) -> float:
    """損切り = max(ブレイク起点の安値, entry × 0.97)（損失が小さい方 = 価格が高い方）"""
    sl_pct   = entry_price * 0.97
    sl_break = low10d if (not np.isnan(low10d)) else sl_pct
    return max(sl_break, sl_pct)


def calc_take_profit_b(entry_price: float) -> float:
    """利確 = entry × 1.08（+8%）"""
    return entry_price * 1.08
