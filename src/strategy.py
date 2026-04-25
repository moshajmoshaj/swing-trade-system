import pandas as pd


def generate_signals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    golden_cross   = df["SMA20"] > df["SMA50"]                         # 中期上昇トレンド
    above_sma200   = df["Close"] > df["SMA200"]                        # 長期上昇トレンド
    rsi_ok         = (df["RSI14"] >= 45) & (df["RSI14"] <= 75)
    rsi_rising     = df["RSI14"] > df["RSI14"].shift(3)                # 3日前比でモメンタム上昇（日次ノイズを除去）
    volume_surge   = df["Volume"] >= df["VOL_MA20"] * 1.2
    bullish_candle = df["Close"] > df["Open"]                          # シグナル日が陽線（当日モメンタム確認）

    df["signal"] = (golden_cross & above_sma200 & rsi_ok & rsi_rising & volume_surge & bullish_candle).astype(int)
    return df
