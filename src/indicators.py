import pandas as pd
import pandas_ta as ta


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["SMA20"]  = ta.sma(df["Close"], length=20)
    df["SMA50"]  = ta.sma(df["Close"], length=50)
    df["SMA200"] = ta.sma(df["Close"], length=200)
    df["RSI14"]  = ta.rsi(df["Close"], length=14)
    df["ATR14"]  = ta.atr(df["High"], df["Low"], df["Close"], length=14)
    df["VOL_MA20"] = ta.sma(df["Volume"], length=20)
    return df
