import pandas as pd


def generate_signals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    golden_cross = (df["SMA20"] > df["SMA50"]) & (df["SMA20"].shift(1) <= df["SMA50"].shift(1))
    rsi_ok       = (df["RSI14"] >= 50) & (df["RSI14"] <= 70)
    volume_surge = df["Volume"] >= df["VOL_MA20"] * 1.5

    df["signal"] = (golden_cross & rsi_ok & volume_surge).astype(int)
    return df
