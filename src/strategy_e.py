import pandas as pd


def generate_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    戦略E：52週高値ブレイクアウト

    設計方針：
    - 行動経済学的根拠：投資家のアンカリング（過去高値への固執）により
      52週高値更新後はモメンタムが継続しやすい
    - 戦略Aの「上昇中に乗る」と補完的：Aは指標ベース、Eは価格水準ベース

    エントリー条件：
    1. 52週高値更新：終値 > 前日までの252営業日最高値
    2. 出来高フィルター：20日平均の1.2倍以上
    3. 長期トレンド：終値 > SMA200
    4. RSIフィルター：50以上80以下（過熱除外）
    5. 陽線確認：終値 > 始値
    """
    df = df.copy()

    high_52w      = df["Close"].shift(1).rolling(252, min_periods=200).max()
    new_52w_high  = df["Close"] > high_52w
    vol_surge     = df["Volume"] >= df["VOL_MA20"] * 1.2
    above_sma200  = df["Close"] > df["SMA200"]
    rsi_ok        = (df["RSI14"] >= 50) & (df["RSI14"] <= 80)
    bullish       = df["Close"] > df["Open"]

    df["signal"] = (
        new_52w_high & vol_surge & above_sma200 & rsi_ok & bullish
    ).astype(int)

    return df
