import pandas as pd


def generate_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    戦略C：平均回帰・逆張り戦略

    設計方針：
    - 戦略Aが「上昇中に乗る」順張りであるのに対し、
      戦略Cは「下がりすぎた優良株を拾う」逆張り
    - 相場環境が異なる局面で補完的に機能することを狙う

    エントリー条件：
    1. 長期上�昇トレンド内：終値 > SMA200
    2. 短期押し目：RSI(14)が30以上40以下
    3. 出来高スパイク：20日平均の1.5倍以上
    """
    df = df.copy()

    # エントリー条件
    above_sma200 = df["Close"] > df["SMA200"]                    # 長期上昇トレンド内
    rsi_dip = (df["RSI14"] >= 30) & (df["RSI14"] <= 40)          # 短期押し目
    volume_spike = df["Volume"] >= df["VOL_MA20"] * 1.5          # 出来高スパイク

    # シグナル：3条件をすべて満たす
    df["signal"] = (above_sma200 & rsi_dip & volume_spike).astype(int)

    return df
