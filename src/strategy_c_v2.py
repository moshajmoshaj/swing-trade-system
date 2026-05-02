import pandas as pd


def generate_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    戦略C v2：底打ち確認型逆張り（Phase 5用・Track A研究）

    現行C の問題：RSI 30-40帯に入った時点でエントリー → 下落継続リスク大
    v2 の改善 ：RSI が 35 を下から上に抜けた瞬間にエントリー → 底打ち確認後

    エントリー条件：
    1. 長期上昇トレンド内：終値 > SMA200
    2. 底打ち確認：RSI(14) が 35 を下から上抜け（前日 < 35 かつ 当日 ≥ 35）
    3. 出来高スパイク：20日平均の 1.5倍以上（底打ちの出来高裏付け）
    4. SMA20 が終値より上（回帰先が存在する）

    エグジット（呼び出し元で実装）：
    - TP：SMA20 回帰（エントリー時の SMA20 値）
    - SL：ATR(14) × 2.0
    - 強制決済：保有10日目
    """
    df = df.copy()

    rsi_prev        = df["RSI14"].shift(1)
    rsi_cross_35    = (rsi_prev < 35) & (df["RSI14"] >= 35)
    above_sma200    = df["Close"] > df["SMA200"]
    volume_spike    = df["Volume"] >= df["VOL_MA20"] * 1.5
    sma20_above     = df["SMA20"] > df["Close"]          # 回帰先が上にある

    df["signal"] = (
        rsi_cross_35 & above_sma200 & volume_spike & sma20_above
    ).astype(int)

    return df
