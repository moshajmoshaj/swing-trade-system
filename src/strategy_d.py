import pandas as pd


def generate_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    戦略D：ギャップアップ翌日戦略

    設計方針：
    - 決算・材料で窓を開けた翌日に、値固まりを確認してからエントリー
    - シグナル頻度は低いが1回あたりの期待値が高い
    - 戦略Aの決算除外ルールとは補完関係にある

    エントリー条件：
    1. 前日ギャップアップ：前日比+3%以上の窓開け上昇
    2. 出来高継続：翌日出来高 ≥ 20日平均×1.5
    3. モメンタム確認：RSI(14) ≥ 50
    4. 長期トレンド確認：終値 > SMA200
    """
    df = df.copy()

    # 前日のギャップアップを検出（当日の始値 vs 前日の終値）
    prev_close = df["Close"].shift(1)
    gap_pct = (df["Open"] - prev_close) / prev_close * 100
    gap_up = gap_pct >= 3.0  # 前日比+3%以上

    # 当日の条件
    volume_surge = df["Volume"] >= df["VOL_MA20"] * 1.5       # 出来高継続
    rsi_strong = df["RSI14"] >= 50                             # モメンタム確認
    above_sma200 = df["Close"] > df["SMA200"]                  # 長期トレンド確認

    # シグナル：ギャップアップした前日の翌日にエントリー可能
    # gap_up.shift(1) で前日のギャップアップを当日で判定
    df["signal"] = (gap_up.shift(1) & volume_surge & rsi_strong & above_sma200).astype(int)

    return df
