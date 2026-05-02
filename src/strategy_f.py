import pandas as pd


def build_earnings_events(fins_path: str) -> pd.DataFrame:
    """
    fins_summary.parquet から「EPS成長20%以上の通期決算開示」を抽出する。

    Returns:
        DataFrame: Code, DiscDate, eps_growth, np_val 列を持つ
    """
    fins = pd.read_parquet(fins_path)
    fins["DiscDate"] = pd.to_datetime(fins["DiscDate"], errors="coerce")

    # 通期決算のみ
    annual = fins[fins["CurPerType"] == "FY"].copy()

    for col in ["EPS", "NP", "OP"]:
        annual[col] = pd.to_numeric(annual[col], errors="coerce")

    # 銘柄×開示日順にソートしてYoY成長率を計算
    annual = annual.sort_values(["Code", "DiscDate"]).reset_index(drop=True)
    annual["eps_prev"] = annual.groupby("Code")["EPS"].shift(1)
    annual["np_prev"]  = annual.groupby("Code")["NP"].shift(1)

    # EPS成長率（両方正の場合のみ）
    valid = (annual["EPS"] > 0) & (annual["eps_prev"] > 0)
    annual["eps_growth"] = 0.0
    annual.loc[valid, "eps_growth"] = (
        (annual.loc[valid, "EPS"] - annual.loc[valid, "eps_prev"])
        / annual.loc[valid, "eps_prev"]
    )

    # フィルター: EPS成長20%以上 かつ 純利益黒字
    events = annual[
        (annual["eps_growth"] >= 0.20) &
        (annual["NP"] > 0)
    ][["DiscDate", "Code", "eps_growth", "NP"]].copy()

    events = events.rename(columns={"NP": "np_val"})
    events["Code"] = events["Code"].astype(str).str.strip()
    return events.reset_index(drop=True)


def add_earnings_flag(price_df: pd.DataFrame, events: pd.DataFrame,
                      window_days: int = 7) -> pd.DataFrame:
    """
    price_df に earnings_flag 列を追加する。
    決算開示日の翌日から window_days 日以内は True。

    Args:
        window_days: 開示後の有効ウィンドウ（カレンダー日）
    """
    df = price_df.copy()
    df["earnings_flag"] = False

    # コードを特定
    code_col = next((c for c in df.columns if c in ("Code", "code")), None)
    if code_col is None:
        return df

    code = str(df[code_col].iloc[0]).strip()
    # 5桁コードで照合
    code_events = events[events["Code"] == code]["DiscDate"]

    if code_events.empty:
        return df

    dates = df["Date"].values
    for disc_date in code_events:
        mask = (
            (df["Date"] > disc_date) &
            (df["Date"] <= disc_date + pd.Timedelta(days=window_days))
        )
        df.loc[mask, "earnings_flag"] = True

    return df


def generate_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    戦略F：決算モメンタム（PEAD）戦略（Phase 5用・Track A研究）

    エントリー条件：
    1. 決算ウィンドウ内：EPS成長20%以上の通期決算開示から7日以内
    2. 長期上昇トレンド：終値 > SMA200
    3. RSIフィルター：45以上70以下（過熱除外）
    4. 出来高確認：20日平均の1.2倍以上
    5. 陽線確認：終値 > 始値

    前提：add_earnings_flag() を呼び出した後に使用すること。

    エグジット（呼び出し元で実装）：
    - TP：ATR(14) × 5
    - SL：ATR(14) × 2
    - 強制決済：15日目
    """
    df = df.copy()

    if "earnings_flag" not in df.columns:
        df["signal"] = 0
        return df

    earnings_ok  = df["earnings_flag"] == True
    above_sma200 = df["Close"] > df["SMA200"]
    rsi_ok       = (df["RSI14"] >= 45) & (df["RSI14"] <= 70)
    vol_ok       = df["Volume"] >= df["VOL_MA20"] * 1.2
    bullish      = df["Close"] > df["Open"]

    df["signal"] = (
        earnings_ok & above_sma200 & rsi_ok & vol_ok & bullish
    ).astype(int)

    return df
