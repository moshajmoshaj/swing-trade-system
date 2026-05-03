"""
analyze_sector_regime.py
業種別レジーム分析（TOPIX 33業種別指数）

目的:
  現行のTOPIX単体レジームフィルターを業種レベルで補完できるか検証する。
  候補銘柄の業種が上昇トレンドか確認し、シグナルの質向上を測る。

使用データ: data/raw/indices.parquet（TOPIX業種別指数）

コード対応（TOPIX 33業種 0040-0060）:
  J-Quantsの業種コードと一致（東証33業種分類）

出力:
  1. 現在の業種別レジーム（BULL/NEUTRAL/BEAR）一覧
  2. 戦略A候補銘柄の業種レジーム確認
  3. 業種レジームフィルターの概念設計
"""
import sys, time, os
sys.stdout.reconfigure(encoding="utf-8")

from pathlib import Path
import pandas as pd
import numpy as np
import pandas_ta as ta
from dotenv import load_dotenv
import jquantsapi

load_dotenv()

INDICES_PATH = Path("data/raw/indices.parquet")
PRICES_PATH  = Path("data/raw/prices_20y.parquet")
CAND_PATH    = Path("logs/final_candidates.csv")

IS_START = pd.Timestamp("2016-04-01")
IS_END   = pd.Timestamp("2020-12-31")
ATR_TP   = 6.0
ATR_SL   = 2.0
MAX_HOLD = 10
ADX_MIN  = 15

# TOPIX 33業種コード → 業種名マッピング
SECTOR_MAP = {
    "0040": "水産・農林業",
    "0041": "鉱業",
    "0042": "建設業",
    "0043": "食料品",
    "0044": "繊維製品",
    "0045": "パルプ・紙",
    "0046": "化学",
    "0047": "医薬品",
    "0048": "石油・石炭製品",
    "0049": "ゴム製品",
    "004A": "ガラス・土石製品",
    "004B": "鉄鋼",
    "004C": "非鉄金属",
    "004D": "金属製品",
    "004E": "機械",
    "004F": "電気機器",
    "0050": "輸送用機器",
    "0051": "精密機器",
    "0052": "その他製品",
    "0053": "電気・ガス業",
    "0054": "陸運業",
    "0055": "海運業",
    "0056": "空運業",
    "0057": "倉庫・運輸関連業",
    "0058": "情報・通信業",
    "0059": "卸売業",
    "005A": "小売業",
    "005B": "銀行業",
    "005C": "証券・商品先物取引業",
    "005D": "保険業",
    "005E": "その他金融業",
    "005F": "不動産業",
    "0060": "サービス業",
}

# 候補銘柄の業種コード（銘柄マスタから取得が本来だがここでは代表例で記載）
# 東証33業種コード（2桁）→ J-Quantsインデックスコードのマッピング
# 参考: 東証業種コード17 → 33のマッピング
TSE33_TO_IDX = {str(i): f"{0x40+j:04X}" for j, i in enumerate(range(1, 34))}


# 業種名 → インデックスコード（SECTOR_MAP の逆引き・半角中点を正規化）
def _normalize(s: str) -> str:
    return s.replace("･", "・")  # 半角中点→全角中点

SECTOR_NAME_TO_IDX = {_normalize(v): k for k, v in SECTOR_MAP.items()}


def classify_regime(close_series: pd.Series) -> str:
    """SMA50/200でBULL/NEUTRAL/BEARを判定"""
    if len(close_series) < 200:
        return "DATA"
    sma50  = float(ta.sma(close_series, length=50).iloc[-1])
    sma200 = float(ta.sma(close_series, length=200).iloc[-1])
    close  = float(close_series.iloc[-1])
    if pd.isna(sma50) or pd.isna(sma200):
        return "DATA"
    if close > sma50:
        return "BULL"
    elif close > sma200:
        return "NEUTRAL"
    else:
        return "BEAR"


def _compute_signals_for_sector(df: pd.DataFrame) -> pd.DataFrame:
    """戦略A シグナル計算（prices_20y 列名前提）"""
    df = df.copy().reset_index(drop=True)
    df["SMA20"]    = df["Close"].rolling(20).mean()
    df["SMA50"]    = df["Close"].rolling(50).mean()
    df["SMA200"]   = df["Close"].rolling(200).mean()
    df["VOL_MA20"] = df["Volume"].rolling(20).mean()
    delta = df["Close"].diff()
    gain  = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
    loss  = (-delta).clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
    df["RSI14"]    = 100 - 100 / (1 + gain / loss.replace(0, 1e-9))
    df["RSI_lag3"] = df["RSI14"].shift(3)
    prev_cl = df["Close"].shift(1)
    tr = pd.concat([df["High"] - df["Low"],
                    (df["High"] - prev_cl).abs(),
                    (df["Low"]  - prev_cl).abs()], axis=1).max(axis=1)
    df["ATR14"] = tr.ewm(alpha=1/14, adjust=False).mean()
    prev_hi = df["High"].shift(1)
    prev_lo = df["Low"].shift(1)
    dm_p = np.where((df["High"] - prev_hi) > (prev_lo - df["Low"]),
                    np.maximum(df["High"] - prev_hi, 0.0), 0.0)
    dm_m = np.where((prev_lo - df["Low"]) > (df["High"] - prev_hi),
                    np.maximum(prev_lo - df["Low"], 0.0), 0.0)
    atr_s = df["ATR14"].replace(0, np.nan)
    di_p  = 100 * pd.Series(dm_p, index=df.index).ewm(alpha=1/14, adjust=False).mean() / atr_s
    di_m  = 100 * pd.Series(dm_m, index=df.index).ewm(alpha=1/14, adjust=False).mean() / atr_s
    dx    = 100 * (di_p - di_m).abs() / (di_p + di_m).replace(0, np.nan)
    df["ADX14"] = dx.ewm(alpha=1/14, adjust=False).mean()
    df["signal"] = (
        df["SMA20"].notna() & df["SMA200"].notna() &
        df["RSI14"].notna() & df["ATR14"].notna() & df["ADX14"].notna() &
        (df["SMA20"] > df["SMA50"]) & (df["Close"] > df["SMA200"]) &
        (df["RSI14"] >= 45) & (df["RSI14"] <= 75) &
        (df["RSI14"] > df["RSI_lag3"]) &
        (df["Volume"] >= df["VOL_MA20"] * 1.2) &
        (df["Close"] > df["Open"]) & (df["ADX14"] > ADX_MIN)
    ).astype(int)
    return df


def _extract_is_trades_for_sector(df: pd.DataFrame, code5: str) -> list[dict]:
    trades = []
    sigs = df[(df["Date"] >= IS_START) & (df["Date"] <= IS_END) & (df["signal"] == 1)]
    for i in sigs.index:
        if i + 1 >= len(df):
            continue
        sig = df.iloc[i]; nxt = df.iloc[i + 1]
        ep = nxt["Open"]; atr = sig["ATR14"]
        if (ep - sig["Close"]) / sig["Close"] < -0.015:
            continue
        tp = ep + atr * ATR_TP; sl = ep - atr * ATR_SL
        exit_px = reason = None
        for k in range(1, MAX_HOLD + 1):
            if i + 1 + k >= len(df): break
            fut = df.iloc[i + 1 + k]
            if fut["Low"] <= sl:    exit_px, reason = sl, "損切り"; break
            elif fut["High"] >= tp: exit_px, reason = tp, "利確"; break
            elif k == MAX_HOLD:     exit_px, reason = fut["Close"], "強制終了"
        if exit_px is None: continue
        trades.append({"code": code5, "signal_date": sig["Date"],
                       "win": exit_px > ep, "pnl_pct": (exit_px - ep) / ep})
    return trades


def run_sector_backtest(idx: pd.DataFrame) -> None:
    """IS期間の戦略Aシグナルに業種レジームフィルターを適用して効果を検証"""
    print(f"\n{'='*65}")
    print("  【Phase 5 研究】業種レジームフィルター IS期間バックテスト")
    print(f"  IS期間: {IS_START.date()} ～ {IS_END.date()}")
    print(f"{'='*65}")

    # ─── 日次セクターレジーム辞書を構築 ───────────────────
    print("\n日次セクターレジーム計算中...")
    sector_regime: dict[str, dict] = {}  # idx_code → {date → regime}
    idx_33 = idx[idx["Code"].isin(set(SECTOR_MAP.keys()))].copy()
    for idx_code, grp in idx_33.groupby("Code"):
        grp = grp.sort_values("Date").copy()
        grp["SMA50"]  = grp["C"].rolling(50).mean()
        grp["SMA200"] = grp["C"].rolling(200).mean()
        regime_map = {}
        for _, row in grp.iterrows():
            if pd.isna(row["SMA50"]) or pd.isna(row["SMA200"]):
                r = "DATA"
            elif row["C"] > row["SMA50"]:
                r = "BULL"
            elif row["C"] > row["SMA200"]:
                r = "NEUTRAL"
            else:
                r = "BEAR"
            regime_map[row["Date"]] = r
        sector_regime[idx_code] = regime_map
    print(f"  {len(sector_regime)}業種の日次レジーム計算完了")

    # ─── 候補銘柄 & セクターマッピング ────────────────────
    api_key = os.getenv("JQUANTS_REFRESH_TOKEN")
    client  = jquantsapi.ClientV2(api_key=api_key)
    master  = client.get_eq_master()
    # code5 → sector index code
    code5_to_idx: dict[str, str] = {}
    for _, row in master.iterrows():
        nm      = _normalize(str(row.get("S33Nm", "")))
        idx_c   = SECTOR_NAME_TO_IDX.get(nm)
        code5   = str(row["Code"]).strip()
        if idx_c:
            code5_to_idx[code5] = idx_c

    cand = pd.read_csv(CAND_PATH, dtype=str)
    col  = next(c for c in cand.columns if "code" in c.lower())
    codes5 = [c.zfill(4) + "0" for c in cand[col]]
    print(f"  候補銘柄: {len(codes5)}銘柄  セクターマップ取得済み")

    # ─── 株価データ & シグナル抽出 ─────────────────────────
    print("IS期間シグナル抽出中...")
    # prices_20y.parquet は Open/High/Low/Close/Volume が既存列（リネーム不要）
    prices = pd.read_parquet(PRICES_PATH)
    prices["Date"] = pd.to_datetime(prices["Date"])
    prices["Code"] = prices["Code4"].astype(str).str.zfill(4) + "0"
    drop_c = [c for c in ["Code4","O","H","L","C","Vo","Va","UL","LL","AdjFactor",
                           "MO","MH","ML","MC","MUL","MLL","MVo","MVa",
                           "AO","AH","AL","AC","AUL","ALL","AVo","AVa"]
              if c in prices.columns]
    prices = prices.drop(columns=drop_c)
    prices = prices[prices["Code"].isin(set(codes5))].copy()
    prices = prices[prices["Date"] >= pd.Timestamp("2015-01-01")].copy()
    prices = prices.sort_values(["Code", "Date"]).reset_index(drop=True)

    all_trades = []
    for code5 in codes5:
        df_s = prices[prices["Code"] == code5].copy().reset_index(drop=True)
        if len(df_s) < 210:
            continue
        df_s = _compute_signals_for_sector(df_s)
        all_trades.extend(_extract_is_trades_for_sector(df_s, code5))

    if not all_trades:
        print("  IS期間のトレードなし - スキップ")
        return

    tdf = pd.DataFrame(all_trades)

    # ─── 業種レジームとジョイン ────────────────────────────
    def get_regime_on_date(code5: str, dt: pd.Timestamp) -> str:
        idx_code = code5_to_idx.get(code5)
        if not idx_code:
            return "UNKNOWN"
        return sector_regime.get(idx_code, {}).get(dt, "UNKNOWN")

    tdf["sector_regime"] = tdf.apply(
        lambda r: get_regime_on_date(r["code"], r["signal_date"]), axis=1)

    n_known = (tdf["sector_regime"] != "UNKNOWN").sum()
    print(f"  シグナル総数: {len(tdf)}  業種レジーム判定済み: {n_known}")

    # ─── 結果比較 ──────────────────────────────────────────
    print(f"\n{'─'*65}")
    print(f"  {'フィルター':20} {'件数':>5} {'勝率':>7} {'平均損益':>9} {'除外率':>7}")
    print(f"  {'-'*20} {'-'*5} {'-'*7} {'-'*9} {'-'*7}")

    def show_row(label: str, subset: pd.DataFrame, total: int) -> None:
        if len(subset) == 0:
            return
        wr  = subset["win"].mean() * 100
        avg = subset["pnl_pct"].mean() * 100
        excl = (1 - len(subset) / total) * 100 if total > 0 else 0
        print(f"  {label:20} {len(subset):>5} {wr:>6.1f}% {avg:>+8.2f}% {excl:>6.1f}%除外")

    base = tdf[tdf["sector_regime"] != "UNKNOWN"]
    show_row("全シグナル（ベース）", base, len(base))
    show_row("BULL 業種のみ",       base[base["sector_regime"] == "BULL"], len(base))
    show_row("NEUTRAL 業種のみ",    base[base["sector_regime"] == "NEUTRAL"], len(base))
    show_row("BEAR 業種のみ",       base[base["sector_regime"] == "BEAR"], len(base))
    show_row("BULL+NEUTRAL",        base[base["sector_regime"].isin(["BULL","NEUTRAL"])], len(base))

    # ─── 結論 ─────────────────────────────────────────────
    bull_sub  = base[base["sector_regime"] == "BULL"]
    base_wr   = base["win"].mean() * 100  if len(base)      > 0 else 0
    bull_wr   = bull_sub["win"].mean() * 100 if len(bull_sub) > 0 else 0
    diff      = bull_wr - base_wr

    print(f"\n{'='*65}")
    print(f"  【結論】")
    print(f"  BULL業種フィルターの勝率変化: {base_wr:.1f}% → {bull_wr:.1f}% "
          f"({diff:+.1f}pt)")
    if diff > 2.0:
        print("  ✅ 勝率改善あり。ただし取引数減少とのトレードオフを確認すること。")
        print("  ⚠️  OOS期間での確認なしに採用不可。Phase 5 で正式検証推奨。")
    elif diff > 0:
        print("  ⚠️  わずかな改善。Track A 結論と同様に追加フィルター=取引数減少。")
        print("  → 現行 TOPIX 単体フィルターで継続推奨。")
    else:
        print("  ❌ 勝率改善なし。業種レジームフィルターは不採用推奨。")
        print("  → 戦略のエッジは業種の強弱とは独立して機能する（Track A 結論と一致）。")

    out_path = Path("logs/sector_regime_backtest.csv")
    tdf.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n  → {out_path} に保存")


def main():
    print("=" * 65)
    print("  業種別レジーム分析（TOPIX 33業種）")
    print("=" * 65)

    # 指数データ読み込み
    idx = pd.read_parquet(INDICES_PATH)
    idx["Date"] = pd.to_datetime(idx["Date"], errors="coerce")
    idx["C"]    = pd.to_numeric(idx["C"], errors="coerce")
    idx = idx.sort_values(["Code", "Date"])

    # TOPIX全体のレジーム
    topix = idx[idx["Code"] == "0000"][["Date","C"]].sort_values("Date")
    topix_regime = classify_regime(topix["C"])
    topix_latest = float(topix["C"].iloc[-1])
    topix_sma50  = float(ta.sma(topix["C"], length=50).iloc[-1])
    topix_sma200 = float(ta.sma(topix["C"], length=200).iloc[-1])
    print(f"\n  TOPIX（全体）: {topix_regime}")
    print(f"  終値={topix_latest:.1f}  SMA50={topix_sma50:.1f}  SMA200={topix_sma200:.1f}")

    # 33業種別レジーム計算
    print(f"\n{'─'*65}")
    print("  業種別レジーム（直近: 2026-05-01）")
    print(f"{'─'*65}")
    print(f"  {'業種':16} {'レジーム':8} {'終値':>8} {'SMA50':>8} {'SMA200':>8} {'SMA50比':>8}")
    print(f"  {'-'*16} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")

    results = []
    for code, name in SECTOR_MAP.items():
        sub = idx[idx["Code"] == code][["Date","C"]].sort_values("Date")
        if len(sub) < 200:
            continue
        regime = classify_regime(sub["C"])
        cl     = float(sub["C"].iloc[-1])
        sma50  = float(ta.sma(sub["C"], length=50).iloc[-1])
        sma200 = float(ta.sma(sub["C"], length=200).iloc[-1])
        pct50  = (cl / sma50 - 1) * 100 if sma50 > 0 else 0

        mark = {"BULL": "✅", "NEUTRAL": "⚠️ ", "BEAR": "❌"}.get(regime, "？")
        print(f"  {name:16} {mark}{regime:6} {cl:>8.1f} {sma50:>8.1f} {sma200:>8.1f} {pct50:>+7.1f}%")
        results.append({"code": code, "name": name, "regime": regime,
                         "close": cl, "sma50": sma50, "sma200": sma200})

    df_res = pd.DataFrame(results)

    # 集計
    print(f"\n{'─'*65}")
    regime_counts = df_res["regime"].value_counts()
    total = len(df_res)
    for r in ["BULL", "NEUTRAL", "BEAR"]:
        n = regime_counts.get(r, 0)
        mark = {"BULL": "✅", "NEUTRAL": "⚠️ ", "BEAR": "❌"}[r]
        print(f"  {mark} {r:8}: {n:2d}業種 ({n/total*100:.0f}%)")

    # BULL業種のトップ
    bull = df_res[df_res["regime"] == "BULL"].sort_values("close", ascending=False)
    print(f"\n  【BULL業種（SMA50上）上位5】")
    for _, r in bull.head(5).iterrows():
        pct = (r["close"] / r["sma50"] - 1) * 100
        print(f"    {r['name']:16} SMA50比 {pct:+.1f}%")

    # Phase 5での活用提案
    print(f"\n{'='*65}")
    print("  業種別レジームフィルター 設計提案")
    print(f"{'='*65}")
    print("""
  現行のレジームフィルター（TOPIX単体）との比較:

  現行:  TOPIX > SMA50 → BULL（A/D/E 有効）
  改善案: TOPIX > SMA50 AND 対象銘柄の業種が BULL → エントリー許可

  効果の見立て:
  - BULLセクターの銘柄のみに絞ることで「市場全体は強気だが
    その業種は弱い」ケースを除外できる可能性
  - ただし戦略A/Eの検証（財務・信用倍率）では
    「フィルター追加=取引数減少=CAGR低下」のパターンが続いている

  推奨: 業種レジームフィルターは backtest で効果を確認してから採用判断。
  実装は Phase 5 移行後、戦略A 20年再選定と同タイミングで検討。
  Phase 4 は現行 TOPIX 単体フィルターで凍結継続。
  """)

    # 現状マップ保存
    print("  業種レジームマップを logs/sector_regime.csv に保存中...")
    df_res.to_csv("logs/sector_regime.csv", index=False, encoding="utf-8-sig")
    print("  保存完了: logs/sector_regime.csv")

    # Phase 5 研究: 業種レジームフィルターの IS 期間バックテスト
    run_sector_backtest(idx)


if __name__ == "__main__":
    main()
