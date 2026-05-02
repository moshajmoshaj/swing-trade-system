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
import sys, time
sys.stdout.reconfigure(encoding="utf-8")

from pathlib import Path
import pandas as pd
import numpy as np
import pandas_ta as ta

INDICES_PATH = Path("data/raw/indices.parquet")
MASTER_PATH  = Path("data/raw/prices_20y.parquet")

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

    # スクリプト保存の提案
    print("  業種レジームマップを logs/sector_regime.csv に保存中...")
    df_res.to_csv("logs/sector_regime.csv", index=False, encoding="utf-8-sig")
    print("  保存完了: logs/sector_regime.csv")


if __name__ == "__main__":
    main()
