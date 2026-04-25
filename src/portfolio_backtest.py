import os
import sys
import time
sys.stdout.reconfigure(encoding="utf-8")

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = "Noto Sans JP"
import matplotlib.pyplot as plt
import matplotlib.ticker
import pandas as pd
from dotenv import load_dotenv
import jquantsapi

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent))
from indicators import add_indicators
from strategy import generate_signals
from utils.risk import calc_position_size

# ── 定数 ──────────────────────────────────────────────────────
INITIAL_CAPITAL  = 1_000_000
MAX_POSITIONS    = 5
MAX_POS_RATIO    = 0.20
ATR_TP_MULT      = 3
ATR_STOP_MULT    = 2
MAX_HOLD_DAYS    = 10

# 期間A（訓練）/ 期間B（検証）
PERIOD_A_START   = pd.Timestamp("2022-01-01")
PERIOD_A_END     = pd.Timestamp("2024-04-23")
PERIOD_B_START   = pd.Timestamp("2024-04-24")
PERIOD_B_END     = pd.Timestamp("2026-04-24")

DATA_START       = "20160401"          # prices_10y.parquet 開始日
DATA_END         = "20260424"

FINAL_30_CACHE   = Path("data/raw/prices_10y.parquet")
CANDIDATE_CSV    = Path("logs/final_candidates.csv")
OUTPUT_PNG       = Path("logs/portfolio_result.png")


@dataclass
class Position:
    code4:       str
    name:        str
    entry_date:  pd.Timestamp
    entry_price: float
    shares:      int
    stop_loss:   float
    take_profit: float
    hold_days:   int = 0
    signal_rsi:  float = 0.0


@dataclass
class Trade:
    code4:       str
    name:        str
    entry_date:  pd.Timestamp
    exit_date:   pd.Timestamp
    entry_price: float
    exit_price:  float
    shares:      int
    pnl:         float
    reason:      str


def _fetch_one(
    client: jquantsapi.ClientV2, code5: str, start: str, end: str
) -> pd.DataFrame:
    try:
        df = client.get_eq_bars_daily(code=code5, from_yyyymmdd=start, to_yyyymmdd=end)
        if df.empty:
            return pd.DataFrame()
        df = df.rename(columns={
            "AdjO": "Open", "AdjH": "High", "AdjL": "Low",
            "AdjC": "Close", "AdjVo": "Volume",
        })
        df["Date"]  = pd.to_datetime(df["Date"])
        df["Code4"] = code5[:-1]
        return df.sort_values("Date").reset_index(drop=True)
    except Exception:
        return pd.DataFrame()


def fetch_final_30(codes: list[str]) -> None:
    """
    最終候補30銘柄を2022-01-01から一括取得して専用キャッシュに保存。
    既存キャッシュが全銘柄・全期間をカバーしていれば再取得しない。
    """
    FINAL_30_CACHE.parent.mkdir(parents=True, exist_ok=True)

    if FINAL_30_CACHE.exists():
        cached = pd.read_parquet(FINAL_30_CACHE)
        cached["Date"] = pd.to_datetime(cached["Date"])
        ok_codes  = set(cached["Code4"].unique())
        ok_start  = cached["Date"].min() <= pd.Timestamp(DATA_START)
        ok_end    = cached["Date"].max() >= pd.Timestamp(DATA_END) - pd.Timedelta(days=10)
        ok_stocks = set(codes) <= ok_codes
        if ok_start and ok_end and ok_stocks:
            print(f"30銘柄キャッシュ利用: {FINAL_30_CACHE}"
                  f"  ({cached['Date'].min().date()} ～ {cached['Date'].max().date()})")
            return

    api_key = os.getenv("JQUANTS_REFRESH_TOKEN")
    client  = jquantsapi.ClientV2(api_key=api_key)
    codes5  = [c + "0" for c in codes]

    print(f"30銘柄データ取得中（2並列）: {DATA_START} ～ {DATA_END}")
    results: list[pd.DataFrame] = []
    done = 0
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = {executor.submit(_fetch_one, client, c5, DATA_START, DATA_END): c5
                   for c5 in codes5}
        for future in as_completed(futures):
            done += 1
            df = future.result()
            if not df.empty:
                results.append(df)
                print(f"  [{done:2d}/30] {futures[future][:-1]} 取得完了")
            time.sleep(0.35)

    combined = pd.concat(results, ignore_index=True)
    combined.to_parquet(FINAL_30_CACHE)
    size_mb = FINAL_30_CACHE.stat().st_size / 1024 / 1024
    print(f"保存: {FINAL_30_CACHE}  ({size_mb:.1f} MB)  {combined['Code4'].nunique()}銘柄")


def load_stock_data(codes: list[str]) -> dict[str, pd.DataFrame]:
    """prices_10y.parquet から読み込み、指標・シグナルを付与して返す。"""
    cache = pd.read_parquet(FINAL_30_CACHE)
    cache["Date"] = pd.to_datetime(cache["Date"])

    # 生列削除（prices_10y.parquet の O/H/L/C 等を除去）
    _raw = [c for c in ["Code", "Code4", "O", "H", "L", "C", "Vo", "Va", "UL", "LL", "AdjFactor"]
            if c in cache.columns]

    # 【修正2】groupby化：事前グループ化でO(1)アクセス
    grouped = cache.drop(columns=_raw).groupby(
        cache["Code4"] if "Code4" in cache.columns else cache["Code"].str[:4]
    )

    result: dict[str, pd.DataFrame] = {}
    for code4 in codes:
        try:
            df = grouped.get_group(code4).copy()
        except KeyError:
            print(f"  スキップ({code4}): データなし")
            continue
        if len(df) < 260:
            print(f"  スキップ({code4}): データ不足 {len(df)}行")
            continue
        df = df.sort_values("Date").reset_index(drop=True)
        df = add_indicators(df)
        df = generate_signals(df)
        result[code4] = df
    return result


def run_portfolio_backtest(
    stock_data:  dict[str, pd.DataFrame],
    names:       dict[str, str],
    bt_start:    pd.Timestamp,
    bt_end:      pd.Timestamp,
) -> tuple[list[Trade], pd.Series]:
    """
    全銘柄の日次データを統合してポートフォリオバックテストを実行する。
    Returns: (trades, equity_series indexed by Date)
    """
    # 全取引日の集合を作成
    all_dates = sorted(set(
        d for df in stock_data.values()
        for d in df["Date"]
        if bt_start <= d <= bt_end
    ))

    # 【修正3】set_index+to_dict：iterrowsより10〜50倍高速
    lookup: dict[str, dict] = {}
    for code4, df in stock_data.items():
        lookup[code4] = df.set_index("Date").to_dict("index")

    capital:   float          = float(INITIAL_CAPITAL)
    positions: list[Position] = []
    trades:    list[Trade]    = []
    equity:    dict[pd.Timestamp, float] = {bt_start: capital}

    for today in all_dates:
        # ── 1. 保有中ポジションの決済判定 ──────────────────────
        next_positions: list[Position] = []
        for pos in positions:
            row = lookup[pos.code4].get(today)
            if row is None:
                next_positions.append(pos)
                continue

            pos.hold_days += 1
            hi, lo, cl = row["High"], row["Low"], row["Close"]
            exit_price = exit_reason = None

            if lo <= pos.stop_loss:
                exit_price, exit_reason = pos.stop_loss, "損切り"
            elif hi >= pos.take_profit:
                exit_price, exit_reason = pos.take_profit, "利確"
            elif pos.hold_days >= MAX_HOLD_DAYS:
                exit_price, exit_reason = cl, "期間満了"

            if exit_price is not None:
                pnl = (exit_price - pos.entry_price) * pos.shares
                capital += pnl
                trades.append(Trade(
                    code4=pos.code4, name=pos.name,
                    entry_date=pos.entry_date, exit_date=today,
                    entry_price=pos.entry_price, exit_price=exit_price,
                    shares=pos.shares, pnl=pnl, reason=exit_reason,
                ))
            else:
                next_positions.append(pos)

        positions = next_positions

        # ── 2. 新規エントリー判定（前日シグナル → 本日始値）────
        slots = MAX_POSITIONS - len(positions)
        if slots > 0:
            holding_codes = {p.code4 for p in positions}
            signals: list[tuple[float, str]] = []   # (RSI, code4)

            for code4, df_lookup in lookup.items():
                if code4 in holding_codes:
                    continue
                # 前営業日を取得
                prev_row = None
                for code4_check, df in stock_data.items():
                    if code4_check == code4:
                        prev_dates = df.loc[df["Date"] < today, "Date"]
                        if not prev_dates.empty:
                            prev_row = df_lookup.get(prev_dates.iloc[-1])
                        break

                if prev_row is None:
                    continue
                if prev_row.get("signal", 0) != 1:
                    continue
                if pd.isna(prev_row.get("ATR14")):
                    continue

                today_row = df_lookup.get(today)
                if today_row is None:
                    continue

                entry_price = today_row["Open"]
                gap_pct = (entry_price - prev_row["Close"]) / prev_row["Close"]
                if gap_pct < -0.015:
                    continue

                signals.append((float(prev_row["RSI14"]), code4, prev_row, entry_price))

            # RSI降順でソートしてスロット分エントリー
            signals.sort(key=lambda x: x[0], reverse=True)

            for rsi_val, code4, prev_row, entry_price in signals[:slots]:
                atr = float(prev_row["ATR14"])
                pos_budget = capital * MAX_POS_RATIO
                shares, stop_loss = calc_position_size(capital, atr, entry_price)
                # 予算キャップ
                shares = min(shares, int(pos_budget / entry_price))
                take_profit = entry_price + atr * ATR_TP_MULT

                if shares > 0 and entry_price * shares <= capital:
                    positions.append(Position(
                        code4=code4,
                        name=names.get(code4, code4),
                        entry_date=today,
                        entry_price=entry_price,
                        shares=shares,
                        stop_loss=stop_loss,
                        take_profit=take_profit,
                        signal_rsi=rsi_val,
                    ))

        equity[today] = capital

    equity_series = pd.Series(equity).sort_index()
    return trades, equity_series


def compute_stats(
    trades: list[Trade],
    equity: pd.Series,
    bt_start: pd.Timestamp,
    bt_end:   pd.Timestamp,
) -> dict:
    total      = len(trades)
    wins       = [t for t in trades if t.pnl > 0]
    win_rate   = len(wins) / total * 100 if total else 0
    avg_pnl    = sum(t.pnl for t in trades) / total if total else 0
    final_pnl  = equity.iloc[-1] - INITIAL_CAPITAL
    years      = (bt_end - bt_start).days / 365
    annual_ret = ((equity.iloc[-1] / INITIAL_CAPITAL) ** (1 / years) - 1) * 100

    peak   = equity.cummax()
    dd     = (equity - peak) / peak * 100
    max_dd = dd.min()

    # 月別損益
    monthly = equity.resample("ME").last().diff()
    monthly.iloc[0] = equity.resample("ME").last().iloc[0] - INITIAL_CAPITAL

    return {
        "total": total, "win_rate": win_rate, "avg_pnl": avg_pnl,
        "final_pnl": final_pnl, "annual_ret": annual_ret,
        "max_dd": max_dd, "monthly": monthly,
        "final_cap": equity.iloc[-1],
    }


def save_chart(stats: dict, equity: pd.Series) -> None:
    monthly = stats["monthly"]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 9),
                                    gridspec_kw={"height_ratios": [2, 1]})

    # 上段: 資産曲線
    ax1.plot(equity.index, equity.values, color="steelblue",
             linewidth=1.8, label="ポートフォリオ資産")
    ax1.axhline(INITIAL_CAPITAL, color="gray", linestyle="--",
                linewidth=0.8, alpha=0.7, label="元本")
    ax1.fill_between(equity.index, equity.values, INITIAL_CAPITAL,
                     where=(equity.values >= INITIAL_CAPITAL),
                     alpha=0.12, color="steelblue")
    ax1.fill_between(equity.index, equity.values, INITIAL_CAPITAL,
                     where=(equity.values < INITIAL_CAPITAL),
                     alpha=0.15, color="tomato")
    ax1.set_title("ポートフォリオバックテスト（30銘柄・最大5同時保有）",
                  fontsize=14)
    ax1.set_ylabel("資産（円）")
    ax1.yaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # 下段: 月次損益棒グラフ
    colors = ["steelblue" if v >= 0 else "tomato" for v in monthly.values]
    ax2.bar(monthly.index, monthly.values, color=colors,
            width=20, align="center")
    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.set_title("月次損益", fontsize=11)
    ax2.set_ylabel("損益（円）")
    ax2.yaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda x, _: f"{int(x):+,}"))
    ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    OUTPUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_PNG, dpi=150)
    plt.close()
    print(f"グラフ保存: {OUTPUT_PNG}")


def main() -> None:
    print("=" * 64)
    print("  ポートフォリオバックテスト（訓練 / 検証 分離）")
    print(f"  期間A（訓練）: {PERIOD_A_START.date()} ～ {PERIOD_A_END.date()}")
    print(f"  期間B（検証）: {PERIOD_B_START.date()} ～ {PERIOD_B_END.date()}  ← 実行")
    print(f"  最大同時保有: {MAX_POSITIONS}銘柄 / 1銘柄上限: {MAX_POS_RATIO*100:.0f}%")
    print("=" * 64)
    print()
    print("  ※ 30銘柄は期間Bのバックテスト結果で選定済み。")
    print("  ※ ここでは「同一期間で再確認」する意味合いがあります。")
    print("  ※ 真のOOSテストには期間外データ（2026年以降）が必要です。")
    print()

    # 候補銘柄読み込み
    cands = pd.read_csv(CANDIDATE_CSV, encoding="utf-8-sig")
    codes = cands["code"].astype(str).tolist()
    names = dict(zip(cands["code"].astype(str), cands["name"]))
    print(f"対象銘柄: {len(codes)}銘柄")

    # Step1: 30銘柄の拡張データを取得（2022-01-01〜）
    fetch_final_30(codes)

    # Step2: データ読み込み・指標計算（期間A含む全期間で計算）
    print("\n指標・シグナル計算中（期間A含む全データで計算）...")
    stock_data = load_stock_data(codes)
    print(f"  完了: {len(stock_data)}銘柄")

    # Step3: 期間Bのみでポートフォリオバックテスト
    print(f"\n期間B（検証期間）バックテスト実行中...")
    trades, equity = run_portfolio_backtest(
        stock_data, names, PERIOD_B_START, PERIOD_B_END
    )

    # 統計
    stats = compute_stats(trades, equity, PERIOD_B_START, PERIOD_B_END)

    # ── 結果表示 ─────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"  検証期間（期間B）バックテスト結果")
    print(f"  {PERIOD_B_START.date()} ～ {PERIOD_B_END.date()}")
    print(f"{'=' * 60}")
    print(f"  初期資金      : {INITIAL_CAPITAL:>12,} 円")
    print(f"  最終資産      : {stats['final_cap']:>12,.0f} 円")
    print(f"  総損益        : {stats['final_pnl']:>+12,.0f} 円")
    print(f"  年利換算      : {stats['annual_ret']:>+11.2f} %")
    print(f"  総取引数      : {stats['total']:>12} 回")
    print(f"  勝率          : {stats['win_rate']:>11.1f} %")
    print(f"  平均損益/取引  : {stats['avg_pnl']:>+12,.0f} 円")
    print(f"  最大DD        : {stats['max_dd']:>11.2f} %")
    print(f"{'=' * 60}")

    # 月別損益
    print("\n【月別損益（期間B）】")
    print(f"  {'年月':8}  {'損益':>12}  {'累計資産':>13}")
    print("  " + "-" * 40)
    eq_monthly = equity.resample("ME").last()
    for ym, cap in eq_monthly.items():
        pnl_m = stats["monthly"].get(ym, 0)
        sign  = "▲" if pnl_m < 0 else "+"
        mark  = " ◀赤字" if pnl_m < 0 else ""
        print(f"  {ym.strftime('%Y-%m'):8}  {sign}{pnl_m:>10,.0f}円  {cap:>12,.0f}円{mark}")

    # 取引一覧（損益上位15件）
    if trades:
        print(f"\n【取引一覧 損益上位15件】（全{stats['total']}件）")
        print(f"  {'銘柄':12} {'エントリー':12} {'決済':12} {'損益':>10}  理由")
        for t in sorted(trades, key=lambda x: x.pnl, reverse=True)[:15]:
            print(
                f"  {t.code4}({t.name[:5]})"
                f" {t.entry_date.strftime('%Y-%m-%d'):12}"
                f" {t.exit_date.strftime('%Y-%m-%d'):12}"
                f" {t.pnl:>+10,.0f}円  {t.reason}"
            )

    save_chart(stats, equity)


if __name__ == "__main__":
    main()
