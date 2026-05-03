# Phase 5 運用手順書
**作成: 2026-05-03**

---

## 1. 日次自動実行フロー（毎営業日 17:00）

| 順序 | スクリプト | 処理 | 所要時間 |
|------|-----------|------|---------|
| ① | `auto_exit.py` | 保有銘柄の決済判定・実売り注文 | ~30秒 |
| ② | `scanner.py` | A/C/D/E/F 全戦略シグナルスキャン | ~5分 |
| ③ | `auto_entry.py` | RSI降順・空き枠にエントリー・実買い注文 | ~30秒 |
| ④ | `auto_report.py` | 月次集計・日次通知 | ~10秒 |

確認先：`logs/scheduler_log.txt`

---

## 2. 障害対応フロー

### ケース A: ntfy通知が届かない

1. `logs/scheduler_log.txt` を確認 → 最終実行時刻をチェック
2. Windows タスクスケジューラを開き「スイングトレード」タスクの状態を確認
3. 手動実行: `run_scanner.bat` をダブルクリック
4. エラーがあれば `logs/scheduler_log.txt` 末尾を確認

---

### ケース B: スケジューラが止まっている

```powershell
# タスクを手動で開始（PowerShell）
Start-ScheduledTask -TaskName "スイングトレード"

# タスク状態確認
Get-ScheduledTask -TaskName "スイングトレード" | Select State
```

---

### ケース C: J-Quants API エラー（ConnectionError）

1. J-Quants ステータスページを確認
2. リフレッシュトークン期限切れの場合:
   ```
   # .env の JQUANTS_REFRESH_TOKEN を更新
   # J-Quants サイト → マイページ → APIキー再発行
   ```
3. 当日データが翌日（翌営業日の 10:00 頃）に再取得される場合は
   翌日の 17:00 自動実行を待つ

---

### ケース D: kabuステーション® が起動していない（Phase 5）

1. `setup_kabu_startup.ps1` を実行（タスクスケジューラ登録済みのはず）
2. または手動で kabuステーション® を起動
3. kabuステーション® → システム設定 → API → API利用: ON を確認
4. `python src/broker_client.py` で接続テスト

---

### ケース E: 実注文が通らない（Phase 5）

- **対応**: `LIVE_TRADING=true` で実注文失敗が出た場合、スクリプトは
  自動的にペーパートレード記録に切り替わる（注文失敗≠システム停止）
- ntfy でエラー通知が届く → `logs/scheduler_log.txt` でエラー詳細確認
- kabu API のエラーコードを確認: [kabuステーション API ドキュメント](https://kabu.com/company/lp/api.html)

---

### ケース F: Excel ファイル（paper_trade_log.xlsx）が開けない

エラー: `PermissionError: [Errno 13] Permission denied`

1. Excel でファイルを開いていないか確認して閉じる
2. スクリプトは try/except で保護済み → 次回実行時に自動リカバリ

---

### ケース G: 月次損失ストップが発動した

- `logs/monthly_stop.txt` に当月の年月（例: `2026-05`）が記録される
- **効果**: `auto_entry.py` が新規エントリーを全停止
- **翌月**: 自動的にファイルが削除され、エントリー再開
- **確認方法**:
  ```powershell
  Get-Content logs\monthly_stop.txt
  ```

---

## 3. 手動操作手順

### 手動でシグナルスキャン実行

```powershell
cd C:\Users\moshaj\swing-trade-system
python src/scanner.py
```

### 手動で月次レポート確認

```powershell
python src/auto_report.py
```

### 保有ポジション確認

Excel で `logs/paper_trade_log.xlsx` → 「取引記録」シートを開き
ステータス列が「保有中」の行を確認

---

## 4. Phase 5 移行チェック（2026-07-27）

Phase 5 移行前に必ず実施:

1. `docs/phase5_checklist.md` の全項目確認
2. `.env` 更新:
   ```
   LIVE_TRADING=true
   KABU_TRADE_PASSWORD=（取引パスワード）
   PHASE5_MAX_CAPITAL=500000
   ```
3. kabuステーション® 起動確認
4. `python src/broker_client.py` で接続・残高照会テスト
5. 候補銘柄更新:
   - `logs/final_candidates_v2.csv` → `logs/final_candidates.csv` にリネーム
   - `logs/strategy_e_candidates_v2.csv` → `logs/strategy_e_candidates.csv` にリネーム
6. Phase 5の1銘柄上限を20万→10万円に縮小する場合は
   `auto_entry.py` と `auto_exit.py` の `MAX_PER_STOCK` を変更

---

## 5. 緊急時・撤退判断

| 状況 | 対応 |
|------|------|
| 月次損失 > 運用資金の10% | 自動停止（monthly_stop.txt）。翌月まで待機。 |
| 年間累計損失 > 運用資金の20% | 手動で全ポジション決済。Phase 5を一時停止し原因分析。 |
| システム障害が3日以上継続 | 手動で保有ポジションの損切り設定を証券口座で直接設定。 |
| 大暴落（TOPIX -5%以上の単日急落）| BEAR レジームへの切替を待ち、新規エントリー自粛。 |

---

*本ドキュメントは Phase 5 移行時・重大な変更時に更新すること*
