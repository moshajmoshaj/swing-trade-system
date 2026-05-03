"""
src/broker_client.py
kabuステーション® API クライアント（Phase 5 実取引用）

前提:
  - kabuステーション® が起動していること（localhost:18080 でAPIが動作）
  - .env に KABU_TRADE_PASSWORD が設定されていること
  - 特定口座（源泉徴収あり）で口座開設済みであること

使用方法:
  from broker_client import KabuClient
  client = KabuClient()
  client.authenticate()
  result = client.buy("7203", qty=100)  # トヨタを100株成行買い
"""
import os
import sys
sys.stdout.reconfigure(encoding="utf-8")

import requests
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# ── 設定 ─────────────────────────────────────────────────────
KABU_API_PORT    = int(os.getenv("KABU_API_PORT", "18080"))
BASE_URL         = f"http://localhost:{KABU_API_PORT}/kabusapi"
TRADE_PASSWORD   = os.getenv("KABU_TRADE_PASSWORD", "")

# 発注パラメータ定数（特定口座・東証・現物）
EXCHANGE         = 1   # 1=東証
SECURITY_TYPE    = 1   # 1=株式
ACCOUNT_TYPE     = 4   # 4=特定口座（2=一般, 12=NISA）
CASH_MARGIN      = 1   # 1=現物
FRONT_ORDER_TYPE = 10  # 10=成行（20=指値）
EXPIRE_DAY       = 0   # 0=当日限り


class KabuClientError(Exception):
    pass


class KabuClient:
    def __init__(self):
        self._token: str | None = None

    # ────────────────────────────────────────────────────────
    # 認証
    # ────────────────────────────────────────────────────────
    def authenticate(self) -> str:
        """トークン取得（kabuステーション起動後に1回実行）"""
        if not TRADE_PASSWORD:
            raise KabuClientError(".env に KABU_TRADE_PASSWORD が設定されていません")
        resp = requests.post(
            f"{BASE_URL}/token",
            json={"Password": TRADE_PASSWORD},
            timeout=10,
        )
        resp.raise_for_status()
        self._token = resp.json()["Token"]
        return self._token

    @property
    def _headers(self) -> dict:
        if not self._token:
            raise KabuClientError("authenticate() を先に呼び出してください")
        return {"X-API-KEY": self._token, "Content-Type": "application/json"}

    # ────────────────────────────────────────────────────────
    # 発注（現物・成行）
    # ────────────────────────────────────────────────────────
    def buy(self, code4: str, qty: int) -> dict:
        """現物成行買い。code4: 4桁銘柄コード（例: '7203'）"""
        return self._send_order(code4, qty, side="2",
                                deliv_type=2, fund_type="AA")

    def sell(self, code4: str, qty: int) -> dict:
        """現物成行売り。code4: 4桁銘柄コード（例: '7203'）"""
        return self._send_order(code4, qty, side="1",
                                deliv_type=0, fund_type="  ")

    def _send_order(self, code4: str, qty: int,
                    side: str, deliv_type: int, fund_type: str) -> dict:
        body = {
            "Password":       TRADE_PASSWORD,
            "Symbol":         code4,
            "Exchange":       EXCHANGE,
            "SecurityType":   SECURITY_TYPE,
            "Side":           side,
            "CashMargin":     CASH_MARGIN,
            "DelivType":      deliv_type,
            "AccountType":    ACCOUNT_TYPE,
            "Qty":            qty,
            "FrontOrderType": FRONT_ORDER_TYPE,
            "FundType":       fund_type,
            "Price":          0,
            "ExpireDay":      EXPIRE_DAY,
        }
        resp = requests.post(f"{BASE_URL}/sendorder",
                             json=body, headers=self._headers, timeout=10)
        resp.raise_for_status()
        result = resp.json()
        if result.get("Result") != 0:
            raise KabuClientError(f"発注失敗: {result}")
        return result

    # ────────────────────────────────────────────────────────
    # 保有残高
    # ────────────────────────────────────────────────────────
    def get_positions(self) -> list[dict]:
        """現物保有残高一覧を返す"""
        resp = requests.get(f"{BASE_URL}/positions",
                            params={"product": 1},   # 1=現物
                            headers=self._headers, timeout=10)
        resp.raise_for_status()
        return resp.json() or []

    def get_position_qty(self, code4: str) -> int:
        """指定銘柄の保有株数を返す（0 = 未保有）"""
        for pos in self.get_positions():
            if str(pos.get("Symbol")) == code4:
                return int(pos.get("LeavesQty", 0))
        return 0

    # ────────────────────────────────────────────────────────
    # 注文照会・取消
    # ────────────────────────────────────────────────────────
    def get_orders(self, active_only: bool = True) -> list[dict]:
        """注文一覧を返す"""
        params = {"product": 0}
        if active_only:
            params["uptime"] = "0"
        resp = requests.get(f"{BASE_URL}/orders",
                            params=params, headers=self._headers, timeout=10)
        resp.raise_for_status()
        return resp.json() or []

    def cancel_order(self, order_id: str) -> dict:
        """注文を取消する"""
        body = {"Password": TRADE_PASSWORD, "OrderId": order_id}
        resp = requests.put(f"{BASE_URL}/cancelorder",
                            json=body, headers=self._headers, timeout=10)
        resp.raise_for_status()
        return resp.json()

    # ────────────────────────────────────────────────────────
    # ユーティリティ
    # ────────────────────────────────────────────────────────
    @staticmethod
    def to_code4(code: str) -> str:
        """J-Quants 5桁コード → kabu API 4桁コードに変換"""
        c = str(code).strip()
        return c[:4] if len(c) == 5 and c.endswith("0") else c

    def is_available(self) -> bool:
        """kabuステーションが起動しているか簡易確認"""
        try:
            resp = requests.get(f"{BASE_URL}/board/1321",
                                headers=self._headers, timeout=3)
            return resp.status_code in (200, 400)
        except Exception:
            return False


# ────────────────────────────────────────────────────────────
# 動作確認（直接実行時）
# ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import json
    print("kabu API 接続テスト")
    client = KabuClient()
    try:
        token = client.authenticate()
        print(f"  認証成功: token={token[:8]}...")
        positions = client.get_positions()
        print(f"  保有銘柄数: {len(positions)}")
        for p in positions[:5]:
            print(f"    {p.get('Symbol')} {p.get('SymbolName')} "
                  f"x{p.get('LeavesQty')}株")
        print("テスト完了 - 発注は行いません")
    except KabuClientError as e:
        print(f"エラー: {e}")
    except requests.exceptions.ConnectionError:
        print(f"接続失敗: kabuステーション® が起動していないか、"
              f"ポート番号が異なります（現在: {KABU_API_PORT}）")
