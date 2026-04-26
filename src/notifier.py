"""
src/notifier.py
LINE Notify 送信ユーティリティ
LINE_NOTIFY_TOKEN が未設定の場合はサイレントスキップ。
"""
import os
import requests
from dotenv import load_dotenv

load_dotenv()

_URL = "https://notify-api.line.me/api/notify"
_LIMIT = 140


def send_line(message: str) -> bool:
    """
    LINE Notify にメッセージを送信する。
    Returns: True=送信成功, False=スキップまたは失敗
    """
    token = os.getenv("LINE_NOTIFY_TOKEN", "").strip()
    if not token:
        return False

    if len(message) > _LIMIT:
        message = message[:_LIMIT - 1] + "…"

    try:
        resp = requests.post(
            _URL,
            headers={"Authorization": f"Bearer {token}"},
            data={"message": message},
            timeout=10,
        )
        return resp.status_code == 200
    except Exception:
        return False
