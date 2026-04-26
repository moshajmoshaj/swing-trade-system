"""
src/notifier.py
Telegram Bot 通知ユーティリティ（requests）
TELEGRAM_BOT_TOKEN・TELEGRAM_CHAT_ID が未設定の場合はサイレントスキップ。
"""
import os
import requests
from dotenv import load_dotenv

load_dotenv()

_API_BASE = "https://api.telegram.org/bot{token}/sendMessage"


def send_notify(subject: str, body: str) -> bool:
    """
    Telegram Bot でメッセージを送信する。
    メッセージ形式: "{subject}\n{body}"
    Returns: True=送信成功, False=スキップまたは失敗
    """
    token   = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "").strip()
    if not token or not chat_id:
        return False

    text = f"{subject}\n{body}"
    try:
        resp = requests.post(
            _API_BASE.format(token=token),
            json={"chat_id": chat_id, "text": text},
            timeout=10,
        )
        return resp.status_code == 200
    except Exception:
        return False
