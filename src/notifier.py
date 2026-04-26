"""
src/notifier.py
ntfy.sh 通知ユーティリティ（requests）
NTFY_TOPIC が未設定の場合はサイレントスキップ。
"""
import os
from urllib.parse import quote
import requests
from dotenv import load_dotenv

load_dotenv()

_NTFY_BASE = "https://ntfy.sh/{topic}"


def send_notify(subject: str, body: str) -> bool:
    """
    ntfy.sh にプッシュ通知を送信する。
    Returns: True=送信成功, False=スキップまたは失敗
    """
    topic = os.getenv("NTFY_TOPIC", "").strip()
    if not topic:
        return False

    try:
        resp = requests.post(
            _NTFY_BASE.format(topic=topic),
            data=body.encode("utf-8"),
            headers={
                "Title": quote(subject),   # 日本語タイトルはURLエンコード必須
                "Content-Type": "text/plain; charset=utf-8",
            },
            timeout=10,
        )
        return resp.status_code == 200
    except Exception:
        return False
