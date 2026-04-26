"""
src/notifier.py
ntfy.sh 通知ユーティリティ（requests）
NTFY_TOPIC が未設定の場合はサイレントスキップ。

Title ヘッダーは ASCII 必須のため英語固定。
日本語タイトル（subject）は本文先頭に含める。
"""
import os
import requests
from dotenv import load_dotenv

load_dotenv()

_NTFY_BASE  = "https://ntfy.sh/{topic}"
_TITLE_ASCII = "[ST] SwingTrade"   # Title ヘッダー用 ASCII 文字列


def send_notify(subject: str, body: str) -> bool:
    """
    ntfy.sh にプッシュ通知を送信する。
    subject は本文先頭に表示し、Title ヘッダーには ASCII 文字列を使用。
    Returns: True=送信成功, False=スキップまたは失敗
    """
    topic = os.getenv("NTFY_TOPIC", "").strip()
    if not topic:
        return False

    full_body = f"{subject}\n{body}"

    try:
        resp = requests.post(
            _NTFY_BASE.format(topic=topic),
            data=full_body.encode("utf-8"),
            headers={
                "Title": _TITLE_ASCII,
                "Content-Type": "text/plain; charset=utf-8",
            },
            timeout=10,
        )
        return resp.status_code == 200
    except Exception:
        return False
