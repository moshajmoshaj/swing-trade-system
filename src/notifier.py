"""
src/notifier.py
ntfy.sh 通知ユーティリティ（requests）
NTFY_TOPIC が未設定の場合はサイレントスキップ。

Title ヘッダーは ASCII 必須のため英語固定。
日本語タイトル（subject）は本文先頭に含める。
"""
import os
import time
import requests
from dotenv import load_dotenv

load_dotenv()

_NTFY_BASE   = "https://ntfy.sh/{topic}"
_TITLE_ASCII = "[ST] SwingTrade"


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


def send_error_notify(script: str, error_type: str, message: str) -> bool:
    """
    エラー発生時のフォーマット通知を送信する。
    subject: [ST] エラー発生
    body:    スクリプト・種別・内容
    """
    subject = "[ST] エラー発生"
    body = f"スクリプト：{script}\n種別：{error_type}\n内容：{message}"
    return send_notify(subject, body)


def call_with_retry(func, *args, max_retries: int = 3, wait_sec: float = 5.0, **kwargs):
    """
    API呼び出しをリトライ付きで実行する。
    OSError / TimeoutError / requests.RequestException を対象にリトライ。
    全て失敗した場合は ConnectionError を送出。
    """
    last_exc: Exception = RuntimeError("未実行")
    for attempt in range(1, max_retries + 1):
        try:
            return func(*args, **kwargs)
        except (OSError, TimeoutError, requests.exceptions.RequestException) as e:
            last_exc = e
            if attempt < max_retries:
                time.sleep(wait_sec)
    raise ConnectionError(
        f"{max_retries}回リトライ後も接続失敗: {last_exc}"
    ) from last_exc
