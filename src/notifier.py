"""
src/notifier.py
Gmail 通知ユーティリティ（smtplib 標準ライブラリ）
GMAIL_ADDRESS・GMAIL_APP_PASSWORD が未設定の場合はサイレントスキップ。
"""
import os
import smtplib
from email.mime.text import MIMEText
from dotenv import load_dotenv

load_dotenv()

_SMTP_HOST = "smtp.gmail.com"
_SMTP_PORT = 587


def send_mail(subject: str, body: str) -> bool:
    """
    Gmail でメールを自分宛に送信する。
    Returns: True=送信成功, False=スキップまたは失敗
    """
    address  = os.getenv("GMAIL_ADDRESS", "").strip()
    password = os.getenv("GMAIL_APP_PASSWORD", "").strip()
    if not address or not password:
        return False

    msg = MIMEText(body, "plain", "utf-8")
    msg["Subject"] = subject
    msg["From"]    = address
    msg["To"]      = address

    try:
        with smtplib.SMTP(_SMTP_HOST, _SMTP_PORT) as smtp:
            smtp.ehlo()
            smtp.starttls()
            smtp.login(address, password)
            smtp.sendmail(address, [address], msg.as_string())
        return True
    except Exception:
        return False
