"""
営業日チェックスクリプト
土日・祝日: exit(1)  営業日: exit(0)
stdout に日付情報を出力する
"""
import sys
import jpholiday
from datetime import date

d = date.today()
weekday_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
w = weekday_names[d.weekday()]
holiday_name = jpholiday.is_holiday_name(d) or ""

if d.weekday() >= 5:
    reason = "Saturday" if d.weekday() == 5 else "Sunday"
    print(f"{d} ({w}) SKIP: {reason}")
    sys.exit(1)

if holiday_name:
    print(f"{d} ({w}) SKIP: {holiday_name}")
    sys.exit(1)

print(f"{d} ({w}) OK: business day")
sys.exit(0)
