@echo off
chcp 65001 > nul
setlocal EnableDelayedExpansion

set PROJECT_DIR=C:\Users\moshaj\swing-trade-system
set VENV_PYTHON=%PROJECT_DIR%\.venv\Scripts\python.exe
set VENV_PIP=%PROJECT_DIR%\.venv\Scripts\pip.exe
set SCANNER=%PROJECT_DIR%\src\scanner.py
set AUTO_EXIT=%PROJECT_DIR%\src\auto_exit.py
set AUTO_ENTRY=%PROJECT_DIR%\src\auto_entry.py
set AUTO_REPORT=%PROJECT_DIR%\src\auto_report.py
set CHECK_BIZDAY=%PROJECT_DIR%\src\check_bizday.py
set BIZDAY_TMP=%PROJECT_DIR%\logs\_bizday.tmp

cd /d "%PROJECT_DIR%"

if not exist "%PROJECT_DIR%\logs" mkdir "%PROJECT_DIR%\logs"

rem --- install jpholiday if missing ---
%VENV_PYTHON% -c "import jpholiday" 2>nul
if !errorlevel! neq 0 (
    %VENV_PIP% install jpholiday > nul 2>&1
    if !errorlevel! neq 0 (
        echo jpholiday install failed
        goto :end
    )
)

rem --- business day check ---
%VENV_PYTHON% %CHECK_BIZDAY% > "!BIZDAY_TMP!" 2>&1
set BIZDAY_EXIT=!errorlevel!
del "!BIZDAY_TMP!" 2>nul

if !BIZDAY_EXIT! neq 0 (
    goto :end
)

rem --- Phase 5: kabu API 起動確認（LIVE_TRADING=true の場合のみ） ---
for /f "tokens=2 delims==" %%A in ('findstr /i "^LIVE_TRADING" "%PROJECT_DIR%\.env" 2^>nul') do set LIVE_TRADING=%%A
set LIVE_TRADING=!LIVE_TRADING: =!
if /i "!LIVE_TRADING!"=="true" (
    %VENV_PYTHON% -c "import requests; r=requests.get('http://localhost:18080/kabusapi/token', timeout=3); exit(0)" 2>nul
    if !errorlevel! neq 0 (
        echo [WARN] kabuステーション が起動していないか API に接続できません
        echo [WARN] LIVE_TRADING=true ですが実注文は発行されません
        %VENV_PYTHON% -c "from src.notifier import send_error_notify; send_error_notify('run_scanner.bat','kabuAPI未起動','kabuステーション(R)を起動してください')" 2>nul
    )
)

rem === 1. auto_exit.py - 保有中銘柄の決済判定 ===
%VENV_PYTHON% "%AUTO_EXIT%"

rem === 2. scanner.py - 新規シグナルスキャン ===
%VENV_PYTHON% "%SCANNER%"

rem === 3. auto_entry.py - エントリー記録 ===
%VENV_PYTHON% "%AUTO_ENTRY%"

rem === 4. auto_report.py - 月次集計更新 ===
%VENV_PYTHON% "%AUTO_REPORT%"

:end
endlocal
exit /b 0
