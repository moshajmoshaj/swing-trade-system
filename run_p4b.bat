@echo off
chcp 65001 > nul
setlocal EnableDelayedExpansion

set PROJECT_DIR=C:\Users\moshaj\swing-trade-system
set VENV_PYTHON=%PROJECT_DIR%\.venv\Scripts\python.exe
set CHECK_BIZDAY=%PROJECT_DIR%\src\check_bizday.py

cd /d "%PROJECT_DIR%"

rem --- business day check ---
for /f "delims=" %%L in ('%VENV_PYTHON% %CHECK_BIZDAY% 2^>nul') do set BIZDAY_MSG=%%L
echo [%date% %time:~0,8%] [P4-2] !BIZDAY_MSG! >> "%PROJECT_DIR%\logs\scheduler_log_p4b.txt"

echo !BIZDAY_MSG! | findstr /i "SKIP" > nul 2>&1
if !errorlevel! equ 0 (
    goto :end
)

rem === 1. auto_exit_p4b.py ===
%VENV_PYTHON% "%PROJECT_DIR%\src\auto_exit_p4b.py"

rem === 2. scanner_p4b.py ===
%VENV_PYTHON% "%PROJECT_DIR%\src\scanner_p4b.py"

rem === 3. auto_entry_p4b.py ===
%VENV_PYTHON% "%PROJECT_DIR%\src\auto_entry_p4b.py"

rem === 4. auto_report_p4b.py ===
%VENV_PYTHON% "%PROJECT_DIR%\src\auto_report_p4b.py"

:end
endlocal
exit /b 0
