@echo off
chcp 65001 > nul
setlocal EnableDelayedExpansion

set PROJECT_DIR=C:\Users\moshaj\swing-trade-system
set VENV_PYTHON=%PROJECT_DIR%\.venv\Scripts\python.exe
set VENV_PIP=%PROJECT_DIR%\.venv\Scripts\pip.exe
set LOG_FILE=%PROJECT_DIR%\logs\scheduler_log.txt
set SCANNER=%PROJECT_DIR%\src\scanner.py
set AUTO_EXIT=%PROJECT_DIR%\src\auto_exit.py
set AUTO_ENTRY=%PROJECT_DIR%\src\auto_entry.py
set AUTO_REPORT=%PROJECT_DIR%\src\auto_report.py
set CHECK_BIZDAY=%PROJECT_DIR%\src\check_bizday.py
set GET_TS=%PROJECT_DIR%\src\get_ts.py
set BIZDAY_TMP=%PROJECT_DIR%\logs\_bizday.tmp

cd /d "%PROJECT_DIR%"

if not exist "%PROJECT_DIR%\logs" mkdir "%PROJECT_DIR%\logs"

rem --- timestamp ---
for /f "delims=" %%i in ('%VENV_PYTHON% %GET_TS%') do set TS=%%i

echo. >> "%LOG_FILE%"
echo ======================================== >> "%LOG_FILE%"
echo [!TS!] run_scanner.bat started >> "%LOG_FILE%"

rem --- install jpholiday if missing ---
%VENV_PYTHON% -c "import jpholiday" 2>nul
if !errorlevel! neq 0 (
    echo [!TS!] INFO: installing jpholiday >> "%LOG_FILE%"
    %VENV_PIP% install jpholiday >> "%LOG_FILE%" 2>&1
    if !errorlevel! neq 0 (
        echo [!TS!] ERROR: jpholiday install failed >> "%LOG_FILE%"
        goto :end
    )
    echo [!TS!] INFO: jpholiday installed >> "%LOG_FILE%"
)

rem --- business day check ---
%VENV_PYTHON% %CHECK_BIZDAY% > "!BIZDAY_TMP!" 2>&1
set BIZDAY_EXIT=!errorlevel!
set /p BIZDAY_MSG=<"!BIZDAY_TMP!"
del "!BIZDAY_TMP!" 2>nul

echo [!TS!] !BIZDAY_MSG! >> "%LOG_FILE%"

if !BIZDAY_EXIT! neq 0 (
    goto :end
)

rem ==========================================
rem  1. auto_exit.py - 保有中銘柄の決済判定
rem ==========================================
for /f "delims=" %%i in ('%VENV_PYTHON% %GET_TS%') do set TS=%%i
echo [!TS!] START auto_exit.py >> "%LOG_FILE%"

%VENV_PYTHON% "%AUTO_EXIT%" >> "%LOG_FILE%" 2>&1
set EXIT_CODE=!errorlevel!

for /f "delims=" %%i in ('%VENV_PYTHON% %GET_TS%') do set TS=%%i
if !EXIT_CODE! neq 0 (
    echo [!TS!] WARN: auto_exit.py exited with code !EXIT_CODE! - continuing >> "%LOG_FILE%"
) else (
    echo [!TS!] END: auto_exit.py finished >> "%LOG_FILE%"
)

rem ==========================================
rem  2. scanner.py - 新規シグナルスキャン
rem ==========================================
for /f "delims=" %%i in ('%VENV_PYTHON% %GET_TS%') do set TS=%%i
echo [!TS!] START scanner.py >> "%LOG_FILE%"

%VENV_PYTHON% "%SCANNER%" >> "%LOG_FILE%" 2>&1
set SCAN_CODE=!errorlevel!

for /f "delims=" %%i in ('%VENV_PYTHON% %GET_TS%') do set TS=%%i
if !SCAN_CODE! neq 0 (
    echo [!TS!] ERROR: scanner.py exited with code !SCAN_CODE! >> "%LOG_FILE%"
) else (
    echo [!TS!] END: scanner.py finished >> "%LOG_FILE%"
)

rem ==========================================
rem  3. auto_entry.py - エントリー記録
rem ==========================================
for /f "delims=" %%i in ('%VENV_PYTHON% %GET_TS%') do set TS=%%i
echo [!TS!] START auto_entry.py >> "%LOG_FILE%"

%VENV_PYTHON% "%AUTO_ENTRY%" >> "%LOG_FILE%" 2>&1
set ENTRY_CODE=!errorlevel!

for /f "delims=" %%i in ('%VENV_PYTHON% %GET_TS%') do set TS=%%i
if !ENTRY_CODE! neq 0 (
    echo [!TS!] WARN: auto_entry.py exited with code !ENTRY_CODE! - continuing >> "%LOG_FILE%"
) else (
    echo [!TS!] END: auto_entry.py finished >> "%LOG_FILE%"
)

rem ==========================================
rem  4. auto_report.py - 月次集計更新
rem ==========================================
for /f "delims=" %%i in ('%VENV_PYTHON% %GET_TS%') do set TS=%%i
echo [!TS!] START auto_report.py >> "%LOG_FILE%"

%VENV_PYTHON% "%AUTO_REPORT%" >> "%LOG_FILE%" 2>&1
set REPORT_CODE=!errorlevel!

for /f "delims=" %%i in ('%VENV_PYTHON% %GET_TS%') do set TS=%%i
if !REPORT_CODE! neq 0 (
    echo [!TS!] WARN: auto_report.py exited with code !REPORT_CODE! - continuing >> "%LOG_FILE%"
) else (
    echo [!TS!] END: auto_report.py finished >> "%LOG_FILE%"
)

:end
for /f "delims=" %%i in ('%VENV_PYTHON% %GET_TS%') do set TS=%%i
echo [!TS!] run_scanner.bat completed >> "%LOG_FILE%"
echo ======================================== >> "%LOG_FILE%"
endlocal
exit /b 0