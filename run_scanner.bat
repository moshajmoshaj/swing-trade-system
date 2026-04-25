@echo off
chcp 65001 > nul
setlocal EnableDelayedExpansion

set PROJECT_DIR=C:\Users\moshaj\swing-trade-system
set VENV_PYTHON=%PROJECT_DIR%\.venv\Scripts\python.exe
set VENV_PIP=%PROJECT_DIR%\.venv\Scripts\pip.exe
set LOG_FILE=%PROJECT_DIR%\logs\scheduler_log.txt
set SCANNER=%PROJECT_DIR%\src\scanner.py
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
rem    for/f does NOT propagate child exit code, so run separately
%VENV_PYTHON% %CHECK_BIZDAY% > "!BIZDAY_TMP!" 2>&1
set BIZDAY_EXIT=!errorlevel!
set /p BIZDAY_MSG=<"!BIZDAY_TMP!"
del "!BIZDAY_TMP!" 2>nul

echo [!TS!] !BIZDAY_MSG! >> "%LOG_FILE%"

if !BIZDAY_EXIT! neq 0 (
    goto :end
)

rem --- run scanner ---
echo [!TS!] START scanner.py >> "%LOG_FILE%"

%VENV_PYTHON% "%SCANNER%" >> "%LOG_FILE%" 2>&1
set SCAN_EXIT=!errorlevel!

for /f "delims=" %%i in ('%VENV_PYTHON% %GET_TS%') do set TS_END=%%i

if !SCAN_EXIT! neq 0 (
    echo [!TS_END!] ERROR: scanner.py exited with code !SCAN_EXIT! >> "%LOG_FILE%"
) else (
    echo [!TS_END!] END: scanner.py finished successfully >> "%LOG_FILE%"
)

:end
echo ======================================== >> "%LOG_FILE%"
endlocal
exit /b 0