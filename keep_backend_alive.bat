@echo off
setlocal enabledelayedexpansion
chcp 65001 >nul
cd /d "%~dp0"

title VieNeu-TTS Backend Keeper

set VENV_NAME=.xpu_venv
set PYTHON_PATH=%VENV_NAME%\Scripts\python.exe
set ENTRY=apps\gradio_xpu.py
set LOG_DIR=logs
set RESTART_DELAY=5
set MAX_FAST_RESTARTS=5
set FAST_WINDOW=60

if not exist "%PYTHON_PATH%" (
    echo [X] Khong tim thay %PYTHON_PATH%. Chay setup_xpu_uv.bat truoc.
    pause
    exit /b 1
)
if not exist "%ENTRY%" (
    echo [X] Khong tim thay %ENTRY%
    pause
    exit /b 1
)
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

set RESTART_COUNT=0
set FAST_COUNT=0
set WINDOW_START=%TIME%

:RUN_LOOP
set /a RESTART_COUNT+=1
for /f "tokens=2 delims==" %%I in ('wmic os get localdatetime /value ^| find "="') do set DT=%%I
set TS=%DT:~0,8%-%DT:~8,6%
set LOG_FILE=%LOG_DIR%\backend-%TS%.log

echo.
echo ==================================================
echo   [%TIME%] LAN CHAY #%RESTART_COUNT%
echo   Log: %LOG_FILE%
echo   Nhan Ctrl+C de dung han keeper nay.
echo ==================================================

"%PYTHON_PATH%" "%ENTRY%" 1>>"%LOG_FILE%" 2>&1
set EXIT_CODE=%ERRORLEVEL%

echo.
echo [%TIME%] Backend da thoat voi ma %EXIT_CODE%

:: Nguoi dung nhan Ctrl+C -> dung han
if %EXIT_CODE%==-1073741510 goto :STOP
if %EXIT_CODE%==3221225786 goto :STOP

:: Dem so lan restart nhanh (crash loop bao ve)
set /a FAST_COUNT+=1
if !FAST_COUNT! GEQ %MAX_FAST_RESTARTS% (
    echo [!] Da restart %MAX_FAST_RESTARTS% lan lien tiep. Dung %FAST_WINDOW%s de tranh crash loop...
    timeout /t %FAST_WINDOW% /nobreak >nul
    set FAST_COUNT=0
)

echo [+] Khoi dong lai sau %RESTART_DELAY% giay... (nhan Ctrl+C de huy)
timeout /t %RESTART_DELAY% /nobreak >nul
goto :RUN_LOOP

:STOP
echo.
echo ==================================================
echo   Da dung keeper. Tong so lan chay: %RESTART_COUNT%
echo ==================================================
pause
endlocal
