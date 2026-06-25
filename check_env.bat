@echo off
setlocal enabledelayedexpansion
chcp 65001 >nul
cd /d "%~dp0"

echo ==================================================
echo        KIEM TRA MOI TRUONG VieNeu-TTS (XPU)
echo ==================================================
echo.

set VENV_NAME=.xpu_venv
set PYTHON_PATH=%VENV_NAME%\Scripts\python.exe
set FAIL=0

:: --- 1. Kiem tra Python he thong ---
echo [1/8] Kiem tra Python he thong...
where python >nul 2>&1
if errorlevel 1 (
    echo     [X] Khong tim thay Python tren PATH. Cai Python 3.12 truoc.
    set /a FAIL+=1
) else (
    for /f "tokens=*" %%v in ('python --version 2^>^&1') do echo     [OK] %%v
)
echo.

:: --- 2. Kiem tra uv ---
echo [2/8] Kiem tra uv (package manager)...
where uv >nul 2>&1
if errorlevel 1 (
    echo     [X] Khong tim thay uv. Chay: pip install uv
    set /a FAIL+=1
) else (
    for /f "tokens=*" %%v in ('uv --version 2^>^&1') do echo     [OK] %%v
)
echo.

:: --- 3. Kiem tra venv ---
echo [3/8] Kiem tra venv %VENV_NAME%...
if not exist "%PYTHON_PATH%" (
    echo     [X] Chua co venv. Chay: setup_xpu_uv.bat
    set /a FAIL+=1
    goto :SKIP_PY_CHECKS
) else (
    echo     [OK] Tim thay %PYTHON_PATH%
)
echo.

:: --- 4. Kiem tra file cau hinh ---
echo [4/8] Kiem tra requirements_xpu.txt...
if not exist requirements_xpu.txt (
    echo     [X] Thieu requirements_xpu.txt
    set /a FAIL+=1
) else (
    echo     [OK] Co requirements_xpu.txt
)
echo.

:: --- 5. Kiem tra entry point ---
echo [5/8] Kiem tra apps\gradio_xpu.py...
if not exist apps\gradio_xpu.py (
    echo     [X] Thieu apps\gradio_xpu.py
    set /a FAIL+=1
) else (
    echo     [OK] Co apps\gradio_xpu.py
)
echo.

:: --- 6. Kiem tra Torch + XPU ---
echo [6/8] Kiem tra Torch / XPU...
> check_tmp.py echo import sys
>> check_tmp.py echo try:
>> check_tmp.py echo     import torch
>> check_tmp.py echo     print(f'    [OK] torch {torch.__version__}')
>> check_tmp.py echo     xpu = hasattr(torch, 'xpu') and torch.xpu.is_available()
>> check_tmp.py echo     if xpu:
>> check_tmp.py echo         print(f'    [OK] XPU available: {torch.xpu.get_device_name(0)}')
>> check_tmp.py echo     else:
>> check_tmp.py echo         print('    [!] XPU khong san sang (chi chay CPU)')
>> check_tmp.py echo except Exception as e:
>> check_tmp.py echo     print(f'    [X] Loi torch: {e}'); sys.exit(1)
"%PYTHON_PATH%" check_tmp.py
if errorlevel 1 set /a FAIL+=1
del check_tmp.py >nul 2>&1
echo.

:: --- 7. Kiem tra cac thu vien chinh ---
echo [7/8] Kiem tra thu vien chinh...
> check_tmp.py echo mods = ['gradio','transformers','accelerate','torchaudio','torchvision','numpy','soundfile']
>> check_tmp.py echo missing = []
>> check_tmp.py echo for m in mods:
>> check_tmp.py echo     try:
>> check_tmp.py echo         __import__(m); print(f'    [OK] {m}')
>> check_tmp.py echo     except Exception as e:
>> check_tmp.py echo         print(f'    [X] {m}: {e}'); missing.append(m)
>> check_tmp.py echo import sys; sys.exit(1 if missing else 0)
"%PYTHON_PATH%" check_tmp.py
if errorlevel 1 set /a FAIL+=1
del check_tmp.py >nul 2>&1
echo.

:: --- 8. Kiem tra cong 7860 (Gradio default) ---
echo [8/8] Kiem tra cong 7860 (Gradio)...
netstat -ano | findstr ":7860" >nul 2>&1
if errorlevel 1 (
    echo     [OK] Cong 7860 dang ranh
) else (
    echo     [!] Cong 7860 dang co tien trinh khac chiem
)
echo.

:SKIP_PY_CHECKS
echo ==================================================
if %FAIL%==0 (
    echo   KET QUA: MOI THANH PHAN SAN SANG ^(OK^)
    echo   Co the chay: keep_backend_alive.bat
) else (
    echo   KET QUA: CO %FAIL% LOI - VUI LONG KHAC PHUC TRUOC
)
echo ==================================================
pause
endlocal
