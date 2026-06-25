@echo off
setlocal enabledelayedexpansion
chcp 65001 >nul 2>&1
cd /d "%~dp0"
title VieNeu-TTS Launcher

:: ============================================================
::  VieNeu-TTS -- 1-Click Launcher (Windows)
::
::  - Menu chon che do: NVIDIA GPU / CPU (ONNX)
::  - Tu dong cai uv, tao venv, cai dependencies
::  - Kiem tra GPU, Python, cong mang
::  - Khoi chay Web UI + tu dong restart khi crash
::  - Ghi log loi vao thu muc logs/
:: ============================================================

set "LOG_DIR=logs"
set "RESTART_DELAY=5"
set "MAX_FAST_RESTARTS=5"
set "FAST_WINDOW=60"

:: Fix loi hf_transfer: torchtune (qua lmdeploy) tu dong bat
:: HF_HUB_ENABLE_HF_TRANSFER=1 nhung hf_transfer 0.1.9 bi loi tren Windows.
:: Set truoc de torchtune khong ghi de.
set "HF_HUB_ENABLE_HF_TRANSFER=0"

echo.
echo ============================================================
echo          VieNeu-TTS -- 1-Click Launcher
echo ============================================================
echo.

:: =============================================================
::  BUOC 1: Kiem tra Python
:: =============================================================
echo [1/6] Kiem tra Python...
where python >nul 2>&1
if errorlevel 1 (
    echo.
    echo   [X] KHONG TIM THAY PYTHON!
    echo.
    echo   Huong dan cai dat:
    echo     1. Vao https://www.python.org/downloads/
    echo     2. Tai Python 3.12 (ban "Windows installer 64-bit")
    echo     3. Khi cai, PHAI TICH CHON "Add Python to PATH"
    echo     4. Khoi dong lai may tinh
    echo     5. Chay lai file start.bat nay
    echo.
    pause
    exit /b 1
)
for /f "tokens=2 delims= " %%v in ('python --version 2^>^&1') do set "PY_VER=%%v"
echo   [OK] Python !PY_VER!

:: Kiem tra phien ban Python >= 3.10
for /f "tokens=1,2 delims=." %%a in ("!PY_VER!") do (
    set "PY_MAJOR=%%a"
    set "PY_MINOR=%%b"
)
if !PY_MAJOR! LSS 3 goto :PY_OLD
if !PY_MAJOR!==3 if !PY_MINOR! LSS 10 goto :PY_OLD
goto :PY_OK

:PY_OLD
echo   [X] Can Python 3.10 tro len. Hien tai: !PY_VER!
echo       Tai ban moi tu: https://www.python.org/downloads/
pause
exit /b 1

:PY_OK
echo.

:: =============================================================
::  BUOC 2: Kiem tra / Cai uv
:: =============================================================
echo [2/6] Kiem tra uv (package manager)...
where uv >nul 2>&1
if errorlevel 1 (
    echo   [!] Chua co uv. Dang tu dong cai dat...
    echo       (Khong can quyen Admin)
    echo.
    powershell -NoProfile -ExecutionPolicy Bypass -Command "irm https://astral.sh/uv/install.ps1 | iex"
    if errorlevel 1 (
        echo.
        echo   [!] Cach 1 that bai. Thu cach 2: pip install uv
        pip install uv >nul 2>&1
    )
    :: Refresh PATH sau khi cai uv
    for /f "tokens=*" %%p in ('powershell -NoProfile -Command "[System.Environment]::GetEnvironmentVariable('Path','User')"') do set "PATH=%%p;!PATH!"
    where uv >nul 2>&1
    if errorlevel 1 (
        echo.
        echo   [X] KHONG THE CAI UV!
        echo       Thu chay thu cong trong PowerShell:
        echo         irm https://astral.sh/uv/install.ps1 ^| iex
        echo       Sau do dong PowerShell va chay lai start.bat
        pause
        exit /b 1
    )
)
for /f "tokens=*" %%v in ('uv --version 2^>^&1') do echo   [OK] %%v
echo.

:: =============================================================
::  BUOC 3: MENU CHON CHE DO
:: =============================================================
echo [3/6] Chon che do chay:
echo.
echo   -------------------------------------------------------

:: Kiem tra NVIDIA GPU de hien thi thong tin
set "GPU_NAME=Khong tim thay"
set "HAS_NVIDIA=0"
where nvidia-smi >nul 2>&1
if not errorlevel 1 (
    nvidia-smi >nul 2>&1
    if not errorlevel 1 (
        set "HAS_NVIDIA=1"
        for /f "tokens=*" %%g in ('nvidia-smi --query-gpu=name --format^=csv^,noheader^,nounits 2^>nul') do set "GPU_NAME=%%g"
    )
)

if "!HAS_NVIDIA!"=="1" (
    echo   GPU phat hien: !GPU_NAME!
) else (
    echo   GPU phat hien: Khong co NVIDIA GPU
)
echo   -------------------------------------------------------
echo.
echo   [1] GPU - NVIDIA CUDA (can NVIDIA GPU + CUDA 12.8)
echo       Chay duoc TAT CA model: v3 Turbo, v2, v1
echo       Toc do: Nhanh nhat
echo.
echo   [2] CPU - ONNX (khong can GPU)
echo       Chi chay v3 Turbo (48kHz) -- van rat tot!
echo       Cai dat nhe, khong can CUDA
echo       Phu hop: may khong co GPU hoac GPU AMD/Intel
echo.

if "!HAS_NVIDIA!"=="1" (
    echo   * De xuat: Chon [1] vi may ban co NVIDIA GPU
) else (
    echo   * De xuat: Chon [2] vi may ban chua co NVIDIA GPU
)
echo.

:ASK_MODE
set "CHOICE="
set /p "CHOICE=  Nhap lua chon (1 hoac 2): "

if "!CHOICE!"=="1" (
    set "MODE=gpu"
    echo.
    echo   ^>^> Da chon: GPU (CUDA)
    if "!HAS_NVIDIA!"=="0" (
        echo.
        echo   [!] CANH BAO: Khong phat hien NVIDIA GPU!
        echo       Neu ban chac chan co GPU, kiem tra:
        echo         - Da cai NVIDIA Driver moi nhat
        echo         - Da cai CUDA Toolkit 12.8:
        echo           https://developer.nvidia.com/cuda-downloads
        echo.
        set /p "CONFIRM=      Van tiep tuc cai GPU? (y/n): "
        if /i not "!CONFIRM!"=="y" goto :ASK_MODE
    )
) else if "!CHOICE!"=="2" (
    set "MODE=cpu"
    echo.
    echo   ^>^> Da chon: CPU (ONNX)
) else (
    echo   [!] Vui long nhap 1 hoac 2.
    goto :ASK_MODE
)
echo.

:: =============================================================
::  BUOC 3b: Kiem tra CUDA Toolkit (GPU mode)
:: =============================================================
if "!MODE!"=="gpu" (
    echo [3b/6] Kiem tra NVIDIA CUDA Toolkit...
    echo.

    set "CUDA_OK=0"
    if defined CUDA_PATH (
        if exist "!CUDA_PATH!\bin\nvcc.exe" (
            set "CUDA_OK=1"
            for /f "tokens=5" %%v in ('"!CUDA_PATH!\bin\nvcc.exe" --version 2^>^&1 ^| findstr "release"') do (
                echo   [OK] CUDA Toolkit %%v
            )
        )
    )

    if "!CUDA_OK!"=="0" (
        echo   [!] Chua cai NVIDIA CUDA Toolkit!
        echo       LMDeploy can CUDA Toolkit de toi uu toc do cho v2/v1.
        echo       (Neu khong cai, model van chay duoc nhung CHAM hon)
        echo.
        echo   Ban muon tu dong cai CUDA Toolkit 12.8 khong?
        echo     [1] Co - Tu dong tai va cai CUDA Toolkit 12.8 (~3 GB, can Admin)
        echo     [2] Khong - Bo qua, chay voi backend cham hon
        echo.
        set "CUDA_CHOICE="
        set /p "CUDA_CHOICE=  Nhap lua chon (1 hoac 2): "

        if "!CUDA_CHOICE!"=="1" (
            echo.
            echo   Dang kiem tra winget...
            where winget >nul 2>&1
            if not errorlevel 1 (
                echo   Dang cai CUDA Toolkit 12.8 qua winget...
                echo   (Can quyen Admin - co the hien hop UAC)
                echo.
                winget install Nvidia.CUDA --version 12.8 --accept-package-agreements --accept-source-agreements
                if errorlevel 1 (
                    echo.
                    echo   [!] Winget that bai. Thu tai truc tiep...
                    goto :CUDA_MANUAL_DL
                )
                echo.
                echo   [OK] CUDA Toolkit da cai xong!
                echo   [!] BAN CAN KHOI DONG LAI MAY TINH de CUDA_PATH co hieu luc.
                echo       Sau khi restart, chay lai start.bat.
                echo.
                pause
                exit /b 0
            ) else (
                :CUDA_MANUAL_DL
                echo   Winget khong co san. Dang mo trang tai CUDA Toolkit...
                echo.
                start "" "https://developer.nvidia.com/cuda-12-8-0-download-archive"
                echo   [!] Trinh duyet da mo trang tai CUDA Toolkit.
                echo       1. Chon: Windows ^> x86_64 ^> 10 ^> exe (local)
                echo       2. Tai va cai dat
                echo       3. Khoi dong lai may tinh
                echo       4. Chay lai start.bat
                echo.
                pause
                exit /b 0
            )
        ) else (
            echo.
            echo   ^>^> Bo qua CUDA Toolkit. LMDeploy se khong hoat dong.
            echo      Model van chay duoc voi standard backend (cham hon).
        )
    )
    echo.
)

:: =============================================================
::  BUOC 4: Cai dat dependencies
:: =============================================================
echo [4/6] Kiem tra va cai dat dependencies...

set "NEED_SYNC=0"
if not exist ".venv\Scripts\python.exe" (
    set "NEED_SYNC=1"
    echo   [!] Chua co .venv -- se tao moi va cai dat.
) else (
    :: Kiem tra mode hien tai co khop khong
    if "!MODE!"=="gpu" (
        .venv\Scripts\python.exe -c "import torch" >nul 2>&1
        if errorlevel 1 (
            echo   [!] .venv hien tai khong co PyTorch. Can cai lai cho GPU.
            echo   Dang xoa .venv cu...
            rmdir /s /q .venv >nul 2>&1
            set "NEED_SYNC=1"
        ) else (
            echo   [OK] .venv san sang (GPU mode)
        )
    ) else (
        echo   [OK] .venv san sang (CPU mode)
    )
)

if "!NEED_SYNC!"=="1" (
    echo.
    if "!MODE!"=="gpu" (
        echo   Dang cai dat (GPU mode)...
        echo   Lenh: uv sync --group gpu
        echo   (Co the mat 5-15 phut tuy toc do mang)
        echo.
        uv sync --group gpu
    ) else (
        echo   Dang cai dat (CPU mode)...
        echo   Lenh: uv sync
        echo   (Co the mat 3-8 phut tuy toc do mang)
        echo.
        uv sync
    )
    if errorlevel 1 (
        echo.
        echo   [X] LOI KHI CAI DAT!
        echo.
        echo   Thu khac phuc:
        echo     1. Xoa thu muc .venv:  rmdir /s /q .venv
        echo     2. Chay lai start.bat
        echo.
        echo   Neu van loi, kiem tra:
        echo     - Ket noi mang
        echo     - Python version (can 3.10-3.13)
        echo     - Quyen ghi file trong thu muc nay
        pause
        exit /b 1
    )
    echo.
    echo   [OK] Cai dat hoan tat!
)
echo.

:: =============================================================
::  BUOC 5: Kiem tra nhanh truoc khi chay
:: =============================================================
echo [5/6] Kiem tra truoc khi chay...

:: Kiem tra Gradio import
.venv\Scripts\python.exe -c "import gradio; print(f'   [OK] Gradio {gradio.__version__}')" 2>nul
if errorlevel 1 (
    echo   [X] Khong import duoc Gradio. Thu xoa .venv va chay lai.
    pause
    exit /b 1
)

:: Kiem tra vieneu import
.venv\Scripts\python.exe -c "import vieneu; print('   [OK] vieneu')" 2>nul
if errorlevel 1 (
    echo   [X] Khong import duoc vieneu. Thu xoa .venv va chay lai.
    pause
    exit /b 1
)

:: GPU mode: kiem tra torch + CUDA
if "!MODE!"=="gpu" (
    .venv\Scripts\python.exe -c "import torch; g=torch.cuda.is_available(); n=torch.cuda.get_device_name(0) if g else 'N/A'; print(f'   [OK] PyTorch {torch.__version__} -- CUDA: {g} -- GPU: {n}')" 2>nul
    if errorlevel 1 (
        echo   [!] PyTorch co van de. Van thu khoi chay...
    )
)

:: Kiem tra cong 7860
netstat -ano 2>nul | findstr ":7860 " | findstr "LISTENING" >nul 2>&1
if not errorlevel 1 (
    echo   [!] Cong 7860 dang bi chiem. Gradio se dung cong khac.
) else (
    echo   [OK] Cong 7860 san sang
)
echo.

:: =============================================================
::  BUOC 6: Khoi chay + Auto-restart
:: =============================================================
echo [6/6] Khoi chay VieNeu-TTS Web UI...
echo.
if not exist "!LOG_DIR!" mkdir "!LOG_DIR!"

set "RESTART_COUNT=0"
set "FAST_COUNT=0"

:RUN_LOOP
set /a RESTART_COUNT+=1

:: Tao ten log file
for /f "tokens=2 delims==" %%I in ('wmic os get localdatetime /value 2^>nul ^| find "="') do set "DT=%%I"
set "TS=!DT:~0,8!-!DT:~8,6!"
set "LOG_FILE=!LOG_DIR!\vieneu-!TS!.log"

echo ============================================================
echo   [!TIME!] Lan chay #!RESTART_COUNT!  ^|  Mode: !MODE!
echo   Log loi: !LOG_FILE!
echo.
echo   Web UI se mo tai: http://127.0.0.1:7860
echo   Nhan Ctrl+C de dung.
echo ============================================================
echo.

:: Mo trinh duyet (chi lan dau)
if !RESTART_COUNT!==1 (
    start "" cmd /c "timeout /t 6 /nobreak >nul 2>&1 && start http://127.0.0.1:7860" >nul 2>&1
)

:: Chay -- hien output tren man hinh, ghi stderr vao log
uv run vieneu-web 2>"!LOG_FILE!"
set "EXIT_CODE=!ERRORLEVEL!"

echo.
echo   [!TIME!] VieNeu-TTS thoat voi ma: !EXIT_CODE!

:: Ctrl+C -> dung
if "!EXIT_CODE!"=="-1073741510" goto :STOP
if "!EXIT_CODE!"=="3221225786" goto :STOP
if "!EXIT_CODE!"=="0" goto :STOP

:: Hien thi loi tu log
echo.
echo   --- LOI GAN NHAT (tu log) ---
if exist "!LOG_FILE!" (
    powershell -NoProfile -Command "Get-Content '!LOG_FILE!' | Select-Object -Last 15"
)
echo   --- HET LOI ---
echo.

:: Crash loop protection
set /a FAST_COUNT+=1
if !FAST_COUNT! GEQ %MAX_FAST_RESTARTS% (
    echo   [!] Da crash !MAX_FAST_RESTARTS! lan lien tiep!
    echo   [!] Kiem tra log: !LOG_FILE!
    echo   [!] Cho %FAST_WINDOW% giay truoc khi thu lai...
    echo.
    timeout /t %FAST_WINDOW% /nobreak >nul
    set "FAST_COUNT=0"
)

echo   [+] Tu dong khoi dong lai sau %RESTART_DELAY% giay...
echo       (Nhan Ctrl+C de huy)
timeout /t %RESTART_DELAY% /nobreak >nul
goto :RUN_LOOP

:STOP
echo.
echo ============================================================
echo   Da dung VieNeu-TTS.
echo   Tong so lan chay: !RESTART_COUNT!
echo   Log cuoi: !LOG_FILE!
echo ============================================================
pause
endlocal
