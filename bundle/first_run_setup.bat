@echo off
REM ============================================================
REM Metabobarcoding - First Run Setup
REM Unpacks the conda environments from .tar.gz archives.
REM This runs automatically on first launch.
REM ============================================================

setlocal
set BUNDLE_ROOT=%~dp0

echo.
echo ============================================================
echo  Metabobarcoding - First Run Setup
echo  Unpacking Python environments (this may take 5-10 minutes)
echo  Please wait...
echo ============================================================
echo.

REM ---- Unpack torch_gpu3 ----
if not exist "%BUNDLE_ROOT%envs\torch_gpu3\python.exe" (
    echo [1/2] Unpacking torch_gpu3 environment...
    mkdir "%BUNDLE_ROOT%envs\torch_gpu3"
    tar -xzf "%BUNDLE_ROOT%envs\torch_gpu3.tar.gz" -C "%BUNDLE_ROOT%envs\torch_gpu3"
    if errorlevel 1 (
        echo ERROR: Failed to unpack torch_gpu3
        pause
        exit /b 1
    )
    REM Run conda-unpack to fix hardcoded paths
    call "%BUNDLE_ROOT%envs\torch_gpu3\Scripts\conda-unpack.exe"
    echo [1/2] torch_gpu3 ready.
) else (
    echo [1/2] torch_gpu3 already unpacked, skipping.
)

REM ---- Unpack mesmer ----
if not exist "%BUNDLE_ROOT%envs\mesmer\python.exe" (
    echo [2/2] Unpacking mesmer environment...
    mkdir "%BUNDLE_ROOT%envs\mesmer"
    tar -xzf "%BUNDLE_ROOT%envs\mesmer.tar.gz" -C "%BUNDLE_ROOT%envs\mesmer"
    if errorlevel 1 (
        echo ERROR: Failed to unpack mesmer
        pause
        exit /b 1
    )
    call "%BUNDLE_ROOT%envs\mesmer\Scripts\conda-unpack.exe"
    echo [2/2] mesmer ready.
) else (
    echo [2/2] mesmer already unpacked, skipping.
)

echo.
echo Setup complete!
echo.
