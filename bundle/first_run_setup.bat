@echo off
REM ============================================================
REM Metabobarcoding - First Run Setup
REM Unpacks the conda environments from .tar.gz archives.
REM This runs automatically on first launch from the bundle folder.
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
if exist "%BUNDLE_ROOT%envs\torch_gpu3\python.exe" (
    echo [1/2] torch_gpu3 already unpacked, skipping.
) else (
    if not exist "%BUNDLE_ROOT%envs\torch_gpu3.tar.gz" (
        echo ERROR: torch_gpu3.tar.gz not found at %BUNDLE_ROOT%envs\
        echo Make sure you extracted the full bundle zip before running.
        pause
        exit /b 1
    )
    echo [1/2] Unpacking torch_gpu3 environment...
    if exist "%BUNDLE_ROOT%envs\torch_gpu3" rmdir /s /q "%BUNDLE_ROOT%envs\torch_gpu3"
    mkdir "%BUNDLE_ROOT%envs\torch_gpu3"
    tar -xzf "%BUNDLE_ROOT%envs\torch_gpu3.tar.gz" -C "%BUNDLE_ROOT%envs\torch_gpu3"
    if errorlevel 1 (
        echo ERROR: Failed to unpack torch_gpu3
        pause
        exit /b 1
    )
    call "%BUNDLE_ROOT%envs\torch_gpu3\Scripts\conda-unpack.exe"
    echo [1/2] torch_gpu3 ready.
)

REM ---- Unpack mesmer ----
if exist "%BUNDLE_ROOT%envs\mesmer\python.exe" (
    echo [2/2] mesmer already unpacked, skipping.
) else (
    if not exist "%BUNDLE_ROOT%envs\mesmer.tar.gz" (
        echo ERROR: mesmer.tar.gz not found at %BUNDLE_ROOT%envs\
        pause
        exit /b 1
    )
    echo [2/2] Unpacking mesmer environment...
    if exist "%BUNDLE_ROOT%envs\mesmer" rmdir /s /q "%BUNDLE_ROOT%envs\mesmer"
    mkdir "%BUNDLE_ROOT%envs\mesmer"
    tar -xzf "%BUNDLE_ROOT%envs\mesmer.tar.gz" -C "%BUNDLE_ROOT%envs\mesmer"
    if errorlevel 1 (
        echo ERROR: Failed to unpack mesmer
        pause
        exit /b 1
    )
    call "%BUNDLE_ROOT%envs\mesmer\Scripts\conda-unpack.exe"
    echo [2/2] mesmer ready.
)

echo.
echo Setup complete!
echo.
