@echo off
REM ============================================================
REM Metabobarcoding - App Code Updater
REM
REM Run this after changing app code (no need to repack envs).
REM Much faster than build_bundle.bat — only copies app/ folder.
REM ============================================================

setlocal

set SCRIPT_DIR=%~dp0
set REPO_ROOT=%SCRIPT_DIR%..
set DIST=%REPO_ROOT%\dist\Metabobarcoding

if not exist "%DIST%" (
    echo ERROR: Bundle not found at %DIST%
    echo Run build_bundle.bat first.
    pause
    exit /b 1
)

echo Updating app code in bundle...
rmdir /s /q "%DIST%\app"
xcopy /E /I /Q "%REPO_ROOT%\app" "%DIST%\app"
xcopy /E /I /Q "%REPO_ROOT%\.streamlit" "%DIST%\.streamlit"

REM Copy deepcell cache if not already present
if not exist "%DIST%\deepcell_cache" (
    echo Copying Mesmer model cache...
    xcopy /E /I /Q "%USERPROFILE%\.deepcell" "%DIST%\deepcell_cache"
)

echo Done. App code updated in %DIST%
