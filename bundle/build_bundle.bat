@echo off
REM ============================================================
REM Metabobarcoding - Bundle Builder
REM Run this ONCE on the developer machine to create the
REM portable bundle folder.
REM
REM Requirements:
REM   - conda-pack installed in base env
REM   - torch_gpu3 and mesmer conda envs exist
REM   - Fiji.app at C:\Users\eozturk7\Desktop\Fiji.app
REM
REM Output: ..\dist\Metabobarcoding\
REM ============================================================

setlocal

set SCRIPT_DIR=%~dp0
set REPO_ROOT=%SCRIPT_DIR%..
set DIST=%REPO_ROOT%\dist\Metabobarcoding

echo.
echo ============================================================
echo  Metabobarcoding Bundle Builder
echo ============================================================
echo  Output: %DIST%
echo.

REM ---- Clean previous build ----
if exist "%DIST%" (
    echo [1/6] Removing previous dist folder...
    rmdir /s /q "%DIST%"
)
mkdir "%DIST%"
mkdir "%DIST%\envs"

REM ---- Pack torch_gpu3 env ----
echo [2/6] Packing torch_gpu3 environment (this takes a few minutes)...
conda-pack -n torch_gpu3 -o "%DIST%\envs\torch_gpu3.tar.gz" --ignore-missing-files
if errorlevel 1 (
    echo ERROR: conda-pack failed for torch_gpu3
    exit /b 1
)

REM ---- Pack mesmer env ----
echo [3/6] Packing mesmer environment...
conda-pack -n mesmer -o "%DIST%\envs\mesmer.tar.gz" --ignore-missing-files
if errorlevel 1 (
    echo ERROR: conda-pack failed for mesmer
    exit /b 1
)

REM ---- Copy Fiji ----
echo [4/6] Copying Fiji.app...
xcopy /E /I /Q "C:\Users\eozturk7\Desktop\Fiji.app" "%DIST%\Fiji.app"
if errorlevel 1 (
    echo ERROR: Could not copy Fiji.app
    exit /b 1
)

REM ---- Copy app code ----
echo [5/6] Copying app code...
xcopy /E /I /Q "%REPO_ROOT%\app" "%DIST%\app"
xcopy /E /I /Q "%REPO_ROOT%\.streamlit" "%DIST%\.streamlit"

REM ---- Copy launchers ----
echo [6/6] Copying launchers...
copy "%SCRIPT_DIR%launch.bat" "%DIST%\launch.bat"
copy "%SCRIPT_DIR%first_run_setup.bat" "%DIST%\first_run_setup.bat"
copy "%SCRIPT_DIR%README.txt" "%DIST%\README.txt"

echo.
echo ============================================================
echo  Bundle created at: %DIST%
echo  Total size:
dir /s "%DIST%" | find "File(s)"
echo ============================================================
echo.
echo Next steps:
echo   1. Zip the dist\Metabobarcoding folder
echo   2. Distribute the zip to users
echo   3. Users extract and double-click launch.bat
echo.
pause
