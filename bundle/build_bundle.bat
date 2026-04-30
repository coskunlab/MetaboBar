@echo off
REM ============================================================
REM Metabobarcoding - Bundle Builder
REM ============================================================

setlocal enabledelayedexpansion

set SCRIPT_DIR=%~dp0
set REPO_ROOT=%SCRIPT_DIR%..
set DIST=%REPO_ROOT%\dist\Metabobarcoding
set CONDA_BASE=C:\Users\eozturk7\AppData\Local\miniconda3
set CONDA_PACK=%CONDA_BASE%\Scripts\conda-pack.exe

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
"%CONDA_PACK%" -n torch_gpu3 -o "%DIST%\envs\torch_gpu3.tar.gz" --ignore-missing-files
if not exist "%DIST%\envs\torch_gpu3.tar.gz" (
    echo ERROR: torch_gpu3.tar.gz was not created. Check conda-pack output above.
    pause
    exit /b 1
)
echo torch_gpu3 packed OK.

REM ---- Pack mesmer env ----
echo [3/6] Packing mesmer environment...
"%CONDA_PACK%" -n mesmer -o "%DIST%\envs\mesmer.tar.gz" --ignore-missing-files
if not exist "%DIST%\envs\mesmer.tar.gz" (
    echo ERROR: mesmer.tar.gz was not created. Check conda-pack output above.
    pause
    exit /b 1
)
echo mesmer packed OK.

REM ---- Copy Fiji ----
echo [4/6] Copying Fiji.app...
xcopy /E /I /Q "C:\Users\eozturk7\Desktop\Fiji.app" "%DIST%\Fiji.app"
echo Fiji copied.

REM ---- Copy Mesmer model cache ----
echo [4b] Copying Mesmer model cache...
xcopy /E /I /Q "%USERPROFILE%\.deepcell" "%DIST%\deepcell_cache"
echo Mesmer model cache copied.

REM ---- Copy app code ----
echo [5/6] Copying app code...
xcopy /E /I /Q "%REPO_ROOT%\app" "%DIST%\app"
xcopy /E /I /Q "%REPO_ROOT%\.streamlit" "%DIST%\.streamlit"
echo App code copied.

REM ---- Copy launchers ----
echo [6/6] Copying launchers...
copy "%SCRIPT_DIR%launch.bat" "%DIST%\launch.bat"
copy "%SCRIPT_DIR%first_run_setup.bat" "%DIST%\first_run_setup.bat"
copy "%SCRIPT_DIR%README.txt" "%DIST%\README.txt"

echo.
echo ============================================================
echo  Bundle created successfully at:
echo  %DIST%
echo ============================================================
echo.
echo Next steps:
echo   1. Zip the dist\Metabobarcoding folder
echo   2. Distribute the zip to users
echo   3. Users extract and double-click launch.bat
echo.
pause
