@echo off
REM ============================================================
REM Metabobarcoding - Launcher
REM Double-click this file to start the app.
REM ============================================================

setlocal

set BUNDLE_ROOT=%~dp0

REM ---- First-run setup (unpacks envs if needed) ----
call "%BUNDLE_ROOT%first_run_setup.bat"
if errorlevel 1 (
    echo Setup failed. Please check the error above.
    pause
    exit /b 1
)

REM ---- Set bundle root so app can find Fiji + mesmer ----
set METABOBARCODING_ROOT=%BUNDLE_ROOT%

REM ---- Activate torch_gpu3 env ----
set PYTHON=%BUNDLE_ROOT%envs\torch_gpu3\python.exe
set PATH=%BUNDLE_ROOT%envs\torch_gpu3;%BUNDLE_ROOT%envs\torch_gpu3\Scripts;%BUNDLE_ROOT%envs\torch_gpu3\Library\bin;%PATH%

REM ---- Start Streamlit ----
echo.
echo ============================================================
echo  Starting Metabobarcoding...
echo  The app will open in your browser at http://localhost:8501
echo  Close this window to stop the app.
echo ============================================================
echo.

"%PYTHON%" -m streamlit run "%BUNDLE_ROOT%app\main.py" ^
    --server.headless true ^
    --server.port 8501 ^
    --browser.gatherUsageStats false

REM If Streamlit exits, keep window open so user can see any errors
if errorlevel 1 (
    echo.
    echo The app exited with an error. See above for details.
    pause
)
