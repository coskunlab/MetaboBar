@echo off
REM ============================================================
REM MetaBar - Launcher
REM Double-click this file to start the app.
REM ============================================================

setlocal

set BUNDLE_ROOT=%~dp0

REM ---- First-run setup (installs packages if needed) ----
call "%BUNDLE_ROOT%first_run_setup.bat"
if errorlevel 1 (
    echo Setup failed. Please check the error above.
    pause
    exit /b 1
)

REM ---- Set bundle root so app can find Fiji + mesmer ----
set METABOBARCODING_ROOT=%BUNDLE_ROOT%

REM ---- Use embedded Python ----
set PYTHON=%BUNDLE_ROOT%python\python.exe
set ENV=%BUNDLE_ROOT%python
set PATH=%ENV%;%ENV%\Scripts;%ENV%\Lib\site-packages\torch\lib;%PATH%

REM ---- Verify Python is present ----
if not exist "%PYTHON%" (
    echo ERROR: Python not found. Please reinstall MetaBar.
    pause
    exit /b 1
)

REM ---- Kill any existing Streamlit instance on port 8501 ----
for /f "tokens=5" %%a in ('netstat -aon ^| findstr ":8501 " 2^>nul') do (
    taskkill /F /PID %%a >nul 2>&1
)

REM ---- Start Streamlit ----
echo.
echo ============================================================
echo  Starting MetaBar...
echo  The app will open in your browser at http://localhost:8501
echo  Close this window to stop the app.
echo ============================================================
echo.

"%PYTHON%" -m streamlit run "%BUNDLE_ROOT%app\main.py" ^
    --server.headless true ^
    --server.port 8501 ^
    --server.maxUploadSize 5120 ^
    --browser.gatherUsageStats false

echo.
echo The app has stopped. See above for any errors.
pause
