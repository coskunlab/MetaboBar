@echo off
REM ============================================================
REM MetaBar - Fix Napari
REM Installs napari 0.6.6 to match the developer environment.
REM ============================================================

setlocal
set BUNDLE_ROOT=%~dp0
set PYTHON=%BUNDLE_ROOT%python\python.exe

if not exist "%PYTHON%" (
    echo ERROR: Python not found. Make sure this file is in the MetaBar folder.
    pause
    exit /b 1
)

echo Removing old napari packages...
"%PYTHON%" -m pip uninstall -y napari napari-console napari-plugin-engine napari-svg napari-plugin-manager vispy superqt numba llvmlite PyQt5 PyQt6 PySide2 PySide6 qtpy magicgui >nul 2>&1

echo Installing PyQt5...
"%PYTHON%" -m pip install --quiet --no-warn-script-location "PyQt5==5.15.11" "PyQt5-Qt5==5.15.2" "PyQt5-sip"
if errorlevel 1 ( echo ERROR: PyQt5 failed. & pause & exit /b 1 )

echo Installing napari 0.6.6...
"%PYTHON%" -m pip install --quiet --no-warn-script-location "napari==0.6.6" "magicgui" "qtpy>=2.0"
if errorlevel 1 ( echo ERROR: napari failed. & pause & exit /b 1 )

echo.
echo Testing...
"%PYTHON%" -c "import napari; print('napari OK:', napari.__version__)"
if errorlevel 1 ( echo ERROR: napari not working. & pause & exit /b 1 )

"%PYTHON%" -c "from napari._qt.qt_event_loop import get_qapp; print('Qt OK')"
if errorlevel 1 ( echo ERROR: Qt not working. & pause & exit /b 1 )

echo.
echo Done! Launch MetaBar from your desktop shortcut.
pause
