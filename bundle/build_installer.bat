@echo off
REM ============================================================
REM Build the Metabobarcoding installer .exe
REM
REM Requirements:
REM   1. Run build_bundle.bat first (creates dist\Metabobarcoding\)
REM   2. Inno Setup 6 installed at default location
REM ============================================================

setlocal

set ISCC="C:\Program Files (x86)\Inno Setup 6\ISCC.exe"

if not exist %ISCC% (
    echo ERROR: Inno Setup not found at %ISCC%
    echo Download and install from: https://jrsoftware.org/isdl.php
    pause
    exit /b 1
)

set SCRIPT_DIR=%~dp0
set BUNDLE=%SCRIPT_DIR%..\dist\Metabobarcoding

if not exist "%BUNDLE%" (
    echo ERROR: Bundle not found at %BUNDLE%
    echo Run build_bundle.bat first.
    pause
    exit /b 1
)

echo Building installer...
%ISCC% "%SCRIPT_DIR%installer.iss"

if errorlevel 1 (
    echo ERROR: Installer build failed.
    pause
    exit /b 1
)

echo.
echo ============================================================
echo  Installer created: dist\Metabobarcoding_Setup.exe
echo  Distribute this single file to users.
echo ============================================================
pause
