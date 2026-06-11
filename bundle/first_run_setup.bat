@echo off
REM ============================================================
REM MetaBar - First Run Setup
REM Sets up Python and installs all packages via pip.
REM Runs automatically on first launch. Takes 10-20 minutes.
REM Internet connection required.
REM ============================================================

setlocal
set BUNDLE_ROOT=%~dp0
set PYTHON_DIR=%BUNDLE_ROOT%python
set PYTHON=%PYTHON_DIR%\python.exe
set PIP_BOOTSTRAP=%BUNDLE_ROOT%get-pip.py
set DONE_FLAG=%BUNDLE_ROOT%envs\.setup_complete

echo.
echo ============================================================
echo  MetaBar - First Run Setup
echo  Please wait, this only happens once (10-20 minutes)...
echo ============================================================
echo.

REM ---- Skip if already done and Python works ----
if exist "%DONE_FLAG%" (
    if exist "%PYTHON%" (
        "%PYTHON%" -c "import streamlit" >nul 2>&1
        if not errorlevel 1 (
            echo Setup already complete. Starting app...
            goto :done
        )
    )
    echo Previous setup incomplete, restarting...
    del "%DONE_FLAG%" >nul 2>&1
)

REM ---- Step 1: Extract embedded Python ----
if exist "%PYTHON%" (
    echo [1/5] Python already extracted, skipping.
) else (
    if not exist "%BUNDLE_ROOT%python311.zip" (
        echo ERROR: python311.zip not found in bundle.
        pause
        exit /b 1
    )
    echo [1/5] Extracting Python...
    if exist "%PYTHON_DIR%" rmdir /s /q "%PYTHON_DIR%"
    mkdir "%PYTHON_DIR%"
    tar -xf "%BUNDLE_ROOT%python311.zip" -C "%PYTHON_DIR%"
    if errorlevel 1 (
        echo ERROR: Failed to extract Python.
        pause
        exit /b 1
    )
    echo [1/5] Python ready.
)

REM ---- Enable site-packages in embedded Python ----
REM The embedded python._pth file disables site by default — fix it
set PTH_FILE=%PYTHON_DIR%\python311._pth
if exist "%PTH_FILE%" (
    powershell -Command "(Get-Content '%PTH_FILE%') -replace '#import site','import site' | Set-Content '%PTH_FILE%'"
)

REM ---- Step 2: Install pip ----
"%PYTHON%" -c "import pip" >nul 2>&1
if errorlevel 1 (
    echo [2/5] Installing pip...
    if not exist "%PIP_BOOTSTRAP%" (
        echo Downloading get-pip.py...
        powershell -Command "Invoke-WebRequest -Uri 'https://bootstrap.pypa.io/get-pip.py' -OutFile '%PIP_BOOTSTRAP%'"
        if not exist "%PIP_BOOTSTRAP%" (
            echo ERROR: Could not download get-pip.py. Check internet connection.
            pause
            exit /b 1
        )
    )
    "%PYTHON%" "%PIP_BOOTSTRAP%" --quiet
    if errorlevel 1 ( echo ERROR: pip install failed. & pause & exit /b 1 )
    echo [2/5] pip ready.
) else (
    echo [2/5] pip already installed, skipping.
)

REM ---- Step 3: Install app dependencies ----
echo [3/5] Installing app dependencies (~10 min)...

REM Remove any partially installed packages that may have locked files
if exist "%PYTHON_DIR%\Lib\site-packages\PIL" rmdir /s /q "%PYTHON_DIR%\Lib\site-packages\PIL" >nul 2>&1
if exist "%PYTHON_DIR%\Lib\site-packages\numpy" rmdir /s /q "%PYTHON_DIR%\Lib\site-packages\numpy" >nul 2>&1

"%PYTHON%" -m pip install --quiet --no-warn-script-location numpy pandas scipy scikit-image scikit-learn matplotlib seaborn pillow tifffile opencv-python-headless pyimzML scikit-posthocs
if errorlevel 1 (
    echo Retrying core packages with force-reinstall...
    "%PYTHON%" -m pip install --quiet --no-warn-script-location --force-reinstall numpy pandas scipy scikit-image scikit-learn matplotlib seaborn pillow tifffile opencv-python-headless pyimzML scikit-posthocs
    if errorlevel 1 ( echo ERROR: Core packages failed. & pause & exit /b 1 )
)

"%PYTHON%" -m pip install --quiet --no-warn-script-location "streamlit==1.56.0" "plotly==6.5.2"
if errorlevel 1 ( echo ERROR: Streamlit failed. & pause & exit /b 1 )

"%PYTHON%" -m pip install --quiet --no-warn-script-location "umap-learn==0.5.9.post2" "scanpy==1.11.5" "anndata==0.12.9" "leidenalg==0.11.0" "igraph==1.0.0"
if errorlevel 1 ( echo ERROR: Clustering packages failed. & pause & exit /b 1 )

REM Napari + Qt backend — same version as developer environment
"%PYTHON%" -m pip install --quiet --no-warn-script-location "PyQt5==5.15.11" "PyQt5-Qt5==5.15.2" "PyQt5-sip" "qtpy>=2.0" "napari==0.6.6" "magicgui"
if errorlevel 1 ( echo WARNING: Napari install failed. Napari viewer will not work. )
echo [3/5] Core packages installed.

REM ---- Step 4: Install PyTorch ----
echo [4/5] Installing PyTorch (GPU first, CPU fallback)...
"%PYTHON%" -m pip install --quiet --extra-index-url https://download.pytorch.org/whl/cu126 "torch==2.9.1+cu126" "torchvision==0.24.1+cu126"
if errorlevel 1 (
    echo GPU version failed, installing CPU version...
    "%PYTHON%" -m pip install --quiet --extra-index-url https://download.pytorch.org/whl/cpu "torch==2.9.1+cpu" "torchvision==0.24.1+cpu"
    if errorlevel 1 ( echo ERROR: PyTorch failed. & pause & exit /b 1 )
)
"%PYTHON%" -m pip install --quiet "torch-geometric==2.7.0"
"%PYTHON%" -m pip install --quiet torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.9.1+cu126.html
if errorlevel 1 ( echo WARNING: PyG extras failed. GNN features may not work. )
echo [4/5] PyTorch installed.

REM ---- Step 5: Unpack mesmer env ----
if exist "%BUNDLE_ROOT%envs\mesmer\python.exe" (
    echo [5/5] Mesmer env already unpacked, skipping.
) else (
    if not exist "%BUNDLE_ROOT%envs\mesmer.tar.gz" (
        echo ERROR: mesmer.tar.gz not found.
        pause
        exit /b 1
    )
    echo [5/5] Unpacking mesmer environment...
    if exist "%BUNDLE_ROOT%envs\mesmer" rmdir /s /q "%BUNDLE_ROOT%envs\mesmer"
    mkdir "%BUNDLE_ROOT%envs\mesmer"
    tar -xzf "%BUNDLE_ROOT%envs\mesmer.tar.gz" -C "%BUNDLE_ROOT%envs\mesmer"
    if errorlevel 1 ( echo ERROR: Failed to unpack mesmer. & pause & exit /b 1 )
    call "%BUNDLE_ROOT%envs\mesmer\Scripts\conda-unpack.exe"
    echo [5/5] Mesmer ready.
)

REM ---- Verify ----
"%PYTHON%" -c "import streamlit, torch; print('OK')" >nul 2>&1
if errorlevel 1 (
    echo ERROR: Verification failed. Please check your internet connection and try again.
    del "%DONE_FLAG%" >nul 2>&1
    pause
    exit /b 1
)

if not exist "%BUNDLE_ROOT%envs" mkdir "%BUNDLE_ROOT%envs"
echo. > "%DONE_FLAG%"
echo.
echo ============================================================
echo  Setup complete! Launching MetaBar...
echo ============================================================
echo.

:done
