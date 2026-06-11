============================================================
 Metabobarcoding
============================================================

QUICK START
-----------
1. Extract this folder anywhere on your computer
   (e.g. C:\Metabobarcoding or your Desktop)

2. Double-click launch.bat

3. The first time you run it, setup will take 5-10 minutes
   to unpack the Python environments. This only happens once.

4. Your browser will open automatically at http://localhost:8501

5. To stop the app, close the black command window.


REQUIREMENTS
------------
- Windows 10 or 11 (64-bit)
- At least 20 GB free disk space
- NVIDIA GPU recommended (app works without one, just slower)
- No internet connection required after first setup


FOLDER STRUCTURE
----------------
Metabobarcoding/
  launch.bat              <- Start the app (double-click this)
  first_run_setup.bat     <- Runs automatically on first launch
  app/                    <- Application code
  envs/
    torch_gpu3.tar.gz     <- Main Python environment (packed)
    mesmer.tar.gz         <- Segmentation environment (packed)
    torch_gpu3/           <- Unpacked on first run
    mesmer/               <- Unpacked on first run
  Fiji.app/               <- Fiji image analysis software
  .streamlit/             <- Streamlit configuration


TROUBLESHOOTING
---------------
- If the browser does not open automatically, go to:
    http://localhost:8501

- If you see "port already in use", another instance may be
  running. Close all command windows and try again.

- If setup fails, make sure you have enough disk space and
  that the folder path contains no special characters.

- For GPU support, ensure your NVIDIA drivers are up to date.
  Download from: https://www.nvidia.com/drivers


CONTACT
-------
Coskun Lab - Georgia Tech
