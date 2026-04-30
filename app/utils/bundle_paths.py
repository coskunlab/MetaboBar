"""
Bundle path resolution.

When running from the portable bundle, the environment variable
METABOBARCODING_ROOT is set to the bundle root folder by launch.bat.

Fallback to the developer machine paths when the variable is not set.
"""

import os
from pathlib import Path


def _bundle_root() -> Path | None:
    root = os.environ.get("METABOBARCODING_ROOT", "").strip()
    return Path(root) if root else None


def get_mesmer_python() -> str:
    root = _bundle_root()
    if root:
        return str(root / "envs" / "mesmer" / "python.exe")
    # Developer fallback
    return r"C:\Users\eozturk7\AppData\Local\miniconda3\envs\mesmer\python.exe"


def get_fiji_exe() -> str:
    root = _bundle_root()
    if root:
        return str(root / "Fiji.app" / "ImageJ-win64.exe")
    # Developer fallback
    return r"C:\Users\eozturk7\Desktop\Fiji.app\ImageJ-win64.exe"
