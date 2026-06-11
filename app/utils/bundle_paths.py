"""
Bundle path resolution.

Priority order for each path:
  1. Explicit environment variable override (MESMER_PYTHON, FIJI_PATH, DEEPCELL_CACHE_DIR)
  2. METABOBARCODING_ROOT-relative paths  (Windows bundle via launch.bat,
                                           or Docker via docker-compose.yml)
  3. Hard-coded developer machine fallbacks

Docker layout (set by docker-compose.yml):
  METABOBARCODING_ROOT=/bundle
    /bundle/envs/mesmer/bin/python3   — mesmer interpreter (Linux)
    /bundle/envs/mesmer/python.exe    — mesmer interpreter (Windows bundle)
    /bundle/deepcell_home/.deepcell   — model cache
  FIJI_PATH=/opt/Fiji.app/ImageJ-linux64  (Linux Docker)
"""

import os
import sys
from pathlib import Path


def _bundle_root() -> Path | None:
    root = os.environ.get("METABOBARCODING_ROOT", "").strip()
    return Path(root) if root else None


def get_mesmer_python() -> str:
    # Explicit override wins
    explicit = os.environ.get("MESMER_PYTHON", "").strip()
    if explicit:
        return explicit

    root = _bundle_root()
    if root:
        # Linux (Docker) layout
        linux_py = root / "envs" / "mesmer" / "bin" / "python3"
        if linux_py.exists():
            return str(linux_py)
        # Windows bundle layout
        win_py = root / "envs" / "mesmer" / "python.exe"
        if win_py.exists():
            return str(win_py)

    # Developer machine fallback
    if sys.platform == "win32":
        return r"C:\Users\eozturk7\AppData\Local\miniconda3\envs\mesmer\python.exe"
    return "/usr/bin/python3"


def get_fiji_exe() -> str:
    # Explicit override wins
    explicit = os.environ.get("FIJI_PATH", "").strip()
    if explicit:
        return explicit

    root = _bundle_root()
    if root:
        # Linux (Docker) layout
        linux_fiji = root / "Fiji.app" / "ImageJ-linux64"
        if linux_fiji.exists():
            return str(linux_fiji)
        # Windows bundle layout
        win_fiji = root / "Fiji.app" / "ImageJ-win64.exe"
        if win_fiji.exists():
            return str(win_fiji)

    # Developer machine fallback
    if sys.platform == "win32":
        return r"C:\Users\eozturk7\Desktop\Fiji.app\ImageJ-win64.exe"
    return "/opt/Fiji.app/ImageJ-linux64"


def get_deepcell_cache() -> str:
    """
    Returns the fake HOME directory containing the .deepcell model cache.
    Setting HOME/USERPROFILE to this path makes Path.home()/.deepcell
    resolve to the bundled/cached models — no token or download needed.
    """
    # Explicit override wins
    explicit = os.environ.get("DEEPCELL_CACHE_DIR", "").strip()
    if explicit:
        return explicit

    root = _bundle_root()
    if root:
        return str(root / "deepcell_home")

    # Developer machine — use actual home
    return str(Path.home())
