"""
File I/O utilities: saving uploads, reading TIFFs, parsing label files.
"""

import os
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import streamlit as st
import tifffile as tiff

MB = 1024 * 1024
CHUNK_SIZE = 64 * MB


# ---------------------------------------------------------------------------
# General helpers
# ---------------------------------------------------------------------------

def safe_name(name: str) -> str:
    name = os.path.basename(str(name))
    name = name.replace("\x00", "")
    return name or "uploaded_file"


def format_size(num_bytes: Optional[int]) -> str:
    if num_bytes is None:
        return "unknown"
    gb = num_bytes / (1024 ** 3)
    mb = num_bytes / (1024 ** 2)
    if gb >= 1:
        return f"{gb:.2f} GB"
    return f"{mb:.1f} MB"


def save_uploaded_file_chunked(uploaded_file, output_path: Path) -> int:
    """
    Save a Streamlit UploadedFile to disk in chunks.
    Avoids getvalue()/getbuffer(), which would duplicate huge files in memory.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    uploaded_file.seek(0)

    total_written = 0
    with open(output_path, "wb") as f:
        while True:
            chunk = uploaded_file.read(CHUNK_SIZE)
            if not chunk:
                break
            f.write(chunk)
            total_written += len(chunk)

    uploaded_file.seek(0)
    return total_written


# ---------------------------------------------------------------------------
# Label parsing
# ---------------------------------------------------------------------------

def parse_label_file(uploaded_label_file) -> Optional[List[str]]:
    """
    Accepts simple line labels or two-column files like:
      C1<TAB>115In_IFNy.ome.tiff
      C2<TAB>127I_127I.ome.tiff

    Returns the display label after the first tab when present.
    """
    if uploaded_label_file is None:
        return None

    raw = uploaded_label_file.read()
    uploaded_label_file.seek(0)

    text = raw.decode("utf-8", errors="replace")
    labels = []

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue

        if "\t" in line:
            parts = line.split("\t")
            label = parts[-1].strip()
        elif "," in line and line.split(",", 1)[0].strip().lower().startswith("c"):
            label = line.split(",", 1)[-1].strip()
        else:
            label = line

        labels.append(label)

    return labels or None


def labels_from_tiff_metadata(tif: tiff.TiffFile, n_channels: int) -> List[str]:
    labels = None

    try:
        meta = tif.imagej_metadata or {}
        if isinstance(meta, dict):
            for key in ["Labels", "labels", "Channel", "Channels"]:
                if key in meta and meta[key] is not None:
                    labels = (
                        list(meta[key])
                        if not isinstance(meta[key], str)
                        else [meta[key]]
                    )
                    break
    except Exception:
        labels = None

    if labels is None or len(labels) != n_channels:
        labels = [f"C{i + 1}" for i in range(n_channels)]

    return [str(x) for x in labels]


# ---------------------------------------------------------------------------
# TIFF reading
# ---------------------------------------------------------------------------

def standardize_stack_shape(arr: np.ndarray) -> np.ndarray:
    """
    Return stack as C x H x W.

    Handles common TIFF shapes:
      H x W                 -> 1 x H x W
      C x H x W             -> C x H x W
      H x W x C             -> C x H x W when C is small
      T/Z/C x H x W-like    -> first non-spatial axes collapsed into channels
    """
    arr = np.asarray(arr)

    if arr.ndim == 2:
        return arr[None, :, :]

    if arr.ndim == 3:
        # HWC RGB / small-channel case
        if arr.shape[-1] <= 64 and arr.shape[0] > 64 and arr.shape[1] > 64:
            return np.moveaxis(arr, -1, 0)
        return arr

    if arr.ndim == 4:
        H, W = arr.shape[-2], arr.shape[-1]
        return arr.reshape((-1, H, W))

    if arr.ndim == 5:
        H, W = arr.shape[-2], arr.shape[-1]
        return arr.reshape((-1, H, W))

    raise ValueError(f"Unsupported image shape: {arr.shape}")


def read_tiff_stack(path: Path, label_file=None) -> Tuple[np.ndarray, List[str]]:
    with tiff.TiffFile(str(path)) as tif:
        arr = tif.asarray()
        stack = standardize_stack_shape(arr)
        metadata_labels = labels_from_tiff_metadata(tif, stack.shape[0])

    user_labels = parse_label_file(label_file)

    if user_labels is not None and len(user_labels) == stack.shape[0]:
        labels = user_labels
    elif user_labels is not None and len(user_labels) != stack.shape[0]:
        st.warning(
            f"Uploaded label file has {len(user_labels)} labels, "
            f"but the image has {stack.shape[0]} channels. "
            "Using TIFF/default labels instead."
        )
        labels = metadata_labels
    else:
        labels = metadata_labels

    return stack, labels
