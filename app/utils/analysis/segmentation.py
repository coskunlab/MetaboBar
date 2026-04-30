"""
Cell segmentation via Mesmer (DeepCell).

Mesmer lives in a separate conda env ('mesmer') so we call it as a
subprocess using that env's Python interpreter.

The subprocess script is written to a temp file, executed, and the
output nuclear mask TIFF is read back.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path
from typing import Optional

import numpy as np
import tifffile as tiff

from app.utils.bundle_paths import get_mesmer_python, get_deepcell_cache

MESMER_PYTHON = get_mesmer_python()

# ---------------------------------------------------------------------------

_MESMER_SCRIPT = textwrap.dedent("""\
import sys, json, os
import numpy as np
import tifffile as tiff
from skimage.draw import polygon2mask

params = json.loads(sys.argv[1])

img_path      = params["img_path"]
out_mask_path = params["out_mask_path"]
out_binary_path = params["out_binary_path"]
if_channel_idx = int(params["if_channel_idx"])
tile_size      = int(params["tile_size"])
image_mpp      = float(params["image_mpp"])
use_area_filter = bool(params["use_area_filter"])
area_method    = params["area_method"]
robust_z_thresh = float(params["robust_z_thresh"])
iqr_multiplier  = float(params["iqr_multiplier"])
use_gpu         = bool(params["use_gpu"])
deepcell_cache  = params.get("deepcell_cache", "")

# ---- Point DeepCell to bundled model cache (no token needed) ----
if deepcell_cache:
    import os
    from pathlib import Path as _Path
    # Override HOME so Path.home() / ".deepcell" resolves to our bundle cache
    # This must happen BEFORE any deepcell import
    _fake_home = str(_Path(deepcell_cache).parent)
    os.environ["USERPROFILE"] = _fake_home
    os.environ["HOME"] = _fake_home
    # Dummy token in case the hash check fails and it tries to download
    if not os.environ.get("DEEPCELL_ACCESS_TOKEN"):
        os.environ["DEEPCELL_ACCESS_TOKEN"] = "bundled_offline"
if not use_gpu:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# ---- Load image ----
img_raw = tiff.imread(img_path).astype(np.float32)
if img_raw.ndim == 2:
    img = img_raw
elif img_raw.shape[0] < img_raw.shape[-1]:   # C x H x W
    img = img_raw[if_channel_idx]
else:                                          # H x W x C
    img = img_raw[:, :, if_channel_idx]

img = img - img.min()
if img.max() > 0:
    img = img / img.max()

# ---- Helpers ----
def compute_area_bounds(areas, method, robust_z_thresh, iqr_multiplier):
    areas = np.asarray(areas, dtype=np.float64)
    areas = areas[areas > 0]
    if len(areas) == 0:
        return None, None
    log_areas = np.log1p(areas)
    if method == "mad_log":
        med = np.median(log_areas)
        mad = np.median(np.abs(log_areas - med))
        if mad == 0:
            q1, q3 = np.percentile(log_areas, [25, 75])
            iqr = q3 - q1
            if iqr == 0:
                return float(areas.min()), float(areas.max())
            lower_log = q1 - 1.5 * iqr
            upper_log = q3 + 1.5 * iqr
        else:
            lower_log = med - (robust_z_thresh * mad / 0.6745)
            upper_log = med + (robust_z_thresh * mad / 0.6745)
    else:
        q1, q3 = np.percentile(log_areas, [25, 75])
        iqr = q3 - q1
        if iqr == 0:
            return float(areas.min()), float(areas.max())
        lower_log = q1 - iqr_multiplier * iqr
        upper_log = q3 + iqr_multiplier * iqr
    return float(max(0.0, np.expm1(lower_log))), float(max(0.0, np.expm1(upper_log)))

def filter_by_area(label_img, method, robust_z_thresh, iqr_multiplier):
    label_img = label_img.astype(np.int64)
    max_label = int(label_img.max())
    if max_label == 0:
        return label_img.astype(np.uint32), 0, 0
    counts = np.bincount(label_img.ravel(), minlength=max_label + 1)
    obj_labels = np.arange(1, max_label + 1)
    obj_areas  = counts[1:]
    valid = obj_areas > 0
    obj_labels = obj_labels[valid]
    obj_areas  = obj_areas[valid]
    lower, upper = compute_area_bounds(obj_areas, method, robust_z_thresh, iqr_multiplier)
    keep = (obj_areas >= lower) & (obj_areas <= upper)
    kept = obj_labels[keep]
    remap = np.zeros(max_label + 1, dtype=np.uint32)
    remap[kept] = np.arange(1, len(kept) + 1, dtype=np.uint32)
    return remap[label_img], int(len(kept)), int((~keep).sum())

def split_tiles(image, tile_size):
    H, W = image.shape
    tiles, coords, shapes = [], [], []
    for y in range(0, H, tile_size):
        for x in range(0, W, tile_size):
            tile = image[y:y+tile_size, x:x+tile_size]
            h, w = tile.shape
            padded = np.zeros((tile_size, tile_size), dtype=image.dtype)
            padded[:h, :w] = tile
            tiles.append(padded)
            coords.append((y, x))
            shapes.append((h, w))
    return tiles, coords, shapes

def stitch(masks, coords, shapes, img_shape):
    combined = np.zeros(img_shape, dtype=np.uint32)
    offset = 0
    for tile, (y, x), (h, w) in zip(masks, coords, shapes):
        tile = tile[:h, :w].astype(np.int32)
        uniq = np.unique(tile)
        uniq = uniq[uniq > 0]
        new_tile = np.zeros_like(tile, dtype=np.uint32)
        for v in uniq:
            offset += 1
            new_tile[tile == v] = offset
        combined[y:y+h, x:x+w] = new_tile
    return combined

# ---- Mesmer ----
from deepcell.applications import Mesmer
app = Mesmer()

tiles, coords, shapes = split_tiles(img, tile_size)
tile_masks = []
for i, tile in enumerate(tiles):
    h, w = tile.shape
    inp = np.concatenate([
        tile.reshape(1, h, w, 1),
        np.zeros((1, h, w, 1), dtype=np.float32)
    ], axis=-1)
    pred = app.predict(inp, image_mpp=image_mpp, compartment="nuclear")
    tile_masks.append(pred[0, :, :, 0].astype(np.uint32))
    # Print progress so the parent process can update a progress bar
    print(f"TILE_PROGRESS {i+1}/{len(tiles)}", flush=True)

combined = stitch(tile_masks, coords, shapes, img.shape)

if use_area_filter:
    combined, kept, removed = filter_by_area(
        combined, area_method, robust_z_thresh, iqr_multiplier
    )

binary = (combined > 0).astype(np.uint8)

tiff.imwrite(out_mask_path,   combined.astype(np.uint32))
tiff.imwrite(out_binary_path, binary)

print(json.dumps({
    "n_cells": int(combined.max()),
    "shape":   list(combined.shape),
}))
""")


# ---------------------------------------------------------------------------
# Public function
# ---------------------------------------------------------------------------

def _run_popen(
    python: str,
    script_path: str,
    params: dict,
    status_cb,
    progress_cb,
    timeout: int,
) -> None:
    """
    Launch the Mesmer script via Popen, read stdout line-by-line,
    and update progress after each tile.
    stderr is drained in a background thread to prevent pipe deadlock.
    Raises RuntimeError on non-zero exit.
    """
    import time
    import threading

    proc = subprocess.Popen(
        [python, script_path, json.dumps(params)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )

    # Drain stderr in a background thread so it never blocks stdout
    stderr_buf = []
    def _drain_stderr():
        for line in proc.stderr:
            stderr_buf.append(line)
    stderr_thread = threading.Thread(target=_drain_stderr, daemon=True)
    stderr_thread.start()

    deadline = time.time() + timeout
    result_info = {}

    for line in proc.stdout:
        if time.time() > deadline:
            proc.kill()
            raise RuntimeError(f"Mesmer timed out after {timeout}s.")

        line = line.rstrip()

        if line.startswith("TILE_PROGRESS"):
            try:
                parts = line.split()[-1].split("/")
                done, total = int(parts[0]), int(parts[1])
                frac = 0.20 + 0.65 * (done / total)
                msg = f"Segmenting tile {done}/{total}…"
                if progress_cb:
                    progress_cb(frac, msg)
                if status_cb:
                    status_cb(msg)
            except Exception:
                pass
        else:
            try:
                result_info = json.loads(line)
            except Exception:
                pass

    proc.stdout.close()
    stderr_thread.join(timeout=10)
    proc.wait()

    if proc.returncode != 0:
        stderr_out = "".join(stderr_buf)
        raise RuntimeError(
            f"Mesmer subprocess failed (exit {proc.returncode}).\n"
            f"STDERR:\n{stderr_out}"
        )

    if result_info and status_cb:
        status_cb(f"Segmentation complete — {result_info.get('n_cells', '?'):,} cells detected.")


def run_mesmer_segmentation(
    if_stack: np.ndarray,
    if_channel_idx: int,
    output_dir: Path,
    tile_size: int = 1024,
    image_mpp: float = 1.0,
    use_area_filter: bool = True,
    area_method: str = "mad_log",
    robust_z_thresh: float = 3.5,
    iqr_multiplier: float = 1.5,
    mesmer_python: str = MESMER_PYTHON,
    status_cb=None,
    progress_cb=None,
    timeout: int = 3600,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run Mesmer nuclear segmentation in the mesmer conda env.
    progress_cb(fraction: float, msg: str) is called after each tile.
    Returns (labeled_mask, binary_mask).
    """
    def _s(msg):
        if status_cb:
            status_cb(msg)

    def _p(frac, msg=""):
        if progress_cb:
            progress_cb(frac, msg)

    output_dir.mkdir(parents=True, exist_ok=True)

    img_path    = output_dir / "_seg_input.tif"
    mask_path   = output_dir / "nuclear_mask.tif"
    binary_path = output_dir / "nuclear_mask_binary.tif"

    _s("Writing IF image for Mesmer…")
    _p(0.05, "Writing IF image…")
    tiff.imwrite(str(img_path), if_stack)

    base_params = dict(
        img_path=str(img_path),
        out_mask_path=str(mask_path),
        out_binary_path=str(binary_path),
        if_channel_idx=if_channel_idx,
        tile_size=tile_size,
        image_mpp=image_mpp,
        use_area_filter=use_area_filter,
        area_method=area_method,
        robust_z_thresh=robust_z_thresh,
        iqr_multiplier=iqr_multiplier,
        use_gpu=True,
        deepcell_cache=get_deepcell_cache(),
    )

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, encoding="utf-8"
    ) as f:
        f.write(_MESMER_SCRIPT)
        script_path = f.name

    try:
        _s("Running Mesmer (GPU)…")
        _p(0.10, "Running Mesmer (GPU)…")
        try:
            _run_popen(mesmer_python, script_path, base_params,
                       status_cb, progress_cb, timeout)
        except RuntimeError as gpu_err:
            if "exit" in str(gpu_err).lower():
                _s("GPU failed, retrying without GPU…")
                _p(0.10, "GPU failed, retrying without GPU…")
                params_cpu = dict(base_params)
                params_cpu["use_gpu"] = False
                _run_popen(mesmer_python, script_path, params_cpu,
                           status_cb, progress_cb, timeout)
            else:
                raise

        _p(0.90, "Reading output masks…")
        if not mask_path.exists():
            raise RuntimeError(f"Mesmer finished but no mask was written to {mask_path}")

    finally:
        Path(script_path).unlink(missing_ok=True)

    labeled = tiff.imread(str(mask_path))
    binary  = tiff.imread(str(binary_path))
    _p(1.0, "Done.")
    return labeled, binary
