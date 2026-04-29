"""
Superpixel segmentation (SLIC) on the MBP fluorescence channel.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, List, Optional

import numpy as np
import pandas as pd
import tifffile as tiff
from skimage.filters import gaussian
from skimage.segmentation import slic
from skimage.transform import resize


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalize01(img: np.ndarray) -> np.ndarray:
    img = img.astype(np.float32)
    mn, mx = img.min(), img.max()
    if mx > mn:
        return (img - mn) / (mx - mn + 1e-8)
    return np.zeros_like(img)


def _resize_nearest(img: np.ndarray, shape: tuple) -> np.ndarray:
    return resize(img, shape, order=0, preserve_range=True, anti_aliasing=False)


def _resize_linear(img: np.ndarray, shape: tuple) -> np.ndarray:
    return resize(img, shape, order=1, preserve_range=True, anti_aliasing=True)


def _relabel(label_img: np.ndarray, start: int = 1) -> np.ndarray:
    out = np.zeros_like(label_img, dtype=np.int32)
    for new_id, old_id in enumerate(np.unique(label_img[label_img > 0]), start=start):
        out[label_img == old_id] = new_id
    return out


# ---------------------------------------------------------------------------
# Superpixel stats
# ---------------------------------------------------------------------------

def compute_superpixel_stats(
    label_mask: np.ndarray,
    analysis_mask: np.ndarray,
    stack_chw: np.ndarray,
    channel_labels: List[str],
) -> pd.DataFrame:
    _, h, w = stack_chw.shape
    masked = np.where(analysis_mask, label_mask, 0).astype(np.int32)
    flat   = masked.ravel()
    max_id = int(masked.max())

    if max_id == 0:
        return pd.DataFrame()

    yy, xx = np.indices((h, w))
    ids = np.unique(masked)
    ids = ids[ids > 0]

    area = np.bincount(flat, minlength=max_id + 1).astype(np.int64)[ids]
    cx   = (np.bincount(flat, weights=xx.ravel().astype(np.float64), minlength=max_id + 1)[ids]
            / np.maximum(area, 1)).astype(np.float32)
    cy   = (np.bincount(flat, weights=yy.ravel().astype(np.float64), minlength=max_id + 1)[ids]
            / np.maximum(area, 1)).astype(np.float32)

    df = pd.DataFrame({
        "superpixel_id": ids.astype(np.int32),
        "area_px": area,
        "centroid_x": cx,
        "centroid_y": cy,
    })

    for c, name in enumerate(channel_labels):
        vals = np.where(analysis_mask, stack_chw[c], 0).ravel().astype(np.float64)
        sums = np.bincount(flat, weights=vals, minlength=max_id + 1)
        df[name] = (sums[ids] / np.maximum(area, 1)).astype(np.float32)

    return df


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_superpixel_segmentation(
    if_stack: np.ndarray,
    mbp_channel_idx: int,
    mbp_mask: np.ndarray,
    nuclear_binary: np.ndarray,
    projected_msi: np.ndarray,
    channel_labels: List[str],
    output_dir: Path,
    n_segments: int = 40_000,
    compactness: float = 0.15,
    gaussian_sigma: float = 1.0,
    downsample_factor: int = 2,
    status_cb: Optional[Callable] = None,
) -> dict:
    """
    Run SLIC superpixel segmentation on the MBP channel.
    Returns dict with label mask array and stats DataFrame.
    """
    def _s(msg):
        if status_cb:
            status_cb(msg)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Extract MBP channel
    if if_stack.shape[0] <= 64:
        mbp = if_stack[mbp_channel_idx].astype(np.float32)
    else:
        mbp = if_stack[:, :, mbp_channel_idx].astype(np.float32)

    mbp01 = _normalize01(mbp)
    if gaussian_sigma > 0:
        mbp01 = gaussian(mbp01, sigma=gaussian_sigma, preserve_range=True).astype(np.float32)

    h, w = mbp01.shape

    # Union mask for segmentation support
    union_mask = mbp_mask | nuclear_binary.astype(bool)
    # Final analysis mask: MBP only, no nuclei
    final_mask = mbp_mask & (~nuclear_binary.astype(bool))

    if final_mask.sum() == 0:
        raise ValueError("Final superpixel mask is empty — check MBP and nuclear masks.")

    # Crop to bounding box
    ys, xs = np.where(union_mask)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1

    mbp_crop   = mbp01[y0:y1, x0:x1]
    union_crop = union_mask[y0:y1, x0:x1]
    final_crop = final_mask[y0:y1, x0:x1]
    ch, cw     = mbp_crop.shape

    # Downsample for speed
    if downsample_factor > 1:
        sh = max(1, ch // downsample_factor)
        sw = max(1, cw // downsample_factor)
        _s(f"Downsampling crop {ch}×{cw} → {sh}×{sw} for SLIC…")
        mbp_small   = _resize_linear(mbp_crop, (sh, sw)).astype(np.float32)
        union_small = _resize_nearest(union_crop.astype(np.float32), (sh, sw)) > 0
    else:
        mbp_small, union_small = mbp_crop, union_crop

    target_n = max(10, int(n_segments * union_small.sum() / max(union_crop.sum(), 1)))
    _s(f"Running SLIC with ~{target_n:,} segments…")

    labels_small = slic(
        mbp_small,
        n_segments=target_n,
        compactness=compactness,
        sigma=0,
        mask=union_small,
        start_label=1,
        channel_axis=None,
    )
    labels_small = _relabel(labels_small)

    # Upsample back
    if downsample_factor > 1:
        _s("Upsampling labels to full resolution…")
        labels_crop = _resize_nearest(labels_small.astype(np.float32), (ch, cw)).astype(np.int32)
    else:
        labels_crop = labels_small.astype(np.int32)

    labels_crop[~union_crop] = 0
    labels_crop[~final_crop] = 0
    labels_crop = _relabel(labels_crop)

    labels_full = np.zeros((h, w), dtype=np.int32)
    labels_full[y0:y1, x0:x1] = labels_crop

    n_sp = int(labels_full.max())
    _s(f"Generated {n_sp:,} superpixels.")

    # Save label mask
    label_path = output_dir / "mbp_superpixels_label_mask.tif"
    tiff.imwrite(str(label_path), labels_full.astype(np.int32))

    # Compute stats
    _s("Computing per-superpixel mean intensities…")
    df = compute_superpixel_stats(labels_full, final_mask, projected_msi, channel_labels)
    csv_path = output_dir / "mbp_superpixels_mean_intensity_matrix.csv"
    df.to_csv(csv_path, index=False)
    _s(f"Saved superpixel stats → {csv_path.name}")

    return {
        "label_mask": labels_full,
        "label_mask_path": label_path,
        "stats_df": df,
        "stats_csv": csv_path,
        "n_superpixels": n_sp,
    }
