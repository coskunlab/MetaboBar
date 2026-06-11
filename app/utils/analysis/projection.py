"""
Weighted Gaussian LR→HR projection of the registered MSI stack.
Produces projected_stack_all_channels__full_hr__gaussian.tif
and per-cell metabolic tables.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, List, Optional

import numpy as np
import pandas as pd
import tifffile as tiff


# ---------------------------------------------------------------------------
# Projection weight precomputation
# ---------------------------------------------------------------------------

def _gaussian_1d_weights(
    n_hr: int, n_lr: int, sigma_lr: float, radius_lr: int
) -> tuple[np.ndarray, np.ndarray]:
    coords = ((np.arange(n_hr, dtype=np.float64) + 0.5) * n_lr / n_hr) - 0.5
    offsets = np.arange(-radius_lr, radius_lr + 1, dtype=np.int64)
    K = len(offsets)
    base = np.floor(coords).astype(np.int64)[:, None]
    idx_raw = np.clip(base + offsets[None, :], 0, n_lr - 1)
    d = coords[:, None] - idx_raw.astype(np.float64)
    w_raw = np.exp(-(d ** 2) / (2.0 * sigma_lr ** 2))

    idx_out = np.zeros((n_hr, K), dtype=np.int64)
    w_out   = np.zeros((n_hr, K), dtype=np.float64)

    for i in range(n_hr):
        inds, vals = idx_raw[i], w_raw[i]
        uniq_i, uniq_v = [], []
        for j, v in zip(inds, vals):
            found = False
            for t in range(len(uniq_i)):
                if uniq_i[t] == j:
                    uniq_v[t] += v
                    found = True
                    break
            if not found:
                uniq_i.append(int(j))
                uniq_v.append(float(v))
        uniq_v_arr = np.asarray(uniq_v)
        uniq_v_arr /= max(uniq_v_arr.sum(), 1e-12)
        nk = len(uniq_i)
        idx_out[i, :nk] = uniq_i
        w_out[i, :nk]   = uniq_v_arr

    return idx_out.astype(np.int64), w_out.astype(np.float32)


def precompute_projection_weights(
    hr_shape: tuple[int, int],
    lr_shape: tuple[int, int],
    sigma_lr: float = 0.75,
    radius_lr: int = 2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    hr_h, hr_w = hr_shape
    lr_h, lr_w = lr_shape
    y_idx, y_w = _gaussian_1d_weights(hr_h, lr_h, sigma_lr, radius_lr)
    x_idx, x_w = _gaussian_1d_weights(hr_w, lr_w, sigma_lr, radius_lr)
    return y_idx, y_w, x_idx, x_w


# ---------------------------------------------------------------------------
# Channel projection
# ---------------------------------------------------------------------------

def _project_channel(
    src: np.ndarray,
    y_idx: np.ndarray, y_w: np.ndarray,
    x_idx: np.ndarray, x_w: np.ndarray,
    row_chunk: int = 512,
) -> np.ndarray:
    hr_h, hr_w = y_idx.shape[0], x_idx.shape[0]
    out = np.zeros((hr_h, hr_w), dtype=np.float32)
    for r0 in range(0, hr_h, row_chunk):
        r1 = min(r0 + row_chunk, hr_h)
        yi = y_idx[r0:r1]
        wy = y_w[r0:r1]
        mixed_y = np.sum(src[yi, :] * wy[:, :, None], axis=1)
        gathered_x = mixed_y[:, x_idx]
        out[r0:r1] = np.sum(gathered_x * x_w[None, :, :], axis=2).astype(np.float32)
    return out


def project_msi_to_if(
    msi_stack: np.ndarray,
    if_shape: tuple[int, int],
    sigma_lr: float = 0.75,
    radius_lr: int = 2,
    row_chunk: int = 512,
    output_dir: Optional[Path] = None,
    status_cb: Optional[Callable] = None,
) -> np.ndarray:
    """
    Project registered MSI stack (C × lr_H × lr_W) to IF resolution
    (C × if_H × if_W) using Gaussian-weighted interpolation.
    Saves projected TIFF to output_dir if provided.
    Returns projected stack.
    """
    def _s(msg):
        if status_cb:
            status_cb(msg)

    C, lr_h, lr_w = msi_stack.shape
    hr_h, hr_w = if_shape

    _s("Precomputing projection weights…")
    y_idx, y_w, x_idx, x_w = precompute_projection_weights(
        (hr_h, hr_w), (lr_h, lr_w), sigma_lr, radius_lr
    )

    projected = np.zeros((C, hr_h, hr_w), dtype=np.float32)
    for c in range(C):
        _s(f"Projecting channel {c + 1}/{C}…")
        projected[c] = _project_channel(
            msi_stack[c], y_idx, y_w, x_idx, x_w, row_chunk
        )

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / "projected_stack_all_channels__full_hr__gaussian.tif"
        tiff.imwrite(str(out_path), projected)
        _s(f"Saved projected stack → {out_path.name}")

    return projected


# ---------------------------------------------------------------------------
# Cell-level quantification
# ---------------------------------------------------------------------------

def quantify_cells(
    projected_stack: np.ndarray,
    label_mask: np.ndarray,
    region_mask: np.ndarray,
    channel_labels: List[str],
    min_pixels: int = 1,
) -> pd.DataFrame:
    """
    Compute mean intensity per cell from the projected HR stack,
    restricted to pixels inside region_mask.
    Returns a DataFrame with cell_id, centroid_x/y, area, and per-channel means.
    """
    effective = np.where(region_mask, label_mask, 0).astype(np.int32)
    flat = effective.ravel()
    max_label = int(label_mask.max())

    if max_label == 0:
        return pd.DataFrame()

    h, w = label_mask.shape
    yy, xx = np.indices((h, w))

    area_all  = np.bincount(flat, minlength=max_label + 1).astype(np.int64)
    sum_x_all = np.bincount(flat, weights=xx.ravel().astype(np.float64), minlength=max_label + 1)
    sum_y_all = np.bincount(flat, weights=yy.ravel().astype(np.float64), minlength=max_label + 1)

    ids = np.unique(effective)
    ids = ids[ids > 0]
    area = area_all[ids]
    keep = area >= min_pixels
    ids  = ids[keep]
    area = area[keep]

    cx = (sum_x_all[ids] / np.maximum(area, 1)).astype(np.float32)
    cy = (sum_y_all[ids] / np.maximum(area, 1)).astype(np.float32)

    rows = {"cell_id": ids.astype(np.int32), "centroid_x": cx, "centroid_y": cy, "area_px": area}
    df = pd.DataFrame(rows)

    flat_eff = effective.ravel()
    for c, name in enumerate(channel_labels):
        vals = projected_stack[c].ravel().astype(np.float64)
        sums = np.bincount(flat_eff, weights=vals, minlength=max_label + 1)
        df[name] = (sums[ids] / np.maximum(area, 1)).astype(np.float32)

    return df


def run_projection_pipeline(
    msi_stack: np.ndarray,
    if_shape: tuple[int, int],
    nuclear_binary: np.ndarray,
    nuclear_labeled: np.ndarray,
    nuclear_expanded: np.ndarray,
    channel_labels: List[str],
    output_dir: Path,
    sigma_lr: float = 0.75,
    radius_lr: int = 2,
    status_cb: Optional[Callable] = None,
) -> dict:
    """
    Full projection pipeline:
      1. Project MSI → IF resolution
      2. Quantify cells (nuclear mask)
      3. Quantify cells (expanded nuclear mask)
    Returns dict with paths to saved files.
    """
    def _s(msg):
        if status_cb:
            status_cb(msg)

    output_dir.mkdir(parents=True, exist_ok=True)

    # 1a. Nearest-neighbour upsample (raw, no smoothing)
    _s("Upsampling MSI to IF resolution (nearest-neighbour)…")
    import cv2 as _cv2
    hr_h, hr_w = if_shape
    C = msi_stack.shape[0]
    nn_stack = np.empty((C, hr_h, hr_w), dtype=msi_stack.dtype)
    for c in range(C):
        nn_stack[c] = _cv2.resize(
            msi_stack[c], (hr_w, hr_h), interpolation=_cv2.INTER_NEAREST
        )
    nn_path = output_dir / "projected_stack_all_channels__full_hr__nearest.tif"
    tiff.imwrite(str(nn_path), nn_stack)
    _s(f"Saved nearest-neighbour stack → {nn_path.name}")

    # 1b. Gaussian-weighted projection
    projected = project_msi_to_if(
        msi_stack, if_shape,
        sigma_lr=sigma_lr, radius_lr=radius_lr,
        output_dir=output_dir, status_cb=status_cb,
    )

    _s("Quantifying cells (nuclear mask)…")
    df_nuclear = quantify_cells(
        projected, nuclear_labeled, nuclear_binary, channel_labels
    )
    nuclear_csv = output_dir / "cell_level_metabolic_table__nuclear__mean.csv"
    df_nuclear.to_csv(nuclear_csv, index=False)
    _s(f"Saved nuclear cell table → {nuclear_csv.name} ({len(df_nuclear):,} cells)")

    _s("Quantifying cells (expanded nuclear mask)…")
    expanded_region = nuclear_expanded > 0
    df_expanded = quantify_cells(
        projected, nuclear_expanded, expanded_region, channel_labels
    )
    expanded_csv = output_dir / "cell_level_metabolic_table__nuclear_expanded__mean.csv"
    df_expanded.to_csv(expanded_csv, index=False)
    _s(f"Saved expanded cell table → {expanded_csv.name} ({len(df_expanded):,} cells)")

    return {
        "projected_tif": output_dir / "projected_stack_all_channels__full_hr__gaussian.tif",
        "nearest_tif":   nn_path,
        "nuclear_csv":   nuclear_csv,
        "expanded_csv":  expanded_csv,
    }
