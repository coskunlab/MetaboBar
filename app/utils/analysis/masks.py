"""
Nuclei expansion and MBP mask generation.
Both run in the current env (torch_gpu3) — no subprocess needed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import tifffile as tiff
from skimage.filters import gaussian
from skimage.morphology import (
    binary_closing,
    binary_opening,
    disk,
    remove_small_holes,
    remove_small_objects,
)
from skimage.segmentation import expand_labels


# ---------------------------------------------------------------------------
# Nuclei expansion
# ---------------------------------------------------------------------------

def expand_nuclear_mask(
    labeled_mask: np.ndarray,
    tissue_mask: Optional[np.ndarray],
    expand_um: float,
    pixel_size_um: float,
    output_dir: Path,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Expand each labeled nucleus outward by expand_um microns.
    Clips to tissue_mask if provided.
    Returns (expanded_labeled, expanded_binary).
    """
    expand_px = expand_um / pixel_size_um
    expanded = expand_labels(labeled_mask.astype(np.int32), distance=expand_px).astype(np.uint32)

    if tissue_mask is not None:
        expanded[~tissue_mask] = 0

    binary = (expanded > 0).astype(np.uint8)

    output_dir.mkdir(parents=True, exist_ok=True)
    tiff.imwrite(str(output_dir / "nuclear_mask_expanded.tif"), expanded)
    tiff.imwrite(str(output_dir / "nuclear_mask_expanded_binary.tif"), binary)

    return expanded, binary


# ---------------------------------------------------------------------------
# MBP mask
# ---------------------------------------------------------------------------

def make_mbp_mask(
    if_stack: np.ndarray,
    mbp_channel_idx: int,
    nuclear_binary: np.ndarray,
    mbp_percentile: float = 50.0,
    gaussian_sigma: float = 1.0,
    min_object_size: int = 20,
    min_hole_size: int = 20,
    open_radius: int = 1,
    close_radius: int = 1,
    output_dir: Optional[Path] = None,
) -> np.ndarray:
    """
    Threshold the MBP IF channel, apply morphological cleanup,
    and exclude nuclear pixels.
    Returns binary MBP mask (bool).
    """
    # Extract channel
    if if_stack.ndim == 3 and if_stack.shape[0] <= 64:
        ch = if_stack[mbp_channel_idx].astype(np.float32)
    else:
        ch = if_stack[:, :, mbp_channel_idx].astype(np.float32)

    # Normalise
    mn, mx = ch.min(), ch.max()
    if mx > mn:
        ch = (ch - mn) / (mx - mn)

    # Smooth
    if gaussian_sigma > 0:
        ch = gaussian(ch, sigma=gaussian_sigma, preserve_range=True).astype(np.float32)

    # Threshold
    thresh = float(np.percentile(ch, mbp_percentile))
    mask = ch > thresh

    # Morphological cleanup
    if open_radius > 0:
        mask = binary_opening(mask, disk(open_radius))
    if close_radius > 0:
        mask = binary_closing(mask, disk(close_radius))
    if min_object_size > 0:
        mask = remove_small_objects(mask, min_size=min_object_size)
    if min_hole_size > 0:
        mask = remove_small_holes(mask, area_threshold=min_hole_size)

    # Exclude nuclei
    mask = mask & (~nuclear_binary.astype(bool))

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        out = (mask.astype(np.uint8) * 255)
        tiff.imwrite(str(output_dir / "mbp_mask_binary.tif"), out)

    return mask
