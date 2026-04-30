"""
MSI-to-IF registration utilities.

Pipeline
--------
1. resize_msi_to_if()          – nearest-neighbour resize MSI → IF dimensions
2. write_registration_tiff()   – save 2-channel ref TIFF (downscaled) for Fiji
3. run_fiji_sift()             – run Fiji headlessly, get aligned output TIFF
4. recover_affine_ecc()        – recover exact 2×3 matrix from aligned output
5. apply_affine_to_stack()     – warpAffine every MSI channel with that matrix
6. downsample_to_original_w()  – resize back to original MSI width (aspect-ratio)
"""

import subprocess
import tempfile
import time
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import tifffile as tiff

# ---------------------------------------------------------------------------
# Fiji executable
# ---------------------------------------------------------------------------
from app.utils.bundle_paths import get_fiji_exe

DEFAULT_FIJI_PATH = get_fiji_exe()

# ---------------------------------------------------------------------------
# SIFT default parameters (matching the screenshot)
# ---------------------------------------------------------------------------
SIFT_DEFAULTS = dict(
    initial_gaussian_blur=1.60,
    steps_per_scale_octave=3,
    minimum_image_size=64,
    maximum_image_size=1024,
    feature_descriptor_size=4,
    feature_descriptor_orientation_bins=8,
    closest_next_closest_ratio=0.92,
    maximal_alignment_error=25.0,
    inlier_ratio=0.05,
    expected_transformation="Affine",
    interpolate=False,
)

# Max px on longest side sent to Fiji — keeps SIFT fast
SIFT_MAX_PX = 1024


# ---------------------------------------------------------------------------
# Step 1 – resize MSI to IF size (no aspect ratio, nearest-neighbour)
# ---------------------------------------------------------------------------

def resize_msi_to_if(
    msi_stack: np.ndarray,
    if_h: int,
    if_w: int,
) -> np.ndarray:
    C = msi_stack.shape[0]
    out = np.empty((C, if_h, if_w), dtype=msi_stack.dtype)
    for c in range(C):
        out[c] = cv2.resize(
            msi_stack[c], (if_w, if_h),
            interpolation=cv2.INTER_NEAREST,
        )
    return out


# ---------------------------------------------------------------------------
# Step 2 – write downscaled 2-channel reference TIFF for Fiji
# ---------------------------------------------------------------------------

def _downscale(img: np.ndarray, max_px: int = SIFT_MAX_PX) -> Tuple[np.ndarray, float]:
    H, W = img.shape
    scale = min(1.0, max_px / max(H, W))
    if scale == 1.0:
        return img, 1.0
    nh, nw = max(1, int(H * scale)), max(1, int(W * scale))
    return cv2.resize(img, (nw, nh), interpolation=cv2.INTER_NEAREST), scale


def _to_uint16(img: np.ndarray) -> np.ndarray:
    img = img.astype(np.float32)
    mn, mx = img.min(), img.max()
    if mx > mn:
        img = (img - mn) / (mx - mn)
    return (img * 65535).clip(0, 65535).astype(np.uint16)


def write_registration_tiff(
    if_channel: np.ndarray,
    msi_channel: np.ndarray,
    output_path: Path,
) -> float:
    """
    Downscale both channels to SIFT_MAX_PX, save as 2-slice uint16 TIFF.
    Slice 0 = IF ref, slice 1 = MSI ref.
    Returns the scale factor (downscaled / original).
    """
    if_small, scale = _downscale(if_channel)
    msi_small, _    = _downscale(msi_channel)
    stack = np.stack([_to_uint16(if_small), _to_uint16(msi_small)], axis=0)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tiff.imwrite(str(output_path), stack, imagej=True, metadata={"axes": "CYX"})
    return scale


# ---------------------------------------------------------------------------
# Step 3 – run Fiji headlessly, get aligned output TIFF
# ---------------------------------------------------------------------------

_MACRO_TEMPLATE = """\
open("{input_tiff}");
run("Linear Stack Alignment with SIFT", "initial_gaussian_blur={initial_gaussian_blur} steps_per_scale_octave={steps_per_scale_octave} minimum_image_size={minimum_image_size} maximum_image_size={maximum_image_size} feature_descriptor_size={feature_descriptor_size} feature_descriptor_orientation_bins={feature_descriptor_orientation_bins} closest/next_closest_ratio={closest_next_closest_ratio} maximal_alignment_error={maximal_alignment_error} inlier_ratio={inlier_ratio} expected_transformation={expected_transformation} interpolate={interpolate_flag} show_info=false show_transformation_matrix=false");
saveAs("Tiff", "{output_tiff}");
"""


def _build_macro(input_tiff: Path, output_tiff: Path, params: dict) -> str:
    return _MACRO_TEMPLATE.format(
        input_tiff=str(input_tiff).replace("\\", "/"),
        output_tiff=str(output_tiff).replace("\\", "/"),
        initial_gaussian_blur=params["initial_gaussian_blur"],
        steps_per_scale_octave=params["steps_per_scale_octave"],
        minimum_image_size=params["minimum_image_size"],
        maximum_image_size=params["maximum_image_size"],
        feature_descriptor_size=params["feature_descriptor_size"],
        feature_descriptor_orientation_bins=params["feature_descriptor_orientation_bins"],
        closest_next_closest_ratio=params["closest_next_closest_ratio"],
        maximal_alignment_error=params["maximal_alignment_error"],
        inlier_ratio=params["inlier_ratio"],
        expected_transformation=params["expected_transformation"],
        interpolate_flag="true" if params["interpolate"] else "false",
    )


def run_fiji_sift(
    input_tiff: Path,
    output_tiff: Path,
    fiji_path: str = DEFAULT_FIJI_PATH,
    params: Optional[dict] = None,
    timeout: int = 300,
) -> None:
    """
    Run Fiji headlessly. Polls for the output TIFF; raises on timeout.
    """
    if params is None:
        params = SIFT_DEFAULTS.copy()

    if output_tiff.exists():
        output_tiff.unlink()

    macro_text = _build_macro(input_tiff, output_tiff, params)

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".ijm", delete=False, encoding="utf-8"
    ) as f:
        f.write(macro_text)
        macro_path = f.name

    try:
        # ImageJ-win64.exe runs headlessly and exits when the macro finishes
        subprocess.run(
            [fiji_path, "--headless", "--console", "--run", macro_path],
            timeout=timeout,
        )

        if not output_tiff.exists():
            raise RuntimeError(
                f"Fiji finished but no output TIFF was written.\n"
                f"Expected: {output_tiff}\n"
                f"Macro:\n{macro_text}"
            )
    finally:
        Path(macro_path).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Step 4 – recover exact affine matrix from Fiji's aligned output via ECC
# ---------------------------------------------------------------------------

def recover_affine_ecc(
    aligned_tiff: Path,
    msi_ref_small: np.ndarray,
) -> np.ndarray:
    """
    Fiji aligned the MSI reference channel and wrote it as slice 1 of the
    output TIFF. We recover the exact 2×3 affine matrix by comparing the
    original downscaled MSI ref (src) with the aligned version (dst) using:
      1. Phase correlation  → initial translation estimate
      2. cv2.findTransformECC → sub-pixel accurate affine refinement

    Returns a (2, 3) float64 affine matrix.
    """
    with tiff.TiffFile(str(aligned_tiff)) as tf:
        aligned_stack = tf.asarray()   # 2 x H x W
    aligned_msi = aligned_stack[1]

    def _u8(img: np.ndarray) -> np.ndarray:
        img = img.astype(np.float32)
        mn, mx = img.min(), img.max()
        if mx > mn:
            img = (img - mn) / (mx - mn)
        return (img * 255).clip(0, 255).astype(np.uint8)

    src = _u8(msi_ref_small)
    dst = _u8(aligned_msi)

    # Phase correlation for initial translation
    src_f = np.fft.fft2(src.astype(np.float32))
    dst_f = np.fft.fft2(dst.astype(np.float32))
    cross = src_f * np.conj(dst_f)
    cross /= (np.abs(cross) + 1e-8)
    shift_img = np.abs(np.fft.ifft2(cross))
    peak = np.unravel_index(np.argmax(shift_img), shift_img.shape)
    h, w = src.shape
    ty = float(peak[0] if peak[0] < h // 2 else peak[0] - h)
    tx = float(peak[1] if peak[1] < w // 2 else peak[1] - w)

    warp = np.array([[1, 0, tx], [0, 1, ty]], dtype=np.float32)

    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 1000, 1e-7)
    try:
        _, warp = cv2.findTransformECC(
            src.astype(np.float32),
            dst.astype(np.float32),
            warp,
            cv2.MOTION_AFFINE,
            criteria,
            None,
            5,
        )
    except cv2.error:
        pass  # keep phase-correlation translation as fallback

    return warp.astype(np.float64)


# ---------------------------------------------------------------------------
# Scale matrix translation back to full resolution
# ---------------------------------------------------------------------------

def scale_affine_matrix(matrix: np.ndarray, sift_scale: float) -> np.ndarray:
    """
    The linear part (rotation/scale/shear) is unitless — unchanged.
    Only the translation column needs dividing by sift_scale to convert
    from downscaled-pixel offsets to full-resolution offsets.
    """
    m = matrix.copy()
    m[0, 2] /= sift_scale
    m[1, 2] /= sift_scale
    return m


# ---------------------------------------------------------------------------
# Step 5 – apply affine to all MSI channels
# ---------------------------------------------------------------------------

def apply_affine_to_stack(
    stack: np.ndarray,
    matrix: np.ndarray,
    out_h: int,
    out_w: int,
) -> np.ndarray:
    """
    Apply the same 2×3 affine matrix to every channel (C × H × W).
    All MSI channels share the same spatial coordinate system, so the
    transform that aligns the reference channel aligns all others too.
    Uses INTER_NEAREST (no smoothing).
    """
    C = stack.shape[0]
    out = np.empty((C, out_h, out_w), dtype=stack.dtype)
    for c in range(C):
        out[c] = cv2.warpAffine(
            stack[c], matrix, (out_w, out_h),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
    return out


# ---------------------------------------------------------------------------
# Step 6 – downsample aligned MSI back to original MSI width
# ---------------------------------------------------------------------------

def downsample_to_original_w(
    stack: np.ndarray,
    orig_w: int,
) -> np.ndarray:
    """
    Resize stack so its width matches orig_w, preserving aspect ratio,
    using INTER_NEAREST.
    """
    current_h, current_w = stack.shape[-2], stack.shape[-1]
    scale = orig_w / current_w
    new_w = orig_w
    new_h = max(1, round(current_h * scale))

    C = stack.shape[0]
    out = np.empty((C, new_h, new_w), dtype=stack.dtype)
    for c in range(C):
        out[c] = cv2.resize(stack[c], (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    return out


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def register_msi_to_if(
    if_stack: np.ndarray,
    msi_stack: np.ndarray,
    if_ref_channel: int,
    msi_ref_channel: int,
    work_dir: Path,
    fiji_path: str = DEFAULT_FIJI_PATH,
    sift_params: Optional[dict] = None,
    status_cb=None,
) -> Tuple[np.ndarray, str]:
    """
    Full registration pipeline. Returns (aligned_msi_stack, status_summary).

    Steps:
      1. Resize MSI → IF size (nearest-neighbour, no aspect ratio)
      2. Downscale reference channels to 1024px, write 2-slice TIFF for Fiji
      3. Run Fiji SIFT headlessly → aligned output TIFF
      4. Recover affine matrix via phase-correlation + ECC on aligned output
      5. Scale translation to full IF resolution; warpAffine all MSI channels
      6. Downsample aligned stack back to original MSI width (aspect-ratio)
    """
    if sift_params is None:
        sift_params = SIFT_DEFAULTS.copy()

    orig_msi_w = msi_stack.shape[-1]
    if_h, if_w = if_stack.shape[-2], if_stack.shape[-1]

    def _s(msg: str) -> None:
        if status_cb:
            status_cb(msg)

    _s("Step 1/5 — Resizing MSI to IF dimensions…")
    msi_up = resize_msi_to_if(msi_stack, if_h, if_w)

    _s("Step 2/5 — Writing reference TIFF for Fiji (1024 px)…")
    ref_tiff     = work_dir / "reg_input.tif"
    aligned_tiff = work_dir / "reg_output.tif"
    sift_scale = write_registration_tiff(
        if_stack[if_ref_channel],
        msi_up[msi_ref_channel],
        ref_tiff,
    )

    _s("Step 3/5 — Running Fiji SIFT alignment…")
    run_fiji_sift(ref_tiff, aligned_tiff, fiji_path, sift_params)

    _s("Step 4/5 — Recovering transformation matrix (ECC)…")
    msi_ref_small, _ = _downscale(msi_up[msi_ref_channel])
    matrix_small = recover_affine_ecc(aligned_tiff, msi_ref_small)
    matrix = scale_affine_matrix(matrix_small, sift_scale)

    _s("Step 5/5 — Applying transform to all MSI channels…")
    aligned_full = apply_affine_to_stack(msi_up, matrix, if_h, if_w)

    _s("Done — Downsampling to original MSI width…")
    aligned_msi = downsample_to_original_w(aligned_full, orig_msi_w)

    summary = (
        f"SIFT scale: {sift_scale:.4f}\n"
        f"Affine matrix (full res):\n{matrix}\n"
        f"Output shape: {aligned_msi.shape}"
    )
    return aligned_msi, summary
