"""
Stack transform utilities: rotation, flipping, and bounding-box crop.
All functions operate on C x H x W numpy arrays and return the same shape.
"""

import io
from typing import List, Tuple

import numpy as np
import tifffile as tiff


# ---------------------------------------------------------------------------
# Rotation / flip
# ---------------------------------------------------------------------------

def rotate_stack(stack: np.ndarray, degrees: int) -> np.ndarray:
    """
    Rotate all channels by 0 / 90 / 180 / 270 degrees (counter-clockwise).
    """
    k = (degrees // 90) % 4
    if k == 0:
        return stack
    # np.rot90 on a C x H x W array: rotate in the H-W plane (axes 1, 2)
    return np.rot90(stack, k=k, axes=(1, 2))


def flip_stack(stack: np.ndarray, axis: str) -> np.ndarray:
    """
    Flip all channels.
    axis: 'horizontal' flips left-right (W axis),
          'vertical'   flips top-bottom (H axis).
    """
    if axis == "horizontal":
        return np.flip(stack, axis=2)
    if axis == "vertical":
        return np.flip(stack, axis=1)
    raise ValueError(f"axis must be 'horizontal' or 'vertical', got {axis!r}")


# ---------------------------------------------------------------------------
# Crop
# ---------------------------------------------------------------------------

def crop_stack(
    stack: np.ndarray,
    y0: int,
    y1: int,
    x0: int,
    x1: int,
) -> np.ndarray:
    """
    Crop all channels to the bounding box [y0:y1, x0:x1].
    Coordinates are clamped to valid range automatically.
    """
    H, W = stack.shape[-2], stack.shape[-1]
    y0 = int(np.clip(y0, 0, H))
    y1 = int(np.clip(y1, 0, H))
    x0 = int(np.clip(x0, 0, W))
    x1 = int(np.clip(x1, 0, W))

    if y1 <= y0 or x1 <= x0:
        raise ValueError(
            f"Invalid crop box: y=[{y0},{y1}), x=[{x0},{x1}). "
            "Make sure the box has positive width and height."
        )

    return stack[:, y0:y1, x0:x1].copy()


# ---------------------------------------------------------------------------
# TIFF serialisation (for in-memory download)
# ---------------------------------------------------------------------------

def stack_to_tiff_bytes(stack: np.ndarray, labels: List[str]) -> bytes:
    """
    Serialise a C x H x W stack to an ImageJ-compatible TIFF in memory.
    Returns raw bytes suitable for st.download_button.
    """
    buf = io.BytesIO()
    tiff.imwrite(
        buf,
        stack,
        imagej=True,
        metadata={
            "axes": "CYX",
            "Labels": labels,
            "Channel": labels,
        },
    )
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Canvas-coord → stack-coord mapping
# ---------------------------------------------------------------------------

def canvas_box_to_stack_coords(
    left: float,
    top: float,
    width: float,
    height: float,
    canvas_w: int,
    canvas_h: int,
    stack_w: int,
    stack_h: int,
) -> Tuple[int, int, int, int]:
    """
    Map a bounding box drawn on a canvas (canvas_w x canvas_h) back to
    pixel coordinates in the original stack (stack_w x stack_h).

    Returns (x0, y0, x1, y1) in stack pixel space.
    """
    sx = stack_w / canvas_w
    sy = stack_h / canvas_h

    x0 = int(left * sx)
    y0 = int(top * sy)
    x1 = int((left + width) * sx)
    y1 = int((top + height) * sy)

    return x0, y0, x1, y1
