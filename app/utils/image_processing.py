"""
Image display helpers: normalisation, single-channel rendering, RGB overlays,
and colormap application (IF → cycling RGB colors, MSI → viridis).
"""

from typing import Optional, Sequence

import numpy as np

# ---------------------------------------------------------------------------
# IF channel color cycle: R, G, B, Cyan, Magenta, Yellow, then repeats
# ---------------------------------------------------------------------------
_IF_COLORS = [
    (1, 0, 0),   # red
    (0, 1, 0),   # green
    (0, 0, 1),   # blue
    (0, 1, 1),   # cyan
    (1, 0, 1),   # magenta
    (1, 1, 0),   # yellow
]


def if_channel_color(channel_idx: int) -> tuple:
    """Return (R, G, B) float color for an IF channel index."""
    return _IF_COLORS[channel_idx % len(_IF_COLORS)]


def apply_if_color(img: np.ndarray, channel_idx: int) -> np.ndarray:
    """
    Apply the IF color cycle to a normalised 2-D float32 image.
    Returns H × W × 3 float32 RGB.
    """
    r, g, b = if_channel_color(channel_idx)
    H, W = img.shape
    rgb = np.zeros((H, W, 3), dtype=np.float32)
    rgb[:, :, 0] = img * r
    rgb[:, :, 1] = img * g
    rgb[:, :, 2] = img * b
    return rgb


def apply_viridis(img: np.ndarray) -> np.ndarray:
    """
    Apply the viridis colormap to a normalised 2-D float32 image.
    Returns H × W × 3 float32 RGB.
    """
    import matplotlib.cm as cm
    rgba = cm.viridis(img)          # H × W × 4
    return rgba[:, :, :3].astype(np.float32)


def robust_display_image(
    img: np.ndarray,
    pmin: float = 1.0,
    pmax: float = 99.5,
    gamma: float = 1.0,
) -> np.ndarray:
    img = np.asarray(img, dtype=np.float32)
    finite = np.isfinite(img)

    if not finite.any():
        return np.zeros(img.shape, dtype=np.float32)

    vals = img[finite]
    lo, hi = np.percentile(vals, [pmin, pmax])

    if hi <= lo:
        lo, hi = float(np.min(vals)), float(np.max(vals))

    if hi <= lo:
        return np.zeros(img.shape, dtype=np.float32)

    out = np.clip((img - lo) / (hi - lo), 0, 1)

    if gamma != 1.0:
        out = np.power(out, gamma)

    return out.astype(np.float32)


def make_rgb_overlay(
    stack: np.ndarray,
    channels: Sequence[int],
    pmin: float,
    pmax: float,
    gamma: float,
) -> np.ndarray:
    """
    Make an RGB overlay from up to 3 selected channels.
    Channel 1 -> red, channel 2 -> green, channel 3 -> blue.
    """
    H, W = stack.shape[-2], stack.shape[-1]
    rgb = np.zeros((H, W, 3), dtype=np.float32)

    for out_c, ch in enumerate(list(channels)[:3]):
        rgb[:, :, out_c] = robust_display_image(
            stack[ch],
            pmin=pmin,
            pmax=pmax,
            gamma=gamma,
        )

    return np.clip(rgb, 0, 1)


def make_merged_if_msi_overlay(
    if_stack: np.ndarray,
    msi_stack: np.ndarray,
    if_channel: Optional[int],
    msi_channel: Optional[int],
    pmin: float,
    pmax: float,
    gamma: float,
) -> Optional[np.ndarray]:
    """
    Optional merged overlay:
      IF channel  -> green
      MSI channel -> magenta

    Only works when IF and MSI image dimensions match.
    """
    if if_channel is None or msi_channel is None:
        return None

    if if_stack.shape[-2:] != msi_stack.shape[-2:]:
        return None

    H, W = if_stack.shape[-2], if_stack.shape[-1]
    rgb = np.zeros((H, W, 3), dtype=np.float32)

    if_img = robust_display_image(if_stack[if_channel], pmin=pmin, pmax=pmax, gamma=gamma)
    msi_img = robust_display_image(msi_stack[msi_channel], pmin=pmin, pmax=pmax, gamma=gamma)

    # MSI -> magenta (R + B)
    rgb[:, :, 0] = msi_img
    rgb[:, :, 2] = msi_img

    # IF -> green
    rgb[:, :, 1] = if_img

    return np.clip(rgb, 0, 1)
