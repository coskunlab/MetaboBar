"""
Cell positivity thresholding for each MSI channel.

For each channel, fits a GMM (or uses Otsu/quantile) to the per-cell
mean intensity distribution and assigns a binary positive/negative label.

Outputs:
  <out_dir>/protein_marker_thresholds.csv
  <out_dir>/cell_binary_labels.csv
  <out_dir>/threshold_histograms/<channel>__hist.png
  <out_dir>/positivity_masks/<channel>__overlay.png   (optional)
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile as tiff
from sklearn.mixture import GaussianMixture


# ---------------------------------------------------------------------------
# Threshold methods
# ---------------------------------------------------------------------------

def _otsu(values: np.ndarray, n_bins: int = 256) -> float:
    x = values[np.isfinite(values)]
    if x.size == 0:
        return 0.0
    vmin, vmax = float(x.min()), float(x.max())
    if vmax <= vmin:
        return vmin
    hist, edges = np.histogram(x, bins=n_bins, range=(vmin, vmax))
    prob = hist.astype(np.float64) / hist.sum()
    centers = (edges[:-1] + edges[1:]) / 2.0
    omega = np.cumsum(prob)
    mu = np.cumsum(prob * centers)
    mu_t = mu[-1]
    sigma_b2 = (mu_t * omega - mu) ** 2 / np.maximum(omega * (1.0 - omega), 1e-12)
    return float(centers[np.argmax(sigma_b2)])


def _gmm(values: np.ndarray, seed: int = 42) -> float:
    x = values[np.isfinite(values)]
    if x.size == 0:
        return 0.0
    if np.allclose(x.min(), x.max()):
        return float(x.min())
    use_log = x.min() >= 0
    xu = np.log1p(x) if use_log else x.copy()
    gm = GaussianMixture(n_components=2, covariance_type="full", random_state=seed)
    gm.fit(xu.reshape(-1, 1))
    means = gm.means_.ravel()
    order = np.argsort(means)
    m1, m2 = means[order[0]], means[order[1]]
    w1, w2 = gm.weights_[order[0]], gm.weights_[order[1]]
    s1 = math.sqrt(float(gm.covariances_[order[0]].ravel()[0]))
    s2 = math.sqrt(float(gm.covariances_[order[1]].ravel()[0]))
    xs = np.linspace(xu.min(), xu.max(), 5000)

    def gpdf(xx, mu, sigma):
        sigma = max(sigma, 1e-8)
        return np.exp(-0.5 * ((xx - mu) / sigma) ** 2) / (sigma * math.sqrt(2 * math.pi))

    d = np.abs(w1 * gpdf(xs, m1, s1) - w2 * gpdf(xs, m2, s2))
    thr_log = float(xs[np.argmin(d)])
    return float(np.expm1(thr_log)) if use_log else thr_log


def _top_component_quantile(
    values: np.ndarray,
    n_components: int = 3,
    component_quantile: float = 0.60,
    fallback_quantile: float = 0.75,
    seed: int = 42,
) -> float:
    x = values[np.isfinite(values)]
    if x.size == 0:
        return 0.0
    if np.allclose(x.min(), x.max()):
        return float(x.min())
    use_log = x.min() >= 0
    xu = np.log1p(x) if use_log else x.copy()
    n_components = min(n_components, xu.shape[0])
    if n_components < 2:
        return float(np.quantile(x, fallback_quantile))
    gm = GaussianMixture(n_components=n_components, covariance_type="full", random_state=seed)
    gm.fit(xu.reshape(-1, 1))
    top_idx = int(np.argmax(gm.means_.ravel()))
    assigned = np.argmax(gm.predict_proba(xu.reshape(-1, 1)), axis=1)
    top_vals = xu[assigned == top_idx]
    if top_vals.size < 10:
        return float(np.quantile(x, fallback_quantile))
    thr_log = float(np.quantile(top_vals, component_quantile))
    return float(np.expm1(thr_log)) if use_log else thr_log


def compute_threshold(
    values: np.ndarray,
    method: str = "top_component_quantile",
    gmm_components: int = 3,
    component_quantile: float = 0.60,
    fallback_quantile: float = 0.75,
    seed: int = 42,
) -> Tuple[float, str]:
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return 0.0, "empty"

    if method == "top_component_quantile":
        thr = _top_component_quantile(x, gmm_components, component_quantile,
                                      fallback_quantile, seed)
        return thr, f"top_component_q{component_quantile:.2f}"

    if method == "gmm":
        return _gmm(x, seed), "gmm"

    if method == "otsu":
        return _otsu(x), "otsu"

    if method == "upper_quantile":
        return float(np.quantile(x, fallback_quantile)), f"upper_q{fallback_quantile:.2f}"

    raise ValueError(f"Unknown threshold method: {method!r}")


# ---------------------------------------------------------------------------
# Overlay PNG
# ---------------------------------------------------------------------------

def _safe_name(s: str) -> str:
    return re.sub(r'[\\/:*?"<>| ]', "_", str(s)).strip(".")


def save_positivity_overlay(
    expanded_mask: np.ndarray,
    positive_ids: np.ndarray,
    negative_ids: np.ndarray,
    out_path: Path,
    positive_color: tuple = (255, 0, 0),
    negative_color: tuple = (64, 64, 64),
) -> None:
    arr = np.asarray(expanded_mask).astype(np.int64)
    rgb = np.zeros((*arr.shape, 3), dtype=np.uint8)
    max_id = int(arr.max()) if arr.size > 0 else 0
    lut = np.zeros((max_id + 1, 3), dtype=np.uint8)
    for cid in positive_ids.astype(np.int64):
        if 0 <= cid <= max_id:
            lut[cid] = positive_color
    for cid in negative_ids.astype(np.int64):
        if 0 <= cid <= max_id and np.all(lut[cid] == 0):
            lut[cid] = negative_color
    valid = arr > 0
    rgb[valid] = lut[arr[valid]]
    tiff.imwrite(str(out_path), rgb)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_positivity_thresholding(
    cell_csv: Path,
    expanded_mask_path: Path,
    channel_labels: List[str],
    output_dir: Path,
    method: str = "top_component_quantile",
    gmm_components: int = 3,
    component_quantile: float = 0.60,
    fallback_quantile: float = 0.75,
    min_pos_fraction: float = 0.01,
    max_pos_fraction: float = 0.99,
    seed: int = 42,
    save_overlays: bool = True,
    status_cb: Optional[Callable] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Threshold each channel in channel_labels using the cell CSV.

    Returns:
        threshold_df  — one row per channel with threshold + stats
        binary_df     — cell_id + one binary column per channel
    """
    def _s(msg: str) -> None:
        if status_cb:
            status_cb(msg)

    output_dir.mkdir(parents=True, exist_ok=True)
    hist_dir = output_dir / "threshold_histograms"
    hist_dir.mkdir(exist_ok=True)

    _s("Loading cell table…")
    cell_df = pd.read_csv(str(cell_csv))

    expanded_mask = None
    if save_overlays and expanded_mask_path.exists():
        _s("Loading expanded cell mask…")
        arr = np.squeeze(tiff.imread(str(expanded_mask_path)))
        expanded_mask = arr.astype(np.int32)
        overlay_dir = output_dir / "positivity_masks"
        overlay_dir.mkdir(exist_ok=True)

    threshold_rows = []
    binary_cols: Dict[str, np.ndarray] = {"cell_id": cell_df["cell_id"].values}

    total = len(channel_labels)
    for i, ch in enumerate(channel_labels):
        _s(f"Thresholding channel {i+1}/{total}: {ch}…")

        if ch not in cell_df.columns:
            _s(f"  [SKIP] {ch} not in cell table")
            continue

        vals = pd.to_numeric(cell_df[ch], errors="coerce").to_numpy(dtype=np.float32)
        vals_finite = np.where(
            np.isfinite(vals), vals,
            float(np.nanmedian(vals[np.isfinite(vals)])) if np.isfinite(vals).any() else 0.0
        )

        thr, method_used = compute_threshold(
            vals_finite, method=method,
            gmm_components=gmm_components,
            component_quantile=component_quantile,
            fallback_quantile=fallback_quantile,
            seed=seed,
        )

        y = (vals_finite >= thr).astype(int)
        pos_frac = float(y.mean()) if len(y) else 0.0

        # Clamp extreme class imbalance
        if pos_frac < min_pos_fraction or pos_frac > max_pos_fraction:
            fallback_thr = float(np.quantile(vals_finite[np.isfinite(vals_finite)],
                                             fallback_quantile))
            y = (vals_finite >= fallback_thr).astype(int)
            thr = fallback_thr
            method_used += "_clamped"
            pos_frac = float(y.mean())

        binary_cols[f"{ch}__positive"] = y.astype(int)

        threshold_rows.append({
            "channel": ch,
            "threshold": float(thr),
            "method": method_used,
            "n_cells": int(len(y)),
            "n_positive": int(y.sum()),
            "n_negative": int((1 - y).sum()),
            "positive_fraction": pos_frac,
            "min_value": float(np.nanmin(vals_finite)),
            "max_value": float(np.nanmax(vals_finite)),
            "mean_value": float(np.nanmean(vals_finite)),
            "median_value": float(np.nanmedian(vals_finite)),
        })

        # Histogram
        plt.figure(figsize=(6, 3))
        plt.hist(vals_finite[np.isfinite(vals_finite)], bins=100, color="steelblue", alpha=0.8)
        plt.axvline(thr, color="red", linestyle="--", label=f"thr={thr:.4f}")
        plt.title(f"{ch}  |  {pos_frac*100:.1f}% positive")
        plt.xlabel("Cell mean intensity")
        plt.ylabel("Count")
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(hist_dir / f"{_safe_name(ch)}__hist.png", dpi=150)
        plt.close()

        # Overlay
        if expanded_mask is not None:
            pos_ids = cell_df.loc[y == 1, "cell_id"].to_numpy(dtype=np.int64)
            neg_ids = cell_df.loc[y == 0, "cell_id"].to_numpy(dtype=np.int64)
            save_positivity_overlay(
                expanded_mask, pos_ids, neg_ids,
                overlay_dir / f"{_safe_name(ch)}__overlay.tif",
            )

    threshold_df = pd.DataFrame(threshold_rows).sort_values("channel").reset_index(drop=True)
    binary_df    = pd.DataFrame(binary_cols)

    threshold_df.to_csv(output_dir / "protein_marker_thresholds.csv", index=False)
    binary_df.to_csv(output_dir / "cell_binary_labels.csv", index=False)

    _s(f"Done — thresholded {len(threshold_rows)} channels.")
    return threshold_df, binary_df
