"""
PCA + UMAP + Leiden + KMeans clustering for cells and superpixels.
"""

from __future__ import annotations

import colorsys
import warnings
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tifffile as tiff

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Color helpers
# ---------------------------------------------------------------------------

def _distinct_colors(n: int) -> list:
    return [
        tuple(int(v * 255) for v in colorsys.hsv_to_rgb(i / max(n, 1), 0.75, 0.95))
        for i in range(n)
    ]


def _color_map(labels) -> dict:
    uniq = sorted(pd.unique(pd.Series(labels).astype(str)))
    return {lab: col for lab, col in zip(uniq, _distinct_colors(len(uniq)))}


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------

def _run_clustering(
    df: pd.DataFrame,
    id_col: str,
    feature_cols: List[str],
    n_pcs: int,
    n_neighbors: int,
    umap_min_dist: float,
    leiden_resolutions: List[float],
    k_values: List[int],
    random_seed: int,
) -> tuple[pd.DataFrame, np.ndarray]:
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans
    import scanpy as sc
    import anndata as ad

    X = df[feature_cols].fillna(0).to_numpy(dtype=np.float32)
    X_scaled = StandardScaler().fit_transform(X)

    n_pcs_use = min(n_pcs, X_scaled.shape[0] - 1, X_scaled.shape[1])
    X_pca = PCA(n_components=n_pcs_use, random_state=random_seed).fit_transform(X_scaled)

    adata = ad.AnnData(X=X_scaled)
    adata.obsm["X_pca"] = X_pca
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, use_rep="X_pca", random_state=random_seed)
    sc.tl.umap(adata, random_state=random_seed, min_dist=umap_min_dist)

    out = df.copy()
    for i in range(X_pca.shape[1]):
        out[f"PC{i+1}"] = X_pca[:, i]
    out["UMAP1"] = adata.obsm["X_umap"][:, 0]
    out["UMAP2"] = adata.obsm["X_umap"][:, 1]

    for res in leiden_resolutions:
        key = f"leiden_res_{res:.2f}"
        sc.tl.leiden(adata, resolution=res, key_added=key, random_state=random_seed)
        out[key] = adata.obs[key].astype(str).values

    for k in k_values:
        if k >= len(df):
            continue
        km = KMeans(n_clusters=k, random_state=random_seed, n_init=20)
        out[f"kmeans_k_{k}"] = km.fit_predict(X_scaled).astype(str)

    return out, adata.obsm["X_umap"]


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def _save_umap(umap_xy, labels, cmap, path, title, dpi=220):
    colors = np.array([np.array(cmap[str(l)]) / 255.0 for l in labels])
    plt.figure(figsize=(7, 6))
    plt.scatter(umap_xy[:, 0], umap_xy[:, 1], c=colors, s=3, alpha=0.9, linewidths=0)
    plt.xlabel("UMAP1"); plt.ylabel("UMAP2"); plt.title(title)
    uniq = sorted(pd.unique(pd.Series(labels).astype(str)))
    if len(uniq) <= 25:
        handles = [plt.Line2D([0],[0], marker="o", linestyle="", markersize=6,
                               markerfacecolor=np.array(cmap[l])/255.0,
                               markeredgecolor=np.array(cmap[l])/255.0, label=l)
                   for l in uniq]
        plt.legend(handles=handles, bbox_to_anchor=(1.02,1), loc="upper left",
                   frameon=False, title="cluster")
    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()


def _save_matrixplot(df, cluster_col, feature_cols, path, title, row_zscore=True, dpi=220):
    mat = df.groupby(cluster_col)[feature_cols].mean().sort_index()
    arr = mat.to_numpy(dtype=np.float32)
    if row_zscore:
        mu = arr.mean(axis=1, keepdims=True)
        sd = arr.std(axis=1, keepdims=True) + 1e-8
        arr = (arr - mu) / sd
        cbar_label = "z-score"
    else:
        cbar_label = "Mean intensity"
    plt.figure(figsize=(12, max(4, 0.45 * arr.shape[0])))
    im = plt.imshow(arr, aspect="auto")
    plt.colorbar(im, fraction=0.03, pad=0.02, label=cbar_label)
    plt.xticks(np.arange(len(feature_cols)), feature_cols, rotation=90)
    plt.yticks(np.arange(mat.shape[0]), mat.index.astype(str))
    plt.xlabel("Channel"); plt.ylabel("Cluster"); plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()


def _save_colored_mask(label_mask, ids, labels, cmap, path, white_boundaries=False, dpi=220):
    h, w = label_mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    max_id = int(label_mask.max())
    lut = np.zeros((max_id + 1, 3), dtype=np.uint8)
    for oid, cl in zip(ids, labels):
        if 0 <= int(oid) <= max_id:
            lut[int(oid)] = np.array(cmap[str(cl)], dtype=np.uint8)
    valid = label_mask > 0
    rgb[valid] = lut[label_mask[valid]]
    if white_boundaries:
        lab = label_mask.astype(np.int32)
        b = np.zeros(lab.shape, dtype=bool)
        for axis in [0, 1]:
            diff = np.diff(lab, axis=axis) != 0
            slices_a = [slice(None)] * 2
            slices_b = [slice(None)] * 2
            slices_a[axis] = slice(1, None)
            slices_b[axis] = slice(None, -1)
            b[tuple(slices_a)] |= diff
            b[tuple(slices_b)] |= diff
        rgb[b & valid] = 255
    fig_w = max(6, w / 350)
    fig_h = max(6, h / 350)
    plt.figure(figsize=(fig_w, fig_h))
    plt.imshow(rgb); plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0)
    plt.close()


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_clustering(
    cell_csv: Path,
    superpixel_csv: Path,
    cell_label_mask_path: Path,
    superpixel_label_mask_path: Path,
    channel_labels: List[str],
    output_dir: Path,
    n_pcs: int = 10,
    n_neighbors: int = 15,
    umap_min_dist: float = 0.3,
    leiden_resolutions: Optional[List[float]] = None,
    k_values: Optional[List[int]] = None,
    random_seed: int = 42,
    row_zscore: bool = True,
    status_cb: Optional[Callable] = None,
) -> None:
    def _s(msg):
        if status_cb:
            status_cb(msg)

    if leiden_resolutions is None:
        leiden_resolutions = [0.25, 0.50, 1.00]
    if k_values is None:
        k_values = [2, 4, 6, 8, 10]

    output_dir.mkdir(parents=True, exist_ok=True)

    configs = [
        ("cells",       cell_csv,       cell_label_mask_path,       "cell_id",       False),
        ("superpixels", superpixel_csv, superpixel_label_mask_path, "superpixel_id", True),
    ]

    for name, csv_path, mask_path, id_col, white_bounds in configs:
        _s(f"Clustering {name}…")
        out_dir = output_dir / name
        out_dir.mkdir(parents=True, exist_ok=True)

        df_raw = pd.read_csv(csv_path)
        missing = [c for c in channel_labels if c not in df_raw.columns]
        if missing:
            _s(f"Warning: {len(missing)} channel(s) not found in {name} table, skipping.")
            continue

        df, umap_xy = _run_clustering(
            df_raw, id_col, channel_labels,
            n_pcs, n_neighbors, umap_min_dist,
            leiden_resolutions, k_values, random_seed,
        )

        clustered_csv = out_dir / f"{name}__clustered.csv"
        df.to_csv(clustered_csv, index=False)

        label_mask = None
        if mask_path.exists():
            arr = tiff.imread(str(mask_path))
            label_mask = np.squeeze(arr).astype(np.int32)

        cluster_cols = [c for c in df.columns
                        if c.startswith("leiden_res_") or c.startswith("kmeans_k_")]
        ids = df[id_col].to_numpy(dtype=np.int32)

        for cc in cluster_cols:
            labels = df[cc].astype(str).to_numpy()
            cmap   = _color_map(labels)

            # Save color assignment CSV (required by napari viewer)
            color_rows = [{"cluster": lab, "R": r, "G": g, "B": b,
                           "hex": "#{:02X}{:02X}{:02X}".format(r,g,b)}
                          for lab, (r,g,b) in cmap.items()]
            pd.DataFrame(color_rows).sort_values("cluster").to_csv(
                out_dir / f"{cc}__colors.csv", index=False
            )

            _save_umap(umap_xy, labels, cmap,
                       out_dir / f"{cc}__umap.png",
                       f"{name}: {cc}")

            _save_matrixplot(df, cc, channel_labels,
                             out_dir / f"{cc}__matrixplot.png",
                             f"{name}: {cc}", row_zscore=row_zscore)

            if label_mask is not None:
                _save_colored_mask(label_mask, ids, labels, cmap,
                                   out_dir / f"{cc}__colored_mask.png",
                                   white_boundaries=white_bounds)

        _s(f"Done clustering {name} → {out_dir}")
