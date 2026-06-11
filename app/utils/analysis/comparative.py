"""
Cross-sample comparative analysis of GNN explainability results.

Loads feature_importance_mean.csv / feature_importance_foldwise.csv
from multiple sample result folders and generates comparison plots and CSVs.

Binary task only: one CSV + plots per marker.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe(s: str) -> str:
    return re.sub(r"[^\w\-.]", "_", str(s))


def _load_binary_importance(sample_dir: Path, sample_name: str) -> Dict[str, pd.DataFrame]:
    """
    Load feature_importance_mean.csv from <sample_dir>/gnn_explainability/binary/<marker>/
    Returns {marker_name: {"mean": df, "foldwise": df}}
    """
    gnn_dir = sample_dir / "gnn_explainability" / "binary"
    out: Dict[str, dict] = {}
    if not gnn_dir.exists():
        return out
    for marker_dir in gnn_dir.iterdir():
        if not marker_dir.is_dir():
            continue
        mean_csv = marker_dir / "feature_importance_mean.csv"
        fold_csv = marker_dir / "feature_importance_foldwise.csv"
        if not mean_csv.exists():
            continue
        mean_df = pd.read_csv(str(mean_csv))
        if "feature" not in mean_df.columns or "mean_importance" not in mean_df.columns:
            continue
        mean_df = mean_df[["feature", "mean_importance"]].copy()
        if "sem" in pd.read_csv(str(mean_csv), nrows=0).columns:
            mean_df["sem"] = pd.read_csv(str(mean_csv))["sem"]
        else:
            mean_df["sem"] = 0.0
        mean_df["sample"] = sample_name

        fold_df = pd.DataFrame()
        if fold_csv.exists():
            fold_df = pd.read_csv(str(fold_csv))
            fold_df["sample"] = sample_name

        out[marker_dir.name] = {"mean": mean_df, "foldwise": fold_df}
    return out


def _load_multiclass_importance(sample_dir: Path, sample_name: str,
                                 source: str = "annotations") -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Load per-class feature_importance_mean_cluster_*.csv files from
    <sample_dir>/gnn_explainability/<source>/<target>/

    source: "annotations" or "multiclass"

    Returns {target_name: {class_name: {"mean": df}}}
    """
    gnn_dir = sample_dir / "gnn_explainability" / source
    out: Dict[str, Dict[str, dict]] = {}
    if not gnn_dir.exists():
        return out
    for target_dir in gnn_dir.iterdir():
        if not target_dir.is_dir():
            continue
        class_data: Dict[str, dict] = {}
        for csv_path in target_dir.glob("feature_importance_mean_cluster_*.csv"):
            class_name = csv_path.stem.replace("feature_importance_mean_cluster_", "")
            df = pd.read_csv(str(csv_path))
            if "feature" not in df.columns or "mean_importance" not in df.columns:
                continue
            mean_df = df[["feature", "mean_importance"]].copy()
            mean_df["sem"] = df["sem"] if "sem" in df.columns else 0.0
            mean_df["sample"] = sample_name
            class_data[class_name] = {"mean": mean_df}
        if class_data:
            out[target_dir.name] = class_data
    return out







# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _grouped_bar(
    merged: pd.DataFrame,
    sample_names: List[str],
    title: str,
    xlabel: str,
    save_path: Path,
    top_n: int = 20,
) -> None:
    """
    Horizontal grouped bar chart: top_n features, one bar per sample.
    merged has columns: feature, <sample1>, <sample1>_sem, <sample2>, ...
    """
    # Rank features by mean importance across samples
    val_cols = [s for s in sample_names if s in merged.columns]
    if not val_cols:
        return
    merged = merged.copy()
    merged["_mean"] = merged[val_cols].mean(axis=1)
    top = merged.nlargest(top_n, "_mean").iloc[::-1]  # ascending for horizontal

    n_samples = len(val_cols)
    bar_h = 0.8 / n_samples
    y_base = np.arange(len(top))

    fig, ax = plt.subplots(figsize=(10, max(5, len(top) * 0.45)))
    colors = plt.cm.tab10(np.linspace(0, 0.9, n_samples))

    for i, sname in enumerate(val_cols):
        sem_col = f"{sname}_sem"
        vals = top[sname].fillna(0).values
        sems = top[sem_col].fillna(0).values if sem_col in top.columns else np.zeros(len(top))
        offset = (i - n_samples / 2 + 0.5) * bar_h
        ax.barh(y_base + offset, vals, height=bar_h * 0.9,
                xerr=sems, color=colors[i], label=sname,
                ecolor="black", capsize=2)

    ax.set_yticks(y_base)
    ax.set_yticklabels(top["feature"].values, fontsize=8)
    ax.set_xlabel(xlabel)
    ax.set_title(title, fontsize=10)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    fig.savefig(str(save_path), dpi=150)
    plt.close(fig)


def _importance_heatmap(
    merged: pd.DataFrame,
    sample_names: List[str],
    title: str,
    save_path: Path,
    top_n: int = 30,
) -> None:
    """
    Heatmap: features (rows) × samples (columns), values = mean_importance.
    """
    val_cols = [s for s in sample_names if s in merged.columns]
    if not val_cols:
        return
    merged = merged.copy()
    merged["_mean"] = merged[val_cols].mean(axis=1)
    top = merged.nlargest(top_n, "_mean")
    mat = top.set_index("feature")[val_cols]

    fig_h = max(5, len(mat) * 0.35)
    fig_w = max(5, len(val_cols) * 1.5)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    sns.heatmap(mat, ax=ax, cmap="viridis", linewidths=0.3,
                cbar_kws={"label": "Mean importance"})
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("")
    ax.set_ylabel("Feature")
    ax.tick_params(axis="x", rotation=90, labelsize=8)
    ax.tick_params(axis="y", labelsize=7)
    # Remove colorbar title (legend title equivalent)
    ax.collections[0].colorbar.set_label("")
    plt.tight_layout()
    fig.savefig(str(save_path), dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main analysis functions
# ---------------------------------------------------------------------------

def run_binary_comparison(
    sample_dirs: Dict[str, Path],
    output_dir: Path,
    top_n: int = 20,
    status_cb=None,
) -> Dict[str, List[Path]]:
    """
    Compare binary GNN importance across samples.

    Parameters
    ----------
    sample_dirs : {sample_name: result_root_path}
    output_dir  : where to save outputs
    top_n       : top features to show in plots

    Returns
    -------
    plots : {marker_name: [list of saved PNG paths]}
    """
    def _s(m):
        if status_cb: status_cb(m)

    output_dir.mkdir(parents=True, exist_ok=True)
    sample_names = list(sample_dirs.keys())

    # Load all importance data
    _s("Loading binary importance data…")
    all_data: Dict[str, Dict[str, pd.DataFrame]] = {}
    for sname, sdir in sample_dirs.items():
        all_data[sname] = _load_binary_importance(sdir, sname)

    # Find all markers present in at least 2 samples
    all_markers = set()
    for sname in sample_names:
        all_markers.update(all_data[sname].keys())

    plots: Dict[str, List[Path]] = {}

    for marker in sorted(all_markers):
        _s(f"  Comparing marker: {marker}…")
        marker_dir = output_dir / _safe(marker)
        marker_dir.mkdir(parents=True, exist_ok=True)

        # Merge importance across samples
        dfs = []
        fold_dfs = []
        for sname in sample_names:
            if marker in all_data[sname]:
                entry = all_data[sname][marker]
                df = entry["mean"][["feature", "mean_importance", "sem"]].copy()
                df = df.rename(columns={"mean_importance": sname, "sem": f"{sname}_sem"})
                dfs.append(df)
                if not entry["foldwise"].empty:
                    fdf = entry["foldwise"][["feature", "importance", "fold"]].copy()
                    fdf["sample"] = sname
                    fold_dfs.append(fdf)

        if len(dfs) < 1:
            continue

        merged = dfs[0]
        for df in dfs[1:]:
            merged = merged.merge(df, on="feature", how="outer")
        merged = merged.fillna(0)

        # Save CSV
        csv_path = marker_dir / f"{_safe(marker)}__importance_comparison.csv"
        merged.to_csv(str(csv_path), index=False)

        saved_pngs = []

        # Grouped bar chart
        bar_path = marker_dir / f"{_safe(marker)}__grouped_bar.png"
        _grouped_bar(merged, sample_names,
                     title=f"{marker} — feature importance across samples",
                     xlabel="Mean importance ± SEM",
                     save_path=bar_path, top_n=top_n)
        saved_pngs.append(bar_path)

        # Heatmap
        hm_path = marker_dir / f"{_safe(marker)}__heatmap.png"
        _importance_heatmap(merged, sample_names,
                            title=f"{marker} — importance heatmap",
                            save_path=hm_path, top_n=top_n)
        saved_pngs.append(hm_path)

        plots[marker] = saved_pngs

    _s(f"Binary comparison done → {output_dir}")
    return plots


def run_multiclass_comparison(
    sample_dirs: Dict[str, Path],
    output_dir: Path,
    source: str = "annotations",
    top_n: int = 20,
    status_cb=None,
) -> Dict[str, List[Path]]:
    """
    Compare multiclass GNN importance (annotations or clustering) across samples.
    One set of plots per target × class.

    Parameters
    ----------
    sample_dirs : {sample_name: result_root_path}
    output_dir  : where to save outputs
    source      : "annotations" or "multiclass"
    top_n       : top features to show
    """
    def _s(m):
        if status_cb: status_cb(m)

    output_dir.mkdir(parents=True, exist_ok=True)
    sample_names = list(sample_dirs.keys())

    _s(f"Loading {source} importance data…")
    all_data: Dict[str, Dict] = {}
    for sname, sdir in sample_dirs.items():
        all_data[sname] = _load_multiclass_importance(sdir, sname, source)

    # Collect all (target, class) pairs present in at least one sample
    all_targets: set = set()
    for sname in sample_names:
        all_targets.update(all_data[sname].keys())

    plots: Dict[str, List[Path]] = {}

    for target in sorted(all_targets):
        _s(f"  Comparing target: {target}…")
        target_dir = output_dir / _safe(target)
        target_dir.mkdir(parents=True, exist_ok=True)

        # Collect all class names across samples
        all_classes: set = set()
        for sname in sample_names:
            if target in all_data[sname]:
                all_classes.update(all_data[sname][target].keys())

        saved_pngs = []

        for class_name in sorted(all_classes):
            dfs = []
            for sname in sample_names:
                if target in all_data[sname] and class_name in all_data[sname][target]:
                    df = all_data[sname][target][class_name]["mean"][
                        ["feature", "mean_importance", "sem"]
                    ].copy()
                    df = df.rename(columns={"mean_importance": sname, "sem": f"{sname}_sem"})
                    dfs.append(df)

            if len(dfs) < 1:
                continue

            merged = dfs[0]
            for df in dfs[1:]:
                merged = merged.merge(df, on="feature", how="outer")
            merged = merged.fillna(0)

            safe_class = _safe(class_name)
            csv_path = target_dir / f"{_safe(target)}__{safe_class}__importance_comparison.csv"
            merged.to_csv(str(csv_path), index=False)

            bar_path = target_dir / f"{_safe(target)}__{safe_class}__grouped_bar.png"
            _grouped_bar(merged, sample_names,
                         title=f"{target} · {class_name} — importance across samples",
                         xlabel="Mean importance ± SEM",
                         save_path=bar_path, top_n=top_n)
            saved_pngs.append(bar_path)

            hm_path = target_dir / f"{_safe(target)}__{safe_class}__heatmap.png"
            _importance_heatmap(merged, sample_names,
                                title=f"{target} · {class_name} — heatmap",
                                save_path=hm_path, top_n=top_n)
            saved_pngs.append(hm_path)

        plots[target] = saved_pngs

    _s(f"{source} comparison done → {output_dir}")
    return plots




