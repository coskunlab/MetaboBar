"""
Cross-sample comparative analysis of GNN explainability results.

Loads feature_importance_mean.csv files from multiple sample result folders
and generates comparison plots and CSVs.

Binary task:  one CSV per marker  → feature × sample importance
Multiclass:   one CSV per cluster → feature × sample importance
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
    Load all feature_importance_mean.csv files from
    <sample_dir>/gnn_explainability/binary/<marker>/
    Returns {marker_name: DataFrame with columns [feature, mean_importance, sem, sample]}
    """
    gnn_dir = sample_dir / "gnn_explainability" / "binary"
    out: Dict[str, pd.DataFrame] = {}
    if not gnn_dir.exists():
        return out
    for marker_dir in gnn_dir.iterdir():
        if not marker_dir.is_dir():
            continue
        csv = marker_dir / "feature_importance_mean.csv"
        if not csv.exists():
            continue
        df = pd.read_csv(str(csv))
        if "feature" not in df.columns or "mean_importance" not in df.columns:
            continue
        df = df[["feature", "mean_importance", "sem"]].copy() if "sem" in df.columns else df[["feature", "mean_importance"]].assign(sem=0.0)
        df["sample"] = sample_name
        df["marker"] = marker_dir.name
        out[marker_dir.name] = df
    return out


def _load_multiclass_importance(sample_dir: Path, sample_name: str) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Load all feature_importance_mean_cluster_*.csv files from
    <sample_dir>/gnn_explainability/multiclass/<target>/
    Returns {target_name: {cluster_name: DataFrame}}
    """
    gnn_dir = sample_dir / "gnn_explainability" / "multiclass"
    out: Dict[str, Dict[str, pd.DataFrame]] = {}
    if not gnn_dir.exists():
        return out
    for target_dir in gnn_dir.iterdir():
        if not target_dir.is_dir():
            continue
        target_name = target_dir.name
        out[target_name] = {}
        for csv in target_dir.glob("feature_importance_mean_cluster_*.csv"):
            cluster_name = csv.stem.replace("feature_importance_mean_cluster_", "")
            df = pd.read_csv(str(csv))
            if "feature" not in df.columns or "mean_importance" not in df.columns:
                continue
            df = df[["feature", "mean_importance", "sem"]].copy() if "sem" in df.columns else df[["feature", "mean_importance"]].assign(sem=0.0)
            df["sample"] = sample_name
            df["cluster"] = cluster_name
            out[target_name][cluster_name] = df
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


def _violin_plot(
    merged: pd.DataFrame,
    sample_names: List[str],
    title: str,
    save_path: Path,
    top_n: int = 10,
) -> None:
    """
    Violin plot: one violin per sample for each of the top-N features.
    Long-format: feature × sample → importance value.
    """
    val_cols = [s for s in sample_names if s in merged.columns]
    if not val_cols:
        return
    merged = merged.copy()
    merged["_mean"] = merged[val_cols].mean(axis=1)
    top = merged.nlargest(top_n, "_mean")

    # Melt to long format
    long = top.melt(id_vars=["feature"], value_vars=val_cols,
                    var_name="sample", value_name="importance")

    fig, ax = plt.subplots(figsize=(max(6, len(val_cols) * 1.5),
                                    max(4, top_n * 0.5)))
    sns.violinplot(
        data=long, x="importance", y="feature", hue="sample",
        ax=ax, orient="h", inner="box", cut=0,
        palette="tab10", linewidth=0.8,
    )
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Mean importance")
    ax.set_ylabel("Feature")
    ax.tick_params(axis="y", labelsize=7)
    # Remove legend title
    legend = ax.get_legend()
    if legend:
        legend.set_title("")
    ax.grid(axis="x", alpha=0.3)
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
        for sname in sample_names:
            if marker in all_data[sname]:
                df = all_data[sname][marker][["feature", "mean_importance", "sem"]].copy()
                df = df.rename(columns={"mean_importance": sname, "sem": f"{sname}_sem"})
                dfs.append(df)

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

        # Violin
        vl_path = marker_dir / f"{_safe(marker)}__violin.png"
        _violin_plot(merged, sample_names,
                     title=f"{marker} — importance distribution",
                     save_path=vl_path, top_n=min(top_n, 10))
        saved_pngs.append(vl_path)

        plots[marker] = saved_pngs

    _s(f"Binary comparison done → {output_dir}")
    return plots


def run_multiclass_comparison(
    sample_dirs: Dict[str, Path],
    output_dir: Path,
    top_n: int = 20,
    status_cb=None,
) -> Dict[str, Dict[str, List[Path]]]:
    """
    Compare multiclass GNN importance across samples.

    Returns
    -------
    plots : {target_name: {cluster_name: [list of saved PNG paths]}}
    """
    def _s(m):
        if status_cb: status_cb(m)

    output_dir.mkdir(parents=True, exist_ok=True)
    sample_names = list(sample_dirs.keys())

    _s("Loading multiclass importance data…")
    all_data: Dict[str, Dict[str, Dict[str, pd.DataFrame]]] = {}
    for sname, sdir in sample_dirs.items():
        all_data[sname] = _load_multiclass_importance(sdir, sname)

    # Find all targets
    all_targets: set = set()
    for sname in sample_names:
        all_targets.update(all_data[sname].keys())

    plots: Dict[str, Dict[str, List[Path]]] = {}

    for target in sorted(all_targets):
        _s(f"  Target: {target}…")
        target_dir = output_dir / _safe(target)
        target_dir.mkdir(parents=True, exist_ok=True)
        plots[target] = {}

        # Find all clusters across samples for this target
        all_clusters: set = set()
        for sname in sample_names:
            if target in all_data[sname]:
                all_clusters.update(all_data[sname][target].keys())

        for cluster in sorted(all_clusters):
            _s(f"    Cluster: {cluster}…")
            cluster_dir = target_dir / f"cluster_{_safe(cluster)}"
            cluster_dir.mkdir(parents=True, exist_ok=True)

            dfs = []
            for sname in sample_names:
                if target in all_data[sname] and cluster in all_data[sname][target]:
                    df = all_data[sname][target][cluster][["feature", "mean_importance", "sem"]].copy()
                    df = df.rename(columns={"mean_importance": sname, "sem": f"{sname}_sem"})
                    dfs.append(df)

            if len(dfs) < 1:
                continue

            merged = dfs[0]
            for df in dfs[1:]:
                merged = merged.merge(df, on="feature", how="outer")
            merged = merged.fillna(0)

            csv_path = cluster_dir / f"{_safe(target)}__cluster_{_safe(cluster)}__importance_comparison.csv"
            merged.to_csv(str(csv_path), index=False)

            saved_pngs = []

            bar_path = cluster_dir / f"{_safe(target)}__cluster_{_safe(cluster)}__grouped_bar.png"
            _grouped_bar(merged, sample_names,
                         title=f"{target} | cluster {cluster} — importance across samples",
                         xlabel="Mean importance ± SEM",
                         save_path=bar_path, top_n=top_n)
            saved_pngs.append(bar_path)

            hm_path = cluster_dir / f"{_safe(target)}__cluster_{_safe(cluster)}__heatmap.png"
            _importance_heatmap(merged, sample_names,
                                title=f"{target} | cluster {cluster} — heatmap",
                                save_path=hm_path, top_n=top_n)
            saved_pngs.append(hm_path)

            vl_path = cluster_dir / f"{_safe(target)}__cluster_{_safe(cluster)}__violin.png"
            _violin_plot(merged, sample_names,
                         title=f"{target} | cluster {cluster} — distribution",
                         save_path=vl_path, top_n=min(top_n, 10))
            saved_pngs.append(vl_path)

            plots[target][cluster] = saved_pngs

    _s(f"Multiclass comparison done → {output_dir}")
    return plots
