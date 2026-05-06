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
    Load feature_importance_mean.csv and feature_importance_foldwise.csv
    from <sample_dir>/gnn_explainability/binary/<marker>/
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
    merged_foldwise: pd.DataFrame,
    sample_names: List[str],
    title: str,
    save_path: Path,
    top_n: int = 10,
) -> None:
    """
    Violin plot matching the notebook style:
    - Features on x-axis (top-N by mean importance)
    - Samples as grouped colored violins per feature
    - Fold-wise attribution on y-axis
    - Statistical significance brackets (Mann-Whitney U)
    """
    from scipy import stats as _stats
    from itertools import combinations

    if merged_foldwise.empty:
        return

    # Find top-N features by mean importance across all samples
    mean_per_feat = (
        merged_foldwise.groupby("feature")["importance"].mean()
        .nlargest(top_n)
    )
    top_features = list(mean_per_feat.index)
    df = merged_foldwise[merged_foldwise["feature"].isin(top_features)].copy()
    df["feature"] = pd.Categorical(df["feature"], categories=top_features, ordered=True)
    df = df.sort_values("feature")

    n_feat = len(top_features)
    n_samp = len(sample_names)
    colors = plt.cm.tab10(np.linspace(0, 0.9, n_samp))
    color_map = {s: colors[i] for i, s in enumerate(sample_names)}

    fig, ax = plt.subplots(figsize=(max(8, n_feat * 1.2), 5))

    # Draw violins manually so we can control x positions
    width = 0.7 / n_samp
    for fi, feat in enumerate(top_features):
        for si, sname in enumerate(sample_names):
            vals = df[(df["feature"] == feat) & (df["sample"] == sname)]["importance"].dropna().values
            if len(vals) < 2:
                # Just plot a dot
                x = fi + (si - n_samp / 2 + 0.5) * width
                ax.scatter([x], [vals[0]] if len(vals) == 1 else [0],
                           color=color_map[sname], s=20, zorder=3)
                continue
            x_center = fi + (si - n_samp / 2 + 0.5) * width
            parts = ax.violinplot([vals], positions=[x_center], widths=width * 0.85,
                                  showmedians=True, showextrema=False)
            for pc in parts["bodies"]:
                pc.set_facecolor(color_map[sname])
                pc.set_alpha(0.75)
            parts["cmedians"].set_color("black")
            parts["cmedians"].set_linewidth(1.2)
            # Overlay jitter
            jitter = np.random.default_rng(42 + fi + si).uniform(-width * 0.2, width * 0.2, len(vals))
            ax.scatter(x_center + jitter, vals, color=color_map[sname],
                       s=12, alpha=0.8, zorder=4, edgecolors="none")

    # Significance brackets between all sample pairs for each feature
    y_max_global = df["importance"].max() if not df.empty else 1.0
    bracket_step = y_max_global * 0.08

    for fi, feat in enumerate(top_features):
        feat_data = {s: df[(df["feature"] == feat) & (df["sample"] == s)]["importance"].dropna().values
                     for s in sample_names}
        pairs = list(combinations(range(n_samp), 2))
        y_bracket = y_max_global + bracket_step

        for (i, j) in pairs:
            a_vals = feat_data[sample_names[i]]
            b_vals = feat_data[sample_names[j]]
            if len(a_vals) < 2 or len(b_vals) < 2:
                continue
            _, p = _stats.mannwhitneyu(a_vals, b_vals, alternative="two-sided")
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"

            xi = fi + (i - n_samp / 2 + 0.5) * width
            xj = fi + (j - n_samp / 2 + 0.5) * width
            ax.plot([xi, xi, xj, xj], [y_bracket - bracket_step * 0.3,
                                        y_bracket, y_bracket,
                                        y_bracket - bracket_step * 0.3],
                    color="black", linewidth=0.8)
            ax.text((xi + xj) / 2, y_bracket + bracket_step * 0.05, sig,
                    ha="center", va="bottom", fontsize=7)
            y_bracket += bracket_step * 1.1

    ax.set_xticks(range(n_feat))
    ax.set_xticklabels(top_features, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Fold-wise attribution")
    ax.set_title(title, fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    # Legend (no title)
    handles = [plt.Rectangle((0, 0), 1, 1, color=color_map[s]) for s in sample_names]
    ax.legend(handles, sample_names, loc="upper right", fontsize=8, framealpha=0.7)

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

        # Violin using foldwise data
        vl_path = marker_dir / f"{_safe(marker)}__violin.png"
        if fold_dfs:
            foldwise_merged = pd.concat(fold_dfs, ignore_index=True)
            _violin_plot(foldwise_merged, sample_names,
                         title=f"{marker} — fold-wise attribution",
                         save_path=vl_path, top_n=min(top_n, 10))
            saved_pngs.append(vl_path)

        plots[marker] = saved_pngs

    _s(f"Binary comparison done → {output_dir}")
    return plots




