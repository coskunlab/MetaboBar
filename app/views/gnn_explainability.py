"""
GNN Explainability view.

Two sub-tabs:
  1. Positivity (binary)   — predict positive/negative per MSI channel
  2. Clustering (multiclass) — predict cluster membership
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional

import pandas as pd
import streamlit as st

from app.utils.analysis.gnn import run_binary_gnn, run_multiclass_gnn


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

def _init_state() -> None:
    for k, v in {
        "gnn_output_dir": "",
        "gnn_binary_results": {},   # marker -> summary dict
        "gnn_multi_results":  {},   # target_name -> summary dict
    }.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _get_paths(out_dir: Path):
    """Return (cell_csv, sp_csv) from the standard output folder structure."""
    cell_csv = out_dir / "projection" / "cell_level_metabolic_table__nuclear_expanded__mean.csv"
    sp_csv   = out_dir / "superpixels" / "mbp_superpixels_mean_intensity_matrix.csv"
    return cell_csv, sp_csv


def _discover_cluster_cols(out_dir: Path) -> List[str]:
    """Find all leiden_res_* and kmeans_k_* columns from the clustered CSV."""
    clustered = out_dir / "clustering" / "cells" / "cells__clustered.csv"
    if not clustered.exists():
        return []
    try:
        df = pd.read_csv(clustered, nrows=0)
        return [c for c in df.columns
                if re.fullmatch(r"leiden_res_.+", c) or re.fullmatch(r"kmeans_k_.+", c)]
    except Exception:
        return []


def _shared_params(key_prefix: str) -> dict:
    """Render shared GNN hyperparameter controls and return a dict."""
    st.markdown("**Graph parameters**")
    c1, c2 = st.columns(2)
    with c1:
        radius_um     = st.number_input("Radius (µm)", 10.0, 500.0, 50.0, 5.0, key=f"{key_prefix}_radius")
        pixel_size_um = st.number_input("Pixel size (µm/px)", 0.1, 20.0, 2.6, 0.1, key=f"{key_prefix}_px")
    with c2:
        n_folds       = st.number_input("K-folds", 2, 10, 5, 1, key=f"{key_prefix}_folds")
        seed          = st.number_input("Random seed", 0, 99999, 42, 1, key=f"{key_prefix}_seed")

    st.markdown("**Model parameters**")
    m1, m2, m3 = st.columns(3)
    with m1:
        model_type  = st.selectbox("Model", ["GraphSAGE", "GCN", "GATv2"], key=f"{key_prefix}_model")
        hidden_dim  = st.number_input("Hidden dim", 16, 512, 64, 16, key=f"{key_prefix}_hidden")
    with m2:
        num_layers  = st.number_input("Layers", 1, 6, 3, 1, key=f"{key_prefix}_layers")
        dropout     = st.slider("Dropout", 0.0, 0.8, 0.2, 0.05, key=f"{key_prefix}_dropout")
    with m3:
        epochs      = st.number_input("Max epochs", 10, 500, 150, 10, key=f"{key_prefix}_epochs")
        patience    = st.number_input("Patience", 5, 100, 20, 5, key=f"{key_prefix}_patience")

    st.markdown("**Explainability**")
    e1, e2, e3 = st.columns(3)
    with e1:
        explain_method = st.selectbox("Method", ["saliency", "occlusion"], key=f"{key_prefix}_explain")
    with e2:
        top_k = st.number_input("Top-k features", 5, 100, 25, 5, key=f"{key_prefix}_topk")
    with e3:
        standardize = st.checkbox("Standardize features", True, key=f"{key_prefix}_std")

    return dict(
        radius_um=float(radius_um), pixel_size_um=float(pixel_size_um),
        n_folds=int(n_folds), seed=int(seed),
        model_type=model_type, hidden_dim=int(hidden_dim),
        num_layers=int(num_layers), dropout=float(dropout),
        epochs=int(epochs), patience=int(patience),
        explain_method=explain_method, top_k=int(top_k),
        standardize=standardize,
    )


def _status_cb(container):
    def _cb(msg: str):
        container.info(msg)
    return _cb


# ---------------------------------------------------------------------------
# Tab 1 – Positivity (binary)
# ---------------------------------------------------------------------------

def _tab_binary(out_dir: Optional[Path], msi_labels: List[str]) -> None:
    st.markdown("#### Positivity GNN (binary classification)")
    st.caption(
        "Trains a GNN to predict whether each cell is positive for a selected MSI channel, "
        "using the binary labels from the Positivity Thresholding step. "
        "Outputs feature importance showing which MSI channels drive positivity."
    )

    if out_dir is None:
        st.warning("Set the results root folder above.")
        return

    cell_csv, sp_csv = _get_paths(out_dir)
    binary_csv = out_dir / "positivity" / "cell_binary_labels.csv"

    missing = [str(p) for p in [cell_csv, sp_csv, binary_csv] if not p.exists()]
    if missing:
        st.warning(f"Missing files:\n" + "\n".join(f"  • {m}" for m in missing))
        return

    # Load binary labels to get available markers
    try:
        bin_df = pd.read_csv(binary_csv, nrows=0)
        available_markers = [c.replace("__positive", "") for c in bin_df.columns if c.endswith("__positive")]
    except Exception as e:
        st.error(f"Could not read binary labels CSV: {e}")
        return

    if not available_markers:
        st.warning("No positivity columns found. Run Positivity Thresholding first.")
        return

    st.markdown("**Select markers to run**")
    selected_markers = st.multiselect(
        "Markers",
        options=available_markers,
        default=available_markers[:min(3, len(available_markers))],
        key="gnn_binary_markers",
        help="Each selected marker will be trained as a separate binary classification task.",
    )

    if not selected_markers:
        st.warning("Select at least one marker.")
        return

    # Feature channel selection — user picks which channels to use as node features
    st.markdown("**Node features (MSI channels)**")
    st.caption(
        "Select which MSI channels to use as GNN node features. "
        "Consider excluding channels that directly define the positivity labels to avoid data leakage."
    )
    feature_cols_selected = st.multiselect(
        "Feature channels",
        options=msi_labels,
        default=msi_labels,
        key="gnn_binary_feat_cols",
        help="All MSI channels are selected by default.",
    )
    if not feature_cols_selected:
        st.warning("Select at least one feature channel.")
        return

    params = _shared_params("bin")

    gnn_out_dir = out_dir / "gnn_explainability" / "binary"

    if st.button("Run Binary GNN", type="primary", key="gnn_binary_run", use_container_width=True):
        status   = st.empty()
        progress = st.progress(0.0, text="Starting…")

        try:
            cell_df = pd.read_csv(cell_csv)
            sp_df   = pd.read_csv(sp_csv)
            bin_df  = pd.read_csv(binary_csv)

            # Merge binary labels into cell_df
            cell_df = cell_df.merge(bin_df, on="cell_id", how="left")

            # Feature cols = user-selected channels present in both tables
            feat_cols = [c for c in feature_cols_selected if c in cell_df.columns and c in sp_df.columns]
            if not feat_cols:
                st.error("None of the selected feature channels were found in the cell/superpixel tables.")
                return
            for i, marker in enumerate(selected_markers):
                col = f"{marker}__positive"
                if col not in cell_df.columns:
                    status.warning(f"Skipping {marker}: column {col} not found.")
                    continue

                progress.progress((i + 0.1) / total, text=f"Running {marker} ({i+1}/{total})…")

                summary = run_binary_gnn(
                    cell_df=cell_df,
                    sp_df=sp_df,
                    feature_cols=feat_cols,
                    binary_labels_col=col,
                    target_name=marker,
                    output_dir=gnn_out_dir,
                    **params,
                    status_cb=_status_cb(status),
                )
                st.session_state.gnn_binary_results[marker] = summary
                progress.progress((i + 1) / total, text=f"{marker} done.")

            progress.progress(1.0, text="All markers complete.")
            status.success(f"Binary GNN complete. Results saved to {gnn_out_dir}")

        except Exception as e:
            status.error("Binary GNN failed.")
            st.exception(e)

    # Results
    if st.session_state.gnn_binary_results:
        st.divider()
        st.markdown("**Results**")
        rows = []
        for marker, s in st.session_state.gnn_binary_results.items():
            rows.append({
                "Marker": marker,
                "ROC-AUC (mean)": f"{s.get('roc_auc_mean', float('nan')):.3f}",
                "F1 (mean)":      f"{s.get('f1_mean', float('nan')):.3f}",
                "Output dir":     s.get("output_dir", ""),
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

        # Show importance plots
        for marker, s in st.session_state.gnn_binary_results.items():
            png = Path(s.get("output_dir", "")) / "feature_importance_topk.png"
            if png.exists():
                with st.expander(f"Feature importance — {marker}", expanded=False):
                    st.image(str(png), use_container_width=True)


# ---------------------------------------------------------------------------
# Tab 2 – Clustering (multiclass)
# ---------------------------------------------------------------------------

def _tab_multiclass(out_dir: Optional[Path], msi_labels: List[str]) -> None:
    st.markdown("#### Clustering GNN (multiclass classification)")
    st.caption(
        "Trains a GNN to predict cluster membership for a selected clustering result. "
        "Per-class feature importance shows which MSI channels characterise each cluster."
    )

    if out_dir is None:
        st.warning("Set the results root folder above.")
        return

    cell_csv, sp_csv = _get_paths(out_dir)
    clustered_csv = out_dir / "clustering" / "cells" / "cells__clustered.csv"

    missing = [str(p) for p in [cell_csv, sp_csv, clustered_csv] if not p.exists()]
    if missing:
        st.warning(f"Missing files:\n" + "\n".join(f"  • {m}" for m in missing))
        return

    cluster_cols = _discover_cluster_cols(out_dir)
    if not cluster_cols:
        st.warning("No clustering columns found. Run the Clustering step first.")
        return

    selected_col = st.selectbox(
        "Clustering result to use",
        options=cluster_cols,
        key="gnn_multi_cluster_col",
        help="Select which Leiden/KMeans result to use as the classification target.",
    )

    target_name = f"cluster_{selected_col}"

    # Feature channel selection
    st.markdown("**Node features (MSI channels)**")
    st.caption(
        "Select which MSI channels to use as GNN node features. "
        "Exclude channels that were used to define the cluster labels to avoid data leakage."
    )
    multi_feat_cols = st.multiselect(
        "Feature channels",
        options=msi_labels,
        default=msi_labels,
        key="gnn_multi_feat_cols",
        help="All MSI channels are selected by default. Deselect any that were used to define the clusters.",
    )
    if not multi_feat_cols:
        st.warning("Select at least one feature channel.")
        return

    gnn_out_dir = out_dir / "gnn_explainability" / "multiclass"

    if st.button("Run Multiclass GNN", type="primary", key="gnn_multi_run", use_container_width=True):
        status   = st.empty()
        progress = st.progress(0.0, text="Starting…")

        try:
            cell_df     = pd.read_csv(cell_csv)
            sp_df       = pd.read_csv(sp_csv)
            clustered   = pd.read_csv(clustered_csv)

            # Merge cluster labels into cell_df
            if "cell_id" in clustered.columns and selected_col in clustered.columns:
                cell_df = cell_df.merge(clustered[["cell_id", selected_col]], on="cell_id", how="left")
            else:
                st.error(f"Clustered CSV missing 'cell_id' or '{selected_col}' column.")
                return

            # Drop rows with missing cluster labels
            cell_df = cell_df.dropna(subset=[selected_col]).reset_index(drop=True)

            feat_cols = [c for c in multi_feat_cols if c in cell_df.columns and c in sp_df.columns]
            if not feat_cols:
                st.error("None of the selected feature channels were found in the cell/superpixel tables.")
                return

            n_classes = cell_df[selected_col].nunique()
            progress.progress(0.05, text=f"Running {target_name} ({n_classes} classes)…")

            summary = run_multiclass_gnn(
                cell_df=cell_df,
                sp_df=sp_df,
                feature_cols=feat_cols,
                cluster_col=selected_col,
                target_name=target_name,
                output_dir=gnn_out_dir,
                **params,
                status_cb=_status_cb(status),
            )
            st.session_state.gnn_multi_results[target_name] = summary
            progress.progress(1.0, text="Done.")
            status.success(f"Multiclass GNN complete. Results saved to {gnn_out_dir}")

        except Exception as e:
            status.error("Multiclass GNN failed.")
            st.exception(e)

    # Results
    if st.session_state.gnn_multi_results:
        st.divider()
        st.markdown("**Results**")
        rows = []
        for tn, s in st.session_state.gnn_multi_results.items():
            rows.append({
                "Target":           tn,
                "Classes":          s.get("num_classes", "?"),
                "Accuracy (mean)":  f"{s.get('accuracy_mean', float('nan')):.3f}",
                "F1-macro (mean)":  f"{s.get('f1_macro_mean', float('nan')):.3f}",
                "Output dir":       s.get("output_dir", ""),
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

        for tn, s in st.session_state.gnn_multi_results.items():
            out_path = Path(s.get("output_dir", ""))
            pngs = sorted(out_path.glob("feature_importance_topk_*.png"))
            if pngs:
                with st.expander(f"Feature importance — {tn}", expanded=False):
                    cols = st.columns(min(3, len(pngs)))
                    for j, png in enumerate(pngs):
                        class_name = png.stem.replace("feature_importance_topk_", "")
                        with cols[j % len(cols)]:
                            st.caption(f"Class: {class_name}")
                            st.image(str(png), use_container_width=True)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def render_gnn_explainability() -> None:
    _init_state()

    msi_labels = (
        st.session_state.get("ana_msi_labels_edited") or
        st.session_state.get("msi_labels") or []
    )
    ana_out = st.session_state.get("ana_output_dir", "").strip()

    if not msi_labels and not ana_out:
        return

    with st.expander("GNN Explainability", expanded=False):

        st.markdown("#### GNN Explainability")
        st.caption(
            "Train Graph Neural Networks on the mixed cell+superpixel spatial graph "
            "to identify which MSI channels drive cell positivity or cluster membership."
        )

        # Output folder
        out_val = st.text_input(
            "Results root folder",
            value=st.session_state.get("gnn_output_dir") or ana_out,
            key="gnn_out_dir_input",
            placeholder=r"e.g. Y:\results\experiment1",
        )
        st.session_state.gnn_output_dir = out_val
        out_dir = Path(out_val.strip()) if out_val.strip() else None

        st.divider()

        tab_bin, tab_multi = st.tabs(["Positivity (binary)", "Clustering (multiclass)"])

        with tab_bin:
            _tab_binary(out_dir, msi_labels)

        with tab_multi:
            _tab_multiclass(out_dir, msi_labels)
