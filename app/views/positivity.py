"""
Positivity thresholding view.

Sits between Analysis and the napari launcher.
Thresholds each MSI channel using GMM/Otsu/quantile methods and
produces binary cell labels + overlay TIFFs for the napari viewer.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st

from app.utils.analysis.positivity import run_positivity_thresholding


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

def _init_state() -> None:
    defaults = {
        "pos_threshold_df": None,
        "pos_binary_df": None,
        "pos_output_dir": "",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def render_positivity() -> None:
    _init_state()

    # Only show when MSI labels and output dir are available
    msi_labels = (
        st.session_state.get("ana_msi_labels_edited") or
        st.session_state.get("msi_labels") or []
    )
    ana_out = st.session_state.get("ana_output_dir", "").strip()

    if not msi_labels:
        return

    with st.expander("Positivity Thresholding", expanded=False):

        st.markdown("#### Per-channel positivity thresholding")
        st.caption(
            "Fits a threshold to each MSI channel's per-cell intensity distribution "
            "and assigns binary positive/negative labels. "
            "Outputs are used by the napari viewer's positivity overlay."
        )

        # ---- Output folder ----
        out_val = st.text_input(
            "Results root folder",
            value=st.session_state.pos_output_dir or ana_out,
            key="pos_out_dir_input",
            placeholder=r"e.g. Y:\results\experiment1",
        )
        st.session_state.pos_output_dir = out_val
        out_dir = Path(out_val.strip()) if out_val.strip() else None

        # Auto-detect cell CSV and expanded mask from output dir — no need to show inputs
        cell_csv_val = ""
        mask_val = ""
        if out_dir:
            candidate = out_dir / "projection" / "cell_level_metabolic_table__nuclear_expanded__mean.csv"
            if candidate.exists():
                cell_csv_val = str(candidate)
            candidate = out_dir / "segmentation" / "nuclear_mask_expanded.tif"
            if candidate.exists():
                mask_val = str(candidate)

        if out_dir and not cell_csv_val:
            st.warning(
                f"Cell metabolic table not found at "
                f"`{out_dir / 'projection' / 'cell_level_metabolic_table__nuclear_expanded__mean.csv'}`. "
                "Run the LR→HR Projection step first."
            )

        st.divider()

        # ---- Parameters ----
        st.markdown("**Thresholding parameters**")

        p1, p2 = st.columns(2)
        with p1:
            method = st.selectbox(
                "Threshold method",
                ["top_component_quantile", "gmm", "otsu", "upper_quantile"],
                index=0,
                key="pos_method",
                help=(
                    "top_component_quantile: GMM → take quantile of top component (recommended)\n"
                    "gmm: 2-component GMM intersection\n"
                    "otsu: Otsu's method\n"
                    "upper_quantile: simple upper percentile"
                ),
            )
            gmm_components = st.number_input(
                "GMM components (top_component_quantile)",
                min_value=2, max_value=10, value=3, step=1,
                key="pos_gmm_components",
            )
            component_quantile = st.slider(
                "Component quantile",
                0.01, 0.99, 0.60, 0.01,
                key="pos_component_quantile",
                help="Quantile within the top GMM component used as threshold.",
            )

        with p2:
            fallback_quantile = st.slider(
                "Fallback quantile",
                0.50, 0.99, 0.75, 0.01,
                key="pos_fallback_quantile",
                help="Used when GMM fails or class balance is extreme.",
            )
            min_pos = st.number_input(
                "Min positive fraction",
                min_value=0.001, max_value=0.5, value=0.01, step=0.005,
                format="%.3f",
                key="pos_min_pos",
            )
            max_pos = st.number_input(
                "Max positive fraction",
                min_value=0.5, max_value=0.999, value=0.99, step=0.005,
                format="%.3f",
                key="pos_max_pos",
            )
            save_overlays = st.checkbox(
                "Save positivity overlay TIFFs",
                value=True,
                key="pos_save_overlays",
                help="Saves a coloured TIFF per channel (red=positive, grey=negative).",
            )

        # ---- Channel selection ----
        st.markdown("**Channels to threshold**")
        selected_channels = st.multiselect(
            "MSI channels",
            options=msi_labels,
            default=msi_labels,
            key="pos_channels",
        )

        if not selected_channels:
            st.warning("Select at least one channel.")

        # ---- Validate ----
        can_run = (
            bool(cell_csv_val) and
            Path(cell_csv_val).exists() and
            out_dir is not None and
            bool(selected_channels)
        )

        # ---- Run ----
        if st.button(
            "Run Positivity Thresholding",
            type="primary",
            key="pos_run_btn",
            disabled=not can_run,
            use_container_width=True,
        ):
            status  = st.empty()
            progress = st.progress(0.0, text="Starting…")
            total = len(selected_channels)
            counter = [0]

            def _cb(msg: str) -> None:
                # Parse "Thresholding channel X/Y" for progress
                import re
                m = re.search(r"(\d+)/(\d+)", msg)
                if m:
                    done, tot = int(m.group(1)), int(m.group(2))
                    progress.progress(done / tot, text=msg)
                else:
                    status.info(msg)

            try:
                pos_dir = out_dir / "positivity"
                threshold_df, binary_df = run_positivity_thresholding(
                    cell_csv=Path(cell_csv_val),
                    expanded_mask_path=Path(mask_val) if mask_val else Path(""),
                    channel_labels=selected_channels,
                    output_dir=pos_dir,
                    method=method,
                    gmm_components=int(gmm_components),
                    component_quantile=float(component_quantile),
                    fallback_quantile=float(fallback_quantile),
                    min_pos_fraction=float(min_pos),
                    max_pos_fraction=float(max_pos),
                    save_overlays=save_overlays,
                    status_cb=_cb,
                )
                st.session_state.pos_threshold_df = threshold_df
                st.session_state.pos_binary_df    = binary_df
                progress.progress(1.0, text="Done.")
                status.success(
                    f"Thresholded {len(threshold_df)} channels. "
                    f"Results saved to {pos_dir}"
                )
            except Exception as e:
                status.error("Positivity thresholding failed.")
                st.exception(e)

        # ---- Results ----
        if st.session_state.pos_threshold_df is not None:
            st.divider()
            st.markdown("**Results**")

            tdf = st.session_state.pos_threshold_df
            st.dataframe(
                tdf[["channel", "threshold", "method",
                     "n_positive", "n_negative", "positive_fraction"]],
                use_container_width=True,
                height=min(400, 35 * (len(tdf) + 1)),
            )

            dl1, dl2 = st.columns(2)
            with dl1:
                st.download_button(
                    "Download threshold CSV",
                    data=tdf.to_csv(index=False).encode("utf-8"),
                    file_name="protein_marker_thresholds.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
            with dl2:
                bdf = st.session_state.pos_binary_df
                if bdf is not None:
                    st.download_button(
                        "Download binary labels CSV",
                        data=bdf.to_csv(index=False).encode("utf-8"),
                        file_name="cell_binary_labels.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )
