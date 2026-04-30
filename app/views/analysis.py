"""
Analysis view — 6-tab pipeline:
  1. Cell Segmentation (Mesmer)
  2. Nuclei Expansion
  3. MBP Mask
  4. LR→HR Projection
  5. Superpixel Segmentation
  6. Clustering
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import List, Optional

import numpy as np
import streamlit as st
import tifffile as tiff

from app.utils.analysis.segmentation import MESMER_PYTHON, run_mesmer_segmentation
from app.utils.analysis.masks import expand_nuclear_mask, make_mbp_mask
from app.utils.analysis.projection import run_projection_pipeline
from app.utils.analysis.superpixels import run_superpixel_segmentation
from app.utils.analysis.clustering import run_clustering


# ---------------------------------------------------------------------------
# Session state init
# ---------------------------------------------------------------------------

def _init_analysis_state() -> None:
    defaults = {
        "ana_output_dir": "",
        "ana_if_labels_edited": None,
        "ana_msi_labels_edited": None,
        "ana_nuclear_labeled": None,
        "ana_nuclear_binary": None,
        "ana_nuclear_expanded": None,
        "ana_nuclear_expanded_binary": None,
        "ana_mbp_mask": None,
        "ana_projected": None,
        "ana_sp_label_mask": None,
        "ana_sp_stats_csv": None,
        "ana_cell_csv": None,
        "ana_expanded_cell_csv": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _get_stacks():
    """Return (if_stack, if_labels, msi_stack, msi_labels) from session state."""
    return (
        st.session_state.get("if_stack"),
        st.session_state.get("if_labels") or [],
        st.session_state.get("msi_stack"),
        st.session_state.get("msi_labels") or [],
    )


def _output_dir() -> Optional[Path]:
    d = st.session_state.get("ana_output_dir", "").strip()
    return Path(d) if d else None


def _status_box(container):
    """Return a status callback that writes to a Streamlit container."""
    def _cb(msg: str):
        container.info(msg)
    return _cb


def _progress_cb(status_container, progress_bar):
    """
    Return a callback(fraction, msg) that updates both a progress bar
    and a status text box.
    fraction: 0.0 – 1.0
    msg: status string
    """
    def _cb(fraction: float, msg: str = ""):
        progress_bar.progress(min(1.0, max(0.0, fraction)), text=msg)
        if msg:
            status_container.info(msg)
    return _cb


def _channel_label_editor(
    key: str,
    labels: List[str],
    title: str,
) -> List[str]:
    """Editable text area for channel labels (one per line)."""
    # Use a separate session-state key for the text content so we can
    # pre-populate it without being overridden by Streamlit's widget cache.
    text_key = f"{key}_text"
    default_text = "\n".join(labels)

    # Only set the default if the key doesn't exist yet, or if the stored
    # text is empty but we now have labels to show.
    if text_key not in st.session_state or (
        not st.session_state[text_key].strip() and default_text.strip()
    ):
        st.session_state[text_key] = default_text

    edited = st.text_area(
        title,
        height=min(300, max(80, 20 * len(labels))),
        key=text_key,
        help="One label per line. Order must match channel order in the stack.",
    )
    return [l.strip() for l in edited.splitlines() if l.strip()]


def _tiff_download_button(label: str, arr: np.ndarray, filename: str, key: str) -> None:
    buf = io.BytesIO()
    tiff.imwrite(buf, arr)
    st.download_button(label, data=buf.getvalue(),
                       file_name=filename, mime="image/tiff",
                       key=key, use_container_width=True)


# ---------------------------------------------------------------------------
# Tab 1 – Cell Segmentation
# ---------------------------------------------------------------------------

def _tab_segmentation(if_stack, if_labels):
    st.markdown("#### Cell Segmentation (Mesmer)")
    st.caption(
        "Runs nuclear segmentation using DeepCell Mesmer in the `mesmer` conda env. "
        "GPU is tried first; falls back to CPU automatically."
    )

    if if_stack is None:
        st.warning("Load an IF image first.")
        return

    out_dir = _output_dir()
    if out_dir is None:
        st.warning("Set an output folder above.")
        return

    c1, c2 = st.columns(2)
    with c1:
        ch_opts = [f"{i}: {if_labels[i]}" if i < len(if_labels) else f"C{i}"
                   for i in range(if_stack.shape[0])]
        ch_sel = st.selectbox("Nuclear channel (DAPI/Hoechst)", ch_opts,
                              key="ana_seg_channel")
        ch_idx = ch_opts.index(ch_sel)
        tile_size = st.number_input("Tile size (px)", 256, 4096, 1024, 256,
                                    key="ana_seg_tile")
        image_mpp = st.number_input("Image resolution (µm/px)", 0.1, 10.0, 1.0, 0.1,
                                    key="ana_seg_mpp")

    with c2:
        use_area_filter = st.checkbox("Filter area outliers", True, key="ana_seg_area_filter")
        area_method = st.selectbox("Area filter method", ["mad_log", "iqr_log"],
                                   key="ana_seg_area_method")
        robust_z = st.number_input("Robust z threshold", 1.0, 10.0, 3.5, 0.5,
                                   key="ana_seg_robust_z")
        iqr_mult = st.number_input("IQR multiplier", 0.5, 5.0, 1.5, 0.5,
                                   key="ana_seg_iqr_mult")
        mesmer_py = st.text_input("Mesmer Python path", MESMER_PYTHON,
                                  key="ana_seg_python")
        deepcell_token = st.text_input(
            "DeepCell access token",
            value=st.session_state.get("ana_deepcell_token", ""),
            key="ana_deepcell_token_input",
            type="password",
            help=(
                "Required on first run to download the Mesmer model (~100 MB). "
                "Create a free token at https://users.deepcell.org — takes 30 seconds. "
                "Leave blank if the model is already cached on this machine."
            ),
        )
        if deepcell_token:
            st.session_state["ana_deepcell_token"] = deepcell_token

    if st.button("Run Segmentation", type="primary", key="ana_seg_run",
                 use_container_width=True):
        status = st.empty()
        progress = st.progress(0.0, text="Starting segmentation…")
        try:
            seg_dir = out_dir / "segmentation"

            def _seg_progress(frac: float, msg: str = ""):
                progress.progress(min(1.0, frac), text=msg)
                if msg:
                    status.info(msg)

            labeled, binary = run_mesmer_segmentation(
                if_stack=if_stack,
                if_channel_idx=ch_idx,
                output_dir=seg_dir,
                tile_size=int(tile_size),
                image_mpp=float(image_mpp),
                use_area_filter=use_area_filter,
                area_method=area_method,
                robust_z_thresh=float(robust_z),
                iqr_multiplier=float(iqr_mult),
                mesmer_python=mesmer_py,
                deepcell_token=deepcell_token,
                status_cb=lambda msg: status.info(msg),
                progress_cb=_seg_progress,
            )
            st.session_state.ana_nuclear_labeled = labeled
            st.session_state.ana_nuclear_binary  = binary
            progress.progress(1.0, text="Segmentation complete.")
            status.success(f"Segmentation complete — {int(labeled.max()):,} cells. "
                           f"Saved to {seg_dir}")
        except Exception as e:
            status.error("Segmentation failed.")
            st.exception(e)

    if st.session_state.ana_nuclear_labeled is not None:
        st.divider()
        st.caption(f"Nuclear mask: {st.session_state.ana_nuclear_labeled.shape}, "
                   f"{int(st.session_state.ana_nuclear_labeled.max()):,} cells")
        dc1, dc2 = st.columns(2)
        with dc1:
            _tiff_download_button("Download nuclear mask (labeled)",
                                  st.session_state.ana_nuclear_labeled,
                                  "nuclear_mask.tif", "dl_nuclear_labeled")
        with dc2:
            _tiff_download_button("Download nuclear mask (binary)",
                                  st.session_state.ana_nuclear_binary,
                                  "nuclear_mask_binary.tif", "dl_nuclear_binary")


# ---------------------------------------------------------------------------
# Tab 2 – Nuclei Expansion
# ---------------------------------------------------------------------------

def _tab_expansion():
    st.markdown("#### Nuclei Expansion")

    if st.session_state.ana_nuclear_labeled is None:
        st.warning("Run cell segmentation first.")
        return

    out_dir = _output_dir()
    if out_dir is None:
        st.warning("Set an output folder above.")
        return

    c1, c2 = st.columns(2)
    with c1:
        expand_um = st.number_input("Expansion distance (µm)", 1.0, 100.0, 15.0, 1.0,
                                    key="ana_exp_um")
        pixel_um  = st.number_input("Pixel size (µm/px)", 0.1, 20.0, 2.6, 0.1,
                                    key="ana_exp_pixel_um")
    with c2:
        st.caption(f"Expansion in pixels: "
                   f"{expand_um / pixel_um:.2f} px")

    if st.button("Run Expansion", type="primary", key="ana_exp_run",
                 use_container_width=True):
        status = st.empty()
        progress = st.progress(0.0, text="Expanding nuclei…")
        try:
            seg_dir = out_dir / "segmentation"
            progress.progress(0.3, text="Expanding labels…")
            expanded, exp_binary = expand_nuclear_mask(
                labeled_mask=st.session_state.ana_nuclear_labeled,
                tissue_mask=None,
                expand_um=float(expand_um),
                pixel_size_um=float(pixel_um),
                output_dir=seg_dir,
            )
            st.session_state.ana_nuclear_expanded        = expanded
            st.session_state.ana_nuclear_expanded_binary = exp_binary
            progress.progress(1.0, text="Expansion complete.")
            status.success(f"Expansion complete — {int(expanded.max()):,} cells. "
                           f"Saved to {seg_dir}")
        except Exception as e:
            status.error("Expansion failed.")
            st.exception(e)

    if st.session_state.ana_nuclear_expanded is not None:
        st.divider()
        dc1, dc2 = st.columns(2)
        with dc1:
            _tiff_download_button("Download expanded mask (labeled)",
                                  st.session_state.ana_nuclear_expanded,
                                  "nuclear_mask_expanded.tif", "dl_exp_labeled")
        with dc2:
            _tiff_download_button("Download expanded mask (binary)",
                                  st.session_state.ana_nuclear_expanded_binary,
                                  "nuclear_mask_expanded_binary.tif", "dl_exp_binary")


# ---------------------------------------------------------------------------
# Tab 3 – MBP Mask
# ---------------------------------------------------------------------------

def _tab_mbp(if_stack, if_labels):
    st.markdown("#### MBP Mask")

    if if_stack is None:
        st.warning("Load an IF image first.")
        return
    if st.session_state.ana_nuclear_binary is None:
        st.warning("Run cell segmentation first.")
        return

    out_dir = _output_dir()
    if out_dir is None:
        st.warning("Set an output folder above.")
        return

    c1, c2 = st.columns(2)
    with c1:
        ch_opts = [f"{i}: {if_labels[i]}" if i < len(if_labels) else f"C{i}"
                   for i in range(if_stack.shape[0])]
        ch_sel = st.selectbox("MBP channel", ch_opts, key="ana_mbp_channel")
        ch_idx = ch_opts.index(ch_sel)
        percentile = st.slider("Threshold percentile", 1.0, 99.0, 50.0, 1.0,
                               key="ana_mbp_percentile")
        sigma = st.number_input("Gaussian sigma", 0.0, 10.0, 1.0, 0.5,
                                key="ana_mbp_sigma")

    with c2:
        min_obj  = st.number_input("Min object size (px)", 0, 1000, 20, 5,
                                   key="ana_mbp_min_obj")
        min_hole = st.number_input("Min hole size (px)", 0, 1000, 20, 5,
                                   key="ana_mbp_min_hole")
        open_r   = st.number_input("Opening radius", 0, 10, 1, 1,
                                   key="ana_mbp_open_r")
        close_r  = st.number_input("Closing radius", 0, 10, 1, 1,
                                   key="ana_mbp_close_r")

    if st.button("Run MBP Mask", type="primary", key="ana_mbp_run",
                 use_container_width=True):
        status = st.empty()
        progress = st.progress(0.0, text="Building MBP mask…")
        try:
            seg_dir = out_dir / "segmentation"
            progress.progress(0.2, text="Normalising and thresholding MBP channel…")
            mbp_mask = make_mbp_mask(
                if_stack=if_stack,
                mbp_channel_idx=ch_idx,
                nuclear_binary=st.session_state.ana_nuclear_binary,
                mbp_percentile=float(percentile),
                gaussian_sigma=float(sigma),
                min_object_size=int(min_obj),
                min_hole_size=int(min_hole),
                open_radius=int(open_r),
                close_radius=int(close_r),
                output_dir=seg_dir,
            )
            st.session_state.ana_mbp_mask = mbp_mask
            progress.progress(1.0, text="MBP mask complete.")
            status.success(f"MBP mask complete — "
                           f"{int(mbp_mask.sum()):,} pixels. Saved to {seg_dir}")
        except Exception as e:
            status.error("MBP mask failed.")
            st.exception(e)

    if st.session_state.ana_mbp_mask is not None:
        st.divider()
        _tiff_download_button("Download MBP mask",
                              (st.session_state.ana_mbp_mask.astype(np.uint8) * 255),
                              "mbp_mask_binary.tif", "dl_mbp")


# ---------------------------------------------------------------------------
# Tab 4 – LR→HR Projection
# ---------------------------------------------------------------------------

def _tab_projection(msi_stack, if_stack, msi_labels):
    st.markdown("#### LR → HR Projection")
    st.caption(
        "Projects the registered MSI stack from its native resolution to IF resolution "
        "using Gaussian-weighted interpolation, then quantifies per-cell mean intensities."
    )

    if msi_stack is None:
        st.warning("Load (or register) an MSI stack first.")
        return
    if if_stack is None:
        st.warning("Load an IF image first.")
        return
    if st.session_state.ana_nuclear_labeled is None:
        st.warning("Run cell segmentation first.")
        return
    if st.session_state.ana_nuclear_expanded is None:
        st.warning("Run nuclei expansion first.")
        return

    out_dir = _output_dir()
    if out_dir is None:
        st.warning("Set an output folder above.")
        return

    c1, c2 = st.columns(2)
    with c1:
        sigma_lr = st.number_input("Gaussian sigma (LR pixels)", 0.1, 5.0, 0.75, 0.05,
                                   key="ana_proj_sigma")
        radius_lr = st.number_input("Gaussian radius (LR pixels)", 1, 10, 2, 1,
                                    key="ana_proj_radius")

    if st.button("Run Projection", type="primary", key="ana_proj_run",
                 use_container_width=True):
        status = st.empty()
        progress = st.progress(0.0, text="Starting projection…")
        try:
            proj_dir = out_dir / "projection"
            C = msi_stack.shape[0]

            def _proj_cb(fraction: float, msg: str = ""):
                progress.progress(min(1.0, fraction), text=msg)
                if msg:
                    status.info(msg)

            # Wrap run_projection_pipeline with per-channel progress
            # The pipeline has: weights (5%), C channels (80%), 2 cell tables (15%)
            channel_counter = [0]
            def _status_with_progress(msg: str):
                if msg.startswith("Projecting channel"):
                    channel_counter[0] += 1
                    frac = 0.05 + 0.80 * (channel_counter[0] / max(C, 1))
                    _proj_cb(frac, msg)
                elif "cell table" in msg.lower() or "saved" in msg.lower():
                    _proj_cb(0.90, msg)
                else:
                    _proj_cb(0.05, msg)

            result = run_projection_pipeline(
                msi_stack=msi_stack,
                if_shape=if_stack.shape[-2:],
                nuclear_binary=st.session_state.ana_nuclear_binary.astype(bool),
                nuclear_labeled=st.session_state.ana_nuclear_labeled,
                nuclear_expanded=st.session_state.ana_nuclear_expanded,
                channel_labels=msi_labels,
                output_dir=out_dir / "projection",
                sigma_lr=float(sigma_lr),
                radius_lr=int(radius_lr),
                status_cb=_status_with_progress,
            )
            st.session_state.ana_projected    = result["projected_tif"]
            st.session_state.ana_cell_csv     = result["nuclear_csv"]
            st.session_state.ana_expanded_cell_csv = result["expanded_csv"]
            progress.progress(1.0, text="Projection complete.")
            status.success("Projection complete.")
        except Exception as e:
            status.error("Projection failed.")
            st.exception(e)

    if st.session_state.ana_cell_csv is not None:
        st.divider()
        import pandas as pd
        dc1, dc2 = st.columns(2)
        with dc1:
            csv_path = st.session_state.ana_cell_csv
            if Path(csv_path).exists():
                st.download_button("Download nuclear cell table",
                                   data=Path(csv_path).read_bytes(),
                                   file_name="cell_level_metabolic_table__nuclear__mean.csv",
                                   mime="text/csv", key="dl_cell_csv",
                                   use_container_width=True)
        with dc2:
            csv_path = st.session_state.ana_expanded_cell_csv
            if csv_path and Path(csv_path).exists():
                st.download_button("Download expanded cell table",
                                   data=Path(csv_path).read_bytes(),
                                   file_name="cell_level_metabolic_table__nuclear_expanded__mean.csv",
                                   mime="text/csv", key="dl_exp_cell_csv",
                                   use_container_width=True)


# ---------------------------------------------------------------------------
# Tab 5 – Superpixel Segmentation
# ---------------------------------------------------------------------------

def _tab_superpixels(if_stack, if_labels, msi_labels):
    st.markdown("#### Superpixel Segmentation (SLIC)")

    if if_stack is None:
        st.warning("Load an IF image first.")
        return
    if st.session_state.ana_mbp_mask is None:
        st.warning("Run MBP mask first.")
        return
    if st.session_state.ana_nuclear_binary is None:
        st.warning("Run cell segmentation first.")
        return
    if st.session_state.ana_projected is None:
        st.warning("Run LR→HR projection first.")
        return

    out_dir = _output_dir()
    if out_dir is None:
        st.warning("Set an output folder above.")
        return

    c1, c2 = st.columns(2)
    with c1:
        ch_opts = [f"{i}: {if_labels[i]}" if i < len(if_labels) else f"C{i}"
                   for i in range(if_stack.shape[0])]
        ch_sel = st.selectbox("MBP channel for SLIC", ch_opts, key="ana_sp_channel")
        ch_idx = ch_opts.index(ch_sel)
        n_seg  = st.number_input("Target segments", 1000, 200000, 40000, 1000,
                                 key="ana_sp_n_seg")
        compact = st.number_input("Compactness", 0.01, 10.0, 0.15, 0.01,
                                  key="ana_sp_compact")

    with c2:
        sigma  = st.number_input("Gaussian sigma", 0.0, 5.0, 1.0, 0.5,
                                 key="ana_sp_sigma")
        ds_fac = st.number_input("Downsample factor", 1, 8, 2, 1,
                                 key="ana_sp_ds")

    if st.button("Run Superpixel Segmentation", type="primary", key="ana_sp_run",
                 use_container_width=True):
        status = st.empty()
        progress = st.progress(0.0, text="Starting superpixel segmentation…")
        try:
            sp_dir = out_dir / "superpixels"
            proj_tif = st.session_state.ana_projected
            import tifffile as _tiff
            status.info("Loading projected MSI stack…")
            progress.progress(0.05, text="Loading projected MSI stack…")
            projected = _tiff.imread(str(proj_tif))

            sp_steps = {
                "Downsampling": 0.15,
                "Running SLIC": 0.50,
                "Upsampling":   0.70,
                "Computing":    0.85,
                "Saved":        0.95,
            }

            def _sp_cb(msg: str):
                frac = 0.10
                for key, f in sp_steps.items():
                    if key.lower() in msg.lower():
                        frac = f
                        break
                progress.progress(frac, text=msg)
                status.info(msg)

            result = run_superpixel_segmentation(
                if_stack=if_stack,
                mbp_channel_idx=ch_idx,
                mbp_mask=st.session_state.ana_mbp_mask,
                nuclear_binary=st.session_state.ana_nuclear_binary.astype(bool),
                projected_msi=projected,
                channel_labels=msi_labels,
                output_dir=sp_dir,
                n_segments=int(n_seg),
                compactness=float(compact),
                gaussian_sigma=float(sigma),
                downsample_factor=int(ds_fac),
                status_cb=_sp_cb,
            )
            st.session_state.ana_sp_label_mask = result["label_mask_path"]
            st.session_state.ana_sp_stats_csv  = result["stats_csv"]
            progress.progress(1.0, text="Superpixel segmentation complete.")
            status.success(f"Superpixel segmentation complete — "
                           f"{result['n_superpixels']:,} superpixels.")
        except Exception as e:
            status.error("Superpixel segmentation failed.")
            st.exception(e)

    if st.session_state.ana_sp_stats_csv is not None:
        st.divider()
        csv_path = st.session_state.ana_sp_stats_csv
        if Path(csv_path).exists():
            st.download_button("Download superpixel stats CSV",
                               data=Path(csv_path).read_bytes(),
                               file_name="mbp_superpixels_mean_intensity_matrix.csv",
                               mime="text/csv", key="dl_sp_csv",
                               use_container_width=True)


# ---------------------------------------------------------------------------
# Tab 6 – Clustering
# ---------------------------------------------------------------------------

def _tab_clustering(msi_labels):
    st.markdown("#### Clustering (PCA + UMAP + Leiden + KMeans)")

    if st.session_state.ana_cell_csv is None:
        st.warning("Run LR→HR projection first.")
        return
    if st.session_state.ana_sp_stats_csv is None:
        st.warning("Run superpixel segmentation first.")
        return

    out_dir = _output_dir()
    if out_dir is None:
        st.warning("Set an output folder above.")
        return

    st.caption(f"{len(msi_labels)} MSI channels available.")

    # Channel selector
    selected_labels = st.multiselect(
        "Channels to use for clustering",
        options=msi_labels,
        default=msi_labels,
        key="ana_cl_channels",
        help="Select which MSI channels to include as features for PCA/UMAP/clustering.",
    )
    if not selected_labels:
        st.warning("Select at least one channel.")
        return

    st.divider()

    c1, c2 = st.columns(2)
    with c1:
        n_pcs      = st.number_input("PCA components", 2, 50, 10, 1, key="ana_cl_pcs")
        n_neighbors = st.number_input("UMAP neighbors", 5, 100, 15, 5, key="ana_cl_nn")
        min_dist   = st.number_input("UMAP min dist", 0.01, 1.0, 0.3, 0.05,
                                     key="ana_cl_min_dist")
        seed       = st.number_input("Random seed", 0, 99999, 42, 1, key="ana_cl_seed")

    with c2:
        leiden_str = st.text_input("Leiden resolutions (comma-separated)",
                                   "0.25, 0.50, 1.00", key="ana_cl_leiden")
        k_str      = st.text_input("KMeans k values (comma-separated)",
                                   "2, 4, 6, 8, 10", key="ana_cl_k")
        row_zscore = st.checkbox("Row z-score in matrixplot", True, key="ana_cl_zscore")

    try:
        leiden_res = [float(x.strip()) for x in leiden_str.split(",") if x.strip()]
        k_values   = [int(x.strip()) for x in k_str.split(",") if x.strip()]
    except ValueError:
        st.error("Invalid Leiden resolutions or k values.")
        return

    if st.button("Run Clustering", type="primary", key="ana_cl_run",
                 use_container_width=True):
        status = st.empty()
        progress = st.progress(0.0, text="Starting clustering…")
        try:
            cl_dir = out_dir / "clustering"
            cluster_steps = {
                "cells":        0.10,
                "superpixels":  0.55,
                "umap":         0.30,
                "leiden":       0.50,
                "kmeans":       0.70,
                "done":         0.90,
            }

            def _cl_cb(msg: str):
                frac = 0.05
                for key, f in cluster_steps.items():
                    if key.lower() in msg.lower():
                        frac = f
                        break
                progress.progress(frac, text=msg)
                status.info(msg)

            run_clustering(
                cell_csv=Path(st.session_state.ana_cell_csv),
                superpixel_csv=Path(st.session_state.ana_sp_stats_csv),
                cell_label_mask_path=Path(out_dir / "segmentation" / "nuclear_mask_expanded.tif"),
                superpixel_label_mask_path=Path(st.session_state.ana_sp_label_mask),
                channel_labels=selected_labels,
                output_dir=cl_dir,
                n_pcs=int(n_pcs),
                n_neighbors=int(n_neighbors),
                umap_min_dist=float(min_dist),
                leiden_resolutions=leiden_res,
                k_values=k_values,
                random_seed=int(seed),
                row_zscore=row_zscore,
                status_cb=_cl_cb,
            )
            progress.progress(1.0, text="Clustering complete.")
            status.success(f"Clustering complete. Results saved to {cl_dir}")
        except Exception as e:
            status.error("Clustering failed.")
            st.exception(e)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def render_analysis() -> None:
    """Render the Analysis expander. Only shown when both stacks are loaded."""
    _init_analysis_state()

    if_stack, if_labels, msi_stack, msi_labels = _get_stacks()

    if if_stack is None and msi_stack is None:
        return

    with st.expander("Analysis Pipeline", expanded=False):

        # ---- Output folder ----
        st.markdown("#### Output folder")
        out_dir_input = st.text_input(
            "Root output directory",
            value=st.session_state.ana_output_dir,
            key="ana_output_dir_input",
            placeholder=r"e.g. Y:\results\experiment1",
        )
        st.session_state.ana_output_dir = out_dir_input

        st.divider()

        # ---- Channel label editors ----
        st.markdown("#### Channel labels")
        st.caption(
            "Edit IF and MSI channel names below. "
            "These are used for all downstream analysis steps."
        )
        lbl_col1, lbl_col2 = st.columns(2)

        with lbl_col1:
            if_stack_c = if_stack.shape[0] if if_stack is not None else 0
            if_text_key = "ana_if_labels_text_text"

            if if_labels and len(if_labels) == if_stack_c:
                default_if = list(if_labels)
            else:
                default_if = [f"ch_{i}" for i in range(if_stack_c)]

            if if_text_key not in st.session_state or not st.session_state[if_text_key].strip():
                st.session_state[if_text_key] = "\n".join(default_if)

            edited_if = _channel_label_editor(
                "ana_if_labels",
                default_if,
                f"IF channel labels ({if_stack_c} channels)",
            )
            st.session_state.ana_if_labels_edited = edited_if

        with lbl_col2:
            # Fall back to auto-generated labels if MSI labels are missing
            msi_stack_c = msi_stack.shape[0] if msi_stack is not None else 0
            msi_text_key = "ana_msi_labels_text_text"

            # Build the default label list
            if msi_labels and len(msi_labels) == msi_stack_c:
                default_msi = list(msi_labels)
            else:
                default_msi = [f"mz_{i}" for i in range(msi_stack_c)]

            # Pre-populate the text area if it's empty or not yet set
            if msi_text_key not in st.session_state or not st.session_state[msi_text_key].strip():
                st.session_state[msi_text_key] = "\n".join(default_msi)

            edited_msi = _channel_label_editor(
                "ana_msi_labels",
                default_msi,
                f"MSI channel labels ({msi_stack_c} channels)",
            )
            st.session_state.ana_msi_labels_edited = edited_msi

        # Combined label list for analysis steps
        all_labels = edited_if + edited_msi

        st.divider()

        # ---- Pipeline tabs ----
        tabs = st.tabs([
            "1 · Cell Segmentation",
            "2 · Nuclei Expansion",
            "3 · MBP Mask",
            "4 · LR→HR Projection",
            "5 · Superpixels",
            "6 · Clustering",
        ])

        with tabs[0]:
            _tab_segmentation(if_stack, edited_if)

        with tabs[1]:
            _tab_expansion()

        with tabs[2]:
            _tab_mbp(if_stack, edited_if)

        with tabs[3]:
            _tab_projection(msi_stack, if_stack, edited_msi)

        with tabs[4]:
            _tab_superpixels(if_stack, edited_if, edited_msi)

        with tabs[5]:
            _tab_clustering(edited_msi)
