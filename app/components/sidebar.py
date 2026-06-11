"""
Sidebar UI: data upload tabs (IF + MSI) and display settings.

Reads from / writes to st.session_state.
Returns display settings as a dict so the viewer page can consume them.
"""

import tempfile
from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st

from app.utils.file_io import safe_name, save_uploaded_file_chunked, read_tiff_stack
from app.utils.imzml import (
    ensure_imzml_ibd_pair_names,
    extract_multichannel_imzml_to_stack,
    read_targets_from_csv_path,
    read_targets_from_text,
    save_imagej_tiff,
)


# ---------------------------------------------------------------------------
# Session-state initialisation (call once at app startup)
# ---------------------------------------------------------------------------

def init_session_state() -> None:
    if "work_dir" not in st.session_state:
        st.session_state.work_dir = tempfile.mkdtemp(prefix="if_msi_streamlit_")

    defaults = {
        "if_stack": None,
        "if_labels": None,
        "if_path": None,
        "msi_stack": None,
        "msi_labels": None,
        "msi_path": None,
        "extracted_tiff_path": None,
        "last_targets_df": None,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# ---------------------------------------------------------------------------
# IF upload panel
# ---------------------------------------------------------------------------

def _render_if_tab(work_dir: Path) -> None:
    st.subheader("IF image")

    if_file = st.file_uploader(
        "Multiplex IF TIFF",
        type=["tif", "tiff"],
        key="if_file",
    )

    if_label_file = st.file_uploader(
        "Optional IF channel labels",
        type=["txt", "csv"],
        key="if_labels_file",
    )

    load_if = st.button(
        "Load IF",
        disabled=if_file is None,
        use_container_width=True,
    )

    if load_if:
        try:
            if_path = work_dir / safe_name(if_file.name)

            with st.spinner("Loading IF image..."):
                save_uploaded_file_chunked(if_file, if_path)
                stack, labels = read_tiff_stack(if_path, if_label_file)

            st.session_state.if_stack = stack
            st.session_state.if_labels = labels
            st.session_state.if_path = str(if_path)

            st.success(f"Loaded: {stack.shape[0]} channel(s)")

        except Exception as e:
            st.error("Failed to load IF image.")
            st.exception(e)


# ---------------------------------------------------------------------------
# MSI upload panel
# ---------------------------------------------------------------------------

def _render_msi_tiff_panel(work_dir: Path) -> None:
    msi_tiff_file = st.file_uploader(
        "MSI TIFF",
        type=["tif", "tiff"],
        key="msi_tiff_file",
    )

    msi_label_file = st.file_uploader(
        "Optional MSI channel labels",
        type=["txt", "csv"],
        key="msi_labels_file",
    )

    load_msi = st.button(
        "Load MSI TIFF",
        disabled=msi_tiff_file is None,
        use_container_width=True,
    )

    if load_msi:
        try:
            msi_path = work_dir / safe_name(msi_tiff_file.name)

            with st.spinner("Loading MSI TIFF..."):
                save_uploaded_file_chunked(msi_tiff_file, msi_path)
                stack, labels = read_tiff_stack(msi_path, msi_label_file)

            st.session_state.msi_stack = stack
            st.session_state.msi_labels = labels
            st.session_state.msi_path = str(msi_path)
            st.session_state.extracted_tiff_path = None
            st.session_state.last_targets_df = None

            st.success(f"Loaded: {stack.shape[0]} channel(s)")

        except Exception as e:
            st.error("Failed to load MSI TIFF.")
            st.exception(e)


def _render_imzml_panel(work_dir: Path) -> None:
    imzml_file = st.file_uploader(
        "imzML file",
        type=["imzML", "imzml"],
        key="imzml_file",
    )

    ibd_file = st.file_uploader(
        "Matching IBD file",
        type=["ibd", "IBD"],
        key="ibd_file",
    )

    st.markdown("###### m/z targets")

    target_mode = st.radio(
        "Target input",
        ["Type/paste", "CSV"],
        horizontal=True,
        key="target_mode",
    )

    csv_file: Optional[object] = None
    mz_text = ""

    if target_mode == "CSV":
        csv_file = st.file_uploader(
            "Target CSV",
            type=["csv"],
            key="target_csv",
            help="Use columns lipid, ion, target_mz; or at least target_mz/mz/m/z.",
        )
    else:
        mz_text = st.text_area(
            "m/z values",
            height=110,
            placeholder="760.5851, 782.5670, 806.5458",
            key="mz_text",
        )

    st.markdown("###### Extraction settings")

    ppm = st.number_input(
        "ppm tolerance",
        min_value=0.1,
        max_value=100.0,
        value=5.0,
        step=0.5,
    )

    dtype_name = st.selectbox(
        "Output dtype",
        ["uint16", "float32", "uint8"],
        index=0,
    )

    normalize_label = st.selectbox(
        "Normalization",
        ["None", "per_channel_max", "global_max"],
        index=0,
    )

    normalize = None if normalize_label == "None" else normalize_label

    output_name = st.text_input(
        "Output TIFF filename",
        value="maldi_extracted.tif",
    )

    can_extract = (
        imzml_file is not None
        and ibd_file is not None
        and (
            (target_mode == "CSV" and csv_file is not None)
            or (target_mode == "Type/paste" and mz_text.strip())
        )
    )

    extract_btn = st.button(
        "Extract MSI",
        type="primary",
        disabled=not can_extract,
        use_container_width=True,
    )

    if extract_btn:
        try:
            output_name_clean = safe_name(output_name)
            if not output_name_clean.lower().endswith((".tif", ".tiff")):
                output_name_clean += ".tif"

            imzml_path = work_dir / safe_name(imzml_file.name)
            ibd_path = work_dir / safe_name(ibd_file.name)
            target_csv_path = (
                work_dir / safe_name(csv_file.name) if csv_file is not None else None
            )
            output_tiff_path = work_dir / output_name_clean

            status_box = st.empty()
            progress_bar = st.progress(0.0, text="Starting extraction...")

            with st.spinner("Saving uploaded files..."):
                save_uploaded_file_chunked(imzml_file, imzml_path)
                save_uploaded_file_chunked(ibd_file, ibd_path)

                if csv_file is not None and target_csv_path is not None:
                    save_uploaded_file_chunked(csv_file, target_csv_path)

            imzml_path, ibd_path = ensure_imzml_ibd_pair_names(imzml_path, ibd_path)

            if target_csv_path is not None:
                targets = read_targets_from_csv_path(target_csv_path)
            else:
                targets = read_targets_from_text(mz_text)

            st.session_state.last_targets_df = pd.DataFrame(targets)

            msi_stack, msi_labels = extract_multichannel_imzml_to_stack(
                imzml_path=imzml_path,
                targets=targets,
                ppm=float(ppm),
                dtype_name=dtype_name,
                normalize=normalize,
                progress_bar=progress_bar,
                status_box=status_box,
            )

            save_imagej_tiff(
                output_tiff=output_tiff_path,
                stack_to_save=msi_stack,
                channel_names=msi_labels,
                source_name=imzml_path.name,
                ppm=float(ppm),
            )

            st.session_state.msi_stack = msi_stack
            st.session_state.msi_labels = msi_labels
            st.session_state.msi_path = str(output_tiff_path)
            st.session_state.extracted_tiff_path = str(output_tiff_path)

            status_box.success(
                f"Extraction complete: {msi_stack.shape[0]} channel(s)"
            )

        except Exception as e:
            st.error("imzML extraction failed.")
            st.exception(e)


def _render_msi_tab(work_dir: Path) -> None:
    st.subheader("MSI image")

    msi_mode = st.radio(
        "Input type",
        ["TIFF stack", "imzML + IBD"],
        horizontal=True,
        key="msi_mode",
    )

    if msi_mode == "TIFF stack":
        _render_msi_tiff_panel(work_dir)
    else:
        _render_imzml_panel(work_dir)


# ---------------------------------------------------------------------------
# Display settings panel
# ---------------------------------------------------------------------------

def _render_display_settings() -> dict:
    st.header("Display")

    view_mode = st.radio(
        "View mode",
        ["Single channels", "RGB overlays"],
        index=0,
    )

    pmin = st.slider("Lower percentile", 0.0, 20.0, 1.0, 0.5)
    pmax = st.slider("Upper percentile", 80.0, 100.0, 99.5, 0.1)
    gamma = st.slider("Gamma", 0.1, 3.0, 1.0, 0.1)

    return {"view_mode": view_mode, "pmin": pmin, "pmax": pmax, "gamma": gamma}


# ---------------------------------------------------------------------------
# Loaded-data summary
# ---------------------------------------------------------------------------

def _render_data_summary() -> None:
    st.header("Loaded data")

    if st.session_state.if_stack is None:
        st.caption("IF: not loaded")
    else:
        s = st.session_state.if_stack
        st.caption(
            f"IF: {s.shape[0]} channels, {s.shape[-2]} × {s.shape[-1]}"
        )

    if st.session_state.msi_stack is None:
        st.caption("MSI: not loaded")
    else:
        s = st.session_state.msi_stack
        st.caption(
            f"MSI: {s.shape[0]} channels, {s.shape[-2]} × {s.shape[-1]}"
        )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def render_sidebar() -> dict:
    """
    Render the full sidebar and return display settings.
    Call this from the main page after init_session_state().
    """
    work_dir = Path(st.session_state.work_dir)

    with st.sidebar:
        st.header("Data")

        data_tab_if, data_tab_msi = st.tabs(["IF", "MSI"])

        with data_tab_if:
            _render_if_tab(work_dir)

        with data_tab_msi:
            _render_msi_tab(work_dir)

        st.divider()
        display_settings = _render_display_settings()

        st.divider()
        _render_data_summary()

    return display_settings
