"""
Napari launch view.

Writes a JSON config and launches app/napari_viewer.py as a subprocess
in the current Python environment (torch_gpu3 has napari installed).
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

import streamlit as st


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

def _init_state() -> None:
    defaults = {
        "napari_proc": None,
        "napari_output_dir": "",
        "napari_if_path": "",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def render_napari_launch() -> None:
    """
    Render the napari launch expander.
    Visible when both IF and MSI stacks are loaded OR when an output dir is set.
    """
    _init_state()

    if_stack  = st.session_state.get("if_stack")
    msi_stack = st.session_state.get("msi_stack")

    # Only show if at least one stack is loaded or an output dir is already set
    if if_stack is None and msi_stack is None and not st.session_state.napari_output_dir:
        return

    with st.expander("Interactive Viewer (napari)", expanded=False):

        st.markdown("#### Launch napari viewer")
        st.caption(
            "Opens a napari window with all analysis results: "
            "fluorescence channels, projected MALDI, cell/superpixel borders, "
            "and cluster overlays. Hover or click to inspect individual cells."
        )

        # ---- Output folder ----
        out_dir_val = st.text_input(
            "Results root folder",
            value=st.session_state.napari_output_dir or
                  st.session_state.get("ana_output_dir", ""),
            key="napari_out_dir_input",
            placeholder=r"e.g. Y:\results\experiment1",
        )
        st.session_state.napari_output_dir = out_dir_val

        # ---- IF stack path — use what's already loaded, no need to re-enter ----
        if_path_val = st.session_state.get("if_path") or st.session_state.napari_if_path
        if if_path_val:
            st.caption(f"IF stack: `{if_path_val}`")
        else:
            if_path_val = st.text_input(
                "IF (fluorescence) TIFF path",
                value="",
                key="napari_if_path_input",
                help="Only needed if no IF stack was loaded in the sidebar.",
            )
        st.session_state.napari_if_path = if_path_val

        # ---- Channel labels ----
        msi_labels = st.session_state.get("ana_msi_labels_edited") or \
                     st.session_state.get("msi_labels") or []
        if_labels  = st.session_state.get("ana_if_labels_edited") or \
                     st.session_state.get("if_labels") or []

        st.caption(
            f"MSI channel labels: {len(msi_labels)} channels  |  "
            f"IF channel labels: {len(if_labels)} channels"
        )

        if not msi_labels:
            st.warning(
                "No MSI channel labels found. "
                "Run the Analysis pipeline first or load an MSI stack."
            )

        # ---- Validate required files ----
        out_dir = Path(out_dir_val.strip()) if out_dir_val.strip() else None

        required_files = []
        if out_dir:
            required_files = [
                out_dir / "projection" / "cell_level_metabolic_table__nuclear_expanded__mean.csv",
                out_dir / "projection" / "projected_stack_all_channels__full_hr__gaussian.tif",
                out_dir / "superpixels" / "mbp_superpixels_mean_intensity_matrix.csv",
                out_dir / "superpixels" / "mbp_superpixels_label_mask.tif",
                out_dir / "segmentation" / "nuclear_mask.tif",
                out_dir / "segmentation" / "nuclear_mask_expanded.tif",
                out_dir / "segmentation" / "nuclear_mask_binary.tif",
            ]
            missing = [str(f) for f in required_files if not f.exists()]
            if missing:
                st.warning(
                    f"{len(missing)} required file(s) not found:\n" +
                    "\n".join(f"  • {m}" for m in missing)
                )

        can_launch = (
            out_dir is not None
            and out_dir.exists()
            and bool(msi_labels)
            and not [f for f in required_files if not f.exists()]
        )

        # ---- Launch button ----
        col1, col2 = st.columns(2)

        with col1:
            if st.button(
                "Launch napari",
                type="primary",
                key="napari_launch_btn",
                disabled=not can_launch,
                use_container_width=True,
            ):
                _launch_napari(
                    out_dir=out_dir,
                    if_path=if_path_val.strip(),
                    msi_labels=msi_labels,
                    if_labels=if_labels,
                )

        with col2:
            proc = st.session_state.napari_proc
            if proc is not None:
                poll = proc.poll()
                if poll is None:
                    st.info("napari is running.")
                    if st.button("Stop napari", key="napari_stop_btn",
                                 use_container_width=True):
                        proc.terminate()
                        st.session_state.napari_proc = None
                        st.rerun()
                else:
                    st.caption(f"napari exited (code {poll}).")
                    st.session_state.napari_proc = None


def _launch_napari(
    out_dir: Path,
    if_path: str,
    msi_labels: list,
    if_labels: list,
) -> None:
    cfg = {
        "output_dir":        str(out_dir),
        "if_stack_path":     if_path,
        "msi_channel_labels": msi_labels,
        "if_channel_labels":  if_labels,
    }
    # Write config to a temp file
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, encoding="utf-8"
    ) as f:
        json.dump(cfg, f, indent=2)
        cfg_path = f.name

    # Find the viewer script relative to this file
    viewer_script = Path(__file__).resolve().parent.parent / "napari_viewer.py"

    proc = subprocess.Popen(
        [sys.executable, str(viewer_script), cfg_path],
        # Don't capture stdout/stderr so napari can use the display
    )
    st.session_state.napari_proc = proc
    st.success(f"napari launched (PID {proc.pid}). The viewer window should open shortly.")
