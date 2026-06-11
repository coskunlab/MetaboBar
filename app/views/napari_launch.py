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
        "napari_log": None,
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

        # ---- IF stack path — use what's already loaded, prefer saved copy in output dir ----
        saved_if = (Path(out_dir_val.strip()) / "segmentation" / "if_stack.tif"
                    if out_dir_val.strip() else None)
        if saved_if and saved_if.exists():
            if_path_val = str(saved_if)
            st.caption(f"IF stack: `{if_path_val}`")
        else:
            if_path_val = st.session_state.get("if_path") or st.session_state.napari_if_path
            if if_path_val and Path(if_path_val).exists():
                st.caption(f"IF stack: `{if_path_val}`")
            else:
                st.warning("No IF image found. Run Cell Segmentation first (it saves the IF stack), or load the IF image from the sidebar.")
                if_path_val = st.text_input(
                    "Or enter IF TIFF path manually",
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

        # ---- Check napari availability ----
        napari_ok = False
        try:
            import importlib
            spec = importlib.util.find_spec("napari")
            if spec is not None:
                napari_ok = True
        except Exception:
            pass

        if not napari_ok:
            st.warning(
                "napari is not available on this machine. "
                "You can still view all results by opening the output folder directly:\n\n"
                "- Cluster maps: `clustering/cells/` and `clustering/superpixels/` — PNG images\n"
                "- Positivity overlays: `positivity/positivity_masks/` — TIFF images\n"
                "- Projected MSI: `projection/` — TIFF stack (open with ImageJ/Fiji)"
            )
            return

        # ---- Validate required files ----
        out_dir = Path(out_dir_val.strip()) if out_dir_val.strip() else None

        # These files are always required
        required_files = []
        # Superpixel files are optional — note their absence but don't block launch
        optional_files = []
        if out_dir:
            required_files = [
                out_dir / "projection" / "cell_level_metabolic_table__nuclear_expanded__mean.csv",
                out_dir / "projection" / "projected_stack_all_channels__full_hr__gaussian.tif",
                out_dir / "segmentation" / "nuclear_mask.tif",
                out_dir / "segmentation" / "nuclear_mask_expanded.tif",
                out_dir / "segmentation" / "nuclear_mask_binary.tif",
            ]
            optional_files = [
                out_dir / "superpixels" / "mbp_superpixels_mean_intensity_matrix.csv",
                out_dir / "superpixels" / "mbp_superpixels_label_mask.tif",
            ]
            missing = [str(f) for f in required_files if not f.exists()]
            if missing:
                st.warning(
                    f"{len(missing)} required file(s) not found:\n" +
                    "\n".join(f"  • {m}" for m in missing)
                )
            missing_opt = [str(f) for f in optional_files if not f.exists()]
            if missing_opt:
                st.info(
                    "Superpixel files not found — superpixel overlay will be disabled in the viewer:\n" +
                    "\n".join(f"  • {m}" for m in missing_opt)
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

        # ---- Live log viewer ----
        log_path = st.session_state.get("napari_log")
        if log_path and Path(log_path).exists():
            log_text = Path(log_path).read_text(encoding="utf-8", errors="replace")
            if log_text.strip():
                with st.expander("napari log (auto-updates)", expanded=False):
                    st.code(log_text[-3000:], language=None)
                    if st.button("Refresh log", key="napari_log_refresh"):
                        st.rerun()


def _launch_napari(
    out_dir: Path,
    if_path: str,
    msi_labels: list,
    if_labels: list,
) -> None:
    import os, time, tempfile as _tmp

    log_file = Path(_tmp.gettempdir()) / "metabar_napari.log"
    log_file.write_text("")  # clear previous log

    cfg = {
        "output_dir":         str(out_dir),
        "if_stack_path":      if_path,
        "msi_channel_labels": msi_labels,
        "if_channel_labels":  if_labels,
        "log_file":           str(log_file),
    }
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, encoding="utf-8"
    ) as f:
        json.dump(cfg, f, indent=2)
        cfg_path = f.name

    viewer_script = Path(__file__).resolve().parent.parent / "napari_viewer.py"

    # Resolve the correct Python interpreter:
    # 1. If running from the bundle, METABOBARCODING_ROOT points to bundle\,
    #    and the embedded Python lives at bundle\python\python.exe.
    # 2. Otherwise fall back to the current interpreter (dev environment).
    bundle_root = os.environ.get("METABOBARCODING_ROOT", "").strip()
    if bundle_root:
        bundle_python = Path(bundle_root) / "python" / "python.exe"
        python_exe = str(bundle_python) if bundle_python.exists() else sys.executable
    else:
        python_exe = sys.executable

    # Build a clean environment — strip all Qt vars that Streamlit may have
    # injected, then let PyQt5 in the subprocess resolve its own plugin paths.
    _QT_VARS_TO_REMOVE = {
        "QT_QPA_PLATFORM_PLUGIN_PATH",
        "QT_PLUGIN_PATH",
        "QT_QPA_PLATFORMTHEME",
        "QT_STYLE_OVERRIDE",
        "QT_SCREEN_SCALE_FACTORS",
        "QT_AUTO_SCREEN_SCALE_FACTOR",
        "QT_SCALE_FACTOR",
        "PYQTDESIGNERPATH",
        "PYQTPATH",
        # On some systems Streamlit sets these
        "MPLBACKEND",
    }
    env = {k: v for k, v in os.environ.items() if k not in _QT_VARS_TO_REMOVE}
    env["QT_QPA_PLATFORM"] = "windows"

    # Write stdout+stderr to the log file so the app can read it
    log_handle = open(str(log_file), "w", encoding="utf-8", errors="replace")

    # DETACHED_PROCESS + CREATE_NO_WINDOW: detaches from the Streamlit console
    # so napari is not suppressed, but no extra CMD window appears for the user.
    # stdout/stderr are redirected to the log file so nothing is lost.
    CREATE_NO_WINDOW  = 0x08000000
    DETACHED_PROCESS  = 0x00000008

    try:
        proc = subprocess.Popen(
            [python_exe, str(viewer_script), cfg_path],
            env=env,
            stdout=log_handle,
            stderr=log_handle,
            creationflags=DETACHED_PROCESS | CREATE_NO_WINDOW,
        )
        time.sleep(4)
        poll = proc.poll()
        if poll is not None:
            log_handle.close()
            error_text = log_file.read_text(encoding="utf-8", errors="replace")
            st.error(
                f"napari failed to start (exit code {poll}).\n\n"
                f"```\n{error_text[-2000:]}\n```"
            )
            return
        st.session_state.napari_proc   = proc
        st.session_state.napari_log    = str(log_file)
        st.session_state.napari_log_fh = log_handle
        st.success(
            f"napari is loading data — the window will appear in ~30–60 seconds. "
            f"If it does not open, expand 'napari log' below."
        )
    except Exception as e:
        log_handle.close()
        st.error(f"Failed to launch napari: {e}")
