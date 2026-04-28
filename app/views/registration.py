"""
Registration view: register MSI stack to IF using Fiji SIFT.

Workflow
--------
1. User picks IF reference channel and MSI reference channel
2. User (optionally) adjusts SIFT parameters
3. Click "Run Registration"
4. Aligned MSI stack is stored in session state and offered for download
"""

import io
from pathlib import Path
from typing import Optional

import numpy as np
import streamlit as st
import tifffile as tiff

from app.utils.registration import (
    DEFAULT_FIJI_PATH,
    SIFT_DEFAULTS,
    register_msi_to_if,
)


# ---------------------------------------------------------------------------
# Session-state helpers
# ---------------------------------------------------------------------------

def _init_reg_state() -> None:
    defaults = {
        "reg_aligned_msi": None,
        "reg_aligned_labels": None,
        "reg_log": None,
        "reg_fiji_path": DEFAULT_FIJI_PATH,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def _stack_to_bytes(stack: np.ndarray, labels: list) -> bytes:
    buf = io.BytesIO()
    tiff.imwrite(
        buf,
        stack,
        imagej=True,
        metadata={"axes": "CYX", "Labels": labels, "Channel": labels},
    )
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def render_registration() -> None:
    """
    Render the registration expander.
    Only visible when both IF and MSI stacks are loaded.
    """
    _init_reg_state()

    if_stack: Optional[np.ndarray] = st.session_state.get("if_stack")
    msi_stack: Optional[np.ndarray] = st.session_state.get("msi_stack")

    if if_stack is None or msi_stack is None:
        return

    with st.expander("Registration — align MSI to IF via Fiji SIFT", expanded=False):

        if_labels = st.session_state.if_labels or []
        msi_labels = st.session_state.msi_labels or []

        # ----------------------------------------------------------------
        # 1. Reference channel selection
        # ----------------------------------------------------------------
        st.markdown("#### Reference channels")

        ref_col1, ref_col2 = st.columns(2)

        with ref_col1:
            if_ch_options = [
                f"{i}: {if_labels[i]}" if i < len(if_labels) else f"C{i}"
                for i in range(if_stack.shape[0])
            ]
            if_ref_label = st.selectbox(
                "IF reference channel",
                if_ch_options,
                index=0,
                key="reg_if_ref_ch",
            )
            if_ref_idx = if_ch_options.index(if_ref_label)

        with ref_col2:
            msi_ch_options = [
                f"{i}: {msi_labels[i]}" if i < len(msi_labels) else f"C{i}"
                for i in range(msi_stack.shape[0])
            ]
            msi_ref_label = st.selectbox(
                "MSI reference channel",
                msi_ch_options,
                index=0,
                key="reg_msi_ref_ch",
            )
            msi_ref_idx = msi_ch_options.index(msi_ref_label)

        st.caption(
            f"IF: {if_stack.shape[-1]} × {if_stack.shape[-2]} px  |  "
            f"MSI (original): {msi_stack.shape[-1]} × {msi_stack.shape[-2]} px  |  "
            f"MSI will be upsampled to IF size, registered, then downsampled back "
            f"to match original MSI height."
        )

        st.divider()

        # ----------------------------------------------------------------
        # 2. SIFT parameters
        # ----------------------------------------------------------------
        st.markdown("#### SIFT parameters")

        with st.expander("Show / edit SIFT parameters", expanded=False):
            p_col1, p_col2 = st.columns(2)

            with p_col1:
                init_blur = st.number_input(
                    "Initial gaussian blur (px)",
                    value=SIFT_DEFAULTS["initial_gaussian_blur"],
                    min_value=0.1, step=0.1, key="reg_init_blur",
                )
                steps = st.number_input(
                    "Steps per scale octave",
                    value=SIFT_DEFAULTS["steps_per_scale_octave"],
                    min_value=1, step=1, key="reg_steps",
                )
                min_size = st.number_input(
                    "Minimum image size (px)",
                    value=SIFT_DEFAULTS["minimum_image_size"],
                    min_value=16, step=8, key="reg_min_size",
                )
                max_size = st.number_input(
                    "Maximum image size (px)",
                    value=SIFT_DEFAULTS["maximum_image_size"],
                    min_value=64, step=64, key="reg_max_size",
                )

            with p_col2:
                fd_size = st.number_input(
                    "Feature descriptor size",
                    value=SIFT_DEFAULTS["feature_descriptor_size"],
                    min_value=1, step=1, key="reg_fd_size",
                )
                fd_bins = st.number_input(
                    "Feature descriptor orientation bins",
                    value=SIFT_DEFAULTS["feature_descriptor_orientation_bins"],
                    min_value=1, step=1, key="reg_fd_bins",
                )
                ratio = st.number_input(
                    "Closest/next closest ratio",
                    value=SIFT_DEFAULTS["closest_next_closest_ratio"],
                    min_value=0.01, max_value=1.0, step=0.01, key="reg_ratio",
                )
                max_err = st.number_input(
                    "Maximal alignment error (px)",
                    value=SIFT_DEFAULTS["maximal_alignment_error"],
                    min_value=1.0, step=1.0, key="reg_max_err",
                )
                inlier = st.number_input(
                    "Inlier ratio",
                    value=SIFT_DEFAULTS["inlier_ratio"],
                    min_value=0.01, max_value=1.0, step=0.01, key="reg_inlier",
                )
                transform_type = st.selectbox(
                    "Expected transformation",
                    ["Translation", "Rigid", "Similarity", "Affine"],
                    index=3,
                    key="reg_transform_type",
                )
                interpolate = st.checkbox(
                    "Interpolate", value=False, key="reg_interpolate"
                )
        sift_params = dict(
            initial_gaussian_blur=float(init_blur),
            steps_per_scale_octave=int(steps),
            minimum_image_size=int(min_size),
            maximum_image_size=int(max_size),
            feature_descriptor_size=int(fd_size),
            feature_descriptor_orientation_bins=int(fd_bins),
            closest_next_closest_ratio=float(ratio),
            maximal_alignment_error=float(max_err),
            inlier_ratio=float(inlier),
            expected_transformation=transform_type,
            interpolate=interpolate,
        )

        st.divider()

        # ----------------------------------------------------------------
        # 3. Fiji path
        # ----------------------------------------------------------------
        fiji_path = st.text_input(
            "Fiji executable path",
            value=st.session_state.reg_fiji_path,
            key="reg_fiji_path_input",
            help="Point to ImageJ-win64.exe inside your Fiji.app folder.",
        )
        st.caption(
            "Fiji will open briefly during registration to run the SIFT macro, "
            "then close automatically. This is normal."
        )
        st.session_state.reg_fiji_path = fiji_path

        st.divider()

        # ----------------------------------------------------------------
        # 4. Run button
        # ----------------------------------------------------------------
        run_btn = st.button(
            "Run Registration",
            type="primary",
            key="reg_run_btn",
            use_container_width=True,
        )

        if run_btn:
            status_box = st.empty()
            progress = st.progress(0.0)
            step_msgs = [
                "Step 1/5 — Resizing MSI to IF dimensions…",
                "Step 2/5 — Writing reference TIFF for Fiji…",
                "Step 3/5 — Running Fiji SIFT alignment…",
                "Step 4/5 — Parsing transformation matrix…",
                "Step 5/5 — Applying transform to all MSI channels…",
                "Done — Downsampling aligned MSI to original height…",
            ]
            step_counter = [0]

            def _status(msg: str) -> None:
                status_box.info(msg)
                idx = next(
                    (i for i, s in enumerate(step_msgs) if s.startswith(msg[:20])),
                    step_counter[0],
                )
                step_counter[0] = idx
                progress.progress(
                    min(1.0, (idx + 1) / len(step_msgs)),
                    text=msg,
                )

            try:
                work_dir = Path(st.session_state.work_dir)

                aligned_msi, log = register_msi_to_if(
                    if_stack=if_stack,
                    msi_stack=msi_stack,
                    if_ref_channel=if_ref_idx,
                    msi_ref_channel=msi_ref_idx,
                    work_dir=work_dir,
                    fiji_path=fiji_path,
                    sift_params=sift_params,
                    status_cb=_status,
                )

                st.session_state.reg_aligned_msi = aligned_msi
                st.session_state.reg_aligned_labels = list(msi_labels)
                st.session_state.reg_log = log

                progress.progress(1.0, text="Registration complete.")
                status_box.success(
                    f"Registration complete. "
                    f"Aligned MSI shape: {aligned_msi.shape[-1]} × {aligned_msi.shape[-2]} px"
                )

            except Exception as e:
                status_box.error("Registration failed.")
                st.exception(e)

        # ----------------------------------------------------------------
        # 5. Results: log + download
        # ----------------------------------------------------------------
        if st.session_state.reg_aligned_msi is not None:
            st.divider()
            st.markdown("#### Results")

            aligned = st.session_state.reg_aligned_msi
            labels = st.session_state.reg_aligned_labels or []

            st.caption(
                f"Aligned MSI: {aligned.shape[0]} channels, "
                f"{aligned.shape[-1]} × {aligned.shape[-2]} px"
            )

            dl_col1, dl_col2 = st.columns(2)

            with dl_col1:
                st.download_button(
                    "Download aligned MSI TIFF",
                    data=_stack_to_bytes(aligned, labels),
                    file_name="MSI_aligned.tif",
                    mime="image/tiff",
                    use_container_width=True,
                    type="primary",
                )

            with dl_col2:
                if st.session_state.reg_log:
                    st.download_button(
                        "Download Fiji log",
                        data=st.session_state.reg_log.encode("utf-8"),
                        file_name="fiji_registration_log.txt",
                        mime="text/plain",
                        use_container_width=True,
                    )

            with st.expander("Fiji log", expanded=False):
                st.code(st.session_state.reg_log or "(empty)", language=None)
