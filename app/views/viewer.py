"""
IF + MSI Viewer page.

Renders the download section and the main viewer (single-channel,
RGB overlay, or merged IF/MSI overlay).
"""

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from app.utils.image_processing import (
    apply_if_color,
    apply_viridis,
    make_merged_if_msi_overlay,
    make_rgb_overlay,
    robust_display_image,
)


# ---------------------------------------------------------------------------
# Downloads
# ---------------------------------------------------------------------------

def render_downloads() -> None:
    extracted_path_str = st.session_state.get("extracted_tiff_path")
    if extracted_path_str is None:
        return

    extracted_path = Path(extracted_path_str)
    if not extracted_path.exists():
        return

    with st.expander("Downloads", expanded=False):
        dl_col1, dl_col2 = st.columns(2)

        with dl_col1:
            with open(extracted_path, "rb") as f:
                st.download_button(
                    "Download extracted MSI TIFF",
                    data=f,
                    file_name=extracted_path.name,
                    mime="image/tiff",
                    use_container_width=True,
                )

        with dl_col2:
            labels = st.session_state.msi_labels or []
            channel_csv = (
                pd.DataFrame(
                    {
                        "channel_index_0based": np.arange(len(labels)),
                        "channel_name": labels,
                    }
                )
                .to_csv(index=False)
                .encode("utf-8")
            )

            st.download_button(
                "Download channel names CSV",
                data=channel_csv,
                file_name="channel_names.csv",
                mime="text/csv",
                use_container_width=True,
            )


# ---------------------------------------------------------------------------
# Individual view modes
# ---------------------------------------------------------------------------

def _render_single_channels(
    if_stack, if_labels, msi_stack, msi_labels, pmin, pmax, gamma
) -> None:
    if_display = [f"{i}: {if_labels[i]}" for i in range(if_stack.shape[0])]
    msi_display = [f"{i}: {msi_labels[i]}" for i in range(msi_stack.shape[0])]

    control_col1, control_col2 = st.columns(2)

    with control_col1:
        if_idx_label = st.selectbox(
            "IF channel", if_display, index=0, key="if_single_channel"
        )
        if_idx = if_display.index(if_idx_label)

    with control_col2:
        msi_idx_label = st.selectbox(
            "MSI channel", msi_display, index=0, key="msi_single_channel"
        )
        msi_idx = msi_display.index(msi_idx_label)

    image_col1, image_col2 = st.columns(2)

    with image_col1:
        norm = robust_display_image(if_stack[if_idx], pmin=pmin, pmax=pmax, gamma=gamma)
        colored = apply_if_color(norm, if_idx)
        st.image(
            colored,
            caption=if_labels[if_idx],
            use_container_width=True,
            clamp=True,
        )

    with image_col2:
        norm = robust_display_image(msi_stack[msi_idx], pmin=pmin, pmax=pmax, gamma=gamma)
        colored = apply_viridis(norm)
        st.image(
            colored,
            caption=msi_labels[msi_idx],
            use_container_width=True,
            clamp=True,
        )


def _render_rgb_overlays(
    if_stack, if_labels, msi_stack, msi_labels, pmin, pmax, gamma
) -> None:
    if_options = list(range(if_stack.shape[0]))
    msi_options = list(range(msi_stack.shape[0]))

    if_display = [f"{i}: {if_labels[i]}" for i in if_options]
    msi_display = [f"{i}: {msi_labels[i]}" for i in msi_options]

    control_col1, control_col2 = st.columns(2)

    with control_col1:
        default_if = if_options[: min(3, len(if_options))]
        if_overlay_labels = st.multiselect(
            "IF overlay channels",
            if_display,
            default=[if_display[i] for i in default_if],
            key="if_overlay_channels",
            help="Up to three channels mapped to red, green, blue.",
        )
        if_overlay_idx = [if_display.index(x) for x in if_overlay_labels][:3]

        if len(if_overlay_labels) > 3:
            st.caption("Only the first three IF channels are shown as RGB.")

    with control_col2:
        default_msi = msi_options[: min(3, len(msi_options))]
        msi_overlay_labels = st.multiselect(
            "MSI overlay channels",
            msi_display,
            default=[msi_display[i] for i in default_msi],
            key="msi_overlay_channels",
            help="Up to three channels mapped to red, green, blue.",
        )
        msi_overlay_idx = [msi_display.index(x) for x in msi_overlay_labels][:3]

        if len(msi_overlay_labels) > 3:
            st.caption("Only the first three MSI channels are shown as RGB.")

    image_col1, image_col2 = st.columns(2)

    with image_col1:
        st.image(
            make_rgb_overlay(if_stack, if_overlay_idx, pmin=pmin, pmax=pmax, gamma=gamma),
            caption="IF RGB overlay",
            use_container_width=True,
            clamp=True,
        )

    with image_col2:
        st.image(
            make_rgb_overlay(msi_stack, msi_overlay_idx, pmin=pmin, pmax=pmax, gamma=gamma),
            caption="MSI RGB overlay",
            use_container_width=True,
            clamp=True,
        )


def _render_merged_overlay(
    if_stack, if_labels, msi_stack, msi_labels, pmin, pmax, gamma
) -> None:
    if_display = [f"{i}: {if_labels[i]}" for i in range(if_stack.shape[0])]
    msi_display = [f"{i}: {msi_labels[i]}" for i in range(msi_stack.shape[0])]

    same_shape = if_stack.shape[-2:] == msi_stack.shape[-2:]

    if not same_shape:
        st.warning(
            "Merged IF/MSI overlay requires IF and MSI images to have the same "
            f"height and width. IF: {if_stack.shape[-2:]}, MSI: {msi_stack.shape[-2:]}."
        )

    control_col1, control_col2 = st.columns(2)

    with control_col1:
        if_merge_label = st.selectbox(
            "IF channel", if_display, index=0, key="if_merge_channel"
        )
        if_merge_idx = if_display.index(if_merge_label)

    with control_col2:
        msi_merge_label = st.selectbox(
            "MSI channel", msi_display, index=0, key="msi_merge_channel"
        )
        msi_merge_idx = msi_display.index(msi_merge_label)

    merged = make_merged_if_msi_overlay(
        if_stack=if_stack,
        msi_stack=msi_stack,
        if_channel=if_merge_idx,
        msi_channel=msi_merge_idx,
        pmin=pmin,
        pmax=pmax,
        gamma=gamma,
    )

    if merged is not None:
        st.image(
            merged,
            caption="Merged overlay: MSI in magenta, IF in green",
            use_container_width=True,
            clamp=True,
        )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def render_viewer(display_settings: dict) -> None:
    """
    Render the main viewer area.
    `display_settings` comes from sidebar.render_sidebar().
    """
    if_stack = st.session_state.if_stack
    if_labels = st.session_state.if_labels
    msi_stack = st.session_state.msi_stack
    msi_labels = st.session_state.msi_labels

    render_downloads()

    # Landing / missing-data messages
    if if_stack is None and msi_stack is None:
        st.info("Load an IF image and an MSI image from the sidebar to begin.")
        return

    if if_stack is None:
        st.info("MSI is loaded. Load an IF image from the sidebar to enable paired viewing.")
        return

    if msi_stack is None:
        st.info("IF is loaded. Load an MSI image from the sidebar to enable paired viewing.")
        return

    # Both loaded — show viewer
    st.subheader("Viewer")

    view_mode = display_settings["view_mode"]
    pmin = display_settings["pmin"]
    pmax = display_settings["pmax"]
    gamma = display_settings["gamma"]

    if view_mode == "Single channels":
        _render_single_channels(if_stack, if_labels, msi_stack, msi_labels, pmin, pmax, gamma)
    else:
        _render_rgb_overlays(if_stack, if_labels, msi_stack, msi_labels, pmin, pmax, gamma)
