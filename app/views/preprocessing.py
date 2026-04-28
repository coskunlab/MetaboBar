"""
Preprocessing panel: rotate, flip, draw bounding box with mouse to crop,
and download the processed IF and MSI stacks.

Uses st.plotly_chart(on_select=...) — available in Streamlit ≥ 1.33 —
so box-select events are captured without interfering with other widgets.
"""

import base64
import io
from typing import Optional, Tuple

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from app.utils.image_processing import apply_if_color, apply_viridis, robust_display_image
from app.utils.transforms import (
    crop_stack,
    flip_stack,
    rotate_stack,
    stack_to_tiff_bytes,
)

MAX_DISPLAY_PX = 800


# ---------------------------------------------------------------------------
# Pure helpers (no Streamlit)
# ---------------------------------------------------------------------------

def _apply_transforms(stack: np.ndarray, rotation: int, flip: str) -> np.ndarray:
    stack = rotate_stack(stack, rotation)
    if flip != "None":
        stack = flip_stack(stack, flip.lower())
    return stack


def _to_png_b64(img: np.ndarray, max_px: int = MAX_DISPLAY_PX) -> Tuple[str, int, int]:
    """
    Downsample float32 image (H×W or H×W×3, values 0-1) to fit within
    max_px on longest side. Returns (base64_png_string, orig_H, orig_W).
    """
    from PIL import Image as PILImage

    if img.ndim == 2:
        H, W = img.shape
    else:
        H, W = img.shape[:2]

    scale = min(1.0, max_px / max(H, W))
    th = max(1, int(H * scale))
    tw = max(1, int(W * scale))

    img_uint8 = (img * 255).clip(0, 255).astype(np.uint8)
    mode = "L" if img.ndim == 2 else "RGB"
    pil = PILImage.fromarray(img_uint8, mode=mode).resize((tw, th), PILImage.LANCZOS)

    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8"), H, W


def _make_figure(
    stack: np.ndarray,
    channel: int,
    box: Optional[Tuple[int, int, int, int]],
    title: str,
    modality: str,          # "IF" or "MSI"
) -> go.Figure:
    norm = robust_display_image(stack[channel], pmin=1.0, pmax=99.5)

    if modality == "IF":
        display = apply_if_color(norm, channel)   # H×W×3
    else:
        display = apply_viridis(norm)             # H×W×3

    b64, H, W = _to_png_b64(display)

    fig = go.Figure()

    # Invisible scatter anchors the axes
    fig.add_trace(go.Scatter(
        x=[0, W], y=[0, H],
        mode="markers",
        marker=dict(opacity=0, size=1),
        showlegend=False,
        hoverinfo="skip",
    ))

    # Image pinned to data coordinates
    fig.add_layout_image(dict(
        source=f"data:image/png;base64,{b64}",
        xref="x", yref="y",
        x=0, y=0,
        sizex=W, sizey=H,
        sizing="stretch",
        layer="below",
        opacity=1.0,
    ))

    # Existing crop box
    if box is not None:
        x0b, y0b, x1b, y1b = box
        fig.add_shape(
            type="rect",
            x0=x0b, y0=y0b, x1=x1b, y1=y1b,
            line=dict(color="#FF6600", width=2, dash="dash"),
            fillcolor="rgba(255,102,0,0.08)",
        )

    fig.update_layout(
        title=dict(text=title, font=dict(size=12, color="#222")),
        xaxis=dict(
            title=dict(text="x (pixels)", font=dict(color="#222")),
            range=[0, W],
            showgrid=True,
            gridcolor="#ccc",
            tickfont=dict(color="#222"),
            zeroline=False,
        ),
        yaxis=dict(
            title=dict(text="y (pixels)", font=dict(color="#222")),
            range=[H, 0],
            scaleanchor="x",
            showgrid=True,
            gridcolor="#ccc",
            tickfont=dict(color="#222"),
            zeroline=False,
        ),
        dragmode="select",
        selectdirection="d",
        margin=dict(l=60, r=10, t=40, b=50),
        height=450,
        paper_bgcolor="#f8f9fa",
        plot_bgcolor="#f8f9fa",
        font=dict(color="#222"),
        newselection=dict(line=dict(color="#FF6600", width=2, dash="dash")),
    )

    return fig


def _box_from_event(event) -> Optional[Tuple[int, int, int, int]]:
    """Extract (x0,y0,x1,y1) from a st.plotly_chart on_select event dict."""
    if event is None:
        return None
    try:
        sel = event.get("selection", {})
        box = sel.get("box", [])
        if not box:
            return None
        b = box[0]
        xs = b.get("x", [])
        ys = b.get("y", [])
        if len(xs) < 2 or len(ys) < 2:
            return None
        return int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Per-modality crop section
# ---------------------------------------------------------------------------

def _render_crop_section(modality: str, stack_key: str, labels_key: str) -> None:
    stack: np.ndarray = st.session_state[stack_key]
    labels = st.session_state[labels_key] or []
    H, W = stack.shape[-2], stack.shape[-1]
    C = stack.shape[0]

    st.markdown(f"**{modality}** — {C} ch, {W} × {H} px")

    # Channel selector — uses its own key, won't be reset by figure events
    ch_options = [
        f"{i}: {labels[i]}" if i < len(labels) else f"C{i}"
        for i in range(C)
    ]
    ch_label = st.selectbox(
        f"Preview channel ({modality})",
        ch_options,
        index=0,
        key=f"pp_{modality.lower()}_preview_ch",
    )
    ch_idx = ch_options.index(ch_label)

    box_key = f"pp_{modality.lower()}_box"
    if box_key not in st.session_state:
        st.session_state[box_key] = None

    current_box: Optional[Tuple[int, int, int, int]] = st.session_state[box_key]

    fig = _make_figure(stack, ch_idx, current_box, f"{modality} — ch {ch_idx}", modality)

    st.caption(
        "Click and drag on the image to draw a crop box, "
        "then click **Apply crop** below."
    )

    # on_select captures box-select without triggering a full rerun of other widgets
    event = st.plotly_chart(
        fig,
        use_container_width=True,
        key=f"pp_{modality.lower()}_chart",
        on_select="rerun",
        selection_mode="box",
    )

    new_box = _box_from_event(event)
    if new_box is not None:
        x0, y0, x1, y1 = new_box
        # Clamp to stack bounds
        x0 = max(0, min(x0, W - 1))
        y0 = max(0, min(y0, H - 1))
        x1 = max(x0 + 1, min(x1, W))
        y1 = max(y0 + 1, min(y1, H))
        if (x0, y0, x1, y1) != current_box:
            st.session_state[box_key] = (x0, y0, x1, y1)
            st.rerun()

    # Info caption
    if current_box is not None:
        x0, y0, x1, y1 = current_box
        st.caption(
            f"Pending crop: x [{x0} → {x1}] ({x1-x0} px)  |  "
            f"y [{y0} → {y1}] ({y1-y0} px)"
        )
    else:
        st.caption("No crop box drawn yet.")

    btn_col1, btn_col2 = st.columns(2)

    with btn_col1:
        if st.button(
            f"Apply crop to {modality}",
            key=f"pp_apply_crop_{modality.lower()}",
            type="primary",
            disabled=current_box is None,
        ):
            try:
                x0, y0, x1, y1 = current_box
                st.session_state[stack_key] = crop_stack(
                    stack, y0=y0, y1=y1, x0=x0, x1=x1
                )
                st.session_state[box_key] = None
                st.success(f"{modality} cropped to {x1-x0} × {y1-y0} px.")
                st.rerun()
            except ValueError as e:
                st.error(str(e))

    with btn_col2:
        if st.button(
            f"Clear box ({modality})",
            key=f"pp_clear_box_{modality.lower()}",
            disabled=current_box is None,
        ):
            st.session_state[box_key] = None
            st.rerun()


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def render_preprocessing() -> None:
    if_stack: Optional[np.ndarray] = st.session_state.get("if_stack")
    msi_stack: Optional[np.ndarray] = st.session_state.get("msi_stack")

    if if_stack is None or msi_stack is None:
        return

    with st.expander("Preprocessing — rotate, flip, crop", expanded=False):

        # ----------------------------------------------------------------
        # 1. Transform
        # ----------------------------------------------------------------
        st.markdown("#### Transform")

        t_col1, t_col2, t_col3 = st.columns(3)
        with t_col1:
            target = st.radio(
                "Apply to", ["IF", "MSI", "Both"],
                horizontal=True, key="pp_target",
            )
        with t_col2:
            rotation = st.selectbox(
                "Rotation (CCW)", [0, 90, 180, 270],
                index=0, key="pp_rotation",
            )
        with t_col3:
            flip = st.selectbox(
                "Flip", ["None", "Horizontal", "Vertical"],
                index=0, key="pp_flip",
            )

        if st.button("Apply transform", key="pp_apply_transform"):
            if target in ("IF", "Both"):
                st.session_state.if_stack = _apply_transforms(
                    st.session_state.if_stack, rotation, flip
                )
            if target in ("MSI", "Both"):
                st.session_state.msi_stack = _apply_transforms(
                    st.session_state.msi_stack, rotation, flip
                )
            st.success("Transform applied.")
            st.rerun()

        st.divider()

        # ----------------------------------------------------------------
        # 2. Crop — independent per modality
        # ----------------------------------------------------------------
        st.markdown("#### Crop — click and drag to draw a box")

        crop_col_if, crop_col_msi = st.columns(2)
        with crop_col_if:
            _render_crop_section("IF", "if_stack", "if_labels")
        with crop_col_msi:
            _render_crop_section("MSI", "msi_stack", "msi_labels")

        st.divider()

        # ----------------------------------------------------------------
        # 3. Download
        # ----------------------------------------------------------------
        st.markdown("#### Download processed stacks")

        dl_col1, dl_col2 = st.columns(2)
        with dl_col1:
            st.download_button(
                "Download processed IF TIFF",
                data=stack_to_tiff_bytes(
                    st.session_state.if_stack,
                    st.session_state.if_labels or [],
                ),
                file_name="IF_processed.tif",
                mime="image/tiff",
                use_container_width=True,
            )
        with dl_col2:
            st.download_button(
                "Download processed MSI TIFF",
                data=stack_to_tiff_bytes(
                    st.session_state.msi_stack,
                    st.session_state.msi_labels or [],
                ),
                file_name="MSI_processed.tif",
                mime="image/tiff",
                use_container_width=True,
            )
