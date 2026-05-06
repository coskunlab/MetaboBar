"""
Cross-sample comparative analysis view.

User adds multiple sample result folders (each produced by the full pipeline),
gives each a name, then runs binary and/or multiclass comparison.
Results (CSVs + plots) are saved and displayed in the page.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import streamlit as st

from app.utils.analysis.comparative import run_binary_comparison


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

def _init_state() -> None:
    for k, v in {
        "comp_samples": {},
        "comp_output_dir": "",
        "comp_binary_plots": {},
    }.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ---------------------------------------------------------------------------
# Sample manager
# ---------------------------------------------------------------------------

def _render_sample_manager() -> Dict[str, Path]:
    st.markdown("**Samples**")
    st.caption(
        "Add one entry per sample (tissue/condition). "
        "Each path should be the results root folder for that sample "
        "(the same folder you used as 'Results root folder' in the Analysis and GNN steps)."
    )

    # Add new sample
    c1, c2, c3 = st.columns([2, 4, 1])
    with c1:
        new_name = st.text_input("Sample name", key="comp_new_name",
                                 placeholder="e.g. no_dose")
    with c2:
        new_path = st.text_input("Results folder", key="comp_new_path",
                                 placeholder=r"e.g. Y:\results\no_dose")
    with c3:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Add", key="comp_add_btn"):
            name = new_name.strip()
            path = new_path.strip()
            if name and path:
                st.session_state.comp_samples[name] = path
                st.rerun()
            else:
                st.warning("Enter both a name and a path.")

    # Show current samples
    if st.session_state.comp_samples:
        to_remove = []
        for sname, spath in list(st.session_state.comp_samples.items()):
            exists = Path(spath).exists()
            icon = "✅" if exists else "❌"
            col_n, col_p, col_r = st.columns([2, 5, 1])
            with col_n:
                st.caption(f"**{sname}**")
            with col_p:
                st.caption(f"{icon} `{spath}`")
            with col_r:
                if st.button("✕", key=f"comp_rm_{sname}"):
                    to_remove.append(sname)
        for s in to_remove:
            del st.session_state.comp_samples[s]
        if to_remove:
            st.rerun()

    valid = {n: Path(p) for n, p in st.session_state.comp_samples.items()
             if Path(p).exists()}
    return valid


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def render_comparative() -> None:
    _init_state()

    with st.expander("Cross-Sample Comparative Analysis", expanded=False):

        st.markdown("#### Cross-Sample GNN Explainability Comparison")
        st.caption(
            "Compare GNN feature importance results across multiple samples. "
            "Each sample must have been processed through the full pipeline "
            "(Analysis → GNN Explainability) with the same MSI channels."
        )

        # ---- Sample manager ----
        valid_samples = _render_sample_manager()

        if len(valid_samples) < 2:
            st.info("Add at least 2 valid sample folders to run comparison.")

        st.divider()

        # ---- Output folder ----
        out_val = st.text_input(
            "Comparison output folder",
            value=st.session_state.comp_output_dir,
            key="comp_out_dir_input",
            placeholder=r"e.g. Y:\results\comparison",
        )
        st.session_state.comp_output_dir = out_val
        out_dir = Path(out_val.strip()) if out_val.strip() else None

        # ---- Parameters ----
        top_n = st.number_input("Top-N features per plot", 5, 50, 20, 5,
                                key="comp_top_n")

        st.divider()

        # ---- Run button ----
        can_run = len(valid_samples) >= 2 and out_dir is not None

        if st.button(
            "Run Comparison",
            type="primary",
            key="comp_run_binary",
            disabled=not can_run,
            use_container_width=True,
        ):
            status   = st.empty()
            progress = st.progress(0.0, text="Starting comparison…")
            try:
                plots = run_binary_comparison(
                    sample_dirs=valid_samples,
                    output_dir=out_dir,
                    top_n=int(top_n),
                    status_cb=lambda m: (status.info(m), progress.progress(0.5, text=m)),
                )
                st.session_state.comp_binary_plots = {
                    k: [str(p) for p in v] for k, v in plots.items()
                }
                progress.progress(1.0, text="Done.")
                status.success(f"Comparison complete → {out_dir}")
            except Exception as e:
                status.error("Comparison failed.")
                st.exception(e)

        # ---- Display results ----
        if st.session_state.comp_binary_plots:
            st.divider()
            st.markdown("#### Comparison results")
            for marker, png_paths in sorted(st.session_state.comp_binary_plots.items()):
                with st.expander(f"Marker: {marker}", expanded=False):
                    img_cols = st.columns(min(3, len(png_paths)))
                    for j, png in enumerate(png_paths):
                        p = Path(png)
                        if p.exists():
                            label = "Grouped bar" if "bar" in p.name else ("Violin" if "violin" in p.name else "Heatmap")
                            with img_cols[j % len(img_cols)]:
                                st.caption(label)
                                st.image(str(p), use_container_width=True)
