"""
Entry point for the Streamlit app.

Run with:
    streamlit run app/main.py
or via the shim:
    streamlit run registration_app.py
"""

import sys
from pathlib import Path

# Ensure the project root (parent of the `app/` package) is on sys.path
# so that `from app.X import Y` works regardless of how Streamlit is invoked.
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import streamlit as st

from app.components.sidebar import init_session_state, render_sidebar
from app.views.analysis import render_analysis
from app.views.comparative import render_comparative
from app.views.gnn_explainability import render_gnn_explainability
from app.views.napari_launch import render_napari_launch
from app.views.positivity import render_positivity
from app.views.preprocessing import render_preprocessing
from app.views.registration import render_registration
from app.views.viewer import render_viewer

# Page config must be the very first Streamlit call
st.set_page_config(
    page_title="MetaBar",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialise session state
init_session_state()

# Header
st.title("MetaBar")

# Sidebar (returns display settings chosen by the user)
display_settings = render_sidebar()

# Main viewer — shown first so data is immediately visible after loading
render_viewer(display_settings)

# Preprocessing (rotate / flip / crop) — only visible when both stacks are loaded
render_preprocessing()

# Registration — align MSI to IF via Fiji SIFT
render_registration()

# Analysis pipeline
render_analysis()

# Positivity thresholding
render_positivity()

# GNN Explainability
render_gnn_explainability()

# Interactive napari viewer
render_napari_launch()

# Cross-sample comparative analysis
render_comparative()
