"""
Credit Risk Evidence Pack - Streamlit Demonstrator.
Main entry point with navigation.

Run: streamlit run src/app.py
"""

import warnings
import streamlit as st

warnings.filterwarnings("ignore", category=FutureWarning)

# --- Page config ---
st.set_page_config(
    page_title="Credit Risk Evidence Pack",
    page_icon="🏦",
    layout="wide",
)

# --- Navigation ---
st.sidebar.title("Credit Risk")
st.sidebar.markdown("**Evidence Pack Demonstrator**")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    [
        "Evidence Pack",
        "Model Comparison",
        "SHAP Analysis",
        "Individual Case Evaluation",
        "Archive",
    ],
)

st.sidebar.markdown("---")
st.sidebar.caption(
    "Prototype developed as part of a Bachelor thesis\n\n"
    "FH Technikum Wien - Business Informatics"
)

# --- Page routing ---
if page == "Evidence Pack":
    from views.page_evidence import render
    render()

elif page == "Model Comparison":
    from views.page_models import render
    render()

elif page == "SHAP Analysis":
    from views.page_shap import render
    render()

elif page == "Individual Case Evaluation":
    from views.page_cases import render
    render()

elif page == "Archive":
    from views.page_archive import render
    render()