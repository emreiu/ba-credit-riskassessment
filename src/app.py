"""
Credit Risk Evidence Pack - Streamlit-Demonstrator.
Haupteinstiegspunkt mit Navigation.

Ausfuehren: streamlit run src/app.py
"""

import warnings
import streamlit as st

warnings.filterwarnings("ignore", category=FutureWarning)

# --- Seiten-Konfiguration ---
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
        "Modellvergleich",
        "SHAP Analyse",
        "Einzelfall-Pruefung",
        "Archiv",
    ],
)

st.sidebar.markdown("---")
st.sidebar.caption(
    "Prototyp im Rahmen der Bachelorarbeit\n\n"
    "FH Technikum Wien - Wirtschaftsinformatik"
)

# --- Seitenweiche ---
if page == "Evidence Pack":
    from views.page_evidence import render
    render()

elif page == "Modellvergleich":
    from views.page_models import render
    render()

elif page == "SHAP Analyse":
    from views.page_shap import render
    render()

elif page == "Einzelfall-Pruefung":
    from views.page_cases import render
    render()

elif page == "Archiv":
    from views.page_archive import render
    render()