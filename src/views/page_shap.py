"""
Seite: SHAP Interpretierbarkeitsanalyse.
Zeigt Feature-Rankings, Summary Plots und lokale Erklärungen.
"""

import os

import streamlit as st

from utils.data_loader import load_evidence_manifest, MODEL_LABELS


def render():
    st.title("🔍 SHAP Interpretierbarkeitsanalyse")

    manifest, _ = load_evidence_manifest()

    # --- Feature-Ranking-Vergleich ---
    st.subheader("Feature-Ranking-Vergleich (alle Modelle)")
    ranking_path = os.path.join("plots", "shap_ranking_comparison.png")
    if os.path.exists(ranking_path):
        st.image(ranking_path)

    if manifest:
        consistency = manifest["interpretability"]["cross_model_consistency"]
        st.success(
            f"**{consistency['n_overlapping']} von 10** Top-Features stimmen "
            f"über alle drei Modelle überein: "
            f"{', '.join(consistency['features_in_all_top10'])}"
        )

    # --- Modellauswahl ---
    st.subheader("Detailanalyse pro Modell")
    selected = st.selectbox(
        "Modell auswählen",
        list(MODEL_LABELS.keys()),
        format_func=lambda x: MODEL_LABELS[x],
        key="shap_page_model",
    )

    col1, col2 = st.columns(2)
    global_path = os.path.join("plots", f"shap_global_{selected}.png")
    summary_path = os.path.join("plots", f"shap_summary_{selected}.png")
    if os.path.exists(global_path):
        col1.subheader("Globale Feature Importance")
        col1.image(global_path)
    if os.path.exists(summary_path):
        col2.subheader("SHAP Summary Plot")
        col2.image(summary_path)

    # --- Lokale Erklärungen ---
    st.subheader("Lokale Erklärungen (Einzelfälle)")
    case_type = st.selectbox(
        "Falltyp auswählen",
        ["true_positive", "false_negative", "true_negative"],
        format_func=lambda x: {
            "true_positive": "✅ True Positive (korrekt als Bad erkannt)",
            "false_negative": "❌ False Negative (Bad als Good klassifiziert)",
            "true_negative": "✅ True Negative (korrekt als Good erkannt)",
        }[x],
        key="shap_page_case",
    )
    local_path = os.path.join("plots", f"shap_local_{selected}_{case_type}.png")
    if os.path.exists(local_path):
        st.image(local_path)