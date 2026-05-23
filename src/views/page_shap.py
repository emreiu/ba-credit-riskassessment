"""
Page: SHAP Interpretability Analysis.
Displays feature rankings, summary plots and local explanations.
"""

import os

import streamlit as st

from utils.data_loader import load_evidence_manifest, MODEL_LABELS


def render():
    st.title("🔍 SHAP Interpretability Analysis")

    manifest, _ = load_evidence_manifest()

    # --- Feature ranking comparison ---
    st.subheader("Feature Ranking Comparison (All Models)")
    ranking_path = os.path.join("plots", "shap_ranking_comparison.png")
    if os.path.exists(ranking_path):
        st.image(ranking_path)

    if manifest:
        consistency = manifest["interpretability"]["cross_model_consistency"]
        st.success(
            f"**{consistency['n_overlapping']} out of 10** top features are consistent "
            f"across all three models: "
            f"{', '.join(consistency['features_in_all_top10'])}"
        )

    # --- Model selection ---
    st.subheader("Detailed Analysis per Model")
    selected = st.selectbox(
        "Select model",
        list(MODEL_LABELS.keys()),
        format_func=lambda x: MODEL_LABELS[x],
        key="shap_page_model",
    )

    col1, col2 = st.columns(2)
    global_path = os.path.join("plots", f"shap_global_{selected}.png")
    summary_path = os.path.join("plots", f"shap_summary_{selected}.png")
    if os.path.exists(global_path):
        col1.subheader("Global Feature Importance")
        col1.image(global_path)
    if os.path.exists(summary_path):
        col2.subheader("SHAP Summary Plot")
        col2.image(summary_path)

    # --- Local explanations ---
    st.subheader("Local Explanations (Individual Cases)")
    case_type = st.selectbox(
        "Select case type",
        ["true_positive", "false_negative", "true_negative"],
        format_func=lambda x: {
            "true_positive": "✅ True Positive (correctly identified as Bad)",
            "false_negative": "❌ False Negative (Bad classified as Good)",
            "true_negative": "✅ True Negative (correctly identified as Good)",
        }[x],
        key="shap_page_case",
    )
    local_path = os.path.join("plots", f"shap_local_{selected}_{case_type}.png")
    if os.path.exists(local_path):
        st.image(local_path)