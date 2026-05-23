"""
Page: Model Comparison.
Displays performance metrics, ROC curves and confusion matrices.
"""

import os

import streamlit as st
import pandas as pd

from utils.data_loader import load_evidence_manifest, MODEL_LABELS


def render():
    st.title("📊 Model Comparison")

    manifest, _ = load_evidence_manifest()
    if manifest is None:
        st.error("No evidence pack found.")
        return

    # --- Metrics table ---
    st.subheader("Performance Metrics (Test Data)")
    rows = []
    for name, label in MODEL_LABELS.items():
        metrics = manifest["evaluation_metrics"][name]
        params = manifest["model_documentation"][name]
        rows.append({
            "Model": label,
            "AUC-ROC": metrics["auc_roc"],
            "Accuracy": metrics["accuracy"],
            "Precision": metrics["precision"],
            "Recall": metrics["recall"],
            "F1-Score": metrics["f1_score"],
            "CV-AUC": params["best_cv_auc"],
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # --- Plots ---
    col1, col2 = st.columns(2)
    roc_path = os.path.join("plots", "roc_curves_comparison.png")
    cm_path = os.path.join("plots", "confusion_matrices.png")
    if os.path.exists(roc_path):
        col1.subheader("ROC Curves")
        col1.image(roc_path)
    if os.path.exists(cm_path):
        col2.subheader("Confusion Matrices")
        col2.image(cm_path)

    # --- Hyperparameters ---
    st.subheader("Best Hyperparameters")
    for name, label in MODEL_LABELS.items():
        with st.expander(label):
            st.json(manifest["model_documentation"][name]["best_hyperparameters"])