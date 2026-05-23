"""
SHAP-Hilfsfunktionen für den Streamlit-Demonstrator.
Berechnung von SHAP-Werten und Erzeugung lesbarer Erklärungen.
"""

import numpy as np
import shap
import matplotlib.pyplot as plt
import streamlit as st


def compute_shap_for_instance(model, model_name: str, instance_processed, X_train):
    """
    Berechnet SHAP-Werte für eine einzelne vorverarbeitete Instanz.
    Verwendet TreeExplainer für Baummodelle, KernelExplainer für LR.
    Gibt (shap_values_1d, base_value_float) zurück.
    """
    np.random.seed(42)

    if model_name in ["random_forest", "xgboost"]:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(instance_processed)
    else:
        background = shap.sample(X_train, 100)
        explainer = shap.KernelExplainer(model.predict_proba, background)
        shap_values = explainer.shap_values(instance_processed, silent=True)

    # Auf Klasse 1 (Bad) reduzieren
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        shap_values = shap_values[:, :, 1]

    # Base Value extrahieren
    base_value = explainer.expected_value
    if isinstance(base_value, (list, np.ndarray)):
        base_value = float(np.array(base_value).flatten()[1])
    else:
        base_value = float(base_value)

    # Auf 1D reduzieren falls nötig
    sv = shap_values[0] if shap_values.ndim == 2 else shap_values

    return sv, base_value


def generate_top_reasons(shap_values_1d, feature_names: list, top_n: int = 5):
    """
    Erzeugt sortierte Listen der risikosteigerden und risikosenkenden Faktoren.
    Gibt (list_increasing, list_decreasing) zurück.
    Jeder Eintrag ist ein Dict mit 'feature', 'shap_value', 'direction'.
    """
    ranked = sorted(
        zip(feature_names, shap_values_1d),
        key=lambda x: abs(x[1]),
        reverse=True,
    )[:top_n]

    increasing = []
    decreasing = []

    for feat, val in ranked:
        entry = {
            "feature": feat.replace("_", " "),
            "shap_value": round(float(val), 4),
            "direction": "↑ Risiko" if val > 0 else "↓ Risiko",
        }
        if val > 0:
            increasing.append(entry)
        else:
            decreasing.append(entry)

    return increasing, decreasing


def render_waterfall_plot(shap_values_1d, base_value: float,
                          instance_data, feature_names: list,
                          max_display: int = 10):
    """
    Erzeugt einen SHAP Waterfall-Plot und zeigt ihn in Streamlit an.
    Kompakte Darstellung für Screenshots.
    """
    explanation = shap.Explanation(
        values=shap_values_1d,
        base_values=base_value,
        data=instance_data,
        feature_names=feature_names,
    )

    fig, ax = plt.subplots(figsize=(6, 3))
    shap.plots.waterfall(explanation, max_display=max_display, show=False)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=False)
    plt.close()