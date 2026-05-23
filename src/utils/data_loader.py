"""
Zentrale Datenlade-Funktionen für den Streamlit-Demonstrator.
Alle Lade-Operationen sind gecached, um Wiederholungen zu vermeiden.
"""

import os
import json
import hashlib

import streamlit as st
import pandas as pd
import joblib

from preprocessing import run_preprocessing


# --- Konstanten ---

MODEL_NAMES = ["logistic_regression", "random_forest", "xgboost"]

MODEL_LABELS = {
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
}


# --- Cached Loaders ---

@st.cache_data
def load_evidence_manifest(run_id: str = None):
    """
    Lädt ein Evidence-Pack-Manifest.
    Ohne run_id wird das neueste geladen.
    Gibt (manifest_dict, run_id_string) zurück.
    """
    evidence_dir = "evidence"
    run_dirs = sorted([d for d in os.listdir(evidence_dir) if d.startswith("run_")])

    if not run_dirs:
        return None, None

    if run_id is None:
        run_id = run_dirs[-1]

    manifest_path = os.path.join(evidence_dir, run_id, "evidence_manifest.json")
    with open(manifest_path, "r", encoding="utf-8") as f:
        return json.load(f), run_id


@st.cache_resource
def load_models_and_data():
    """Lädt alle Modelle und die vorverarbeiteten Daten."""
    data = run_preprocessing()
    models = {}
    for name in MODEL_NAMES:
        models[name] = joblib.load(os.path.join("models", f"{name}.joblib"))
    return models, data


@st.cache_data
def load_raw_data():
    """Lädt den Rohdatensatz als DataFrame."""
    return pd.read_csv(os.path.join("data", "german_credit.csv"))


def get_available_runs() -> list:
    """Gibt eine sortierte Liste aller verfügbaren Run-IDs zurück."""
    evidence_dir = "evidence"
    return sorted([d for d in os.listdir(evidence_dir) if d.startswith("run_")])


def get_model_hash(model_name: str) -> str:
    """Berechnet den SHA-256-Hash eines serialisierten Modells."""
    path = os.path.join("models", f"{model_name}.joblib")
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def get_data_hash() -> str:
    """Liest den gespeicherten SHA-256-Hash des Datensatzes."""
    path = os.path.join("data", "german_credit.sha256")
    if os.path.exists(path):
        with open(path, "r") as f:
            return f.read().strip()
    return "unbekannt"