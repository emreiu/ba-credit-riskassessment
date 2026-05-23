"""
Speicher- und Ladefunktionen für archivierte Kreditentscheidungen.
Fälle werden als JSON-Dateien im Ordner evidence/cases/ gespeichert.
"""

import os
import json
import glob
from datetime import datetime


CASES_DIR = os.path.join("evidence", "cases")


def ensure_cases_dir():
    """Stellt sicher, dass der Cases-Ordner existiert."""
    os.makedirs(CASES_DIR, exist_ok=True)


def save_case(case_record: dict) -> str:
    """
    Speichert einen Fall als JSON-Datei.
    Gibt den Dateipfad zurück.
    """
    ensure_cases_dir()
    case_id = case_record["case_id"]
    filepath = os.path.join(CASES_DIR, f"{case_id}.json")
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(case_record, f, indent=2, ensure_ascii=False, default=str)
    return filepath


def load_all_cases() -> list:
    """Lädt alle archivierten Fälle, sortiert nach Datum (neueste zuerst)."""
    ensure_cases_dir()
    cases = []
    for filepath in glob.glob(os.path.join(CASES_DIR, "CASE-*.json")):
        with open(filepath, "r", encoding="utf-8") as f:
            cases.append(json.load(f))
    cases.sort(key=lambda c: c.get("timestamp", ""), reverse=True)
    return cases


def load_case_by_id(case_id: str) -> dict:
    """Lädt einen einzelnen Fall anhand der Case-ID."""
    filepath = os.path.join(CASES_DIR, f"{case_id}.json")
    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def build_case_record(
    applicant_data: dict,
    source_index: int,
    reference: str,
    actual_label,
    production_model: str,
    threshold: float,
    decision: str,
    all_predictions: dict,
    shap_values_dict: dict,
    shap_base_value: float,
    shap_method: str,
    top_reasons_increasing: list,
    top_reasons_decreasing: list,
    manifest: dict,
    model_hash: str,
    data_hash: str,
) -> dict:
    """
    Baut ein vollständiges, self-contained Case-Record.
    Enthält alle Informationen, die ein Auditor braucht –
    auch wenn das Evidence Pack später gelöscht oder überschrieben wird.
    """
    prod_pred = all_predictions[production_model]

    return {
        # --- Identifikation ---
        "case_id": f"CASE-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        "timestamp": datetime.now().isoformat(),
        "reference": reference,
        "source_dataset_index": source_index,

        # --- Antragsdaten ---
        "applicant_data": applicant_data,
        "actual_label": actual_label,

        # --- Entscheidung ---
        "decision": {
            "production_model": production_model,
            "threshold": threshold,
            "result": decision,
            "p_good": prod_pred["p_good"],
            "p_bad": prod_pred["p_bad"],
        },
        "all_model_predictions": all_predictions,

        # --- SHAP-Erklärung ---
        "shap_explanation": {
            "method": shap_method,
            "base_value": shap_base_value,
            "shap_values": shap_values_dict,
            "top_reasons_increasing": top_reasons_increasing,
            "top_reasons_decreasing": top_reasons_decreasing,
        },

        # --- Evidence-Pack-Snapshot (self-contained) ---
        "evidence_snapshot": {
            "run_id": manifest.get("run_id", "unbekannt") if manifest else "unbekannt",
            "generated_at": manifest.get("generated_at", "unbekannt") if manifest else "unbekannt",
            "model_hash_sha256": model_hash,
            "dataset_hash_sha256": data_hash,
            "hyperparameters": (
                manifest["model_documentation"][production_model]["best_hyperparameters"]
                if manifest and "model_documentation" in manifest
                else {}
            ),
            "preprocessing": (
                manifest["dataset_documentation"]["preprocessing"]
                if manifest and "dataset_documentation" in manifest
                else {}
            ),
            "software_versions": (
                manifest.get("reproducibility", {}).get("python_packages", {})
                if manifest else {}
            ),
        },
    }