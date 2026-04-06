"""
Schritt 7: Automatische Evidence-Pack-Erzeugung.
Führt die gesamte Pipeline aus und erzeugt ein vollständiges,
strukturiertes Evidence Pack pro Run.

Regulatorische Grundlage: EU AI Act Art. 9-15, Annex IV
Methodische Grundlage: Sculley et al. (2015), Breck et al. (2017),
                        Mitchell et al. (2019), Gebru et al. (2021)
"""

import os
import json
import hashlib
import shutil
import time
from datetime import datetime

import numpy as np
import joblib

from preprocessing import run_preprocessing
from train import run_training
from shap_analysis import run_shap_analysis


def generate_run_id():
    """Erzeugt eine eindeutige Run-ID basierend auf Zeitstempel."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def compute_file_hash(filepath: str) -> str:
    """Berechnet SHA-256 Hash einer Datei."""
    with open(filepath, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def compute_model_hash(model_path: str) -> str:
    """Berechnet SHA-256 Hash eines serialisierten Modells."""
    return compute_file_hash(model_path)


def collect_evidence(run_id: str, training_results: dict, data: dict,
                     shap_rankings: dict, shap_overlap: set):
    """
    Sammelt alle Artefakte und erzeugt das Evidence Pack.

    Struktur orientiert sich an:
    - EU AI Act Annex IV (Technische Dokumentation)
    - Model Cards (Mitchell et al., 2019) für die Modellseite
    - Datasheets (Gebru et al., 2021) für die Datenseite
    - ML Test Score (Breck et al., 2017) für die Prüfkriterien
    """

    # Evidence-Pack-Ordner erstellen
    pack_dir = os.path.join("evidence", f"run_{run_id}")
    os.makedirs(pack_dir, exist_ok=True)
    artifacts_dir = os.path.join(pack_dir, "artifacts")
    os.makedirs(artifacts_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"EVIDENCE PACK GENERIERUNG: run_{run_id}")
    print(f"{'='*60}")

    # --- 1. Datensatz-Dokumentation (Gebru et al. / EU AI Act Art. 10) ---
    print("\n[1/7] Datensatz-Dokumentation...")

    data_hash_path = os.path.join("data", "german_credit.sha256")
    with open(data_hash_path, "r") as f:
        data_hash = f.read().strip()

    preprocessing_log_path = os.path.join("data", "preprocessing_log.json")
    with open(preprocessing_log_path, "r") as f:
        preprocessing_config = json.load(f)

    dataset_documentation = {
        "name": "German Credit Dataset (Statlog)",
        "source": "UCI Machine Learning Repository",
        "doi": "https://doi.org/10.24432/C5NC77",
        "license": "CC BY 4.0",
        "n_instances": preprocessing_config["n_train"] + preprocessing_config["n_test"],
        "n_features_original": preprocessing_config["n_features_original"],
        "n_features_encoded": preprocessing_config["n_features_encoded"],
        "target_variable": "credit_risk (0=good, 1=bad)",
        "class_distribution": {
            "train_good": preprocessing_config["n_train"] - int(preprocessing_config["n_train"] * 0.3),
            "train_bad": int(preprocessing_config["n_train"] * 0.3),
            "test_good": preprocessing_config["n_test"] - int(preprocessing_config["n_test"] * 0.3),
            "test_bad": int(preprocessing_config["n_test"] * 0.3),
        },
        "preprocessing": {
            "encoding": preprocessing_config["encoding_method"],
            "scaling": preprocessing_config["scaling_method"],
            "train_test_split": f"{1 - preprocessing_config['test_size']:.0%} / {preprocessing_config['test_size']:.0%}",
            "random_seed": preprocessing_config["random_seed"],
        },
        "data_hash_sha256": data_hash,
        "known_limitations": [
            "Small dataset (1000 instances) - limited generalizability",
            "Historical data from 1994 - may not reflect current lending practices",
            "German market specific - cultural and regulatory context",
            "No temporal component - static snapshot",
        ],
    }

    # --- 2. Modell-Dokumentation (Mitchell et al. / EU AI Act Art. 11) ---
    print("[2/7] Modell-Dokumentation...")

    model_documentation = {}
    for name, result in training_results.items():
        model_path = os.path.join("models", f"{name}.joblib")
        model_hash = compute_model_hash(model_path)

        # Modell-Datei ins Evidence Pack kopieren
        shutil.copy2(model_path, os.path.join(artifacts_dir, f"{name}.joblib"))

        model_documentation[name] = {
            "model_type": name,
            "intended_use": "Credit risk assessment - binary classification (good/bad)",
            "not_intended_for": [
                "Production deployment without further validation",
                "Automated decision-making without human oversight",
                "Application outside the German credit market context",
            ],
            "best_hyperparameters": result["best_params"],
            "best_cv_auc": result["best_cv_auc"],
            "training_time_seconds": result["training_time_seconds"],
            "n_combinations_tested": result["n_combinations_tested"],
            "model_hash_sha256": model_hash,
        }

    # --- 3. Evaluations-Metriken (EU AI Act Art. 9, 15) ---
    print("[3/7] Evaluations-Metriken...")

    evaluation_metrics = {}
    for name, result in training_results.items():
        evaluation_metrics[name] = result["test_metrics"]

    # --- 4. SHAP-Ergebnisse (EU AI Act Art. 13, 14) ---
    print("[4/7] SHAP-Interpretierbarkeit...")

    interpretability = {
        "method": "SHAP (SHapley Additive exPlanations)",
        "reference": "Lundberg & Lee (2017)",
        "explainer_types": {
            "logistic_regression": "KernelExplainer (model-agnostic)",
            "random_forest": "TreeExplainer (tree-optimized)",
            "xgboost": "TreeExplainer (tree-optimized)",
        },
        "feature_rankings_top10": shap_rankings,
        "cross_model_consistency": {
            "features_in_all_top10": sorted(list(shap_overlap)),
            "n_overlapping": len(shap_overlap),
            "consistency_ratio": f"{len(shap_overlap)}/10",
        },
    }

    # SHAP-Plots ins Evidence Pack kopieren
    shap_plots = [f for f in os.listdir("plots") if f.startswith("shap_")]
    for plot_file in shap_plots:
        shutil.copy2(
            os.path.join("plots", plot_file),
            os.path.join(artifacts_dir, plot_file),
        )

    # ROC-Kurven und Confusion Matrices auch kopieren
    for plot_file in ["roc_curves_comparison.png", "confusion_matrices.png"]:
        src = os.path.join("plots", plot_file)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(artifacts_dir, plot_file))

    # --- 5. Reproduzierbarkeits-Informationen (Sculley et al. / Breck et al.) ---
    print("[5/7] Reproduzierbarkeits-Dokumentation...")

    reproducibility = {
        "random_seed": 42,
        "python_packages": {},
        "data_hash_sha256": data_hash,
        "model_hashes_sha256": {
            name: doc["model_hash_sha256"]
            for name, doc in model_documentation.items()
        },
        "deterministic_training": True,
        "rerun_instructions": "python src/generate_evidence_pack.py",
    }

    # Paketversionen erfassen
    import sklearn, xgboost, shap as shap_lib, pandas, numpy
    reproducibility["python_packages"] = {
        "scikit-learn": sklearn.__version__,
        "xgboost": xgboost.__version__,
        "shap": shap_lib.__version__,
        "pandas": pandas.__version__,
        "numpy": numpy.__version__,
    }

    # --- 6. Regulatory Mapping (EU AI Act Art. 9-15) ---
    print("[6/7] Regulatory Mapping...")

    regulatory_mapping = {
        "regulation": "EU AI Act - Regulation (EU) 2024/1689",
        "classification": "High-Risk AI System (Annex III - Credit Scoring)",
        "applicable_from": "2 August 2026",
        "addressed_requirements": {
            "Art. 9 - Risk Management": {
                "status": "partially addressed",
                "evidence": "Evaluation metrics per run document model performance",
                "artifact": "evaluation_metrics",
            },
            "Art. 10 - Data Governance": {
                "status": "partially addressed",
                "evidence": "Dataset documentation, preprocessing log, data hash",
                "artifact": "dataset_documentation",
            },
            "Art. 11 - Technical Documentation": {
                "status": "partially addressed",
                "evidence": "This Evidence Pack serves as structured technical documentation",
                "artifact": "complete evidence pack",
            },
            "Art. 12 - Record-Keeping": {
                "status": "partially addressed",
                "evidence": "Run ID, timestamps, artifact hashes, configuration snapshots",
                "artifact": "reproducibility",
            },
            "Art. 13 - Transparency": {
                "status": "partially addressed",
                "evidence": "SHAP-based global and local explanations",
                "artifact": "interpretability",
            },
            "Art. 14 - Human Oversight": {
                "status": "partially addressed",
                "evidence": "SHAP feature importance enables human interpretation",
                "artifact": "interpretability",
            },
            "Art. 15 - Accuracy": {
                "status": "partially addressed",
                "evidence": "Declared performance metrics (AUC, Accuracy, etc.)",
                "artifact": "evaluation_metrics",
            },
        },
        "not_addressed": [
            "Post-market monitoring (Annex IV, Point 9)",
            "Cybersecurity measures (Art. 15)",
            "Formal EU Declaration of Conformity (Art. 47)",
            "Drift detection and live monitoring",
        ],
    }

    # --- 7. Evidence Pack Manifest zusammenstellen ---
    print("[7/7] Evidence Pack Manifest...")

    manifest = {
        "evidence_pack_version": "1.0.0",
        "run_id": run_id,
        "generated_at": datetime.now().isoformat(),
        "generator": "generate_evidence_pack.py",
        "dataset_documentation": dataset_documentation,
        "model_documentation": model_documentation,
        "evaluation_metrics": evaluation_metrics,
        "interpretability": interpretability,
        "reproducibility": reproducibility,
        "regulatory_mapping": regulatory_mapping,
    }

    # Manifest als JSON speichern
    manifest_path = os.path.join(pack_dir, "evidence_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False, default=str)

    # --- Zusammenfassung ---
    print(f"\n{'='*60}")
    print(f"EVIDENCE PACK ERFOLGREICH ERZEUGT")
    print(f"{'='*60}")
    print(f"Run-ID:          {run_id}")
    print(f"Zeitstempel:     {manifest['generated_at']}")
    print(f"Speicherort:     {pack_dir}/")
    print(f"Manifest:        {manifest_path}")
    print(f"Artefakte:       {artifacts_dir}/")

    # Inhalt auflisten
    print(f"\nInhalt des Evidence Packs:")
    print(f"  evidence_manifest.json (Hauptdokument)")
    artifacts = os.listdir(artifacts_dir)
    for a in sorted(artifacts):
        size_kb = os.path.getsize(os.path.join(artifacts_dir, a)) / 1024
        print(f"  artifacts/{a} ({size_kb:.1f} KB)")

    print(f"\nRegulatorisches Mapping:")
    for art, info in regulatory_mapping["addressed_requirements"].items():
        print(f"  {art}: {info['status']}")

    print(f"\nModellvergleich (AUC-ROC):")
    for name, metrics in evaluation_metrics.items():
        print(f"  {name}: {metrics['auc_roc']}")

    print(f"\nSHAP-Konsistenz: {len(shap_overlap)}/10 Features in allen Modellen")

    return manifest, pack_dir


def run_full_pipeline():
    """Führt die gesamte Pipeline aus und erzeugt das Evidence Pack."""

    run_id = generate_run_id()
    print(f"Pipeline gestartet: Run {run_id}")
    start_time = time.time()

    # 1. Training (inkl. Preprocessing)
    training_results, data = run_training()

    # 2. SHAP-Analyse
    shap_values, shap_importances, shap_rankings = run_shap_analysis()

    # Overlap berechnen
    sets = {name: set(feats) for name, feats in shap_rankings.items()}
    overlap = sets["logistic_regression"] & sets["random_forest"] & sets["xgboost"]

    # 3. Evidence Pack erzeugen
    manifest, pack_dir = collect_evidence(
        run_id=run_id,
        training_results=training_results,
        data=data,
        shap_rankings=shap_rankings,
        shap_overlap=overlap,
    )

    total_time = time.time() - start_time
    print(f"\nGesamtlaufzeit: {total_time:.1f} Sekunden")

    return manifest, pack_dir


if __name__ == "__main__":
    manifest, pack_dir = run_full_pipeline()