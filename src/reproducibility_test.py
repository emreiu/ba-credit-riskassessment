"""
Schritt 8: Reproduzierbarkeitsnachweis.
Führt die Pipeline zweimal aus und vergleicht die Ergebnisse.
Adressiert: Sculley et al. (2015) Reproducibility Debt,
            Breck et al. (2017) Infra 1 - Training is reproducible.
"""

import os
import json
import hashlib
import numpy as np
import joblib
from datetime import datetime

from preprocessing import run_preprocessing
from train import PARAM_GRIDS, RANDOM_SEED, CV_FOLDS, SCORING
from train import create_models, train_single_model, evaluate_model


def compute_file_hash(filepath: str) -> str:
    with open(filepath, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def run_single_pipeline(run_label: str):
    """Führt die Pipeline einmal durch und gibt alle Ergebnisse zurück."""
    print(f"\n{'='*60}")
    print(f"  {run_label}")
    print(f"{'='*60}")

    # Preprocessing
    data = run_preprocessing()

    # Training + Evaluation
    models_dict = create_models()
    results = {}

    for name in models_dict:
        train_result = train_single_model(
            name=name,
            model=models_dict[name],
            param_grid=PARAM_GRIDS[name],
            X_train=data["X_train"],
            y_train=data["y_train"],
            cv_folds=CV_FOLDS,
            scoring=SCORING,
        )
        test_metrics = evaluate_model(name, train_result["model"], data["X_test"], data["y_test"])

        # Modell temporär speichern für Hash-Vergleich
        model_path = os.path.join("models", f"{name}_rerun.joblib")
        joblib.dump(train_result["model"], model_path)

        results[name] = {
            "best_params": train_result["best_params"],
            "best_cv_auc": train_result["best_cv_auc"],
            "test_metrics": test_metrics,
            "model_hash": compute_file_hash(model_path),
        }

        # Temporäre Datei aufräumen
        os.remove(model_path)

    return results, data


def compare_runs(run1_results: dict, run2_results: dict):
    """Vergleicht die Ergebnisse zweier Runs."""

    print(f"\n{'='*60}")
    print("REPRODUZIERBARKEITS-VERGLEICH")
    print(f"{'='*60}")

    all_match = True
    comparison = {}

    for name in run1_results:
        print(f"\n--- {name} ---")
        model_comparison = {"metrics_match": True, "params_match": True, "hash_match": True, "details": {}}

        # Hyperparameter vergleichen
        params_match = run1_results[name]["best_params"] == run2_results[name]["best_params"]
        model_comparison["params_match"] = params_match
        status = "✓" if params_match else "✗"
        print(f"  Hyperparameter:  {status}")
        if not params_match:
            all_match = False
            print(f"    Run 1: {run1_results[name]['best_params']}")
            print(f"    Run 2: {run2_results[name]['best_params']}")

        # CV-AUC vergleichen
        cv_auc_match = run1_results[name]["best_cv_auc"] == run2_results[name]["best_cv_auc"]
        status = "✓" if cv_auc_match else "✗"
        print(f"  CV-AUC:          {status} (Run1={run1_results[name]['best_cv_auc']:.4f}, Run2={run2_results[name]['best_cv_auc']:.4f})")
        if not cv_auc_match:
            all_match = False

        # Test-Metriken vergleichen
        metrics_match = True
        for metric in run1_results[name]["test_metrics"]:
            val1 = run1_results[name]["test_metrics"][metric]
            val2 = run2_results[name]["test_metrics"][metric]
            match = val1 == val2
            if not match:
                metrics_match = False
                all_match = False
            status = "✓" if match else "✗"
            print(f"  {metric:15s}: {status} (Run1={val1:.4f}, Run2={val2:.4f})")

        model_comparison["metrics_match"] = metrics_match

        # Modell-Hash vergleichen
        hash_match = run1_results[name]["model_hash"] == run2_results[name]["model_hash"]
        model_comparison["hash_match"] = hash_match
        status = "✓" if hash_match else "✗"
        print(f"  Modell-Hash:     {status}")
        if not hash_match:
            all_match = False
            print(f"    Run 1: {run1_results[name]['model_hash'][:16]}...")
            print(f"    Run 2: {run2_results[name]['model_hash'][:16]}...")

        comparison[name] = model_comparison

    # Gesamtergebnis
    print(f"\n{'='*60}")
    if all_match:
        print("ERGEBNIS: ✓ ALLE RUNS IDENTISCH – REPRODUZIERBARKEIT BESTÄTIGT")
    else:
        print("ERGEBNIS: ✗ ABWEICHUNGEN FESTGESTELLT")
    print(f"{'='*60}")

    # Ergebnis als JSON speichern (Evidence-Baustein)
    report = {
        "test_timestamp": datetime.now().isoformat(),
        "runs_compared": 2,
        "all_identical": all_match,
        "comparison_details": comparison,
    }

    report_path = os.path.join("evidence", "reproducibility_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nReport gespeichert: {report_path}")

    return all_match, report


def run_reproducibility_test():
    """Führt den vollständigen Reproduzierbarkeitstest durch."""

    print("REPRODUZIERBARKEITSTEST")
    print("Führe Pipeline zweimal mit identischer Konfiguration aus...")

    # Run 1
    run1_results, _ = run_single_pipeline("RUN 1 (Original)")

    # Run 2
    run2_results, _ = run_single_pipeline("RUN 2 (Re-Run)")

    # Vergleich
    all_match, report = compare_runs(run1_results, run2_results)

    return all_match


if __name__ == "__main__":
    run_reproducibility_test()