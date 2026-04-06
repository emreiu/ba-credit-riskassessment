"""
Schritt 4: Modelltraining.
Drei Modelle (LR, RF, XGBoost) mit Hyperparameter-Tuning via GridSearchCV.
Optimierung auf AUC-ROC, stratified 5-fold Cross-Validation.
"""

import os
import json
import time
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score,
    recall_score, f1_score, classification_report
)

# Preprocessing importieren
from preprocessing import run_preprocessing


# --- Konfiguration ---
RANDOM_SEED = 42
CV_FOLDS = 5
SCORING = "roc_auc"

# Hyperparameter-Suchräume
PARAM_GRIDS = {
    "logistic_regression": {
        "C": [0.01, 0.1, 1.0, 10.0],
        "max_iter": [1000],
        "solver": ["lbfgs"],
        "random_state": [RANDOM_SEED],
    },
    "random_forest": {
        "n_estimators": [100, 200],
        "max_depth": [5, 10, None],
        "min_samples_split": [2, 5],
        "random_state": [RANDOM_SEED],
    },
    "xgboost": {
        "n_estimators": [100, 200],
        "max_depth": [3, 5],
        "learning_rate": [0.05, 0.1],
        "subsample": [0.8, 1.0],
        "random_state": [RANDOM_SEED],
        "eval_metric": ["logloss"],
    },
}


def create_models():
    """Erstellt die drei Basis-Modelle."""
    return {
        "logistic_regression": LogisticRegression(),
        "random_forest": RandomForestClassifier(),
        "xgboost": XGBClassifier(),
    }


def train_single_model(name, model, param_grid, X_train, y_train, cv_folds, scoring):
    """
    Trainiert ein einzelnes Modell mit GridSearchCV.
    Gibt das beste Modell und die Ergebnisse zurück.
    """
    print(f"\n{'='*60}")
    print(f"Training: {name}")
    print(f"{'='*60}")
    print(f"Hyperparameter-Suchraum: {param_grid}")

    # Stratified K-Fold: gleiche Klassenverteilung in jedem Fold
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_SEED)

    # GridSearch: alle Kombinationen durchprobieren
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,       # alle CPU-Kerne nutzen
        refit=True,      # bestes Modell am Ende auf gesamten Trainingsdaten fitten
        return_train_score=True,
    )

    start_time = time.time()
    grid_search.fit(X_train, y_train)
    training_time = time.time() - start_time

    print(f"\nBeste Hyperparameter: {grid_search.best_params_}")
    print(f"Beste CV-AUC:         {grid_search.best_score_:.4f}")
    print(f"Trainingszeit:        {training_time:.1f} Sekunden")
    print(f"Getestete Kombinationen: {len(grid_search.cv_results_['mean_test_score'])}")

    return {
        "model": grid_search.best_estimator_,
        "best_params": grid_search.best_params_,
        "best_cv_auc": float(grid_search.best_score_),
        "training_time_seconds": round(training_time, 2),
        "n_combinations_tested": len(grid_search.cv_results_["mean_test_score"]),
    }


def evaluate_model(name, model, X_test, y_test):
    """Evaluiert ein trainiertes Modell auf den Testdaten."""

    # Vorhersagen
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # Metriken berechnen
    metrics = {
        "auc_roc": round(float(roc_auc_score(y_test, y_pred_proba)), 4),
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "precision": round(float(precision_score(y_test, y_pred)), 4),
        "recall": round(float(recall_score(y_test, y_pred)), 4),
        "f1_score": round(float(f1_score(y_test, y_pred)), 4),
    }

    print(f"\n--- Evaluation: {name} ---")
    for metric_name, value in metrics.items():
        print(f"  {metric_name:12s}: {value:.4f}")

    return metrics


def save_model(name, model, output_dir="models"):
    """Speichert ein trainiertes Modell als Joblib-Datei."""
    path = os.path.join(output_dir, f"{name}.joblib")
    joblib.dump(model, path)
    print(f"  Modell gespeichert: {path}")
    return path


def run_training():
    """Führt das gesamte Training und die Evaluation durch."""

    # 1. Preprocessing
    print("=" * 60)
    print("SCHRITT 1: PREPROCESSING")
    print("=" * 60)
    data = run_preprocessing()

    X_train = data["X_train"]
    X_test = data["X_test"]
    y_train = data["y_train"]
    y_test = data["y_test"]
    feature_names = data["feature_names"]

    # 2. Modelle trainieren
    print("\n" + "=" * 60)
    print("SCHRITT 2: MODELLTRAINING")
    print("=" * 60)

    models = create_models()
    results = {}

    for name in models:
        train_result = train_single_model(
            name=name,
            model=models[name],
            param_grid=PARAM_GRIDS[name],
            X_train=X_train,
            y_train=y_train,
            cv_folds=CV_FOLDS,
            scoring=SCORING,
        )

        # 3. Auf Testdaten evaluieren
        test_metrics = evaluate_model(name, train_result["model"], X_test, y_test)

        # 4. Modell speichern
        model_path = save_model(name, train_result["model"])

        # Ergebnisse zusammenführen
        results[name] = {
            "best_params": train_result["best_params"],
            "best_cv_auc": train_result["best_cv_auc"],
            "training_time_seconds": train_result["training_time_seconds"],
            "n_combinations_tested": train_result["n_combinations_tested"],
            "test_metrics": test_metrics,
            "model_path": model_path,
        }

    # 5. Vergleichstabelle ausgeben
    print("\n" + "=" * 60)
    print("MODELLVERGLEICH (Testdaten)")
    print("=" * 60)
    print(f"{'Modell':<25s} {'AUC':>7s} {'Acc':>7s} {'Prec':>7s} {'Rec':>7s} {'F1':>7s}")
    print("-" * 60)
    for name, res in results.items():
        m = res["test_metrics"]
        print(f"{name:<25s} {m['auc_roc']:>7.4f} {m['accuracy']:>7.4f} {m['precision']:>7.4f} {m['recall']:>7.4f} {m['f1_score']:>7.4f}")

    # 6. Ergebnisse als JSON speichern (Evidence-Baustein)
    results_path = os.path.join("evidence", "training_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nTraining-Ergebnisse gespeichert: {results_path}")

    return results, data


if __name__ == "__main__":
    results, data = run_training()