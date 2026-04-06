"""
Schritt 5: Evaluation und Visualisierung.
ROC-Kurven, Vergleichstabelle und Confusion Matrices.
"""

print("evaluate.py wird gestartet...")

import os
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import (
    roc_curve, roc_auc_score, confusion_matrix,
    ConfusionMatrixDisplay
)

from preprocessing import run_preprocessing


def load_models(model_names: list, model_dir: str = "models"):
    """Lädt alle trainierten Modelle."""
    models = {}
    for name in model_names:
        path = os.path.join(model_dir, f"{name}.joblib")
        models[name] = joblib.load(path)
        print(f"Modell geladen: {path}")
    return models


def plot_roc_curves(models: dict, X_test, y_test, output_dir: str = "plots"):
    """Erzeugt einen kombinierten ROC-Kurven-Plot für alle Modelle."""

    model_labels = {
        "logistic_regression": "Logistic Regression",
        "random_forest": "Random Forest",
        "xgboost": "XGBoost",
    }
    colors = {
        "logistic_regression": "#2ecc71",
        "random_forest": "#3498db",
        "xgboost": "#e74c3c",
    }

    fig, ax = plt.subplots(figsize=(8, 6))

    for name, model in models.items():
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        auc = roc_auc_score(y_test, y_pred_proba)

        label = f"{model_labels[name]} (AUC = {auc:.4f})"
        ax.plot(fpr, tpr, color=colors[name], linewidth=2, label=label)

    # Zufallslinie
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1, label="Random (AUC = 0.5)")

    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC-Kurven: Modellvergleich", fontsize=14)
    ax.legend(loc="lower right", fontsize=10)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    path = os.path.join(output_dir, "roc_curves_comparison.png")
    plt.savefig(path, dpi=150)
    plt.show()
    print(f"ROC-Kurven gespeichert: {path}")


def plot_confusion_matrices(models: dict, X_test, y_test, output_dir: str = "plots"):
    """Erzeugt Confusion Matrices für alle Modelle nebeneinander."""

    model_labels = {
        "logistic_regression": "Logistic Regression",
        "random_forest": "Random Forest",
        "xgboost": "XGBoost",
    }

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    for i, (name, model) in enumerate(models.items()):
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(cm, display_labels=["Good", "Bad"])
        disp.plot(ax=axes[i], cmap="Blues", colorbar=False)
        axes[i].set_title(model_labels[name], fontsize=12)

    plt.suptitle("Confusion Matrices: Modellvergleich", fontsize=14)
    plt.tight_layout()

    path = os.path.join(output_dir, "confusion_matrices.png")
    plt.savefig(path, dpi=150)
    plt.show()
    print(f"Confusion Matrices gespeichert: {path}")


def create_comparison_table(models: dict, X_test, y_test):
    """Erstellt eine Vergleichstabelle als DataFrame."""

    model_labels = {
        "logistic_regression": "Logistic Regression",
        "random_forest": "Random Forest",
        "xgboost": "XGBoost",
    }

    rows = []
    for name, model in models.items():
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()

        rows.append({
            "Modell": model_labels[name],
            "AUC-ROC": round(roc_auc_score(y_test, y_pred_proba), 4),
            "Accuracy": round((tp + tn) / (tp + tn + fp + fn), 4),
            "Precision": round(tp / (tp + fp) if (tp + fp) > 0 else 0, 4),
            "Recall": round(tp / (tp + fn) if (tp + fn) > 0 else 0, 4),
            "F1-Score": round(2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0, 4),
            "True Pos.": tp,
            "False Pos.": fp,
            "True Neg.": tn,
            "False Neg.": fn,
        })

    df_results = pd.DataFrame(rows)
    print("\n" + "=" * 80)
    print("VERGLEICHSTABELLE")
    print("=" * 80)
    print(df_results.to_string(index=False))

    # Als CSV speichern
    path = os.path.join("evidence", "model_comparison.csv")
    df_results.to_csv(path, index=False)
    print(f"\nTabelle gespeichert: {path}")

    return df_results


def run_evaluation():
    """Führt die gesamte Evaluation durch."""

    # 1. Daten vorbereiten
    data = run_preprocessing()
    X_test = data["X_test"]
    y_test = data["y_test"]

    # 2. Modelle laden
    model_names = ["logistic_regression", "random_forest", "xgboost"]
    models = load_models(model_names)

    # 3. ROC-Kurven
    print("\n--- ROC-Kurven ---")
    plot_roc_curves(models, X_test, y_test)

    # 4. Confusion Matrices
    print("\n--- Confusion Matrices ---")
    plot_confusion_matrices(models, X_test, y_test)

    # 5. Vergleichstabelle
    df_results = create_comparison_table(models, X_test, y_test)

    return df_results


if __name__ == "__main__":
    run_evaluation()