"""
Schritt 6: SHAP-Analyse.
Globale und lokale Interpretierbarkeitsanalyse für alle drei Modelle.
Methodische Grundlage: Lundberg & Lee (2017).
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import joblib

from preprocessing import run_preprocessing

# SHAP-Warnungen unterdrücken (bekannte Kompatibilitätswarnungen)
warnings.filterwarnings("ignore", category=FutureWarning)


# --- Konfiguration ---
TOP_N_FEATURES = 15  # Anzahl Features in Summary Plots
N_LOCAL_EXAMPLES = 3  # Anzahl lokaler Erklärungen


def compute_shap_values(name: str, model, X_train, X_test, feature_names: list):
    """
    Berechnet SHAP-Werte für ein Modell.
    - Tree SHAP für Random Forest und XGBoost (schnell, exakt für Baummodelle)
    - Kernel SHAP für Logistic Regression (modellunabhängig)
    """
    print(f"\n{'='*60}")
    print(f"SHAP-Analyse: {name}")
    print(f"{'='*60}")

    if name in ["random_forest", "xgboost"]:
        print("Verwende TreeExplainer (optimiert für Baummodelle)...")
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
    else:
        print("Verwende KernelExplainer (modellunabhängig)...")
        print("  (Das dauert ca. 1-2 Minuten...)")
        background = shap.sample(X_train, min(100, len(X_train)))
        explainer = shap.KernelExplainer(model.predict_proba, background)
        shap_values = explainer.shap_values(X_test, silent=True)

    # SHAP-Werte auf Klasse 1 (bad) reduzieren.
    # Je nach SHAP-Version kommen die Werte als:
    #   - Liste [class_0_array, class_1_array]
    #   - 3D-Array mit Shape (n_samples, n_features, n_classes)
    #   - 2D-Array mit Shape (n_samples, n_features) -> bereits korrekt
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        shap_values = shap_values[:, :, 1]

    print(f"SHAP-Werte berechnet: Shape = {shap_values.shape}")

    return explainer, shap_values

def plot_global_importance(name: str, shap_values, X_test, feature_names: list,
                           top_n: int = TOP_N_FEATURES, output_dir: str = "plots"):
    """
    Globale Feature Importance: Mean |SHAP| Bar Plot.
    Zeigt, welche Features im Durchschnitt den größten Einfluss haben.
    """
    model_labels = {
        "logistic_regression": "Logistic Regression",
        "random_forest": "Random Forest",
        "xgboost": "XGBoost",
    }

    # Mean absolute SHAP values
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    feature_importance = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": mean_abs_shap,
    }).sort_values("mean_abs_shap", ascending=False)

    # Top N Features plotten
    top_features = feature_importance.head(top_n)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(
        range(len(top_features)),
        top_features["mean_abs_shap"].values,
        color="#3498db",
    )
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features["feature"].values)
    ax.invert_yaxis()
    ax.set_xlabel("Mean |SHAP Value|", fontsize=11)
    ax.set_title(f"Globale Feature Importance: {model_labels[name]}", fontsize=13)
    plt.tight_layout()

    path = os.path.join(output_dir, f"shap_global_{name}.png")
    plt.savefig(path, dpi=150)
    plt.show()
    print(f"Globale Importance gespeichert: {path}")

    return feature_importance


def plot_summary(name: str, shap_values, X_test, feature_names: list,
                 top_n: int = TOP_N_FEATURES, output_dir: str = "plots"):
    """
    SHAP Summary Plot: Zeigt für jedes Feature die Verteilung der SHAP-Werte.
    Punkte = einzelne Instanzen, Farbe = Feature-Wert (rot=hoch, blau=niedrig).
    """
    model_labels = {
        "logistic_regression": "Logistic Regression",
        "random_forest": "Random Forest",
        "xgboost": "XGBoost",
    }

    # Top N Features nach mean |SHAP| auswählen
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    top_indices = np.argsort(mean_abs_shap)[-top_n:][::-1]

    # DataFrame für SHAP erstellen
    X_test_df = pd.DataFrame(X_test, columns=feature_names)

    plt.figure(figsize=(10, 7))
    plt.title(f"SHAP Summary Plot: {model_labels[name]}")
    shap.summary_plot(
        shap_values[:, top_indices],
        X_test_df.iloc[:, top_indices],
        show=False,
        max_display=top_n,
    )
    plt.tight_layout()

    path = os.path.join(output_dir, f"shap_summary_{name}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()
    print(f"Summary Plot gespeichert: {path}")


def plot_local_explanations(name: str, explainer, shap_values, X_test,
                            feature_names: list, y_test,
                            n_examples: int = N_LOCAL_EXAMPLES,
                            output_dir: str = "plots"):
    """
    Lokale Erklärungen: Waterfall Plots für einzelne Kreditentscheidungen.
    Zeigt, welche Features bei einer konkreten Vorhersage den Ausschlag geben.
    """
    model_labels = {
        "logistic_regression": "Logistic Regression",
        "random_forest": "Random Forest",
        "xgboost": "XGBoost",
    }

    # Base value für Klasse 1 (bad) extrahieren
    base_value = explainer.expected_value
    if isinstance(base_value, (list, np.ndarray)):
        base_value = float(np.array(base_value).flatten()[1])
    else:
        base_value = float(base_value)

    # Drei verschiedene Fälle auswählen
    example_indices = []
    labels = []

    # True Positive: Modell sagt korrekt "bad"
    for i in range(len(y_test)):
        if y_test[i] == 1 and shap_values[i].sum() > 0:
            example_indices.append(i)
            labels.append("True Positive (korrekt als Bad erkannt)")
            break

    # False Negative: Modell sagt fälschlich "good", obwohl "bad"
    for i in range(len(y_test)):
        if y_test[i] == 1 and shap_values[i].sum() <= 0:
            example_indices.append(i)
            labels.append("False Negative (Bad als Good klassifiziert)")
            break

    # True Negative: Modell sagt korrekt "good"
    for i in range(len(y_test)):
        if y_test[i] == 0 and shap_values[i].sum() <= 0:
            example_indices.append(i)
            labels.append("True Negative (korrekt als Good erkannt)")
            break

    for idx, label in zip(example_indices, labels):
        print(f"\n  Lokale Erklärung: Instanz {idx} – {label}")

        explanation = shap.Explanation(
            values=shap_values[idx],
            base_values=base_value,
            data=X_test[idx],
            feature_names=feature_names,
        )

        plt.figure(figsize=(10, 6))
        plt.title(f"{model_labels[name]}: {label}")
        shap.plots.waterfall(explanation, max_display=12, show=False)
        plt.tight_layout()

        safe_label = label.split("(")[0].strip().replace(" ", "_").lower()
        path = os.path.join(output_dir, f"shap_local_{name}_{safe_label}.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.show()
        plt.close()
        print(f"  Gespeichert: {path}")

def compare_feature_rankings(all_importances: dict, top_n: int = 10, output_dir: str = "plots"):
    """
    Vergleicht die Feature-Rankings über alle Modelle.
    Das ist der Kern der Interpretierbarkeitsanalyse.
    """
    model_labels = {
        "logistic_regression": "Logistic Regression",
        "random_forest": "Random Forest",
        "xgboost": "XGBoost",
    }

    print(f"\n{'='*60}")
    print("FEATURE-RANKING-VERGLEICH (Top 10)")
    print(f"{'='*60}")

    # Top N pro Modell
    rankings = {}
    for name, importance_df in all_importances.items():
        top = importance_df.head(top_n)["feature"].tolist()
        rankings[name] = top
        print(f"\n{model_labels[name]}:")
        for i, feat in enumerate(top):
            print(f"  {i+1:2d}. {feat}")

    # Überschneidung berechnen
    sets = {name: set(feats) for name, feats in rankings.items()}
    all_three = sets["logistic_regression"] & sets["random_forest"] & sets["xgboost"]
    print(f"\nIn allen drei Top-{top_n} enthalten ({len(all_three)} Features):")
    for feat in sorted(all_three):
        print(f"  - {feat}")

    # Vergleichsplot: Heatmap der Rankings
    all_features = set()
    for feats in rankings.values():
        all_features.update(feats)
    all_features = sorted(all_features)

    rank_matrix = []
    for feat in all_features:
        row = {}
        for name, importance_df in all_importances.items():
            feat_list = importance_df["feature"].tolist()
            if feat in feat_list:
                row[model_labels[name]] = feat_list.index(feat) + 1
            else:
                row[model_labels[name]] = len(feat_list)
        rank_matrix.append(row)

    df_ranks = pd.DataFrame(rank_matrix, index=all_features)

    fig, ax = plt.subplots(figsize=(8, max(6, len(all_features) * 0.4)))
    im = ax.imshow(df_ranks.values, cmap="YlOrRd_r", aspect="auto")

    ax.set_xticks(range(len(df_ranks.columns)))
    ax.set_xticklabels(df_ranks.columns, fontsize=11)
    ax.set_yticks(range(len(df_ranks.index)))
    ax.set_yticklabels(df_ranks.index, fontsize=9)

    # Werte in die Zellen schreiben
    for i in range(len(df_ranks.index)):
        for j in range(len(df_ranks.columns)):
            ax.text(j, i, str(df_ranks.values[i, j]),
                    ha="center", va="center", fontsize=9, fontweight="bold")

    ax.set_title("Feature-Ranking-Vergleich (niedrigerer Rang = wichtiger)", fontsize=12)
    plt.colorbar(im, ax=ax, label="Rang")
    plt.tight_layout()

    path = os.path.join(output_dir, "shap_ranking_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"\nRanking-Vergleich gespeichert: {path}")

    # Rankings als JSON speichern (Evidence-Baustein)
    rankings_serializable = {name: feats for name, feats in rankings.items()}
    rankings_serializable["overlap_all_three"] = sorted(list(all_three))

    rankings_path = os.path.join("evidence", "shap_feature_rankings.json")
    with open(rankings_path, "w") as f:
        json.dump(rankings_serializable, f, indent=2)
    print(f"Rankings gespeichert: {rankings_path}")

    return rankings, all_three


def run_shap_analysis():
    """Führt die gesamte SHAP-Analyse durch."""

    # 1. Daten und Modelle laden
    data = run_preprocessing()
    X_train = data["X_train"]
    X_test = data["X_test"]
    y_test = data["y_test"]
    feature_names = data["feature_names"]

    model_names = ["logistic_regression", "random_forest", "xgboost"]
    models = {}
    for name in model_names:
        models[name] = joblib.load(os.path.join("models", f"{name}.joblib"))
        print(f"Modell geladen: {name}")

    # 2. SHAP-Werte berechnen für jedes Modell
    all_shap_values = {}
    all_explainers = {}
    all_importances = {}

    for name in model_names:
        explainer, shap_values = compute_shap_values(
            name, models[name], X_train, X_test, feature_names
        )
        all_shap_values[name] = shap_values
        all_explainers[name] = explainer

        # 3. Globale Feature Importance
        importance_df = plot_global_importance(name, shap_values, X_test, feature_names)
        all_importances[name] = importance_df

        # 4. Summary Plot
        plot_summary(name, shap_values, X_test, feature_names)

        # 5. Lokale Erklärungen
        plot_local_explanations(name, explainer, shap_values, X_test, feature_names, y_test)

    # 6. Feature-Ranking-Vergleich über alle Modelle
    rankings, overlap = compare_feature_rankings(all_importances)

    print("\n" + "=" * 60)
    print("SHAP-ANALYSE ABGESCHLOSSEN")
    print("=" * 60)
    print(f"Plots gespeichert in: plots/")
    print(f"Rankings gespeichert in: evidence/shap_feature_rankings.json")

    return all_shap_values, all_importances, rankings


if __name__ == "__main__":
    run_shap_analysis()