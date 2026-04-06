"""
Preprocessing-Pipeline.
Kategoriale Features encodieren, numerische Features skalieren,
Train-Test-Split mit festem Seed und stratified Sampling.
"""

import os
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# --- Konfiguration ---
CONFIG = {
    "random_seed": 42,
    "test_size": 0.3,
    "input_path": os.path.join("data", "german_credit.csv"),
}


def load_data(path: str) -> pd.DataFrame:
    """Lädt den bereinigten Datensatz."""
    df: pd.DataFrame = pd.read_csv(path)
    print(f"Datensatz geladen: {df.shape[0]} Zeilen, {df.shape[1]} Spalten")
    return df


def identify_feature_types(df: pd.DataFrame, target_col: str = "credit_risk"):
    """Identifiziert numerische und kategoriale Features automatisch."""
    feature_cols = [c for c in df.columns if c != target_col]

    numeric_features = []
    categorical_features = []

    for col in feature_cols:
        if df[col].dtype in ["int64", "float64"]:
            numeric_features.append(col)
        else:
            categorical_features.append(col)

    print(f"\nFeature-Typen identifiziert:")
    print(f"  Numerisch ({len(numeric_features)}):  {numeric_features}")
    print(f"  Kategorial ({len(categorical_features)}): {categorical_features}")

    return numeric_features, categorical_features


def build_preprocessor(numeric_features: list, categorical_features: list):
    """
    Baut einen ColumnTransformer:
    - Numerische Features: StandardScaler (Mittelwert=0, Standardabweichung=1)
    - Kategoriale Features: OneHotEncoder (eine Spalte pro Kategorie)
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="error"), categorical_features),
        ],
        remainder="drop",
    )
    return preprocessor


def split_data(df: pd.DataFrame, target_col: str, test_size: float, random_seed: int):
    """Stratified Train-Test-Split."""
    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_seed,
        stratify=y,
    )

    print(f"\nTrain-Test-Split (seed={random_seed}, test_size={test_size}):")
    print(f"  Training:  {X_train.shape[0]} Instanzen (good={sum(y_train==0)}, bad={sum(y_train==1)})")
    print(f"  Test:      {X_test.shape[0]} Instanzen (good={sum(y_test==0)}, bad={sum(y_test==1)})")

    return X_train, X_test, y_train, y_test


def run_preprocessing():
    """Führt die gesamte Preprocessing-Pipeline aus und gibt alle Artefakte zurück."""

    # 1. Daten laden
    df = load_data(CONFIG["input_path"])

    # 2. Feature-Typen identifizieren
    numeric_features, categorical_features = identify_feature_types(df)

    # 3. Train-Test-Split (VOR dem Fitting des Preprocessors – Data Leakage vermeiden!)
    X_train, X_test, y_train, y_test = split_data(
        df,
        target_col="credit_risk",
        test_size=CONFIG["test_size"],
        random_seed=CONFIG["random_seed"],
    )

    # 4. Preprocessor bauen und NUR auf Trainingsdaten fitten
    preprocessor = build_preprocessor(numeric_features, categorical_features)
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    # 5. Feature-Namen nach Encoding zusammensetzen
    cat_encoder = preprocessor.named_transformers_["cat"]
    cat_feature_names = cat_encoder.get_feature_names_out(categorical_features).tolist()
    all_feature_names = numeric_features + cat_feature_names

    print(f"\nNach Preprocessing:")
    print(f"  Features vorher:  {len(numeric_features) + len(categorical_features)}")
    print(f"  Features nachher: {len(all_feature_names)} (durch One-Hot-Encoding)")
    print(f"  Shape Training:   {X_train_processed.shape}")
    print(f"  Shape Test:       {X_test_processed.shape}")

    # 6. Preprocessing-Konfiguration speichern (Evidence-Baustein)
    preprocessing_log = {
        "random_seed": CONFIG["random_seed"],
        "test_size": CONFIG["test_size"],
        "n_train": int(X_train_processed.shape[0]),
        "n_test": int(X_test_processed.shape[0]),
        "n_features_original": len(numeric_features) + len(categorical_features),
        "n_features_encoded": len(all_feature_names),
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "encoding_method": "OneHotEncoder(drop='first')",
        "scaling_method": "StandardScaler",
        "feature_names_after_encoding": all_feature_names,
    }

    log_path = os.path.join("data", "preprocessing_log.json")
    with open(log_path, "w") as f:
        json.dump(preprocessing_log, f, indent=2)
    print(f"\nPreprocessing-Log gespeichert: {log_path}")

    return {
        "X_train": X_train_processed,
        "X_test": X_test_processed,
        "y_train": y_train.values,
        "y_test": y_test.values,
        "feature_names": all_feature_names,
        "preprocessor": preprocessor,
        "config": preprocessing_log,
    }


# --- Ausführen zum Testen ---
if __name__ == "__main__":
    result = run_preprocessing()
    print("\n" + "=" * 60)
    print("PREPROCESSING ABGESCHLOSSEN")
    print("=" * 60)
    print(f"Training: {result['X_train'].shape}")
    print(f"Test:     {result['X_test'].shape}")
    print(f"Features: {len(result['feature_names'])}")
    print(f"\nErste 10 Feature-Namen:")
    for i, name in enumerate(result["feature_names"][:10]):
        print(f"  {i+1:2d}. {name}")
    print(f"  ...")
    print(f"  {len(result['feature_names'])}. {result['feature_names'][-1]}")