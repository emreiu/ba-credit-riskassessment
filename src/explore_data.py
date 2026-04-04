"""
Explorative Datenanalyse: Verteilungen und Korrelationen visualisieren.
"""

import pandas as pd
import matplotlib.pyplot as plt
import os

# Daten laden
df = pd.read_csv(os.path.join("data", "german_credit.csv"))

# --- Klassenverteilung ---
fig, ax = plt.subplots(figsize=(6, 4))
df["credit_risk"].value_counts().plot(kind="bar", ax=ax, color=["#2ecc71", "#e74c3c"])
ax.set_xticklabels(["Good (0)", "Bad (1)"], rotation=0)
ax.set_title("Klassenverteilung")
ax.set_ylabel("Anzahl")
plt.tight_layout()
plt.savefig(os.path.join("plots", "class_distribution.png"), dpi=150)
plt.show()
print("Plot gespeichert: plots/class_distribution.png")

# --- Numerische Features: Verteilung nach Klasse ---
numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
numeric_cols = [c for c in numeric_cols if c != "credit_risk"]

if len(numeric_cols) > 0:
    n_cols = min(len(numeric_cols), 6)
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()
    for i, col in enumerate(numeric_cols[:n_cols]):
        df.boxplot(column=col, by="credit_risk", ax=axes[i])
        axes[i].set_title(col)
        axes[i].set_xlabel("Credit Risk")
    plt.suptitle("Numerische Features nach Kreditrisiko", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join("plots", "numeric_features_by_class.png"), dpi=150)
    plt.show()
    print("Plot gespeichert: plots/numeric_features_by_class.png")

# --- Zusammenfassung ---
print("\n" + "=" * 60)
print("ZUSAMMENFASSUNG")
print("=" * 60)
print(f"Shape: {df.shape}")
print(f"Numerische Spalten: {len(numeric_cols)}")
print(f"Kategoriale Spalten: {len(df.select_dtypes(include=['object']).columns)}")
print(f"Fehlende Werte: {df.isnull().sum().sum()}")
print(f"Klassenverteilung: {dict(df['credit_risk'].value_counts())}")