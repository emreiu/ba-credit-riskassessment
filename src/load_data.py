"""
Daten laden, Spalten umbenennen, als CSV speichern, SHA-256 Hash erzeugen.
Quelle: UCI Machine Learning Repository - Statlog (German Credit Data)
Hofmann, H. (1994). Statlog (German Credit Data) [Dataset].
https://doi.org/10.24432/C5NC77
"""

import os
import hashlib
import pandas as pd
from ucimlrepo import fetch_ucirepo

# --- Daten von UCI laden ---
print("Lade German Credit Dataset von UCI...")
dataset = fetch_ucirepo(id=144)

X = dataset.data.features
y = dataset.data.targets

df = pd.concat([X, y], axis=1)

# --- Spalten umbenennen ---
column_names = {
    "Attribute1":  "checking_account",
    "Attribute2":  "duration_months",
    "Attribute3":  "credit_history",
    "Attribute4":  "purpose",
    "Attribute5":  "credit_amount",
    "Attribute6":  "savings_account",
    "Attribute7":  "employment_years",
    "Attribute8":  "installment_rate",
    "Attribute9":  "personal_status",
    "Attribute10": "other_debtors",
    "Attribute11": "residence_years",
    "Attribute12": "property",
    "Attribute13": "age",
    "Attribute14": "other_installments",
    "Attribute15": "housing",
    "Attribute16": "num_existing_credits",
    "Attribute17": "job",
    "Attribute18": "num_dependents",
    "Attribute19": "telephone",
    "Attribute20": "foreign_worker",
}

target_col = y.columns[0]
df = df.rename(columns=column_names)
df = df.rename(columns={target_col: "credit_risk"})

# --- Zielvariable umkodieren: 1=good -> 0, 2=bad -> 1 ---
df["credit_risk"] = df["credit_risk"].map({1: 0, 2: 1})

# --- Kategoriale Codes in lesbare Labels übersetzen ---
label_mappings = {
    "checking_account": {
        "A11": "< 0 DM",
        "A12": "0-200 DM",
        "A13": ">= 200 DM",
        "A14": "no checking account",
    },
    "credit_history": {
        "A30": "no credits / all paid",
        "A31": "all paid at this bank",
        "A32": "existing credits paid",
        "A33": "delay in past",
        "A34": "critical account",
    },
    "purpose": {
        "A40": "car (new)",
        "A41": "car (used)",
        "A42": "furniture/equipment",
        "A43": "radio/television",
        "A44": "domestic appliances",
        "A45": "repairs",
        "A46": "education",
        "A47": "vacation",
        "A48": "retraining",
        "A49": "business",
        "A410": "others",
    },
    "savings_account": {
        "A61": "< 100 DM",
        "A62": "100-500 DM",
        "A63": "500-1000 DM",
        "A64": ">= 1000 DM",
        "A65": "unknown / none",
    },
    "employment_years": {
        "A71": "unemployed",
        "A72": "< 1 year",
        "A73": "1-4 years",
        "A74": "4-7 years",
        "A75": ">= 7 years",
    },
    "personal_status": {
        "A91": "male: divorced/separated",
        "A92": "female: divorced/married",
        "A93": "male: single",
        "A94": "male: married/widowed",
        "A95": "female: single",
    },
    "other_debtors": {
        "A101": "none",
        "A102": "co-applicant",
        "A103": "guarantor",
    },
    "property": {
        "A121": "real estate",
        "A122": "savings/life insurance",
        "A123": "car or other",
        "A124": "unknown / none",
    },
    "other_installments": {
        "A141": "bank",
        "A142": "stores",
        "A143": "none",
    },
    "housing": {
        "A151": "rent",
        "A152": "own",
        "A153": "for free",
    },
    "job": {
        "A171": "unemployed/unskilled (non-resident)",
        "A172": "unskilled (resident)",
        "A173": "skilled employee",
        "A174": "management/self-employed",
    },
    "telephone": {
        "A191": "none",
        "A192": "yes (registered)",
    },
    "foreign_worker": {
        "A201": "yes",
        "A202": "no",
    },
}

for col, mapping in label_mappings.items():
    if col in df.columns:
        df[col] = df[col].map(mapping).fillna(df[col])

# --- Als CSV speichern ---
output_path = os.path.join("data", "german_credit.csv")
df.to_csv(output_path, index=False)
print(f"Datensatz gespeichert unter: {output_path}")

# --- SHA-256 Hash erzeugen ---
with open(output_path, "rb") as f:
    data_hash = hashlib.sha256(f.read()).hexdigest()

hash_path = os.path.join("data", "german_credit.sha256")
with open(hash_path, "w") as f:
    f.write(data_hash)

print(f"SHA-256 Hash: {data_hash}")
print(f"Hash gespeichert unter: {hash_path}")

# --- Übersicht ausgeben ---
print("\n" + "=" * 60)
print("DATENSATZ-ÜBERSICHT")
print("=" * 60)
print(f"Anzahl Instanzen: {len(df)}")
print(f"Anzahl Features:  {len(df.columns) - 1}")

print(f"\nKlassenverteilung:")
print(df["credit_risk"].value_counts().rename({0: "good (0)", 1: "bad (1)"}))

print(f"\nDatentypen:")
print(f"  Numerisch:   {len(df.select_dtypes(include=['int64', 'float64']).columns)}")
print(f"  Kategorial:  {len(df.select_dtypes(include=['object', 'str']).columns)}")

print(f"\nFehlende Werte: {df.isnull().sum().sum()}")

print(f"\nNumerische Features:")
for col in df.select_dtypes(include=["int64", "float64"]).columns:
    if col != "credit_risk":
        print(f"  {col}: min={df[col].min()}, max={df[col].max()}, mean={df[col].mean():.1f}")

print(f"\nKategoriale Features:")
for col in df.select_dtypes(include=["object", "str"]).columns:
    print(f"  {col}: {df[col].nunique()} Kategorien → {list(df[col].unique()[:4])}...")

print(f"\nDie ersten 5 Zeilen:")
print(df.head().to_string())