# Automatisierte Compliance-Dokumentation und Benchmarking von Interpretierbarkeit im Credit Risk Assessment

Prototyp im Rahmen der Bachelorarbeit an der FH Technikum Wien, Studiengang Wirtschaftsinformatik.

## Übersicht

Eine Python-basierte ML-Pipeline, die automatisiert auditierbares Evidence für Credit-Scoring-Modelle erzeugt und deren Interpretierbarkeit mittels SHAP systematisch analysiert. Der Prototyp vergleicht drei Modelle (Logistic Regression, Random Forest, XGBoost) auf dem German Credit Dataset und mappt die erzeugten Artefakte auf die Anforderungen des EU AI Acts (Art. 9–15).

## Installation

```bash
git clone https://github.com/DEIN-USERNAME/bachelorarbeit-credit-risk.git
cd bachelorarbeit-credit-risk
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Mac/Linux
pip install -r requirements.txt
```

## Ausführung

```bash
# Gesamte Pipeline mit Evidence Pack
python src/generate_evidence_pack.py

# Reproduzierbarkeitstest
python src/reproducibility_test.py

# Streamlit-Demonstrator
streamlit run src/app.py
```

## Projektstruktur

```
src/           → Pipeline-Code und Streamlit-Demonstrator
data/          → Datensatz, Hash, Preprocessing-Log
models/        → Serialisierte Modelle
plots/         → Erzeugte SHAP- und Evaluationsplots
evidence/      → Evidence Packs und archivierte Einzelfallentscheidungen
```

## Technologie

Python 3.11 · scikit-learn · XGBoost · SHAP · Streamlit
