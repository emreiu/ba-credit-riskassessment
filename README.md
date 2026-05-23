# Automated Compliance Documentation and Interpretability Benchmarking in Credit Risk Assessment
 
Prototype developed as part of a Bachelor thesis at FH Technikum Wien, Business Informatics.
 
## Overview
 
A Python-based ML pipeline that automatically generates auditable evidence for credit scoring models and systematically analyzes their interpretability using SHAP. The prototype compares three models (Logistic Regression, Random Forest, XGBoost) on the German Credit Dataset and maps the generated artifacts to the requirements of the EU AI Act (Art. 9–15).
 
## Installation
 
```bash
git clone https://github.com/DEIN-USERNAME/bachelorarbeit-credit-risk.git
cd bachelorarbeit-credit-risk
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Mac/Linux
pip install -r requirements.txt
```
 
## Usage
 
```bash
# Full pipeline with evidence pack
python src/generate_evidence_pack.py
 
# Reproducibility test
python src/reproducibility_test.py
 
# Streamlit demonstrator
streamlit run src/app.py
```
 
## Project Structure
 
```
src/           → Pipeline code and Streamlit demonstrator
data/          → Dataset, hash, preprocessing log
models/        → Serialized models
plots/         → Generated SHAP and evaluation plots
evidence/      → Evidence packs and archived case decisions
```
 
## Technology
 
Python 3.11 · scikit-learn · XGBoost · SHAP · Streamlit
 
