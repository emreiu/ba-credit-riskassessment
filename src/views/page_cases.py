"""
Seite: Einzelfall-Prüfung.
Kreditentscheidung berechnen und archivieren.
"""

import json

import streamlit as st
import pandas as pd
import numpy as np

from utils.data_loader import (
    load_models_and_data, load_raw_data, load_evidence_manifest,
    get_model_hash, get_data_hash, MODEL_LABELS,
)
from utils.shap_utils import (
    compute_shap_for_instance, generate_top_reasons, render_waterfall_plot,
)
from utils.case_store import save_case, build_case_record


# --- Konstanten ---
THRESHOLD = 0.50


def _render_applicant_input(df_raw):
    """Rendert die Antragsdaten-Eingabe."""

    input_method = st.radio(
        "Eingabemethode", ["Aus Datensatz auswählen", "Manuell eingeben"],
        horizontal=True, key="case_input_method",
    )

    if input_method == "Aus Datensatz auswählen":
        idx = st.number_input(
            "Instanz-Index (0-999)", min_value=0,
            max_value=len(df_raw) - 1, value=9, key="case_idx",
        )
        applicant = df_raw.iloc[idx].drop("credit_risk")
        actual_label = int(df_raw.iloc[idx]["credit_risk"])
        reference = f"Datensatz Instanz #{idx}"
        st.dataframe(applicant.to_frame().T, use_container_width=True, hide_index=True)
        label_text = "Bad (1) - ausgefallen" if actual_label == 1 else "Good (0) - zurueckgezahlt"
        st.info(f"Tatsaechliches Label: **{label_text}**")
        return applicant, idx, reference, actual_label

    else:
        ref = st.text_input("Referenz / Antragsnummer", value="Manueller Test", key="case_ref")
        col1, col2, col3 = st.columns(3)
        with col1:
            checking = st.selectbox("Girokonto", ["< 0 DM", "0-200 DM", ">= 200 DM", "no checking account"], key="m_chk")
            duration = st.number_input("Laufzeit (Monate)", 1, 72, 24, key="m_dur")
            credit_hist = st.selectbox("Kredithistorie", ["critical account", "existing credits paid", "delay in past", "all paid at this bank", "no credits / all paid"], key="m_hist")
            purpose = st.selectbox("Zweck", ["car (new)", "car (used)", "furniture/equipment", "radio/television", "domestic appliances", "repairs", "education", "retraining", "business", "others"], key="m_purp")
            amount = st.number_input("Kreditbetrag (DM)", 250, 20000, 3000, key="m_amt")
        with col2:
            savings = st.selectbox("Sparkonto", ["< 100 DM", "100-500 DM", "500-1000 DM", ">= 1000 DM", "unknown / none"], key="m_sav")
            employment = st.selectbox("Beschaeftigung", ["unemployed", "< 1 year", "1-4 years", "4-7 years", ">= 7 years"], key="m_emp")
            installment = st.slider("Ratenhoehe (% Einkommen)", 1, 4, 3, key="m_inst")
            personal = st.selectbox("Persoenlicher Status", ["male: single", "female: divorced/married", "male: divorced/separated", "male: married/widowed"], key="m_pers")
            age = st.number_input("Alter", 18, 80, 35, key="m_age")
        with col3:
            housing = st.selectbox("Wohnsituation", ["rent", "own", "for free"], key="m_hous")
            foreign = st.selectbox("Auslaendische*r AN", ["yes", "no"], key="m_for")
            other_debtors = st.selectbox("Weitere Schuldner", ["none", "co-applicant", "guarantor"], key="m_debt")
            property_val = st.selectbox("Eigentum", ["real estate", "savings/life insurance", "car or other", "unknown / none"], key="m_prop")
            other_inst = st.selectbox("Weitere Ratenkredite", ["none", "bank", "stores"], key="m_oinst")

        applicant = pd.Series({
            "checking_account": checking, "duration_months": duration,
            "credit_history": credit_hist, "purpose": purpose,
            "credit_amount": amount, "savings_account": savings,
            "employment_years": employment, "installment_rate": installment,
            "personal_status": personal, "other_debtors": other_debtors,
            "residence_years": 3, "property": property_val,
            "age": age, "other_installments": other_inst,
            "housing": housing, "num_existing_credits": 1,
            "job": "skilled employee", "num_dependents": 1,
            "telephone": "none", "foreign_worker": foreign,
        })
        return applicant, None, ref, None


def render():
    st.title("Einzelfall-Pruefung")
    st.markdown(
        "Simuliere eine Kreditentscheidung mit vollstaendiger Dokumentation. "
        "Archivierte Faelle koennen im Tab *Archiv* eingesehen und auditiert werden."
    )

    models, data = load_models_and_data()
    df_raw = load_raw_data()
    manifest, _ = load_evidence_manifest()

    # --- Konfiguration (kompakt) ---
    col_cfg1, col_cfg2 = st.columns([3, 1])
    with col_cfg1:
        production_model = st.selectbox(
            "Produktionsmodell",
            list(MODEL_LABELS.keys()),
            format_func=lambda x: MODEL_LABELS[x],
            key="prod_model",
        )
    with col_cfg2:
        st.metric("Schwellenwert", f"{THRESHOLD:.0%}")

    # --- Antragsdaten ---
    applicant, source_index, reference, actual_label = _render_applicant_input(df_raw)

    # --- Berechnung ---
    if st.button("Kreditentscheidung berechnen", type="primary", key="calc_btn"):

        preprocessor = data["preprocessor"]
        applicant_df = applicant.to_frame().T
        applicant_processed = preprocessor.transform(applicant_df)
        feature_names = data["feature_names"]

        # Alle Modelle vorhersagen
        all_predictions = {}
        for name in MODEL_LABELS:
            proba = models[name].predict_proba(applicant_processed)[0]
            all_predictions[name] = {
                "p_good": round(float(proba[0]), 4),
                "p_bad": round(float(proba[1]), 4),
            }

        st.markdown("---")
        st.subheader("Ergebnisse")

        # --- Alle drei Modelle gleich anzeigen ---
        cols = st.columns(3)
        for i, (name, label) in enumerate(MODEL_LABELS.items()):
            pred = all_predictions[name]
            model_decision = "ABGELEHNT" if pred["p_bad"] >= THRESHOLD else "GENEHMIGT"
            is_prod = name == production_model
            with cols[i]:
                header = f"**{label}**"
                if is_prod:
                    header += " (Prod.)"
                st.markdown(header)
                if model_decision == "ABGELEHNT":
                    st.error(f"ABGELEHNT ({pred['p_bad']:.1%})")
                else:
                    st.success(f"GENEHMIGT ({pred['p_good']:.1%})")
                st.caption(f"P(good)={pred['p_good']:.3f} | P(bad)={pred['p_bad']:.3f}")

        # --- SHAP fuer Produktionsmodell ---
        st.markdown("---")
        st.subheader("Erklaerung der Entscheidung")
        st.caption(f"SHAP-Analyse fuer {MODEL_LABELS[production_model]} (Produktionsmodell)")

        with st.spinner("SHAP wird berechnet..."):
            shap_vals, base_val = compute_shap_for_instance(
                models[production_model], production_model,
                applicant_processed, data["X_train"],
            )

        # Top-Gruende kompakt
        increasing, decreasing = generate_top_reasons(shap_vals, feature_names)
        col_r1, col_r2 = st.columns(2)
        with col_r1:
            st.markdown("**Risikosteigernd**")
            for r in increasing:
                st.markdown(f"- {r['feature']} (`{r['shap_value']:+.4f}`)")
            if not increasing:
                st.caption("Keine dominanten Faktoren.")
        with col_r2:
            st.markdown("**Risikosenkend**")
            for r in decreasing:
                st.markdown(f"- {r['feature']} (`{r['shap_value']:+.4f}`)")
            if not decreasing:
                st.caption("Keine dominanten Faktoren.")

        with st.expander("Detaillierter SHAP Waterfall-Plot"):
            render_waterfall_plot(shap_vals, base_val, applicant_processed[0], feature_names)

        with st.expander("SHAP-Erklaerung der anderen Modelle"):
            other_models = [n for n in MODEL_LABELS if n != production_model]
            for other_name in other_models:
                st.markdown(f"**{MODEL_LABELS[other_name]}:**")
                with st.spinner(f"Berechne SHAP fuer {MODEL_LABELS[other_name]}..."):
                    other_sv, other_bv = compute_shap_for_instance(
                        models[other_name], other_name,
                        applicant_processed, data["X_train"],
                    )
                render_waterfall_plot(other_sv, other_bv, applicant_processed[0], feature_names)

        # --- Archivieren ---
        st.markdown("---")
        shap_method = "TreeExplainer" if production_model in ["random_forest", "xgboost"] else "KernelExplainer"
        shap_values_dict = {fn: round(float(sv), 6) for fn, sv in zip(feature_names, shap_vals)}

        st.session_state["pending_case"] = build_case_record(
            applicant_data=applicant.to_dict(),
            source_index=source_index,
            reference=reference,
            actual_label=actual_label,
            production_model=production_model,
            threshold=THRESHOLD,
            decision="ABGELEHNT" if all_predictions[production_model]["p_bad"] >= THRESHOLD else "GENEHMIGT",
            all_predictions=all_predictions,
            shap_values_dict=shap_values_dict,
            shap_base_value=base_val,
            shap_method=shap_method,
            top_reasons_increasing=increasing,
            top_reasons_decreasing=decreasing,
            manifest=manifest,
            model_hash=get_model_hash(production_model),
            data_hash=get_data_hash(),
        )

    # Speichern-Button
    if "pending_case" in st.session_state:
        if st.button("Entscheidung speichern", key="save_btn"):
            filepath = save_case(st.session_state["pending_case"])
            st.success(f"Archiviert: {st.session_state['pending_case']['case_id']}")
            del st.session_state["pending_case"]
            st.rerun()