"""
Page: Individual Case Evaluation.
Calculate credit decisions and archive them.
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


# --- Constants ---
THRESHOLD = 0.50


def _render_applicant_input(df_raw):
    """Renders the applicant data input."""

    input_method = st.radio(
        "Input method", ["Select from dataset", "Enter manually"],
        horizontal=True, key="case_input_method",
    )

    if input_method == "Select from dataset":
        idx = st.number_input(
            "Instance index (0-999)", min_value=0,
            max_value=len(df_raw) - 1, value=9, key="case_idx",
        )
        applicant = df_raw.iloc[idx].drop("credit_risk")
        actual_label = int(df_raw.iloc[idx]["credit_risk"])
        reference = f"Dataset instance #{idx}"
        st.dataframe(applicant.to_frame().T, use_container_width=True, hide_index=True)
        label_text = "Bad (1) - defaulted" if actual_label == 1 else "Good (0) - repaid"
        st.info(f"Actual label: **{label_text}**")
        return applicant, idx, reference, actual_label

    else:
        ref = st.text_input("Reference / application number", value="Manual test", key="case_ref")
        col1, col2, col3 = st.columns(3)
        with col1:
            checking = st.selectbox("Checking account", ["< 0 DM", "0-200 DM", ">= 200 DM", "no checking account"], key="m_chk")
            duration = st.number_input("Duration (months)", 1, 72, 24, key="m_dur")
            credit_hist = st.selectbox("Credit history", ["critical account", "existing credits paid", "delay in past", "all paid at this bank", "no credits / all paid"], key="m_hist")
            purpose = st.selectbox("Purpose", ["car (new)", "car (used)", "furniture/equipment", "radio/television", "domestic appliances", "repairs", "education", "retraining", "business", "others"], key="m_purp")
            amount = st.number_input("Credit amount (DM)", 250, 20000, 3000, key="m_amt")
        with col2:
            savings = st.selectbox("Savings account", ["< 100 DM", "100-500 DM", "500-1000 DM", ">= 1000 DM", "unknown / none"], key="m_sav")
            employment = st.selectbox("Employment", ["unemployed", "< 1 year", "1-4 years", "4-7 years", ">= 7 years"], key="m_emp")
            installment = st.slider("Installment rate (% of income)", 1, 4, 3, key="m_inst")
            personal = st.selectbox("Personal status", ["male: single", "female: divorced/married", "male: divorced/separated", "male: married/widowed"], key="m_pers")
            age = st.number_input("Age", 18, 80, 35, key="m_age")
        with col3:
            housing = st.selectbox("Housing", ["rent", "own", "for free"], key="m_hous")
            foreign = st.selectbox("Foreign worker", ["yes", "no"], key="m_for")
            other_debtors = st.selectbox("Other debtors", ["none", "co-applicant", "guarantor"], key="m_debt")
            property_val = st.selectbox("Property", ["real estate", "savings/life insurance", "car or other", "unknown / none"], key="m_prop")
            other_inst = st.selectbox("Other installment plans", ["none", "bank", "stores"], key="m_oinst")

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
    st.title("Individual Case Evaluation")
    st.markdown(
        "Simulate a credit decision with full documentation. "
        "Archived cases can be reviewed and audited in the *Archive* tab."
    )

    models, data = load_models_and_data()
    df_raw = load_raw_data()
    manifest, _ = load_evidence_manifest()

    # --- Configuration (compact) ---
    col_cfg1, col_cfg2 = st.columns([3, 1])
    with col_cfg1:
        production_model = st.selectbox(
            "Production model",
            list(MODEL_LABELS.keys()),
            format_func=lambda x: MODEL_LABELS[x],
            key="prod_model",
        )
    with col_cfg2:
        st.metric("Threshold", f"{THRESHOLD:.0%}")

    # --- Applicant data ---
    applicant, source_index, reference, actual_label = _render_applicant_input(df_raw)

    # --- Calculation ---
    if st.button("Calculate credit decision", type="primary", key="calc_btn"):

        preprocessor = data["preprocessor"]
        applicant_df = applicant.to_frame().T
        applicant_processed = preprocessor.transform(applicant_df)
        feature_names = data["feature_names"]

        # All models predict
        all_predictions = {}
        for name in MODEL_LABELS:
            proba = models[name].predict_proba(applicant_processed)[0]
            all_predictions[name] = {
                "p_good": round(float(proba[0]), 4),
                "p_bad": round(float(proba[1]), 4),
            }

        st.markdown("---")
        st.subheader("Results")

        # --- All three models displayed equally ---
        cols = st.columns(3)
        for i, (name, label) in enumerate(MODEL_LABELS.items()):
            pred = all_predictions[name]
            model_decision = "REJECTED" if pred["p_bad"] >= THRESHOLD else "APPROVED"
            is_prod = name == production_model
            with cols[i]:
                header = f"**{label}**"
                if is_prod:
                    header += " (Prod.)"
                st.markdown(header)
                if model_decision == "REJECTED":
                    st.error(f"REJECTED ({pred['p_bad']:.1%})")
                else:
                    st.success(f"APPROVED ({pred['p_good']:.1%})")
                st.caption(f"P(good)={pred['p_good']:.3f} | P(bad)={pred['p_bad']:.3f}")

        # --- SHAP for production model ---
        st.markdown("---")
        st.subheader("Decision Explanation")
        st.caption(f"SHAP analysis for {MODEL_LABELS[production_model]} (production model)")

        with st.spinner("Calculating SHAP explanation..."):
            shap_vals, base_val = compute_shap_for_instance(
                models[production_model], production_model,
                applicant_processed, data["X_train"],
            )

        # Top reasons compact
        increasing, decreasing = generate_top_reasons(shap_vals, feature_names)
        col_r1, col_r2 = st.columns(2)
        with col_r1:
            st.markdown("**Risk-increasing**")
            for r in increasing:
                st.markdown(f"- {r['feature']} (`{r['shap_value']:+.4f}`)")
            if not increasing:
                st.caption("No dominant factors.")
        with col_r2:
            st.markdown("**Risk-decreasing**")
            for r in decreasing:
                st.markdown(f"- {r['feature']} (`{r['shap_value']:+.4f}`)")
            if not decreasing:
                st.caption("No dominant factors.")

        with st.expander("Detailed SHAP waterfall plot"):
            render_waterfall_plot(shap_vals, base_val, applicant_processed[0], feature_names)

        with st.expander("SHAP explanations for other models"):
            other_models = [n for n in MODEL_LABELS if n != production_model]
            for other_name in other_models:
                st.markdown(f"**{MODEL_LABELS[other_name]}:**")
                with st.spinner(f"Calculating SHAP for {MODEL_LABELS[other_name]}..."):
                    other_sv, other_bv = compute_shap_for_instance(
                        models[other_name], other_name,
                        applicant_processed, data["X_train"],
                    )
                render_waterfall_plot(other_sv, other_bv, applicant_processed[0], feature_names)

        # --- Archive ---
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
            decision="REJECTED" if all_predictions[production_model]["p_bad"] >= THRESHOLD else "APPROVED",
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

    # Save button
    if "pending_case" in st.session_state:
        if st.button("Save decision", key="save_btn"):
            filepath = save_case(st.session_state["pending_case"])
            st.success(f"Archived: {st.session_state['pending_case']['case_id']}")
            del st.session_state["pending_case"]
            st.rerun()