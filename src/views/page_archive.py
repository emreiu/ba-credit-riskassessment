"""
Seite: Archiv.
Zeigt alle archivierten Kreditentscheidungen als Tabelle.
Detail-Ansicht mit Audit-Neuberechnung zum Reproduzierbarkeitsnachweis.
"""

import json

import streamlit as st
import pandas as pd
import numpy as np

from utils.data_loader import (
    load_models_and_data, load_evidence_manifest,
    get_model_hash, get_data_hash, MODEL_LABELS,
)
from utils.shap_utils import (
    compute_shap_for_instance, generate_top_reasons, render_waterfall_plot,
)
from utils.case_store import load_all_cases


# --- Konstanten ---
THRESHOLD = 0.50


def _render_detail(case):
    """Rendert die Detail-Ansicht eines archivierten Falls mit Audit-Vergleich."""

    st.markdown("---")
    st.subheader(f"Detail: {case['case_id']}")
    st.caption(f"Referenz: {case.get('reference', '–')} | Archiviert: {case['timestamp'][:19]}")

    # --- Antragsdaten ---
    with st.expander("Antragsdaten", expanded=False):
        st.dataframe(
            pd.DataFrame([case["applicant_data"]]),
            use_container_width=True, hide_index=True,
        )
        if case.get("actual_label") is not None:
            label = "Bad (1)" if case["actual_label"] == 1 else "Good (0)"
            st.caption(f"Tatsächliches Label: {label}")

    # --- Audit-Button ---
    do_audit = st.button(
        "Audit: Jetzt neu berechnen",
        key=f"audit_{case['case_id']}",
    )

    col_left, col_right = st.columns(2)

    # === LINKS: Archivierte Entscheidung ===
    with col_left:
        st.markdown("#### Archivierte Entscheidung")

        dec = case["decision"]
        if dec["result"] == "ABGELEHNT":
            st.error(f"ABGELEHNT – P(Bad) = {dec['p_bad']:.1%}")
        else:
            st.success(f"GENEHMIGT – P(Good) = {dec['p_good']:.1%}")

        st.caption(
            f"Modell: {MODEL_LABELS.get(dec['production_model'], dec['production_model'])} | "
            f"Schwellenwert: {dec['threshold']:.0%}"
        )

        # Alle Modellvorhersagen
        if "all_model_predictions" in case:
            st.markdown("**Alle Modelle:**")
            for name, pred in case["all_model_predictions"].items():
                icon = "❌" if pred["p_bad"] >= dec["threshold"] else "✅"
                st.caption(f"{icon} {MODEL_LABELS.get(name, name)}: P(Bad)={pred['p_bad']:.1%}")

        # SHAP
        st.markdown("**SHAP Top-Faktoren:**")
        shap_exp = case.get("shap_explanation", {})
        for r in shap_exp.get("top_reasons_increasing", []):
            st.caption(f"🔴 {r['feature']} ({r['shap_value']:+.4f})")
        for r in shap_exp.get("top_reasons_decreasing", []):
            st.caption(f"🔵 {r['feature']} ({r['shap_value']:+.4f})")

        # Evidence-Snapshot – vollständige Hashes
        with st.expander("Evidence-Pack-Snapshot"):
            snap = case.get("evidence_snapshot", {})
            st.markdown(f"**Run-ID:** {snap.get('run_id', '–')}")
            st.markdown(f"**Modell-Hash:**")
            st.code(snap.get("model_hash_sha256", "–"), language=None)
            st.markdown(f"**Datensatz-Hash:**")
            st.code(snap.get("dataset_hash_sha256", "–"), language=None)
            if snap.get("hyperparameters"):
                st.markdown("**Hyperparameter:**")
                st.json(snap["hyperparameters"])
            if snap.get("preprocessing"):
                st.markdown("**Preprocessing:**")
                st.json(snap["preprocessing"])
            if snap.get("software_versions"):
                st.markdown("**Softwareversionen:**")
                st.json(snap["software_versions"])

    # === RECHTS: Audit-Neuberechnung ===
    with col_right:
        st.markdown("#### Audit-Neuberechnung")

        if not do_audit:
            st.info("Klicke oben auf 'Audit: Jetzt neu berechnen'.")
        else:
            models, data_obj = load_models_and_data()
            prod_model = case["decision"]["production_model"]
            manifest_current, _ = load_evidence_manifest()
            feature_names = data_obj["feature_names"]

            # Antragsdaten rekonstruieren
            applicant_df = pd.DataFrame([case["applicant_data"]])
            applicant_processed = data_obj["preprocessor"].transform(applicant_df)

            # Vorhersage
            model = models[prod_model]
            proba = model.predict_proba(applicant_processed)[0]
            p_bad_new = float(proba[1])
            p_good_new = float(proba[0])
            decision_new = "ABGELEHNT" if p_bad_new >= THRESHOLD else "GENEHMIGT"

            if decision_new == "ABGELEHNT":
                st.error(f"ABGELEHNT – P(Bad) = {p_bad_new:.1%}")
            else:
                st.success(f"GENEHMIGT – P(Good) = {p_good_new:.1%}")

            st.caption(
                f"Modell: {MODEL_LABELS.get(prod_model, prod_model)} | "
                f"Schwellenwert: {THRESHOLD:.0%}"
            )

            # Alle Modelle neu
            st.markdown("**Alle Modelle:**")
            for name in MODEL_LABELS:
                proba_other = models[name].predict_proba(applicant_processed)[0]
                p_bad_other = float(proba_other[1])
                icon = "❌" if p_bad_other >= THRESHOLD else "✅"
                st.caption(f"{icon} {MODEL_LABELS[name]}: P(Bad)={p_bad_other:.1%}")

            # SHAP
            with st.spinner("SHAP wird berechnet..."):
                shap_vals_new, base_val_new = compute_shap_for_instance(
                    model, prod_model, applicant_processed, data_obj["X_train"],
                )
            increasing_new, decreasing_new = generate_top_reasons(shap_vals_new, feature_names)

            st.markdown("**SHAP Top-Faktoren:**")
            for r in increasing_new:
                st.caption(f"🔴 {r['feature']} ({r['shap_value']:+.4f})")
            for r in decreasing_new:
                st.caption(f"🔵 {r['feature']} ({r['shap_value']:+.4f})")

            # Aktuelle Evidence-Daten – vollständige Hashes
            with st.expander("Aktuelle Evidence-Pack-Daten"):
                current_model_hash = get_model_hash(prod_model)
                current_data_hash = get_data_hash()
                st.markdown(f"**Run-ID:** {manifest_current['run_id'] if manifest_current else '–'}")
                st.markdown("**Modell-Hash:**")
                st.code(current_model_hash, language=None)
                st.markdown("**Datensatz-Hash:**")
                st.code(current_data_hash, language=None)

    # === VERGLEICHSERGEBNIS ===
    if do_audit:
        st.markdown("---")
        st.subheader("Vergleichsergebnis")

        snap = case.get("evidence_snapshot", {})
        dec_old = case["decision"]
        current_model_hash = get_model_hash(case["decision"]["production_model"])
        current_data_hash = get_data_hash()

        checks = {
            "Entscheidung": dec_old["result"] == decision_new,
            "P(Bad)": abs(dec_old["p_bad"] - p_bad_new) < 0.0001,
            "Modell-Hash": snap.get("model_hash_sha256", "") == current_model_hash,
            "Datensatz-Hash": snap.get("dataset_hash_sha256", "") == current_data_hash,
        }

        all_ok = all(checks.values())
        for label, ok in checks.items():
            icon = "✅" if ok else "❌"
            st.markdown(f"{icon} **{label}** – {'identisch' if ok else 'ABWEICHUNG'}")

        if all_ok:
            st.success(
                "Reproduzierbarkeit bestätigt – "
                "die Entscheidung kann vollständig nachvollzogen werden."
            )
        else:
            st.warning(
                "Abweichung festgestellt – "
                "das Modell oder die Daten wurden seit der Originalentscheidung verändert."
            )

    # --- Download ---
    st.markdown("---")
    st.download_button(
        f"Herunterladen: {case['case_id']} (JSON)",
        data=json.dumps(case, indent=2, ensure_ascii=False),
        file_name=f"{case['case_id']}.json",
        mime="application/json",
        key=f"dl_{case['case_id']}",
    )


def render():
    st.title("Archiv")
    st.markdown("Alle archivierten Kreditentscheidungen mit Audit-Funktion.")

    cases = load_all_cases()

    if not cases:
        st.info(
            "Noch keine Fälle archiviert. "
            "Berechne im Tab 'Einzelfall-Prüfung' eine Entscheidung und klicke 'Speichern'."
        )
        return

    # Chronologisch sortieren (älteste zuerst) für aufsteigende Nummerierung
    cases_sorted = sorted(cases, key=lambda c: c.get("timestamp", ""))

    # --- Tabelle mit Detail-Buttons ---
    st.markdown("##### Gespeicherte Entscheidungen")

    # Header
    cols = st.columns([0.5, 2, 1.2, 2, 1.2, 1.8, 0.8, 0.8])
    cols[0].markdown("**Nr**")
    cols[1].markdown("**Case-ID**")
    cols[2].markdown("**Datum**")
    cols[3].markdown("**Referenz**")
    cols[4].markdown("**Entscheidung**")
    cols[5].markdown("**Modell**")
    cols[6].markdown("**P(Bad)**")
    cols[7].markdown("**Detail**")

    # Zeilen
    for i, c in enumerate(cases_sorted):
        nr = i + 1
        dec = c["decision"]
        cols = st.columns([0.5, 2, 1.2, 2, 1.2, 1.8, 0.8, 0.8])
        cols[0].write(nr)
        cols[1].write(c["case_id"])
        cols[2].write(c["timestamp"][:10])
        cols[3].write(c.get("reference", "–"))
        cols[4].write(dec["result"])
        cols[5].write(MODEL_LABELS.get(dec["production_model"], "–"))
        cols[6].write(f"{dec['p_bad']:.1%}")
        if cols[7].button("🔍", key=f"detail_{c['case_id']}"):
            st.session_state["selected_case_id"] = c["case_id"]

    # --- Detail-Ansicht ---
    if "selected_case_id" in st.session_state:
        selected_id = st.session_state["selected_case_id"]
        selected_case = next((c for c in cases_sorted if c["case_id"] == selected_id), None)
        if selected_case:
            _render_detail(selected_case)