"""
Seite: Evidence Pack Übersicht.
Zeigt das Manifest, regulatorisches Mapping und Integritätsprüfung.
"""

import os
import json

import streamlit as st
import pandas as pd

from utils.data_loader import load_evidence_manifest, get_available_runs, get_data_hash


def render():
    st.title("📋 Evidence Pack Übersicht")

    manifest, run_id = load_evidence_manifest()
    if manifest is None:
        st.error("Kein Evidence Pack gefunden. Bitte erst `generate_evidence_pack.py` ausführen.")
        return

    # --- Run-Auswahl (falls mehrere vorhanden) ---
    run_dirs = get_available_runs()
    if len(run_dirs) > 1:
        selected_run = st.selectbox("Run auswählen", run_dirs, index=len(run_dirs) - 1)
        manifest, run_id = load_evidence_manifest(selected_run)

    # --- Metadaten ---
    col1, col2, col3 = st.columns(3)
    col1.metric("Run-ID", manifest["run_id"])
    col2.metric("Erzeugt am", manifest["generated_at"][:10])
    col3.metric("Version", manifest["evidence_pack_version"])

    # --- Datensatz ---
    st.subheader("Datensatz")
    ds = manifest["dataset_documentation"]
    col1, col2, col3 = st.columns(3)
    col1.metric("Instanzen", ds["n_instances"])
    col2.metric("Features (Original)", ds["n_features_original"])
    col3.metric("Features (Encoded)", ds["n_features_encoded"])
    st.code(f"SHA-256: {ds['data_hash_sha256']}", language=None)

    # --- Integritätsprüfung ---
    st.subheader("Integritätsprüfung")
    current_hash = get_data_hash()
    stored_hash = ds["data_hash_sha256"]
    if current_hash == stored_hash:
        st.success("✅ Daten-Hash stimmt überein – Datensatz unverändert seit dem Run.")
    else:
        st.error("❌ Daten-Hash stimmt NICHT überein – Datensatz wurde seit dem Run verändert!")

    # --- Regulatorisches Mapping ---
    st.subheader("Regulatorisches Mapping – EU AI Act")
    reg = manifest["regulatory_mapping"]["addressed_requirements"]
    reg_rows = []
    for article, info in reg.items():
        reg_rows.append({
            "Artikel": article,
            "Status": info["status"],
            "Nachweis": info["evidence"],
            "Artefakt": info["artifact"],
        })
    st.dataframe(pd.DataFrame(reg_rows), use_container_width=True, hide_index=True)

    with st.expander("Nicht abgedeckte Anforderungen"):
        for item in manifest["regulatory_mapping"]["not_addressed"]:
            st.markdown(f"- {item}")

    # --- Softwareversionen ---
    with st.expander("Softwareversionen"):
        if "reproducibility" in manifest and "python_packages" in manifest["reproducibility"]:
            st.json(manifest["reproducibility"]["python_packages"])

    # --- Download ---
    st.subheader("Export")
    st.download_button(
        "📥 Evidence Manifest (JSON)",
        data=json.dumps(manifest, indent=2, ensure_ascii=False),
        file_name=f"evidence_manifest_{manifest['run_id']}.json",
        mime="application/json",
    )