"""
streamlit_app.py
------------------
Streamlit UI — entry point for users of the Agentic Medallion Pipeline.

Flow:
1. Upload raw CSV files
2. Enter a business question in plain English
3. Run Phase 1 (Profiler -> Bronze STTM), review & approve the Bronze STTM
4. Run Phase 2 (Bronze ingestion -> Silver STTM), review & approve the Silver STTM
5. Run Phase 3 (Silver cleansing -> Gold STTM), review & approve the Gold STTM
6. Run Phase 4 (Gold materialisation -> Report), view the executive report
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from agents.orchestrator import (
    PipelineState,
    run_bronze_to_silver_sttm,
    run_gold_and_report,
    run_silver_to_gold_sttm,
    run_until_bronze_sttm,
)
from core.config import UPLOADS_DIR

st.set_page_config(page_title="Agentic Medallion Pipeline", page_icon="🏅", layout="wide")

RETRY_MAP = {
    "phase_1_complete": run_bronze_to_silver_sttm,
    "phase_2_complete": run_silver_to_gold_sttm,
    "phase_3_complete": run_gold_and_report,
}

APPROVAL_CONFIG = {
    "awaiting_bronze_approval": {
        "sttm_key": "bronze_sttm_path",
        "layer_label": "Bronze",
        "next_layer_label": "Silver",
        "next_fn": run_bronze_to_silver_sttm,
    },
    "awaiting_silver_approval": {
        "sttm_key": "silver_sttm_path",
        "layer_label": "Silver",
        "next_layer_label": "Gold",
        "next_fn": run_silver_to_gold_sttm,
    },
    "awaiting_gold_approval": {
        "sttm_key": "gold_sttm_path",
        "layer_label": "Gold",
        "next_layer_label": "Report",
        "next_fn": run_gold_and_report,
    },
}

if "pipeline_state" not in st.session_state:
    st.session_state.pipeline_state: PipelineState | None = None


def _save_uploaded_files(uploaded_files) -> list[str]:
    paths = []
    for uploaded_file in uploaded_files:
        dest = UPLOADS_DIR / uploaded_file.name
        dest.write_bytes(uploaded_file.getvalue())
        paths.append(str(dest))
    return paths


st.title("🏅 Agentic Medallion Pipeline")
st.caption("Intent-Driven Agentic Data Engineering for Retail Sales Analytics")

state: PipelineState | None = st.session_state.pipeline_state

# ---------------------------------------------------------------------------
# Step 1 & 2: Upload files + business question (only before Phase 1 runs)
# ---------------------------------------------------------------------------
if state is None:
    st.subheader("1. Upload raw CSV files")
    uploaded_files = st.file_uploader(
        "Drag and drop your raw sales CSV files", type=["csv"], accept_multiple_files=True
    )

    st.subheader("2. Ask your business question")
    business_intent = st.text_input(
        "e.g. Which product category had the highest sales in Q4?",
        placeholder="Which store had the highest revenue last month?",
    )

    if st.button("🚀 Run Phase 1 — Profile & Generate Bronze STTM", type="primary"):
        if not uploaded_files:
            st.error("Please upload at least one CSV file.")
        elif not business_intent.strip():
            st.error("Please enter a business question.")
        else:
            input_paths = _save_uploaded_files(uploaded_files)
            with st.spinner("Profiler and STTM agents are working..."):
                st.session_state.pipeline_state = run_until_bronze_sttm(input_paths, business_intent)
            st.rerun()

# ---------------------------------------------------------------------------
# Failed state — allow retry of the phase that failed
# ---------------------------------------------------------------------------
elif state.get("status") == "failed":
    st.error(f"Phase failed: {state.get('error')}")
    st.json({k: v for k, v in state.items() if k != "error"})

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔁 Retry this phase"):
            retry_fn = RETRY_MAP.get(state.get("phase"))
            with st.spinner("Retrying..."):
                if retry_fn is None:
                    st.session_state.pipeline_state = run_until_bronze_sttm(
                        state["input_files"], state["business_intent"]
                    )
                else:
                    st.session_state.pipeline_state = retry_fn(state)
            st.rerun()
    with col2:
        if st.button("🗑️ Start over"):
            st.session_state.pipeline_state = None
            st.rerun()

# ---------------------------------------------------------------------------
# STTM human-approval gates (Bronze / Silver / Gold)
# ---------------------------------------------------------------------------
elif state.get("status") in APPROVAL_CONFIG:
    cfg = APPROVAL_CONFIG[state["status"]]
    sttm_path = state[cfg["sttm_key"]]

    st.subheader(f"⏸ Review the {cfg['layer_label']} STTM before it runs")
    st.caption(f"Business question: {state['business_intent']}")

    sttm_df = pd.read_csv(sttm_path)
    edited_df = st.data_editor(sttm_df, num_rows="dynamic", use_container_width=True, key=sttm_path)

    col1, col2 = st.columns(2)
    with col1:
        if st.button(f"✅ Approve {cfg['layer_label']} STTM & Run {cfg['next_layer_label']}", type="primary"):
            edited_df.to_csv(sttm_path, index=False)
            with st.spinner(f"Running {cfg['next_layer_label']} phase..."):
                st.session_state.pipeline_state = cfg["next_fn"](state)
            st.rerun()
    with col2:
        if st.button("🗑️ Start over"):
            st.session_state.pipeline_state = None
            st.rerun()

# ---------------------------------------------------------------------------
# Pipeline complete — show the executive report
# ---------------------------------------------------------------------------
elif state.get("status") == "complete":
    st.success("✅ Pipeline complete!")
    st.caption(f"Business question: {state['business_intent']}")

    report_path = Path(state["report_html_path"])
    if report_path.exists():
        st.components.v1.html(report_path.read_text(encoding="utf-8"), height=900, scrolling=True)
        st.download_button(
            "⬇️ Download HTML report", report_path.read_bytes(), file_name=report_path.name
        )

    report_json_path = Path(state.get("report_json_path", ""))
    if report_json_path.exists():
        st.download_button(
            "⬇️ Download JSON report",
            report_json_path.read_bytes(),
            file_name=report_json_path.name,
        )

    with st.expander("Pipeline run details"):
        st.json(dict(state))

    if st.button("🔄 Run a new pipeline"):
        st.session_state.pipeline_state = None
        st.rerun()

else:
    st.info("Unrecognised pipeline state — starting over.")
    st.session_state.pipeline_state = None