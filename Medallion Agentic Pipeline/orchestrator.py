"""
agents/orchestrator.py
------------------------
Supervisor Orchestrator. The pipeline coordinator — not a simple script but
an autonomous LLM agent that receives a phase goal, plans which specialist
agents to call, dispatches them with rich goal descriptions, and verifies
outputs before handing control back to the Streamlit UI at a human approval
gate.

Entry points (called by the Streamlit UI):
    run_until_bronze_sttm(uploaded_files, business_intent) -> PipelineState
    run_bronze_to_silver_sttm(state) -> PipelineState
    run_silver_to_gold_sttm(state) -> PipelineState
    run_gold_and_report(state) -> PipelineState

Key patterns used: PipelineState (TypedDict), tool factories with closures,
scratchpad dict for intra-phase handoffs between specialist agent calls.
"""

from __future__ import annotations

import json
import uuid
from typing import Any, TypedDict

from langchain_core.tools import tool

from agents.bronze_agent import execute_bronze
from agents.gold_agent import execute_gold
from agents.profiler import profile_datasets
from agents.reporter import generate_report
from agents.silver_agent import execute_silver
from agents.sttm_generator import generate_bronze_sttm, generate_gold_sttm, generate_silver_sttm
from core.audit import AuditLogger
from core.config import make_llm
from core.memory import memory_store
from core.observability import run_react_agent

SUPERVISOR_SYSTEM_PROMPT = """You are the Supervisor Orchestrator of an \
agentic Medallion data engineering pipeline (Bronze -> Silver -> Gold -> Report). \
You do not transform data yourself — you dispatch specialist agents in the \
correct order for the current phase and verify their outputs.

For the phase you are given, call each available tool exactly once, in the \
order they are listed to you, and stop once both have returned a result. \
Then report back a one-paragraph summary of what each specialist produced.
"""


class PipelineState(TypedDict, total=False):
    run_id: str
    business_intent: str
    input_files: list[str]
    profile_path: str
    bronze_sttm_path: str
    bronze_output_paths: list[str]
    silver_sttm_path: str
    silver_output_paths: list[str]
    gold_sttm_path: str
    gold_output_paths: list[str]
    report_html_path: str
    report_json_path: str
    phase: str
    status: str
    error: str


def _make_phase1_tools(
    input_files: list[str], business_intent: str, run_id: str, scratchpad: dict[str, Any]
) -> list:
    @tool
    def profiler_agent_tool() -> str:
        """Dispatch the Profiler agent to profile the uploaded raw CSV files."""
        path = profile_datasets(
            input_files, run_id, business_intent, "Profile raw CSV files for STTM generation"
        )
        scratchpad["profile_path"] = path
        return json.dumps({"profile_path": path})

    @tool
    def sttm_agent_tool() -> str:
        """Dispatch the STTM agent to generate the Bronze ingestion STTM from
        the data profile. Requires profiler_agent_tool to have run first."""
        profile_path = scratchpad.get("profile_path")
        if not profile_path:
            return json.dumps({"error": "profile_path unavailable — call profiler_agent_tool first"})
        path = generate_bronze_sttm(
            profile_path, business_intent, run_id, "Generate Bronze ingestion STTM"
        )
        scratchpad["bronze_sttm_path"] = path
        return json.dumps({"bronze_sttm_path": path})

    return [profiler_agent_tool, sttm_agent_tool]


def _make_phase2_tools(
    input_files: list[str],
    bronze_sttm_path: str,
    business_intent: str,
    run_id: str,
    scratchpad: dict[str, Any],
) -> list:
    @tool
    def bronze_agent_tool() -> str:
        """Dispatch the Bronze agent to ingest raw CSV files using the
        approved Bronze STTM."""
        paths = execute_bronze(
            input_files, bronze_sttm_path, run_id, "Ingest raw CSV files into Bronze Parquet"
        )
        scratchpad["bronze_output_paths"] = paths
        return json.dumps({"bronze_output_paths": paths})

    @tool
    def sttm_agent_tool() -> str:
        """Dispatch the STTM agent to generate the Silver cleansing STTM from
        Bronze outputs. Requires bronze_agent_tool to have run first."""
        bronze_paths = scratchpad.get("bronze_output_paths")
        if not bronze_paths:
            return json.dumps({"error": "bronze_output_paths unavailable — call bronze_agent_tool first"})
        path = generate_silver_sttm(
            bronze_paths, bronze_sttm_path, business_intent, run_id, "Generate Silver cleansing STTM"
        )
        scratchpad["silver_sttm_path"] = path
        return json.dumps({"silver_sttm_path": path})

    return [bronze_agent_tool, sttm_agent_tool]


def _make_phase3_tools(
    bronze_output_paths: list[str],
    silver_sttm_path: str,
    business_intent: str,
    run_id: str,
    scratchpad: dict[str, Any],
) -> list:
    @tool
    def silver_agent_tool() -> str:
        """Dispatch the Silver agent to cleanse Bronze Parquet files using the
        approved Silver STTM."""
        paths = execute_silver(
            bronze_output_paths, silver_sttm_path, run_id, "Cleanse Bronze Parquet into Silver Parquet"
        )
        scratchpad["silver_output_paths"] = paths
        return json.dumps({"silver_output_paths": paths})

    @tool
    def sttm_agent_tool() -> str:
        """Dispatch the STTM agent to generate the Gold materialisation STTM
        from Silver outputs. Requires silver_agent_tool to have run first."""
        silver_paths = scratchpad.get("silver_output_paths")
        if not silver_paths:
            return json.dumps({"error": "silver_output_paths unavailable — call silver_agent_tool first"})
        path = generate_gold_sttm(
            silver_paths, silver_sttm_path, business_intent, run_id, "Generate Gold materialisation STTM"
        )
        scratchpad["gold_sttm_path"] = path
        return json.dumps({"gold_sttm_path": path})

    return [silver_agent_tool, sttm_agent_tool]


def _make_phase4_tools(
    silver_output_paths: list[str],
    gold_sttm_path: str,
    business_intent: str,
    run_id: str,
    scratchpad: dict[str, Any],
) -> list:
    @tool
    def gold_agent_tool() -> str:
        """Dispatch the Gold agent to materialise Gold Parquet tables using
        the approved Gold STTM."""
        paths = execute_gold(
            silver_output_paths,
            gold_sttm_path,
            business_intent,
            run_id,
            "Materialise Gold Parquet tables from Silver Parquet",
        )
        scratchpad["gold_output_paths"] = paths
        return json.dumps({"gold_output_paths": paths})

    @tool
    def reporter_agent_tool() -> str:
        """Dispatch the Reporter agent to answer the business question from
        Gold tables and produce the HTML/JSON executive report. Requires
        gold_agent_tool to have run first."""
        gold_paths = scratchpad.get("gold_output_paths")
        if not gold_paths:
            return json.dumps({"error": "gold_output_paths unavailable — call gold_agent_tool first"})
        path = generate_report(
            gold_paths, business_intent, run_id, "Answer the business question with SQL and charts"
        )
        scratchpad["report_html_path"] = path
        return json.dumps({"report_html_path": path})

    return [gold_agent_tool, reporter_agent_tool]


def run_until_bronze_sttm(uploaded_files: list[str], business_intent: str) -> PipelineState:
    """Phase 1: Profiler -> STTM(Bronze). Pauses for human approval of the Bronze STTM."""
    run_id = str(uuid.uuid4())
    state: PipelineState = {
        "run_id": run_id,
        "business_intent": business_intent,
        "input_files": uploaded_files,
        "phase": "phase_1",
        "status": "running",
    }
    audit = AuditLogger(run_id)
    audit.start("phase_1_profile_bronze_sttm", {"input_files": uploaded_files})
    try:
        scratchpad: dict[str, Any] = {}
        llm = make_llm()
        tools = _make_phase1_tools(uploaded_files, business_intent, run_id, scratchpad)
        goal = (
            f"Phase 1 of 4: Profile the uploaded raw CSV files, then generate the Bronze "
            f"ingestion STTM. Business question: {business_intent}"
        )
        run_react_agent(
            "orchestrator_phase1", run_id, llm, tools, SUPERVISOR_SYSTEM_PROMPT, goal,
            {"input_files": uploaded_files, "business_intent": business_intent},
        )

        profile_path = scratchpad.get("profile_path") or profile_datasets(
            uploaded_files, run_id, business_intent, "Profile raw CSV files"
        )
        bronze_sttm_path = scratchpad.get("bronze_sttm_path") or generate_bronze_sttm(
            profile_path, business_intent, run_id, "Generate Bronze ingestion STTM"
        )

        state.update(
            {
                "profile_path": profile_path,
                "bronze_sttm_path": bronze_sttm_path,
                "phase": "phase_1_complete",
                "status": "awaiting_bronze_approval",
            }
        )
        memory_store.add(f"{run_id}_business_intent", business_intent, {"run_id": run_id})
        audit.complete("phase_1_profile_bronze_sttm", dict(state))
    except Exception as exc:  # noqa: BLE001 - phase boundary; surfaced to the UI via state
        state.update({"status": "failed", "error": str(exc)})
        audit.fail("phase_1_profile_bronze_sttm", str(exc))
    return state


def run_bronze_to_silver_sttm(state: PipelineState) -> PipelineState:
    """Phase 2: Bronze -> STTM(Silver). Pauses for human approval of the Silver STTM."""
    run_id = state["run_id"]
    business_intent = state["business_intent"]
    audit = AuditLogger(run_id)
    audit.start("phase_2_bronze_silver_sttm", {})
    try:
        scratchpad: dict[str, Any] = {}
        llm = make_llm()
        tools = _make_phase2_tools(
            state["input_files"], state["bronze_sttm_path"], business_intent, run_id, scratchpad
        )
        goal = (
            f"Phase 2 of 4: Ingest raw CSV files into Bronze using the approved STTM, then "
            f"generate the Silver cleansing STTM. Business question: {business_intent}"
        )
        run_react_agent(
            "orchestrator_phase2", run_id, llm, tools, SUPERVISOR_SYSTEM_PROMPT, goal,
            {"bronze_sttm_path": state["bronze_sttm_path"]},
        )

        bronze_output_paths = scratchpad.get("bronze_output_paths") or execute_bronze(
            state["input_files"], state["bronze_sttm_path"], run_id, "Ingest raw CSVs into Bronze"
        )
        silver_sttm_path = scratchpad.get("silver_sttm_path") or generate_silver_sttm(
            bronze_output_paths, state["bronze_sttm_path"], business_intent, run_id,
            "Generate Silver cleansing STTM",
        )

        state.update(
            {
                "bronze_output_paths": bronze_output_paths,
                "silver_sttm_path": silver_sttm_path,
                "phase": "phase_2_complete",
                "status": "awaiting_silver_approval",
            }
        )
        audit.complete("phase_2_bronze_silver_sttm", dict(state))
    except Exception as exc:  # noqa: BLE001
        state.update({"status": "failed", "error": str(exc)})
        audit.fail("phase_2_bronze_silver_sttm", str(exc))
    return state


def run_silver_to_gold_sttm(state: PipelineState) -> PipelineState:
    """Phase 3: Silver -> STTM(Gold). Pauses for human approval of the Gold STTM."""
    run_id = state["run_id"]
    business_intent = state["business_intent"]
    audit = AuditLogger(run_id)
    audit.start("phase_3_silver_gold_sttm", {})
    try:
        scratchpad: dict[str, Any] = {}
        llm = make_llm()
        tools = _make_phase3_tools(
            state["bronze_output_paths"], state["silver_sttm_path"], business_intent, run_id, scratchpad
        )
        goal = (
            f"Phase 3 of 4: Cleanse Bronze Parquet into Silver using the approved STTM, then "
            f"generate the Gold materialisation STTM. Business question: {business_intent}"
        )
        run_react_agent(
            "orchestrator_phase3", run_id, llm, tools, SUPERVISOR_SYSTEM_PROMPT, goal,
            {"silver_sttm_path": state["silver_sttm_path"]},
        )

        silver_output_paths = scratchpad.get("silver_output_paths") or execute_silver(
            state["bronze_output_paths"], state["silver_sttm_path"], run_id,
            "Cleanse Bronze Parquet into Silver Parquet",
        )
        gold_sttm_path = scratchpad.get("gold_sttm_path") or generate_gold_sttm(
            silver_output_paths, state["silver_sttm_path"], business_intent, run_id,
            "Generate Gold materialisation STTM",
        )

        state.update(
            {
                "silver_output_paths": silver_output_paths,
                "gold_sttm_path": gold_sttm_path,
                "phase": "phase_3_complete",
                "status": "awaiting_gold_approval",
            }
        )
        audit.complete("phase_3_silver_gold_sttm", dict(state))
    except Exception as exc:  # noqa: BLE001
        state.update({"status": "failed", "error": str(exc)})
        audit.fail("phase_3_silver_gold_sttm", str(exc))
    return state


def run_gold_and_report(state: PipelineState) -> PipelineState:
    """Phase 4: Gold -> Reporter. Pipeline complete."""
    run_id = state["run_id"]
    business_intent = state["business_intent"]
    audit = AuditLogger(run_id)
    audit.start("phase_4_gold_report", {})
    try:
        scratchpad: dict[str, Any] = {}
        llm = make_llm()
        tools = _make_phase4_tools(
            state["silver_output_paths"], state["gold_sttm_path"], business_intent, run_id, scratchpad
        )
        goal = (
            f"Phase 4 of 4: Materialise Gold Parquet tables using the approved STTM, then "
            f"generate the executive report answering: {business_intent}"
        )
        run_react_agent(
            "orchestrator_phase4", run_id, llm, tools, SUPERVISOR_SYSTEM_PROMPT, goal,
            {"gold_sttm_path": state["gold_sttm_path"]},
        )

        gold_output_paths = scratchpad.get("gold_output_paths") or execute_gold(
            state["silver_output_paths"], state["gold_sttm_path"], business_intent, run_id,
            "Materialise Gold Parquet tables",
        )
        report_html_path = scratchpad.get("report_html_path") or generate_report(
            gold_output_paths, business_intent, run_id, "Answer the business question with SQL and charts"
        )

        state.update(
            {
                "gold_output_paths": gold_output_paths,
                "report_html_path": report_html_path,
                "report_json_path": str(report_html_path).replace(".html", ".json"),
                "phase": "phase_4_complete",
                "status": "complete",
            }
        )
        audit.complete("phase_4_gold_report", dict(state))
    except Exception as exc:  # noqa: BLE001
        state.update({"status": "failed", "error": str(exc)})
        audit.fail("phase_4_gold_report", str(exc))
    return state
