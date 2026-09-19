"""
agents/bronze_agent.py
------------------------
Bronze layer ingestion agent. Ingests raw CSV files into the Bronze Parquet
layer using approved STTM rules: column renaming, type casting, and metadata
injection. No null handling happens here — Bronze is a faithful raw copy.

Tools:
- inspect_task_tool     — previews CSV file shapes and STTM transformation rules
- bronze_ingestion_tool — applies rules and writes Parquet files

Entry point:
    execute_bronze(input_files, sttm_path, run_id, task_description) -> list[str]
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from langchain_core.tools import tool

from core.audit import AuditLogger
from core.config import BRONZE_DIR, make_llm
from core.observability import run_react_agent

SYSTEM_PROMPT = """You are the Bronze Ingestion agent in an agentic Medallion \
data engineering pipeline. You ingest raw CSV files into the Bronze Parquet \
layer using STTM rules that a human has already approved. You do not decide \
the rules — you execute them faithfully. No null handling happens at this layer.

Follow this loop strictly: THINK -> INSPECT -> PLAN -> ACT -> VERIFY.
1. THINK about the ingestion task you were given.
2. INSPECT first by calling `inspect_task_tool` to see file shapes and STTM rules.
3. PLAN out loud: state which files map to which target tables.
4. ACT by calling `bronze_ingestion_tool` to execute the approved rules.
5. VERIFY by reporting the output Parquet file paths.

Never fabricate file paths — the tools return the authoritative paths themselves.
"""


def _cast_column(series: pd.Series, transformation_type: str) -> pd.Series:
    if transformation_type == "cast_numeric":
        return pd.to_numeric(series, errors="coerce")
    if transformation_type == "cast_datetime":
        return pd.to_datetime(series, errors="coerce")
    if transformation_type == "cast_string":
        return series.astype(str)
    return series


def _find_source_file(input_files: list[str], source_table: str) -> str | None:
    for f in input_files:
        if Path(f).stem == source_table:
            return f
    return None


def _apply_bronze_rules(input_files: list[str], sttm_path: str, run_id: str) -> list[str]:
    sttm = pd.read_csv(sttm_path)
    output_paths: list[str] = []
    load_timestamp = datetime.now(timezone.utc).isoformat()

    for target_table, group in sttm.groupby("target_table"):
        source_table = group["source_table"].iloc[0]
        source_file = _find_source_file(input_files, source_table)
        if source_file is None:
            continue

        df = pd.read_csv(source_file, low_memory=False)

        column_rules = group[group["source_column"].notna() & (group["source_column"] != "")]
        rename_map = dict(zip(column_rules["source_column"], column_rules["target_column"]))
        df = df.rename(columns=rename_map)

        for _, rule in column_rules.iterrows():
            target_col = rule["target_column"]
            if target_col in df.columns:
                df[target_col] = _cast_column(df[target_col], rule["transformation_type"])

        # Metadata injection rules (source_column is blank).
        metadata_rules = group[group["source_column"].isna() | (group["source_column"] == "")]
        for _, rule in metadata_rules.iterrows():
            if rule["target_column"] == "_load_timestamp":
                df["_load_timestamp"] = load_timestamp
            elif rule["target_column"] == "_source_file":
                df["_source_file"] = source_file

        out_path = BRONZE_DIR / f"{target_table}.parquet"
        df.to_parquet(out_path, index=False)
        output_paths.append(str(out_path))

    return output_paths


def _make_bronze_tools(
    input_files: list[str], sttm_path: str, run_id: str, scratchpad: dict[str, Any]
) -> list:
    @tool
    def inspect_task_tool() -> str:
        """Preview the shapes of the raw CSV files and the STTM transformation
        rules that will be applied. Call this first."""
        sttm = pd.read_csv(sttm_path)
        preview = {
            "sttm_rule_count": len(sttm),
            "target_tables": sorted(sttm["target_table"].unique().tolist()),
            "file_shapes": {
                Path(f).stem: pd.read_csv(f, nrows=5, low_memory=False).shape for f in input_files
            },
        }
        return json.dumps(preview, default=str)

    @tool
    def bronze_ingestion_tool(confirmation: str = "execute") -> str:
        """Execute the approved Bronze STTM rules against the raw CSV files:
        rename columns, cast types, inject _load_timestamp and _source_file
        metadata, and write one Parquet file per target table."""
        output_paths = _apply_bronze_rules(input_files, sttm_path, run_id)
        scratchpad["output_paths"] = output_paths
        return json.dumps({"output_paths": output_paths})

    return [inspect_task_tool, bronze_ingestion_tool]


def execute_bronze(
    input_files: list[str], sttm_path: str, run_id: str, task_description: str
) -> list[str]:
    """
    Run the Bronze agent to ingest `input_files` per the approved STTM at
    `sttm_path`. Returns the list of Bronze Parquet output paths. Falls back
    to deterministic execution if the LLM fails to call the required tool.
    """
    audit = AuditLogger(run_id)
    audit.start("bronze_ingestion", {"input_files": input_files, "sttm_path": sttm_path})

    scratchpad: dict[str, Any] = {}
    llm = make_llm()
    tools = _make_bronze_tools(input_files, sttm_path, run_id, scratchpad)

    user_message = (
        f"Task: {task_description}\n"
        f"Ingest the raw CSV files into Bronze Parquet using the approved STTM rules."
    )

    run_react_agent(
        agent_name="bronze_agent",
        run_id=run_id,
        llm=llm,
        tools=tools,
        system_prompt=SYSTEM_PROMPT,
        user_message=user_message,
        input_summary={"input_files": input_files, "sttm_path": sttm_path},
    )

    if "output_paths" not in scratchpad:
        scratchpad["output_paths"] = _apply_bronze_rules(input_files, sttm_path, run_id)

    audit.complete("bronze_ingestion", {"output_paths": scratchpad["output_paths"]})
    return scratchpad["output_paths"]
