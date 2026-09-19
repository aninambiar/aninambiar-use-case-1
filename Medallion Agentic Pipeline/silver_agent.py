"""
agents/silver_agent.py
------------------------
Silver layer cleansing agent. Cleanses Bronze Parquet files into trusted
Silver Parquet files: null handling, deduplication, type standardisation,
date formatting, text normalisation, surrogate key injection, and column
filtering to STTM-approved columns only.

Tools:
- inspect_task_tool     — previews Bronze schemas, null counts, and STTM rules
- silver_ingestion_tool — applies cleansing and writes Silver Parquet files

Entry point:
    execute_silver(input_files, sttm_path, run_id, task_description) -> list[str]
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from langchain_core.tools import tool

from core.audit import AuditLogger
from core.config import SILVER_DIR, make_llm
from core.observability import run_react_agent

SYSTEM_PROMPT = """You are the Silver Cleansing agent in an agentic Medallion \
data engineering pipeline. You cleanse Bronze Parquet files into trusted \
Silver Parquet files using STTM rules that a human has already approved.

Follow this loop strictly: THINK -> INSPECT -> PLAN -> ACT -> VERIFY.
1. THINK about the cleansing task you were given.
2. INSPECT first by calling `inspect_task_tool` to see Bronze schemas, null \
   counts, and the approved STTM cleansing rules.
3. PLAN out loud: state which cleansing operations you will apply per column.
4. ACT by calling `silver_ingestion_tool` to execute the approved rules.
5. VERIFY by reporting the output Parquet file paths.

Never fabricate file paths — the tools return the authoritative paths themselves.
"""


def _find_bronze_file(input_files: list[str], source_table: str) -> str | None:
    for f in input_files:
        if Path(f).stem == source_table:
            return f
    return None


def _apply_column_rule(df: pd.DataFrame, target_col: str, transformation_type: str) -> pd.DataFrame:
    if target_col not in df.columns:
        return df
    if transformation_type == "dropna":
        df = df[df[target_col].notna()]
    elif transformation_type == "fillna_mean":
        df[target_col] = df[target_col].fillna(df[target_col].mean())
    elif transformation_type == "fillna_median":
        df[target_col] = df[target_col].fillna(df[target_col].median())
    elif transformation_type == "fillna_mode":
        mode = df[target_col].mode()
        if not mode.empty:
            df[target_col] = df[target_col].fillna(mode.iloc[0])
    elif transformation_type == "fillna_constant":
        df[target_col] = df[target_col].fillna("unknown")
    elif transformation_type == "date_format":
        df[target_col] = pd.to_datetime(df[target_col], errors="coerce").dt.strftime("%Y-%m-%d")
    elif transformation_type == "text_normalize":
        df[target_col] = df[target_col].astype(str).str.strip().str.lower()
    # "type_cast" is a passthrough — the type was already cast at Bronze.
    return df


def _apply_silver_rules(input_files: list[str], sttm_path: str, run_id: str) -> list[str]:
    sttm = pd.read_csv(sttm_path)
    output_paths: list[str] = []

    for target_table, group in sttm.groupby("target_table"):
        source_table = group["source_table"].iloc[0]
        bronze_file = _find_bronze_file(input_files, source_table)
        if bronze_file is None:
            continue

        df = pd.read_parquet(bronze_file)

        column_rules = group[group["source_column"].notna() & (group["source_column"] != "")]
        rename_map = dict(zip(column_rules["source_column"], column_rules["target_column"]))
        df = df.rename(columns=rename_map)

        for _, rule in column_rules.iterrows():
            df = _apply_column_rule(df, rule["target_column"], rule["transformation_type"])

        # Keep only STTM-approved target columns (drops Bronze metadata columns
        # such as _load_timestamp / _source_file unless explicitly mapped).
        approved_columns = [c for c in column_rules["target_column"] if c in df.columns]
        df = df[approved_columns]

        control_rules = group[group["source_column"].isna() | (group["source_column"] == "")]
        if (control_rules["transformation_type"] == "dedup").any():
            df = df.drop_duplicates()

        pk_rows = control_rules[control_rules["transformation_type"] == "surrogate_key"]
        pk_name = pk_rows["target_column"].iloc[0] if not pk_rows.empty else f"pk_{target_table}_id"
        df = df.reset_index(drop=True)
        df.insert(0, pk_name, range(1, len(df) + 1))

        out_path = SILVER_DIR / f"{target_table}.parquet"
        df.to_parquet(out_path, index=False)
        output_paths.append(str(out_path))

    return output_paths


def _make_silver_tools(
    input_files: list[str], sttm_path: str, run_id: str, scratchpad: dict[str, Any]
) -> list:
    @tool
    def inspect_task_tool() -> str:
        """Preview Bronze Parquet schemas, null counts per column, and the
        approved STTM cleansing rules. Call this first."""
        sttm = pd.read_csv(sttm_path)
        preview = {
            "sttm_rule_count": len(sttm),
            "target_tables": sorted(sttm["target_table"].unique().tolist()),
            "bronze_schemas": {
                Path(f).stem: {
                    "columns": list(pd.read_parquet(f).columns),
                    "null_counts": pd.read_parquet(f).isna().sum().to_dict(),
                }
                for f in input_files
            },
        }
        return json.dumps(preview, default=str)

    @tool
    def silver_ingestion_tool(confirmation: str = "execute") -> str:
        """Execute the approved Silver STTM rules against the Bronze Parquet
        files: null handling, dedup, type/date/text standardisation, surrogate
        key injection, and column filtering, then write Silver Parquet files."""
        output_paths = _apply_silver_rules(input_files, sttm_path, run_id)
        scratchpad["output_paths"] = output_paths
        return json.dumps({"output_paths": output_paths})

    return [inspect_task_tool, silver_ingestion_tool]


def execute_silver(
    input_files: list[str], sttm_path: str, run_id: str, task_description: str
) -> list[str]:
    """
    Run the Silver agent to cleanse `input_files` (Bronze Parquet paths) per
    the approved STTM at `sttm_path`. Returns Silver Parquet output paths.
    Falls back to deterministic execution if the LLM fails to call the
    required tool.
    """
    audit = AuditLogger(run_id)
    audit.start("silver_ingestion", {"input_files": input_files, "sttm_path": sttm_path})

    scratchpad: dict[str, Any] = {}
    llm = make_llm()
    tools = _make_silver_tools(input_files, sttm_path, run_id, scratchpad)

    user_message = (
        f"Task: {task_description}\n"
        f"Cleanse the Bronze Parquet files into Silver Parquet using the approved STTM rules."
    )

    run_react_agent(
        agent_name="silver_agent",
        run_id=run_id,
        llm=llm,
        tools=tools,
        system_prompt=SYSTEM_PROMPT,
        user_message=user_message,
        input_summary={"input_files": input_files, "sttm_path": sttm_path},
    )

    if "output_paths" not in scratchpad:
        scratchpad["output_paths"] = _apply_silver_rules(input_files, sttm_path, run_id)

    audit.complete("silver_ingestion", {"output_paths": scratchpad["output_paths"]})
    return scratchpad["output_paths"]
