"""
agents/gold_agent.py
-----------------------
Gold layer materialisation agent. Builds analytics-ready Gold Parquet tables
from Silver inputs: multi-source joins, groupby aggregations (sum/avg/count/
max/min), and surrogate key injection. Produces one Parquet file per Gold
target table defined in the STTM.

Tools:
- inspect_task_tool  — previews Silver schemas and STTM rules grouped by Gold target table
- gold_ingestion_tool — applies joins, aggregations, and writes Gold Parquet files

Entry point:
    execute_gold(input_files, sttm_path, business_intent, run_id, task_description) -> list[str]
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from langchain_core.tools import tool

from core.audit import AuditLogger
from core.config import GOLD_DIR, make_llm
from core.observability import run_react_agent

SYSTEM_PROMPT = """You are the Gold Materialisation agent in an agentic \
Medallion data engineering pipeline. You build analytics-ready Gold Parquet \
tables from Silver inputs using STTM rules that a human has already approved: \
joins, aggregations, and surrogate keys.

Follow this loop strictly: THINK -> INSPECT -> PLAN -> ACT -> VERIFY.
1. THINK about the business question this Gold table must be able to answer.
2. INSPECT first by calling `inspect_task_tool` to see Silver schemas and the \
   approved STTM rules grouped by Gold target table.
3. PLAN out loud: state the joins and aggregations you will apply.
4. ACT by calling `gold_ingestion_tool` to execute the approved rules.
5. VERIFY by reporting the output Parquet file paths.

Never fabricate file paths — the tools return the authoritative paths themselves.
"""


_AGG_FUNC_MAP = {
    "aggregate_sum": "sum",
    "aggregate_avg": "mean",
    "aggregate_max": "max",
    "aggregate_min": "min",
    "aggregate_count": "count",
}


def _find_silver_file(input_files: list[str], source_table: str) -> str | None:
    for f in input_files:
        if Path(f).stem == source_table:
            return f
    return None


def _load_tables(input_files: list[str], table_names: set[str]) -> dict[str, pd.DataFrame]:
    tables: dict[str, pd.DataFrame] = {}
    for name in table_names:
        path = _find_silver_file(input_files, name)
        if path:
            tables[name] = pd.read_parquet(path)
    return tables


def _merge_tables(tables: dict[str, pd.DataFrame], join_keys: dict[str, list[str]]) -> pd.DataFrame:
    if not tables:
        return pd.DataFrame()
    names = list(tables.keys())
    merged = tables[names[0]].copy()
    for name in names[1:]:
        next_df = tables[name]
        common_join_cols = [
            col
            for col, tbls in join_keys.items()
            if name in tbls and col in merged.columns and col in next_df.columns
        ]
        if common_join_cols:
            merged = merged.merge(next_df, on=common_join_cols, how="outer", suffixes=("", f"_{name}"))
        else:
            # No declared join key between these tables — fall back to a
            # positional outer join so no data is silently dropped.
            merged = merged.join(next_df, how="outer", rsuffix=f"_{name}")
    return merged


def _apply_gold_rules(
    input_files: list[str], sttm_path: str, business_intent: str, run_id: str
) -> list[str]:
    sttm = pd.read_csv(sttm_path)
    output_paths: list[str] = []

    for target_table, group in sttm.groupby("target_table"):
        join_rows = group[group["transformation_type"] == "join_key"]
        groupby_rows = group[group["transformation_type"] == "groupby"]
        agg_rows = group[group["transformation_type"].isin(_AGG_FUNC_MAP.keys())]
        pk_rows = group[group["transformation_type"] == "surrogate_key"]

        involved_tables = set(join_rows["source_table"]) | set(groupby_rows["source_table"]) | set(
            agg_rows["source_table"]
        )
        involved_tables.discard("")
        tables = _load_tables(input_files, involved_tables)
        if not tables:
            continue

        join_keys: dict[str, list[str]] = {}
        for join_col, sub in join_rows.groupby("target_column"):
            join_keys[join_col] = sub["source_table"].tolist()

        merged = _merge_tables(tables, join_keys)
        if merged.empty:
            continue

        dim_cols = [c for c in groupby_rows["target_column"].tolist() if c in merged.columns]

        agg_kwargs: dict[str, Any] = {}
        record_count_needed = False
        for _, rule in agg_rows.iterrows():
            src_col, tgt_col, ttype = rule["source_column"], rule["target_column"], rule["transformation_type"]
            if ttype == "aggregate_count" and (not isinstance(src_col, str) or not src_col):
                record_count_needed = True
                continue
            if src_col in merged.columns:
                agg_kwargs[tgt_col] = pd.NamedAgg(column=src_col, aggfunc=_AGG_FUNC_MAP[ttype])

        if dim_cols:
            if agg_kwargs:
                result = merged.groupby(dim_cols, dropna=False).agg(**agg_kwargs).reset_index()
            else:
                result = merged[dim_cols].drop_duplicates().reset_index(drop=True)
            if record_count_needed:
                counts = merged.groupby(dim_cols, dropna=False).size().reset_index(name="record_count")
                result = result.merge(counts, on=dim_cols, how="left")
        else:
            summary: dict[str, Any] = {}
            for tgt_col, named_agg in agg_kwargs.items():
                col_series = merged[named_agg.column]
                func = named_agg.aggfunc
                summary[tgt_col] = getattr(col_series, func)()
            if record_count_needed:
                summary["record_count"] = len(merged)
            result = pd.DataFrame([summary]) if summary else merged.copy()

        pk_name = pk_rows["target_column"].iloc[0] if not pk_rows.empty else "pk_gold_id"
        result = result.reset_index(drop=True)
        result.insert(0, pk_name, range(1, len(result) + 1))

        out_path = GOLD_DIR / f"{target_table}.parquet"
        result.to_parquet(out_path, index=False)
        output_paths.append(str(out_path))

    return output_paths


def _make_gold_tools(
    input_files: list[str],
    sttm_path: str,
    business_intent: str,
    run_id: str,
    scratchpad: dict[str, Any],
) -> list:
    @tool
    def inspect_task_tool() -> str:
        """Preview Silver Parquet schemas and the approved STTM rules grouped
        by Gold target table. Call this first."""
        sttm = pd.read_csv(sttm_path)
        preview = {
            "target_tables": sorted(sttm["target_table"].unique().tolist()),
            "rule_count": len(sttm),
            "silver_schemas": {
                Path(f).stem: list(pd.read_parquet(f).columns) for f in input_files
            },
        }
        return json.dumps(preview, default=str)

    @tool
    def gold_ingestion_tool(confirmation: str = "execute") -> str:
        """Execute the approved Gold STTM rules against the Silver Parquet
        files: joins, groupby aggregations, and surrogate key injection, then
        write one Parquet file per Gold target table."""
        output_paths = _apply_gold_rules(input_files, sttm_path, business_intent, run_id)
        scratchpad["output_paths"] = output_paths
        return json.dumps({"output_paths": output_paths})

    return [inspect_task_tool, gold_ingestion_tool]


def execute_gold(
    input_files: list[str],
    sttm_path: str,
    business_intent: str,
    run_id: str,
    task_description: str,
) -> list[str]:
    """
    Run the Gold agent to materialise `input_files` (Silver Parquet paths) per
    the approved STTM at `sttm_path`. Returns Gold Parquet output paths. Falls
    back to deterministic execution if the LLM fails to call the required tool.
    """
    audit = AuditLogger(run_id)
    audit.start("gold_materialisation", {"input_files": input_files, "sttm_path": sttm_path})

    scratchpad: dict[str, Any] = {}
    llm = make_llm()
    tools = _make_gold_tools(input_files, sttm_path, business_intent, run_id, scratchpad)

    user_message = (
        f"Business question: {business_intent}\n"
        f"Task: {task_description}\n"
        f"Materialise Gold Parquet tables from the Silver Parquet files using the approved STTM."
    )

    run_react_agent(
        agent_name="gold_agent",
        run_id=run_id,
        llm=llm,
        tools=tools,
        system_prompt=SYSTEM_PROMPT,
        user_message=user_message,
        input_summary={"input_files": input_files, "sttm_path": sttm_path},
    )

    if "output_paths" not in scratchpad:
        scratchpad["output_paths"] = _apply_gold_rules(input_files, sttm_path, business_intent, run_id)

    audit.complete("gold_materialisation", {"output_paths": scratchpad["output_paths"]})
    return scratchpad["output_paths"]
