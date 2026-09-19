"""
agents/profiler.py
-------------------
Data Profiler agent. Profiles raw CSV files to understand structure,
semantics, and quality before any transformation rules are written.

Tools:
- inspect_files_tool  — lightweight preview (shape, columns, dtypes, 3 samples)
- profiler_tool        — full statistics (null count/%, unique count, min/max/mean)

Entry point:
    profile_datasets(input_files, run_id, business_intent, task_description) -> str
    (returns path to data/profiles/profile_combined_{run_id[:8]}.json)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from langchain_core.tools import tool

from core.audit import AuditLogger
from core.config import PROFILES_DIR, make_llm
from core.memory import memory_store
from core.observability import run_react_agent

SYSTEM_PROMPT = """You are the Data Profiler agent in an agentic Medallion data \
engineering pipeline. Your job is to understand the shape, semantics, and \
quality of raw CSV files before any transformation rules are written.

Follow this loop strictly: THINK -> INSPECT -> PLAN -> ACT -> VERIFY.
1. THINK about the business question you were given.
2. INSPECT the files first by calling `inspect_files_tool`. Never skip this step.
3. PLAN out loud: state what you observed and what full statistics you need.
4. ACT by calling `profiler_tool` to compute full statistics and persist the profile.
5. VERIFY by reporting the profile file path and a one-paragraph summary of data \
quality issues you noticed (nulls, mixed types, likely join keys).

Only call each tool once. Do not fabricate file paths — the tools return the \
authoritative path themselves.
"""


def _read_csv_safely(path: str) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)


def _inspect_files(input_files: list[str]) -> dict[str, Any]:
    preview: dict[str, Any] = {}
    for file_path in input_files:
        df = _read_csv_safely(file_path)
        samples = {
            col: df[col].dropna().astype(str).unique()[:3].tolist() for col in df.columns
        }
        preview[Path(file_path).stem] = {
            "file_path": file_path,
            "shape": {"rows": int(df.shape[0]), "columns": int(df.shape[1])},
            "columns": list(df.columns),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "sample_values": samples,
        }
    return preview


def _compute_full_profile(input_files: list[str]) -> dict[str, Any]:
    profile: dict[str, Any] = {}
    for file_path in input_files:
        df = _read_csv_safely(file_path)
        table_name = Path(file_path).stem
        column_stats: dict[str, Any] = {}
        for col in df.columns:
            series = df[col]
            null_count = int(series.isna().sum())
            stats: dict[str, Any] = {
                "dtype": str(series.dtype),
                "null_count": null_count,
                "null_pct": round(null_count / max(len(series), 1) * 100, 2),
                "unique_count": int(series.nunique(dropna=True)),
            }
            if pd.api.types.is_numeric_dtype(series):
                stats.update(
                    {
                        "min": float(series.min()) if series.notna().any() else None,
                        "max": float(series.max()) if series.notna().any() else None,
                        "mean": float(series.mean()) if series.notna().any() else None,
                    }
                )
            column_stats[col] = stats

        likely_keys = [c for c in df.columns if c.lower().endswith("_id") or c.lower() == "id"]

        profile[table_name] = {
            "file_path": file_path,
            "row_count": int(df.shape[0]),
            "column_count": int(df.shape[1]),
            "columns": column_stats,
            "likely_join_keys": likely_keys,
        }
    return profile


def _make_profiler_tools(
    input_files: list[str], run_id: str, scratchpad: dict[str, Any]
) -> list:
    @tool
    def inspect_files_tool() -> str:
        """Preview every raw CSV file: shape, column names, dtypes, and 3 sample
        values per column. Call this first, before profiler_tool."""
        preview = _inspect_files(input_files)
        return json.dumps(preview, default=str)

    @tool
    def profiler_tool() -> str:
        """Compute full column-level statistics (null count/%, unique count,
        min/max/mean for numeric columns) across all files, persist the combined
        profile to disk, and return its path. Call this after inspect_files_tool."""
        profile = _compute_full_profile(input_files)
        out_path = PROFILES_DIR / f"profile_combined_{run_id[:8]}.json"
        out_path.write_text(json.dumps(profile, indent=2, default=str), encoding="utf-8")
        scratchpad["profile_path"] = str(out_path)
        scratchpad["profile"] = profile
        return json.dumps({"profile_path": str(out_path)})

    return [inspect_files_tool, profiler_tool]


def profile_datasets(
    input_files: list[str],
    run_id: str,
    business_intent: str,
    task_description: str,
) -> str:
    """
    Run the Profiler agent over `input_files` and return the path to the
    combined profile JSON. Falls back to deterministic execution if the LLM
    fails to call the required tool, so the pipeline never stalls.
    """
    audit = AuditLogger(run_id)
    audit.start("profiler", {"input_files": input_files})

    scratchpad: dict[str, Any] = {}
    llm = make_llm()
    tools = _make_profiler_tools(input_files, run_id, scratchpad)

    user_message = (
        f"Business question: {business_intent}\n"
        f"Task: {task_description}\n"
        f"Files to profile: {input_files}"
    )

    result = run_react_agent(
        agent_name="profiler",
        run_id=run_id,
        llm=llm,
        tools=tools,
        system_prompt=SYSTEM_PROMPT,
        user_message=user_message,
        input_summary={"input_files": input_files, "business_intent": business_intent},
    )

    if "profile_path" not in scratchpad:
        # Deterministic fallback — guarantees the phase always produces output
        # even if the LLM never invoked profiler_tool.
        profile = _compute_full_profile(input_files)
        out_path = PROFILES_DIR / f"profile_combined_{run_id[:8]}.json"
        out_path.write_text(json.dumps(profile, indent=2, default=str), encoding="utf-8")
        scratchpad["profile_path"] = str(out_path)

    memory_store.add(
        f"{run_id}_profile_summary",
        result.get("final_text", ""),
        {"run_id": run_id, "phase": "profiler"},
    )

    audit.complete("profiler", {"profile_path": scratchpad["profile_path"]})
    return scratchpad["profile_path"]
