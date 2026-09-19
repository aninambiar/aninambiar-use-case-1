"""
agents/sttm_generator.py
-------------------------
Unified STTM (Source-to-Target Mapping) agent. Generates transformation
rules for all three Medallion layers. The Orchestrator tells this agent
which layer to generate for; the agent inspects context and calls the
matching generation tool.

Tools:
- inspect_context_tool        — previews source data for the requested layer
- generate_bronze_sttm_tool   — ingestion rules (rename, type cast, metadata)
- generate_silver_sttm_tool   — cleansing rules (nulls, dedup, types, surrogate key)
- generate_gold_sttm_tool     — materialisation rules (joins, aggregations, surrogate key)

Entry points:
    generate_bronze_sttm(profile_path, business_intent, run_id, task_description) -> str
    generate_silver_sttm(bronze_output_paths, bronze_sttm_path, business_intent, run_id, task_description) -> str
    generate_gold_sttm(silver_output_paths, silver_sttm_path, business_intent, run_id, task_description) -> str

STTM CSV columns:
    source_schema, source_table, source_column, target_schema, target_table,
    target_column, transformation_type, transformation_logic
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd
from langchain_core.tools import tool

from core.audit import AuditLogger
from core.config import STTM_DIR, make_llm
from core.memory import memory_store
from core.observability import run_react_agent

SYSTEM_PROMPT = """You are the STTM (Source-to-Target Mapping) agent in an \
agentic Medallion data engineering pipeline. You generate the transformation \
recipe that specialist agents (Bronze, Silver, Gold) will execute exactly.

Follow this loop strictly: THINK -> INSPECT -> PLAN -> ACT -> VERIFY.
1. THINK about which Medallion layer you were asked to generate rules for.
2. INSPECT the source data first by calling `inspect_context_tool`. Never skip this.
3. PLAN out loud: state the mapping/cleansing/aggregation approach you will use.
4. ACT by calling exactly ONE of the generation tools matching the requested layer:
   - generate_bronze_sttm_tool for raw CSV -> Bronze Parquet ingestion rules
   - generate_silver_sttm_tool for Bronze -> Silver cleansing rules
   - generate_gold_sttm_tool for Silver -> Gold aggregation rules
5. VERIFY by reporting the STTM file path and a short summary of the rules generated.

Never fabricate file paths — the tools return the authoritative path themselves.
"""

STTM_COLUMNS = [
    "source_schema",
    "source_table",
    "source_column",
    "target_schema",
    "target_table",
    "target_column",
    "transformation_type",
    "transformation_logic",
]


def _snake_case(name: str) -> str:
    name = re.sub(r"[^0-9a-zA-Z]+", "_", str(name)).strip("_")
    return name.lower()


def _write_sttm(rows: list[dict[str, Any]], out_path: Path) -> None:
    df = pd.DataFrame(rows, columns=STTM_COLUMNS)
    df.to_csv(out_path, index=False)


# ---------------------------------------------------------------------------
# Bronze STTM generation
# ---------------------------------------------------------------------------
def _infer_bronze_cast(dtype: str, colname: str) -> tuple[str, str]:
    dtype_l = dtype.lower()
    name_l = colname.lower()
    if "date" in name_l or "time" in name_l:
        return "cast_datetime", "pd.to_datetime(value, errors='coerce')"
    if "int" in dtype_l or "float" in dtype_l:
        return "cast_numeric", "pd.to_numeric(value, errors='coerce')"
    return "cast_string", "value.astype(str)"


def _generate_bronze_sttm_rows(profile: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for table_name, table_info in profile.items():
        target_table = f"{table_name}_bronze"
        for col, stats in table_info["columns"].items():
            transformation_type, logic = _infer_bronze_cast(stats["dtype"], col)
            rows.append(
                {
                    "source_schema": "raw",
                    "source_table": table_name,
                    "source_column": col,
                    "target_schema": "bronze",
                    "target_table": target_table,
                    "target_column": _snake_case(col),
                    "transformation_type": transformation_type,
                    "transformation_logic": logic,
                }
            )
        # Metadata columns injected by the Bronze agent on every table.
        for meta_col, logic in [
            ("_load_timestamp", "UTC ISO-8601 timestamp of ingestion"),
            ("_source_file", "original source CSV file path"),
        ]:
            rows.append(
                {
                    "source_schema": "raw",
                    "source_table": table_name,
                    "source_column": "",
                    "target_schema": "bronze",
                    "target_table": target_table,
                    "target_column": meta_col,
                    "transformation_type": "metadata_inject",
                    "transformation_logic": logic,
                }
            )
    return rows


# ---------------------------------------------------------------------------
# Silver STTM generation
# ---------------------------------------------------------------------------
def _generate_silver_sttm_rows(
    bronze_output_paths: list[str], profile: dict[str, Any]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    # Build a lookup of null_pct per (table, column) from the profile, keyed by
    # the original raw table name (bronze tables are named "{raw_table}_bronze").
    for bronze_path in bronze_output_paths:
        bronze_df = pd.read_parquet(bronze_path)
        bronze_table = Path(bronze_path).stem  # e.g. "sales_bronze"
        raw_table = re.sub(r"_bronze$", "", bronze_table)
        silver_table = f"{raw_table}_silver"
        raw_stats = profile.get(raw_table, {}).get("columns", {})

        for col in bronze_df.columns:
            if col in ("_load_timestamp", "_source_file"):
                continue  # dropped at Silver; Bronze metadata is not analytics-facing
            dtype = str(bronze_df[col].dtype)
            null_pct = raw_stats.get(col, {}).get("null_pct", 0.0)
            name_l = col.lower()

            if "date" in name_l or "time" in name_l:
                transformation_type, logic = "date_format", "standardise to YYYY-MM-DD"
            elif "int" in dtype or "float" in dtype:
                if null_pct > 20:
                    transformation_type, logic = "fillna_mean", "fill nulls with column mean"
                elif null_pct > 0:
                    transformation_type, logic = "dropna", "drop rows with nulls in this column"
                else:
                    transformation_type, logic = "type_cast", "retain numeric type"
            elif dtype == "object":
                if null_pct > 20:
                    transformation_type, logic = "fillna_mode", "fill nulls with column mode"
                elif null_pct > 0:
                    transformation_type, logic = "dropna", "drop rows with nulls in this column"
                else:
                    transformation_type, logic = "text_normalize", "strip whitespace, lower-case"
            else:
                transformation_type, logic = "type_cast", "retain existing type"

            rows.append(
                {
                    "source_schema": "bronze",
                    "source_table": bronze_table,
                    "source_column": col,
                    "target_schema": "silver",
                    "target_table": silver_table,
                    "target_column": _snake_case(col),
                    "transformation_type": transformation_type,
                    "transformation_logic": logic,
                }
            )

        rows.append(
            {
                "source_schema": "bronze",
                "source_table": bronze_table,
                "source_column": "",
                "target_schema": "silver",
                "target_table": silver_table,
                "target_column": f"pk_{raw_table}_silver_id",
                "transformation_type": "surrogate_key",
                "transformation_logic": "sequential surrogate key, inserted as first column",
            }
        )
        rows.append(
            {
                "source_schema": "bronze",
                "source_table": bronze_table,
                "source_column": "",
                "target_schema": "silver",
                "target_table": silver_table,
                "target_column": "__dedup__",
                "transformation_type": "dedup",
                "transformation_logic": "drop_duplicates() across all target columns",
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Gold STTM generation
# ---------------------------------------------------------------------------
_ID_PATTERN = re.compile(r"_id$", re.IGNORECASE)


def _generate_gold_sttm_rows(
    silver_output_paths: list[str], business_intent: str
) -> list[dict[str, Any]]:
    intent_l = business_intent.lower()
    tables: dict[str, pd.DataFrame] = {}
    for path in silver_output_paths:
        table_name = Path(path).stem  # e.g. "sales_silver"
        tables[table_name] = pd.read_parquet(path)

    # Identify join keys shared across 2+ tables.
    columns_by_table = {t: set(df.columns) for t, df in tables.items()}
    id_columns_seen: dict[str, list[str]] = {}
    for table, cols in columns_by_table.items():
        for col in cols:
            if _ID_PATTERN.search(col) and not col.startswith("pk_"):
                id_columns_seen.setdefault(col, []).append(table)
    join_keys = {col: tbls for col, tbls in id_columns_seen.items() if len(tbls) >= 2}

    gold_table = "sales_summary_gold"
    rows: list[dict[str, Any]] = []

    # Join key rows
    for join_col, tbls in join_keys.items():
        for table in tbls:
            rows.append(
                {
                    "source_schema": "silver",
                    "source_table": table,
                    "source_column": join_col,
                    "target_schema": "gold",
                    "target_table": gold_table,
                    "target_column": join_col,
                    "transformation_type": "join_key",
                    "transformation_logic": f"outer join across {', '.join(tbls)} on {join_col}",
                }
            )

    # Dimension (group-by) columns: prefer categorical columns mentioned in the
    # business question; fall back to the first categorical column per table.
    dimension_cols: list[tuple[str, str]] = []  # (table, column)
    for table, df in tables.items():
        # Non-numeric, non-key columns are treated as categorical dimensions.
        # Note: string columns round-tripped through Parquet may report as
        # dtype "object" or pandas' native "str" dtype depending on version —
        # checking is_numeric_dtype() is the version-safe discriminator.
        categorical = [
            c
            for c in df.columns
            if not pd.api.types.is_numeric_dtype(df[c])
            and not c.startswith("pk_")
            and not _ID_PATTERN.search(c)
        ]
        matched = [c for c in categorical if c.lower() in intent_l or any(
            word in c.lower() for word in intent_l.split() if len(word) > 3
        )]
        chosen = matched or categorical[:1]
        for c in chosen[:2]:
            dimension_cols.append((table, c))

    for table, col in dimension_cols:
        rows.append(
            {
                "source_schema": "silver",
                "source_table": table,
                "source_column": col,
                "target_schema": "gold",
                "target_table": gold_table,
                "target_column": col,
                "transformation_type": "groupby",
                "transformation_logic": "retained as a grouping dimension",
            }
        )

    # Measure (numeric) columns: aggregate with sum, avg, count.
    for table, df in tables.items():
        numeric_cols = [
            c
            for c in df.columns
            if pd.api.types.is_numeric_dtype(df[c]) and not c.startswith("pk_")
        ]
        for col in numeric_cols:
            for agg, agg_type in [
                ("sum", "aggregate_sum"),
                ("avg", "aggregate_avg"),
            ]:
                rows.append(
                    {
                        "source_schema": "silver",
                        "source_table": table,
                        "source_column": col,
                        "target_schema": "gold",
                        "target_table": gold_table,
                        "target_column": f"{agg}_{col}",
                        "transformation_type": agg_type,
                        "transformation_logic": f"{agg}({col}) grouped by dimension columns",
                    }
                )
        rows.append(
            {
                "source_schema": "silver",
                "source_table": table,
                "source_column": "",
                "target_schema": "gold",
                "target_table": gold_table,
                "target_column": "record_count",
                "transformation_type": "aggregate_count",
                "transformation_logic": "count(*) grouped by dimension columns",
            }
        )

    rows.append(
        {
            "source_schema": "gold",
            "source_table": "",
            "source_column": "",
            "target_schema": "gold",
            "target_table": gold_table,
            "target_column": "pk_gold_id",
            "transformation_type": "surrogate_key",
            "transformation_logic": "sequential surrogate key, inserted as first column",
        }
    )
    return rows


# ---------------------------------------------------------------------------
# Tool factory
# ---------------------------------------------------------------------------
def _make_sttm_tools(
    layer: str,
    run_id: str,
    business_intent: str,
    scratchpad: dict[str, Any],
    *,
    profile_path: str | None = None,
    bronze_output_paths: list[str] | None = None,
    silver_output_paths: list[str] | None = None,
) -> list:
    @tool
    def inspect_context_tool() -> str:
        """Preview the source data available for the requested STTM layer
        (columns, dtypes, row counts). Call this first."""
        if layer == "bronze" and profile_path:
            profile = json.loads(Path(profile_path).read_text(encoding="utf-8"))
            preview = {
                t: {"columns": list(info["columns"].keys()), "row_count": info["row_count"]}
                for t, info in profile.items()
            }
            return json.dumps(preview, default=str)
        if layer == "silver" and bronze_output_paths:
            preview = {
                Path(p).stem: {"columns": list(pd.read_parquet(p).columns)}
                for p in bronze_output_paths
            }
            return json.dumps(preview, default=str)
        if layer == "gold" and silver_output_paths:
            preview = {
                Path(p).stem: {"columns": list(pd.read_parquet(p).columns)}
                for p in silver_output_paths
            }
            return json.dumps(preview, default=str)
        return json.dumps({"error": "No source context available for this layer."})

    @tool
    def generate_bronze_sttm_tool() -> str:
        """Generate the Bronze ingestion STTM (rename, type cast, metadata
        injection) from the data profile and persist it as CSV."""
        profile = json.loads(Path(profile_path).read_text(encoding="utf-8"))
        rows = _generate_bronze_sttm_rows(profile)
        out_path = STTM_DIR / f"sttm_bronze_{run_id[:8]}.csv"
        _write_sttm(rows, out_path)
        scratchpad["sttm_path"] = str(out_path)
        return json.dumps({"sttm_path": str(out_path), "rule_count": len(rows)})

    @tool
    def generate_silver_sttm_tool() -> str:
        """Generate the Silver cleansing STTM (null handling, dedup, type
        standardisation, surrogate key) from Bronze outputs and persist it as CSV."""
        profile: dict[str, Any] = {}
        if profile_path and Path(profile_path).exists():
            profile = json.loads(Path(profile_path).read_text(encoding="utf-8"))
        rows = _generate_silver_sttm_rows(bronze_output_paths or [], profile)
        out_path = STTM_DIR / f"sttm_silver_{run_id[:8]}.csv"
        _write_sttm(rows, out_path)
        scratchpad["sttm_path"] = str(out_path)
        return json.dumps({"sttm_path": str(out_path), "rule_count": len(rows)})

    @tool
    def generate_gold_sttm_tool() -> str:
        """Generate the Gold materialisation STTM (joins, aggregations,
        surrogate key) from Silver outputs and the business question, and
        persist it as CSV."""
        rows = _generate_gold_sttm_rows(silver_output_paths or [], business_intent)
        out_path = STTM_DIR / f"sttm_gold_{run_id[:8]}.csv"
        _write_sttm(rows, out_path)
        scratchpad["sttm_path"] = str(out_path)
        return json.dumps({"sttm_path": str(out_path), "rule_count": len(rows)})

    return [
        inspect_context_tool,
        generate_bronze_sttm_tool,
        generate_silver_sttm_tool,
        generate_gold_sttm_tool,
    ]


def _run_sttm_agent(
    layer: str,
    run_id: str,
    business_intent: str,
    task_description: str,
    fallback_rows_fn,
    **context,
) -> str:
    audit = AuditLogger(run_id)
    audit.start(f"sttm_{layer}", {"layer": layer})

    scratchpad: dict[str, Any] = {}
    llm = make_llm()
    tools = _make_sttm_tools(layer, run_id, business_intent, scratchpad, **context)

    user_message = (
        f"Business question: {business_intent}\n"
        f"Task: {task_description}\n"
        f"Generate the STTM for the {layer.upper()} layer only, then stop."
    )

    run_react_agent(
        agent_name=f"sttm_{layer}",
        run_id=run_id,
        llm=llm,
        tools=tools,
        system_prompt=SYSTEM_PROMPT,
        user_message=user_message,
        input_summary={"layer": layer, "business_intent": business_intent},
    )

    if "sttm_path" not in scratchpad:
        rows = fallback_rows_fn()
        out_path = STTM_DIR / f"sttm_{layer}_{run_id[:8]}.csv"
        _write_sttm(rows, out_path)
        scratchpad["sttm_path"] = str(out_path)

    memory_store.add(
        f"{run_id}_sttm_{layer}",
        f"STTM generated for {layer}: {scratchpad['sttm_path']}",
        {"run_id": run_id, "phase": f"sttm_{layer}"},
    )
    audit.complete(f"sttm_{layer}", {"sttm_path": scratchpad["sttm_path"]})
    return scratchpad["sttm_path"]


def generate_bronze_sttm(
    profile_path: str, business_intent: str, run_id: str, task_description: str
) -> str:
    """Generate the Bronze STTM CSV and return its path."""
    profile = json.loads(Path(profile_path).read_text(encoding="utf-8"))
    return _run_sttm_agent(
        "bronze",
        run_id,
        business_intent,
        task_description,
        fallback_rows_fn=lambda: _generate_bronze_sttm_rows(profile),
        profile_path=profile_path,
    )


def generate_silver_sttm(
    bronze_output_paths: list[str],
    bronze_sttm_path: str,
    business_intent: str,
    run_id: str,
    task_description: str,
) -> str:
    """Generate the Silver STTM CSV and return its path."""
    profile_path = STTM_DIR.parent / "profiles" / f"profile_combined_{run_id[:8]}.json"
    profile = (
        json.loads(profile_path.read_text(encoding="utf-8")) if profile_path.exists() else {}
    )
    return _run_sttm_agent(
        "silver",
        run_id,
        business_intent,
        task_description,
        fallback_rows_fn=lambda: _generate_silver_sttm_rows(bronze_output_paths, profile),
        profile_path=str(profile_path) if profile_path.exists() else None,
        bronze_output_paths=bronze_output_paths,
    )


def generate_gold_sttm(
    silver_output_paths: list[str],
    silver_sttm_path: str,
    business_intent: str,
    run_id: str,
    task_description: str,
) -> str:
    """Generate the Gold STTM CSV and return its path."""
    return _run_sttm_agent(
        "gold",
        run_id,
        business_intent,
        task_description,
        fallback_rows_fn=lambda: _generate_gold_sttm_rows(silver_output_paths, business_intent),
        silver_output_paths=silver_output_paths,
    )
