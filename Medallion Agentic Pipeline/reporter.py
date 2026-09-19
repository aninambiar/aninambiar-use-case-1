"""
agents/reporter.py
---------------------
Reporter agent. Answers the business question by querying Gold tables with
SQL (via DuckDB) and produces a self-contained HTML executive report with
embedded Plotly charts, plus a structured JSON analysis.

Tools:
- inspect_gold_tables_tool — previews Gold table schemas and sample rows (pandas only)
- load_gold_data_tool      — registers Gold Parquet files as DuckDB views
- execute_query_tool       — runs agent-written SQL, returns JSON rows
- finalize_report_tool     — renders the HTML + JSON executive report

Entry point:
    generate_report(gold_files, business_intent, run_id, task_description) -> str
    (returns path to data/reports/report_{run_id[:8]}.html)
"""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd
import plotly.express as px
from langchain_core.tools import tool

from core.audit import AuditLogger
from core.config import REPORTS_DIR, make_llm
from core.observability import run_react_agent

SYSTEM_PROMPT = """You are the Reporter agent in an agentic Medallion data \
engineering pipeline. Your job is to answer the user's business question by \
querying Gold analytics tables with SQL and producing an executive report.

Follow this loop strictly: THINK -> INSPECT -> PLAN -> ACT -> VERIFY.
1. THINK about the exact business question you must answer.
2. INSPECT first by calling `inspect_gold_tables_tool` to see what Gold tables \
   and columns are available.
3. Call `load_gold_data_tool` to register the Gold tables in DuckDB.
4. PLAN out loud: state the SQL you will run and which chart(s) will best \
   communicate the answer (bar, line, pie, or scatter).
5. ACT by calling `execute_query_tool` with valid DuckDB SQL (you may call it \
   multiple times to explore the data) to gather the evidence you need.
6. VERIFY and finish by calling `finalize_report_tool` with:
   - direct_answer: a one or two sentence answer to the business question
   - detailed_analysis: a few paragraphs of supporting analysis
   - charts_json: a JSON array of chart specs, each with "title", "chart_type" \
     (bar|line|pie|scatter), "sql" (a DuckDB SELECT statement that returns the \
     data needed for the chart), "x" (column for the x-axis / labels), and "y" \
     (column for the y-axis / values)

Only call finalize_report_tool once, after you have queried the data.
Table names available in DuckDB match the Gold Parquet file names exactly.
"""


def _register_gold_tables(con: duckdb.DuckDBPyConnection, gold_files: list[str]) -> list[str]:
    table_names = []
    for path in gold_files:
        table_name = Path(path).stem
        con.execute(
            f"CREATE OR REPLACE VIEW {table_name} AS SELECT * FROM read_parquet('{path}')"
        )
        table_names.append(table_name)
    return table_names


def _dataframe_to_records(df: pd.DataFrame, max_rows: int = 200) -> list[dict[str, Any]]:
    return json.loads(df.head(max_rows).to_json(orient="records", date_format="iso"))


def _build_chart_html(chart_type: str, df: pd.DataFrame, x: str, y: str, title: str) -> str:
    chart_type = (chart_type or "bar").lower()
    if chart_type == "line":
        fig = px.line(df, x=x, y=y, title=title, markers=True)
    elif chart_type == "pie":
        fig = px.pie(df, names=x, values=y, title=title)
    elif chart_type == "scatter":
        fig = px.scatter(df, x=x, y=y, title=title)
    else:
        fig = px.bar(df, x=x, y=y, title=title)
    return fig.to_html(full_html=False, include_plotlyjs="cdn")


def _first_matching_column(columns: list[str], candidates: list[str]) -> str | None:
    lowered = {c.lower(): c for c in columns}
    for candidate in candidates:
        if candidate in lowered:
            return lowered[candidate]
    return None


def _question_aware_fallback_answer(
    con: duckdb.DuckDBPyConnection, table_name: str, business_intent: str
) -> tuple[str, str]:
    question = business_intent.lower()
    columns_df = con.execute(f"SELECT * FROM {table_name} LIMIT 1").fetchdf()
    columns = list(columns_df.columns)

    metric_col = _first_matching_column(
        columns,
        [
            "sum_sales_amount",
            "total_sales_amount",
            "sales_amount",
            "sum_revenue",
            "revenue",
            "amount",
            "sum_quantity",
            "quantity",
        ],
    )
    group_col = _first_matching_column(
        columns,
        [
            "product_name",
            "product",
            "item_name",
            "category",
            "store_name",
            "region",
        ],
    )

    wants_highest = any(token in question for token in ["highest", "top", "most", "maximum", "max"])
    wants_lowest = any(token in question for token in ["lowest", "least", "minimum", "min"])

    if metric_col and group_col and (wants_highest or wants_lowest):
        sort_dir = "DESC" if wants_highest else "ASC"
        rank_label = "highest" if wants_highest else "lowest"
        sql = (
            f"SELECT {group_col} AS dimension, SUM({metric_col}) AS metric "
            f"FROM {table_name} "
            f"GROUP BY {group_col} "
            f"ORDER BY metric {sort_dir} "
            f"LIMIT 1"
        )
        top_df = con.execute(sql).fetchdf()
        if not top_df.empty:
            dimension = str(top_df.iloc[0]["dimension"])
            metric = top_df.iloc[0]["metric"]
            direct_answer = (
                f"Based on {table_name}, the {rank_label} {metric_col.replace('_', ' ')} "
                f"is for {dimension} at {metric}."
            )
            detailed_analysis = (
                f"Fallback mode was used because the reporting agent did not finalise. "
                f"Computed answer with SQL: {sql}."
            )
            return direct_answer, detailed_analysis

    sample_df = con.execute(f"SELECT * FROM {table_name} LIMIT 5").fetchdf()
    sample_rows = _dataframe_to_records(sample_df, max_rows=5)
    direct_answer = (
        f"Unable to derive a deterministic metric-specific answer for this question from {table_name}. "
        f"Sample final output rows: {json.dumps(sample_rows, default=str)}"
    )
    detailed_analysis = (
        "Fallback mode was used because the reporting agent did not finalise, "
        "and the question could not be mapped to a deterministic aggregate rule."
    )
    return direct_answer, detailed_analysis


def _render_html_report(
    business_intent: str,
    direct_answer: str,
    detailed_analysis: str,
    chart_blocks: list[str],
    run_id: str,
) -> str:
    safe_business_intent = html.escape(business_intent)
    safe_direct_answer = html.escape(direct_answer)
    safe_detailed_analysis = html.escape(detailed_analysis)
    charts_html = "\n".join(f'<div class="chart">{c}</div>' for c in chart_blocks)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>Executive Report — {run_id[:8]}</title>
<style>
  body {{ font-family: -apple-system, Segoe UI, Roboto, Arial, sans-serif; margin: 40px; color: #1a1a1a; background: #fafafa; }}
  h1 {{ font-size: 26px; }}
  .question {{ color: #555; font-style: italic; margin-bottom: 24px; }}
  .answer {{ background: #eef6ff; border-left: 4px solid #2563eb; padding: 16px 20px; margin: 20px 0; border-radius: 4px; font-size: 18px; }}
  .analysis {{ line-height: 1.6; white-space: pre-wrap; margin-bottom: 32px; }}
  .chart {{ margin-bottom: 32px; background: white; border-radius: 8px; padding: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
  footer {{ margin-top: 40px; color: #999; font-size: 12px; }}
</style>
</head>
<body>
  <h1>📊 Executive Report</h1>
    <div class="question">Business question: {safe_business_intent}</div>
    <div class="answer"><strong>Direct answer:</strong> {safe_direct_answer}</div>
  <h2>Detailed Analysis</h2>
    <div class="analysis">{safe_detailed_analysis}</div>
  <h2>Charts</h2>
  {charts_html}
  <footer>Generated by the Agentic Medallion Pipeline — run {run_id}</footer>
</body>
</html>"""


def _make_reporter_tools(
    gold_files: list[str],
    business_intent: str,
    run_id: str,
    scratchpad: dict[str, Any],
) -> list:
    @tool
    def inspect_gold_tables_tool() -> str:
        """Preview Gold table schemas, dtypes, and 5 sample rows for every
        Gold Parquet file, without using DuckDB. Call this first."""
        preview = {}
        for path in gold_files:
            df = pd.read_parquet(path)
            preview[Path(path).stem] = {
                "columns": list(df.columns),
                "dtypes": {c: str(t) for c, t in df.dtypes.items()},
                "sample_rows": _dataframe_to_records(df, max_rows=5),
            }
        return json.dumps(preview, default=str)

    @tool
    def load_gold_data_tool() -> str:
        """Register every Gold Parquet file as a queryable DuckDB table/view.
        Call this before execute_query_tool."""
        con = duckdb.connect(database=":memory:")
        table_names = _register_gold_tables(con, gold_files)
        scratchpad["con"] = con
        scratchpad["table_names"] = table_names
        return json.dumps({"registered_tables": table_names})

    @tool
    def execute_query_tool(sql_query: str) -> str:
        """Execute a DuckDB SELECT statement against the registered Gold
        tables and return up to 200 result rows as JSON."""
        con = scratchpad.get("con")
        if con is None:
            con = duckdb.connect(database=":memory:")
            scratchpad["table_names"] = _register_gold_tables(con, gold_files)
            scratchpad["con"] = con
        df = con.execute(sql_query).fetchdf()
        records = _dataframe_to_records(df)
        scratchpad.setdefault("query_history", []).append({"sql": sql_query, "rows": len(df)})
        return json.dumps(records, default=str)

    @tool
    def finalize_report_tool(direct_answer: str, detailed_analysis: str, charts_json: str) -> str:
        """Render and persist the final HTML + JSON executive report. Call
        this exactly once, after gathering evidence with execute_query_tool.
        `charts_json` must be a JSON array of {title, chart_type, sql, x, y}."""
        con = scratchpad.get("con")
        if con is None:
            con = duckdb.connect(database=":memory:")
            _register_gold_tables(con, gold_files)
            scratchpad["con"] = con

        try:
            chart_specs = json.loads(charts_json)
        except json.JSONDecodeError:
            chart_specs = []

        chart_blocks: list[str] = []
        chart_data_snapshot: list[dict[str, Any]] = []
        for spec in chart_specs:
            try:
                df = con.execute(spec["sql"]).fetchdf()
                html = _build_chart_html(
                    spec.get("chart_type", "bar"), df, spec["x"], spec["y"], spec.get("title", "")
                )
                chart_blocks.append(html)
                chart_data_snapshot.append(
                    {
                        "title": spec.get("title", ""),
                        "chart_type": spec.get("chart_type", "bar"),
                        "sql": spec["sql"],
                        "rows": _dataframe_to_records(df, max_rows=50),
                    }
                )
            except Exception as exc:  # noqa: BLE001 - one bad chart spec must not fail the report
                chart_blocks.append(f"<p><em>Chart '{spec.get('title', '?')}' failed: {exc}</em></p>")

        html_report = _render_html_report(
            business_intent, direct_answer, detailed_analysis, chart_blocks, run_id
        )
        html_path = REPORTS_DIR / f"report_{run_id[:8]}.html"
        html_path.write_text(html_report, encoding="utf-8")

        json_report = {
            "run_id": run_id,
            "business_intent": business_intent,
            "direct_answer": direct_answer,
            "detailed_analysis": detailed_analysis,
            "charts": chart_data_snapshot,
        }
        json_path = REPORTS_DIR / f"report_{run_id[:8]}.json"
        json_path.write_text(json.dumps(json_report, indent=2, default=str), encoding="utf-8")

        scratchpad["report_html_path"] = str(html_path)
        scratchpad["report_json_path"] = str(json_path)
        return json.dumps({"report_html_path": str(html_path), "report_json_path": str(json_path)})

    return [inspect_gold_tables_tool, load_gold_data_tool, execute_query_tool, finalize_report_tool]


def _fallback_report(gold_files: list[str], business_intent: str, run_id: str) -> str:
    """Deterministic minimal report used only if the LLM never finalises one."""
    con = duckdb.connect(database=":memory:")
    table_names = _register_gold_tables(con, gold_files)
    direct_answer = "No Gold output available to summarise."
    analysis_suffix = (
        "This is a fallback summary generated deterministically because the reporting "
        "agent did not finalise a report."
    )
    chart_blocks: list[str] = []
    chart_data_snapshot: list[dict[str, Any]] = []
    if table_names:
        df = con.execute(f"SELECT * FROM {table_names[0]} LIMIT 20").fetchdf()
        direct_answer, answer_analysis = _question_aware_fallback_answer(
            con, table_names[0], business_intent
        )
        analysis_suffix = f"{analysis_suffix} {answer_analysis}"
        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c != df.columns[0]]
        non_numeric_cols = [c for c in df.columns if c not in numeric_cols]
        if numeric_cols and non_numeric_cols:
            x, y = non_numeric_cols[0], numeric_cols[0]
            html = _build_chart_html("bar", df, x, y, f"{y} by {x}")
            chart_blocks.append(html)
            chart_data_snapshot.append(
                {"title": f"{y} by {x}", "chart_type": "bar", "sql": f"SELECT * FROM {table_names[0]} LIMIT 20", "rows": _dataframe_to_records(df)}
            )
    detailed_analysis = (
        f"Gold tables available: {', '.join(table_names) if table_names else 'none'}. "
        f"{analysis_suffix}"
    )
    html_report = _render_html_report(business_intent, direct_answer, detailed_analysis, chart_blocks, run_id)
    html_path = REPORTS_DIR / f"report_{run_id[:8]}.html"
    html_path.write_text(html_report, encoding="utf-8")
    json_path = REPORTS_DIR / f"report_{run_id[:8]}.json"
    json_path.write_text(
        json.dumps(
            {
                "run_id": run_id,
                "business_intent": business_intent,
                "direct_answer": direct_answer,
                "detailed_analysis": detailed_analysis,
                "charts": chart_data_snapshot,
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    return str(html_path)


def generate_report(
    gold_files: list[str], business_intent: str, run_id: str, task_description: str
) -> str:
    """
    Run the Reporter agent to answer `business_intent` from `gold_files` and
    return the path to the generated HTML report. Falls back to a
    deterministic minimal report if the LLM never calls finalize_report_tool.
    """
    audit = AuditLogger(run_id)
    audit.start("reporting", {"gold_files": gold_files, "business_intent": business_intent})

    scratchpad: dict[str, Any] = {}
    llm = make_llm()
    tools = _make_reporter_tools(gold_files, business_intent, run_id, scratchpad)

    user_message = (
        f"Business question: {business_intent}\n"
        f"Task: {task_description}\n"
        f"Gold tables are ready. Query them and produce the executive report."
    )

    run_react_agent(
        agent_name="reporter",
        run_id=run_id,
        llm=llm,
        tools=tools,
        system_prompt=SYSTEM_PROMPT,
        user_message=user_message,
        input_summary={"gold_files": gold_files, "business_intent": business_intent},
    )

    if "report_html_path" not in scratchpad:
        scratchpad["report_html_path"] = _fallback_report(gold_files, business_intent, run_id)

    audit.complete("reporting", {"report_html_path": scratchpad["report_html_path"]})
    return scratchpad["report_html_path"]
