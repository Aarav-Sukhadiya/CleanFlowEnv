"""
MCP Tool Server for CleanFlowEnv.

Exposes cleaning actions as MCP tools so any MCP-compatible LLM agent
(Claude, GPT, etc.) can interact with the environment directly.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastmcp import FastMCP

from cleanflow_env.env.environment import CleanFlowEnv, build_observation
from cleanflow_env.env.grader import final_score
from cleanflow_env.tasks.task_easy import generate_easy_task
from cleanflow_env.tasks.task_expert import generate_expert_task
from cleanflow_env.tasks.task_hard import generate_hard_task
from cleanflow_env.tasks.task_medium import generate_medium_task
from cleanflow_env.tasks.task_multi import generate_multi_task

TASK_REGISTRY = {
    "task_easy": generate_easy_task,
    "task_medium": generate_medium_task,
    "task_hard": generate_hard_task,
    "task_expert": generate_expert_task,
    "task_multi": generate_multi_task,
}

mcp = FastMCP(
    "CleanFlowEnv",
    instructions="MCP tools for an OpenEnv data-cleaning environment. "
    "Reset a task, apply cleaning actions, and check scores.",
)

# Shared environment instance
_env = CleanFlowEnv(task_registry=TASK_REGISTRY)


# ── Resources ────────────────────────────────────────────────────────────────


@mcp.resource("cleanflow://tasks")
def list_tasks() -> str:
    """List all available cleaning tasks with descriptions."""
    tasks = [
        ("task_easy", "Basic Cleaning", "Easy — Employee survey, 200 rows, nulls + duplicates"),
        ("task_medium", "Schema Normalization", "Medium — Transactions, 300 rows, mixed formats"),
        ("task_hard", "Advanced Cleaning", "Hard — Medical trials, 400 rows, outliers + typos"),
        ("task_expert", "Budget-Constrained", "Expert — E-commerce, 500 rows, tight budget"),
        ("task_multi", "Multi-Table Cleaning", "Expert+ — Customers + Orders, FK relationships"),
    ]
    lines = ["Available Tasks:", ""]
    for tid, name, desc in tasks:
        lines.append(f"- {tid}: {name} — {desc}")
    return "\n".join(lines)


@mcp.resource("cleanflow://actions")
def list_actions() -> str:
    """List all available cleaning action types with costs."""
    actions = [
        ("fill_null", 1, "Fill missing values (method: mean/median/mode/constant/forward_fill/backward_fill/sequential)"),
        ("drop_duplicates", 1, "Remove all fully duplicate rows"),
        ("strip_whitespace", 1, "Strip leading/trailing whitespace from a string column"),
        ("replace_substring", 1, "Replace a substring in string values (e.g. remove '$')"),
        ("convert_type", 2, "Convert column dtype (target_type: int/float/datetime/string)"),
        ("map_values", 2, "Map categorical values (e.g. yes/no -> True/False)"),
        ("normalize", 2, "Scale column values (method: minmax/zscore)"),
        ("validate_foreign_key", 2, "Remove rows with orphan FK references (multi-table)"),
        ("lookup_fill", 2, "Fill nulls via FK lookup from another table (multi-table)"),
        ("remove_outliers", 3, "Remove outliers using IQR x 1.5 rule"),
    ]
    lines = ["Available Actions:", ""]
    for name, cost, desc in actions:
        lines.append(f"- {name} (cost: {cost}) — {desc}")
    return "\n".join(lines)


# ── Tools ────────────────────────────────────────────────────────────────────


@mcp.tool()
def reset_environment(task_id: str = "task_easy") -> Dict[str, Any]:
    """
    Reset the environment and start a new cleaning episode.

    Args:
        task_id: One of task_easy, task_medium, task_hard, task_expert, task_multi
    """
    if task_id not in TASK_REGISTRY:
        return {"error": f"Unknown task: {task_id}. Use one of: {list(TASK_REGISTRY.keys())}"}

    obs = _env.reset(task_id)
    obs_dict = obs.model_dump()

    return {
        "status": "reset",
        "task_id": task_id,
        "rows": obs_dict.get("stats", {}).get("row_count", "?"),
        "columns": list(obs_dict.get("table_schema", {}).keys()),
        "null_counts": obs_dict.get("null_counts", {}),
        "duplicate_count": obs_dict.get("duplicate_count", 0),
        "budget_remaining": obs_dict.get("budget_remaining", 0),
        "column_descriptions": obs_dict.get("column_descriptions", {}),
        "hint": "Use apply_action to clean the data. Check get_status for current state.",
    }


@mcp.tool()
def apply_action(
    action_type: str,
    column: Optional[str] = None,
    method: Optional[str] = None,
    constant_value: Optional[str] = None,
    old_value: Optional[str] = None,
    new_value: Optional[str] = None,
    target_type: Optional[str] = None,
    mapping: Optional[Dict[str, Any]] = None,
    table: Optional[str] = None,
    foreign_key_column: Optional[str] = None,
    lookup_table: Optional[str] = None,
    lookup_key_column: Optional[str] = None,
    lookup_value_column: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Apply a cleaning action to the dataset.

    Args:
        action_type: One of fill_null, drop_duplicates, strip_whitespace, replace_substring,
                     convert_type, map_values, normalize, remove_outliers,
                     validate_foreign_key, lookup_fill
        column: Target column name (required for most actions)
        method: Fill method (mean/median/mode/constant/forward_fill/backward_fill/sequential)
                or normalize method (minmax/zscore)
        constant_value: Value to use with method="constant"
        old_value: Substring to find (for replace_substring)
        new_value: Replacement string (for replace_substring)
        target_type: Target dtype (int/float/datetime/string) for convert_type
        mapping: Value mapping dict for map_values (e.g. {"yes": true, "no": false})
        table: Target table name (for multi-table tasks)
        foreign_key_column: FK column in source table
        lookup_table: Referenced table name
        lookup_key_column: Key column in lookup table
        lookup_value_column: Value column to pull from lookup table
    """
    if _env._state is None:
        return {"error": "No active episode. Call reset_environment first."}

    action_dict: Dict[str, Any] = {"action_type": action_type}
    if column is not None:
        action_dict["column"] = column
    if method is not None:
        action_dict["method"] = method
    if constant_value is not None:
        action_dict["constant_value"] = constant_value
    if old_value is not None:
        action_dict["old_value"] = old_value
    if new_value is not None:
        action_dict["new_value"] = new_value
    if target_type is not None:
        action_dict["target_type"] = target_type
    if mapping is not None:
        action_dict["mapping"] = mapping
    if table is not None:
        action_dict["table"] = table
    if foreign_key_column is not None:
        action_dict["foreign_key_column"] = foreign_key_column
    if lookup_table is not None:
        action_dict["lookup_table"] = lookup_table
    if lookup_key_column is not None:
        action_dict["lookup_key_column"] = lookup_key_column
    if lookup_value_column is not None:
        action_dict["lookup_value_column"] = lookup_value_column

    try:
        obs, reward = _env.step(action_dict)
        obs_dict = obs.model_dump()
        return {
            "success": True,
            "reward": round(reward.reward, 4),
            "cumulative_quality": round(reward.cumulative_quality, 4),
            "done": reward.done,
            "null_counts": obs_dict.get("null_counts", {}),
            "duplicate_count": obs_dict.get("duplicate_count", 0),
            "budget_remaining": obs_dict.get("budget_remaining", 0),
            "step_count": obs_dict.get("step_count", 0),
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@mcp.tool()
def get_status() -> Dict[str, Any]:
    """Get current environment status: null counts, duplicates, budget, schema, and column hints."""
    if _env._state is None:
        return {"error": "No active episode. Call reset_environment first."}

    state = _env._state
    obs = build_observation(state)
    obs_dict = obs.model_dump()

    result = {
        "task_id": state.task_id,
        "step_count": state.step_count,
        "budget_remaining": state.budget_remaining,
        "null_counts": obs_dict.get("null_counts", {}),
        "duplicate_count": obs_dict.get("duplicate_count", 0),
        "table_schema": obs_dict.get("table_schema", {}),
        "column_descriptions": obs_dict.get("column_descriptions", {}),
    }

    if state.is_multi_table:
        result["tables"] = {}
        for name, t_obs in (obs_dict.get("tables") or {}).items():
            result["tables"][name] = {
                "null_counts": t_obs.get("null_counts", {}),
                "duplicate_count": t_obs.get("duplicate_count", 0),
                "table_schema": t_obs.get("table_schema", {}),
            }
        result["table_relationships"] = obs_dict.get("table_relationships", [])

    return result


@mcp.tool()
def get_score() -> Dict[str, Any]:
    """Get the current grading score for the episode."""
    if _env._state is None:
        return {"error": "No active episode. Call reset_environment first."}

    result = final_score(_env._state)
    # Remove validation_details to keep response clean
    result.pop("validation_details", None)
    return {k: round(v, 4) if isinstance(v, float) else v for k, v in result.items()}


@mcp.tool()
def get_data_preview(table_name: Optional[str] = None, rows: int = 5) -> Dict[str, Any]:
    """
    Preview the current state of the data (first N rows).

    Args:
        table_name: For multi-table tasks, specify which table. None = primary table.
        rows: Number of rows to preview (default 5, max 20).
    """
    if _env._state is None:
        return {"error": "No active episode. Call reset_environment first."}

    rows = min(max(1, rows), 20)

    if table_name and _env._state.is_multi_table:
        df = _env._state.get_table(table_name)
        if df is None:
            return {"error": f"Unknown table: {table_name}"}
    else:
        df = _env._state.current_table

    preview = df.head(rows).fillna("NULL").to_dict(orient="records")
    return {
        "table": table_name or _env._state.primary_table or "main",
        "total_rows": len(df),
        "preview": preview,
    }
