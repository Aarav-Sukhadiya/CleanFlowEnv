from __future__ import annotations

from cleanflow_env.models.action import ActionModel

BUDGET_COSTS: dict[str, int] = {
    "fill_null": 1,
    "drop_duplicates": 1,
    "convert_type": 2,
    "normalize": 2,
    "remove_outliers": 3,
    "strip_whitespace": 1,
    "map_values": 2,
    "replace_substring": 1,
    "standardize_format": 2,
    "lookup_fill": 2,
    "validate_foreign_key": 2,
}

TASK_BUDGETS: dict[str, int] = {
    "task_easy": 20,
    "task_medium": 20,
    "task_hard": 20,
    "task_expert": 15,
    "task_multi": 25,
}


def get_action_cost(action: ActionModel) -> int:
    """Return the budget cost for a given action."""
    return BUDGET_COSTS.get(action.action_type, 1)
