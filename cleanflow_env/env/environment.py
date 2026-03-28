from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Optional, Tuple

import pandas as pd
from pydantic import ValidationError

from cleanflow_env.env.actions import InvalidActionError, apply_action
from cleanflow_env.env.budget import get_action_cost
from cleanflow_env.env.rewards import compute_quality, compute_reward, _is_redundant
from cleanflow_env.env.state import EnvironmentState
from cleanflow_env.models.action import ActionModel
from cleanflow_env.models.observation import ObservationModel, TablePreviewRow
from cleanflow_env.models.reward import RewardModel

logger = logging.getLogger("cleanflow")

MAX_STEPS = 20


def build_observation(state: EnvironmentState) -> ObservationModel:
    """
    Build an ObservationModel from the current environment state.

    All decision-critical fields (null_counts, duplicate_count, stats,
    distribution, schema) use current_table so the agent sees the true
    state and can react to changes (e.g. new duplicates created by filling).
    Table preview uses prev_table for before/after comparison in the UI.
    """
    prev = state.prev_table
    cur = state.current_table

    # Table preview: first 5 rows from prev_table (for before/after display)
    preview_rows = []
    for i, (idx, row) in enumerate(prev.head(5).iterrows()):
        values = {}
        for col in prev.columns:
            val = row[col]
            if pd.isna(val):
                values[col] = None
            else:
                # Convert numpy types to Python natives for JSON serialization
                if hasattr(val, "item"):
                    values[col] = val.item()
                else:
                    values[col] = val
        preview_rows.append(TablePreviewRow(row_index=int(idx), values=values))

    # Schema from current_table
    dtype_map = {
        "float64": "float",
        "float32": "float",
        "int64": "int",
        "int32": "int",
        "Int64": "int",
        "Int32": "int",
        "object": "string",
        "bool": "bool",
        "boolean": "bool",
        "category": "string",
    }
    schema = {}
    for col in cur.columns:
        dt = str(cur[col].dtype)
        if "datetime" in dt:
            schema[col] = "datetime"
        else:
            schema[col] = dtype_map.get(dt, "string")

    # Null counts from current_table (agent needs true state to decide next action)
    null_counts = {col: int(cur[col].isnull().sum()) for col in cur.columns}

    # Duplicate count from current_table
    duplicate_count = int(cur.duplicated().sum())

    # Stats from current_table (mean, std for numeric columns only)
    stats: Dict[str, float] = {}
    distribution: Dict[str, Dict[str, float]] = {}
    numeric_cols = cur.select_dtypes(include=["number"]).columns
    for col in numeric_cols:
        col_data = cur[col].dropna()
        if len(col_data) > 0:
            stats[f"{col}_mean"] = round(float(col_data.mean()), 4)
            stats[f"{col}_std"] = round(float(col_data.std()), 4)
            distribution[col] = {
                "min": round(float(col_data.min()), 4),
                "q1": round(float(col_data.quantile(0.25)), 4),
                "median": round(float(col_data.median()), 4),
                "q3": round(float(col_data.quantile(0.75)), 4),
                "max": round(float(col_data.max()), 4),
                "skew": round(float(col_data.skew()), 4),
            }

    return ObservationModel(
        table_preview=preview_rows,
        table_schema=schema,
        null_counts=null_counts,
        duplicate_count=duplicate_count,
        stats=stats,
        distribution=distribution,
        step_count=state.step_count,
        budget_remaining=state.budget_remaining,
        task_id=state.task_id,
        column_descriptions=state.column_descriptions,
    )


class CleanFlowEnv:
    """
    OpenEnv-compliant environment for data cleaning workflows.

    Provides reset(), step(), and state() methods.
    """

    def __init__(self, task_registry: Dict[str, Callable] | None = None) -> None:
        """Initialize with a task registry mapping task_id → generator function."""
        self.task_registry: Dict[str, Callable] = task_registry or {}
        self._state: Optional[EnvironmentState] = None
        self._initial_budget: Optional[int] = None

    def reset(self, task_id: str) -> ObservationModel:
        """
        Initialize the environment for a new episode.

        Loads the dataset for the given task_id and returns the initial observation.
        """
        if task_id not in self.task_registry:
            raise ValueError(
                f"Unknown task_id '{task_id}'. Available: {list(self.task_registry.keys())}"
            )

        task_fn = self.task_registry[task_id]
        raw_table, ground_truth, budget, column_descriptions = task_fn()

        self._state = EnvironmentState(
            task_id=task_id,
            raw_table=raw_table,
            ground_truth=ground_truth,
            budget=budget,
            column_descriptions=column_descriptions,
        )
        self._initial_budget = budget

        logger.info(f"Episode reset for task {task_id}")
        return build_observation(self._state)

    def step(self, action_dict: dict) -> Tuple[ObservationModel, RewardModel]:
        """
        Apply an action and return the new observation and reward.

        Never raises unhandled exceptions — errors become penalty rewards.
        """
        if self._state is None:
            raise RuntimeError("Call reset() before step().")

        state = self._state

        # Parse action
        try:
            action = ActionModel(**action_dict)
        except (ValidationError, Exception) as e:
            logger.warning(f"Invalid action attempted: {e}")
            # Invalid action penalty — still costs 1 budget
            state.step_count += 1
            state.budget_remaining = max(0, state.budget_remaining - 1)
            state.operations_history.append(
                {**action_dict, "valid": False, "error": str(e)}
            )
            normalized_cost = 1 / max(state.initial_budget, 1)
            done = state.step_count >= MAX_STEPS or state.budget_remaining == 0
            reward = RewardModel.from_step(
                quality_delta=0.0,
                penalty=0.5,
                budget_cost=round(normalized_cost, 6),
                cumulative_quality=state.best_quality_so_far,
                done=done,
                info={"error": str(e), "action": action_dict},
            )
            return build_observation(state), reward

        # Get budget cost
        cost = get_action_cost(action)

        # Budget check
        if cost > state.budget_remaining:
            logger.warning(f"Budget exhausted at step {state.step_count}")
            reward = RewardModel.from_step(
                quality_delta=0.0,
                penalty=0.5,
                budget_cost=0.0,
                cumulative_quality=state.best_quality_so_far,
                done=True,
                info={"reason": "budget_exhausted"},
            )
            return build_observation(state), reward

        # Update prev_table BEFORE applying action
        state.prev_table = state.current_table.copy()

        # Apply action
        invalid = False
        try:
            state.current_table = apply_action(state.current_table, action)
        except InvalidActionError as e:
            logger.warning(f"Action failed: {e}")
            invalid = True
            # Revert — keep current_table as-is (action didn't apply)
            state.current_table = state.prev_table.copy()

        # Update state
        state.step_count += 1
        state.budget_remaining -= cost
        state.operations_history.append(
            {
                "action_type": action.action_type,
                "column": action.column,
                "method": action.method,
                "step": state.step_count,
                "valid": not invalid,
            }
        )

        # Compute reward using calibrated formula
        reward, new_best = compute_reward(
            current=state.current_table,
            ground_truth=state.ground_truth,
            best_quality_so_far=state.best_quality_so_far,
            action_dict=action.model_dump(),
            budget_cost=cost,
            initial_budget=state.initial_budget,
            operations_history=state.operations_history[:-1],  # exclude current
            invalid=invalid,
        )
        state.best_quality_so_far = new_best

        # Check termination
        done = (
            state.step_count >= MAX_STEPS
            or state.budget_remaining <= 0
            or reward.cumulative_quality >= 1.0
        )

        # Update done flag on the reward
        if done and not reward.done:
            reward = RewardModel.from_step(
                quality_delta=reward.quality_delta,
                penalty=reward.penalty,
                budget_cost=reward.budget_cost,
                cumulative_quality=reward.cumulative_quality,
                done=True,
                info=reward.info,
            )

        logger.info(
            f"Step {state.step_count}: action={action.action_type} "
            f"col={action.column} reward={reward.reward:+.3f}"
        )

        return build_observation(state), reward

    def state(self) -> Dict[str, Any]:
        """Return the current internal state as a serializable dict."""
        if self._state is None:
            return {"status": "not_initialized"}
        return self._state.to_dict()

    @property
    def current_quality(self) -> float:
        """Compute and return the current overall quality score."""
        if self._state is None:
            return 0.0
        q = compute_quality(self._state.current_table, self._state.ground_truth)
        return q["overall"]

    @property
    def is_done(self) -> bool:
        """Check if the current episode has ended."""
        if self._state is None:
            return True
        return (
            self._state.step_count >= MAX_STEPS
            or self._state.budget_remaining <= 0
            or self.current_quality >= 1.0
        )


