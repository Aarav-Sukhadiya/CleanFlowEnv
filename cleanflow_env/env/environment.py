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
from cleanflow_env.models.observation import ObservationModel, TableObservation, TablePreviewRow
from cleanflow_env.models.reward import RewardModel

logger = logging.getLogger("cleanflow")

MAX_STEPS = 20


def _build_table_obs(
    cur: pd.DataFrame,
    prev: pd.DataFrame,
    col_descriptions: Dict[str, str],
) -> Dict[str, Any]:
    """Build observation fields for a single table (reusable for multi-table)."""
    preview_rows = []
    for i, (idx, row) in enumerate(prev.head(5).iterrows()):
        values = {}
        for col in prev.columns:
            val = row[col]
            if pd.isna(val):
                values[col] = None
            elif hasattr(val, "item"):
                values[col] = val.item()
            else:
                values[col] = val
        preview_rows.append(TablePreviewRow(row_index=int(idx), values=values))

    dtype_map = {
        "float64": "float", "float32": "float",
        "int64": "int", "int32": "int", "Int64": "int", "Int32": "int",
        "object": "string", "bool": "bool", "boolean": "bool", "category": "string",
    }
    schema = {}
    for col in cur.columns:
        dt = str(cur[col].dtype)
        schema[col] = "datetime" if "datetime" in dt else dtype_map.get(dt, "string")

    null_counts = {col: int(cur[col].isnull().sum()) for col in cur.columns}
    duplicate_count = int(cur.duplicated().sum())

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

    return {
        "preview_rows": preview_rows,
        "schema": schema,
        "null_counts": null_counts,
        "duplicate_count": duplicate_count,
        "stats": stats,
        "distribution": distribution,
        "column_descriptions": col_descriptions,
    }


def build_observation(state: EnvironmentState) -> ObservationModel:
    """
    Build an ObservationModel from the current environment state.

    All decision-critical fields (null_counts, duplicate_count, stats,
    distribution, schema) use current_table so the agent sees the true
    state and can react to changes (e.g. new duplicates created by filling).
    Table preview uses prev_table for before/after comparison in the UI.
    """
    # Build primary table observation
    primary = _build_table_obs(state.current_table, state.prev_table, state.column_descriptions)

    # Build multi-table observations if applicable
    tables_obs = None
    table_relationships = None
    if state.is_multi_table:
        tables_obs = {}
        for name, cur_df in state.tables.items():
            prev_df = state.prev_tables.get(name, cur_df)
            col_desc = (state.column_descriptions_multi or {}).get(name, {})
            t = _build_table_obs(cur_df, prev_df, col_desc)
            tables_obs[name] = TableObservation(
                table_preview=t["preview_rows"],
                table_schema=t["schema"],
                null_counts=t["null_counts"],
                duplicate_count=t["duplicate_count"],
                stats=t["stats"],
                distribution=t["distribution"],
                column_descriptions=t["column_descriptions"],
            )
        table_relationships = state.table_relationships

    return ObservationModel(
        table_preview=primary["preview_rows"],
        table_schema=primary["schema"],
        null_counts=primary["null_counts"],
        duplicate_count=primary["duplicate_count"],
        stats=primary["stats"],
        distribution=primary["distribution"],
        step_count=state.step_count,
        budget_remaining=state.budget_remaining,
        task_id=state.task_id,
        column_descriptions=state.column_descriptions,
        tables=tables_obs,
        table_relationships=table_relationships,
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
        Supports both single-table (4-tuple) and multi-table (6-tuple) task generators.
        """
        if task_id not in self.task_registry:
            raise ValueError(
                f"Unknown task_id '{task_id}'. Available: {list(self.task_registry.keys())}"
            )

        task_fn = self.task_registry[task_id]
        result = task_fn()

        if len(result) == 4:
            # Single-table task: (raw, gt, budget, col_desc)
            raw_table, ground_truth, budget, column_descriptions = result
            self._state = EnvironmentState(
                task_id=task_id,
                raw_table=raw_table,
                ground_truth=ground_truth,
                budget=budget,
                column_descriptions=column_descriptions,
            )
        else:
            # Multi-table task: (raw_tables, gt_tables, budget, col_desc_multi, relationships, primary)
            raw_tables, gt_tables, budget, col_desc_multi, relationships, primary = result
            # Use primary table for backwards-compatible fields
            self._state = EnvironmentState(
                task_id=task_id,
                raw_table=raw_tables[primary],
                ground_truth=gt_tables[primary],
                budget=budget,
                column_descriptions=col_desc_multi.get(primary, {}),
                raw_tables=raw_tables,
                ground_truth_tables=gt_tables,
                column_descriptions_multi=col_desc_multi,
                table_relationships=relationships,
                primary_table=primary,
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

        # Update prev_table(s) BEFORE applying action
        state.prev_table = state.current_table.copy()
        if state.is_multi_table:
            state.prev_tables = {k: v.copy() for k, v in state.tables.items()}

        # Apply action
        invalid = False
        try:
            result = apply_action(state.current_table, action, tables=state.tables)
            if isinstance(result, dict):
                # Multi-table result
                state.tables = result
                if state.primary_table and state.primary_table in result:
                    state.current_table = result[state.primary_table]
            else:
                state.current_table = result
                # Sync back to tables dict if in multi-table mode
                if state.is_multi_table and action.table:
                    state.tables[action.table] = result
        except InvalidActionError as e:
            logger.warning(f"Action failed: {e}")
            invalid = True
            # Revert
            state.current_table = state.prev_table.copy()
            if state.is_multi_table:
                state.tables = {k: v.copy() for k, v in state.prev_tables.items()}

        # Update state
        state.step_count += 1
        state.budget_remaining -= cost
        op_entry = {
            "action_type": action.action_type,
            "column": action.column,
            "method": action.method,
            "step": state.step_count,
            "valid": not invalid,
        }
        if action.table:
            op_entry["table"] = action.table
        state.operations_history.append(op_entry)

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

    def preview_action(self, action_dict: dict) -> Dict[str, Any]:
        """Dry-run an action and return what would change, without spending budget.

        Returns a dict with before/after stats so the agent can decide whether
        to commit the action.
        """
        if self._state is None:
            raise RuntimeError("Call reset() before preview_action().")

        state = self._state

        try:
            action = ActionModel(**action_dict)
        except Exception as e:
            return {"valid": False, "error": str(e)}

        cost = get_action_cost(action)
        if cost > state.budget_remaining:
            return {"valid": False, "error": "Insufficient budget", "cost": cost}

        # Snapshot — copy to guarantee no in-place mutation leaks to state
        before_df = state.current_table.copy()
        before_tables = (
            {k: v.copy() for k, v in state.tables.items()}
            if state.is_multi_table and state.tables
            else None
        )

        # For multi-table actions targeting a specific table, diff against that table
        target_table = action.table if state.is_multi_table and action.table else state.primary_table
        if state.is_multi_table and before_tables and target_table in before_tables:
            before_df = before_tables[target_table].copy()

        try:
            result = apply_action(
                state.current_table.copy() if not state.is_multi_table else before_df,
                action,
                tables=before_tables,
            )
            if isinstance(result, dict):
                after_df = result.get(target_table, before_df)
            else:
                after_df = result
        except InvalidActionError as e:
            return {"valid": False, "error": str(e)}

        # Compute diff stats
        before_nulls = int(before_df.isnull().sum().sum())
        after_nulls = int(after_df.isnull().sum().sum())
        before_rows = len(before_df)
        after_rows = len(after_df)
        before_dups = int(before_df.duplicated().sum())
        after_dups = int(after_df.duplicated().sum())

        return {
            "valid": True,
            "cost": cost,
            "rows_before": before_rows,
            "rows_after": after_rows,
            "rows_removed": before_rows - after_rows,
            "nulls_before": before_nulls,
            "nulls_after": after_nulls,
            "nulls_fixed": before_nulls - after_nulls,
            "duplicates_before": before_dups,
            "duplicates_after": after_dups,
        }

    def undo(self) -> Optional[ObservationModel]:
        """Revert the last action. Restores data but does NOT refund budget.

        Returns the new observation, or None if nothing to undo.
        """
        if self._state is None:
            return None

        state = self._state
        if state.step_count == 0 or state.prev_table is None:
            return None

        # Guard: if the most recent op is already undone, there's nothing to undo
        if state.operations_history and state.operations_history[-1].get("undone"):
            return None

        # Revert data
        state.current_table = state.prev_table.copy()
        if state.is_multi_table and state.prev_tables:
            state.tables = {k: v.copy() for k, v in state.prev_tables.items()}

        # Mark the last operation as undone
        if state.operations_history:
            state.operations_history[-1]["undone"] = True

        return build_observation(state)

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


