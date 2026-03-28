from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd


class EnvironmentState:
    """
    Holds all mutable and immutable state for a single CleanFlowEnv episode.

    raw_table and ground_truth are NEVER mutated — they are read-only references.
    current_table is the agent's working copy, modified by actions.
    prev_table stores the state from the previous step for 1-step observation lag.
    """

    def __init__(
        self,
        task_id: str,
        raw_table: pd.DataFrame,
        ground_truth: pd.DataFrame,
        budget: int,
        column_descriptions: Dict[str, str],
    ) -> None:
        self.task_id = task_id
        # Read-only references — never copy, never mutate
        self._raw_table = raw_table
        self._ground_truth = ground_truth
        # Working copies
        self.current_table: pd.DataFrame = raw_table.copy()
        self.prev_table: pd.DataFrame = raw_table.copy()
        # Episode tracking
        self.step_count: int = 0
        self.budget_remaining: int = budget
        self.initial_budget: int = budget
        self.best_quality_so_far: float = 0.0
        self.operations_history: List[Dict[str, Any]] = []
        self.column_descriptions = column_descriptions

    @property
    def raw_table(self) -> pd.DataFrame:
        """Read-only access to the original messy dataset."""
        return self._raw_table

    @property
    def ground_truth(self) -> pd.DataFrame:
        """Read-only access to the target clean dataset."""
        return self._ground_truth

    def reset(self) -> None:
        """Restore all mutable fields to initial values without regenerating data."""
        self.current_table = self._raw_table.copy()
        self.prev_table = self._raw_table.copy()
        self.step_count = 0
        self.budget_remaining = self.initial_budget
        self.best_quality_so_far = 0.0
        self.operations_history = []

    def snapshot(self) -> pd.DataFrame:
        """Return a deep copy of current_table (used before applying an action)."""
        return self.current_table.copy()

    def to_dict(self) -> Dict[str, Any]:
        """Serializable dict of all state fields (DataFrames replaced with shape info)."""
        return {
            "task_id": self.task_id,
            "step_count": self.step_count,
            "budget_remaining": self.budget_remaining,
            "initial_budget": self.initial_budget,
            "best_quality_so_far": self.best_quality_so_far,
            "operations_history": self.operations_history,
            "current_table_shape": list(self.current_table.shape),
            "raw_table_shape": list(self._raw_table.shape),
            "ground_truth_shape": list(self._ground_truth.shape),
            "column_descriptions": self.column_descriptions,
        }
