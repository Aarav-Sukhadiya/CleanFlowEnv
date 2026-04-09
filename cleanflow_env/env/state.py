from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd


class EnvironmentState:
    """
    Holds all mutable and immutable state for a single CleanFlowEnv episode.

    raw_table and ground_truth are NEVER mutated — they are read-only references.
    current_table is the agent's working copy, modified by actions.
    prev_table stores the state from the previous step for 1-step observation lag.

    Multi-table mode: when tables/ground_truth_tables are set, the environment
    manages multiple named DataFrames. current_table points to the primary table
    for backwards compatibility.
    """

    def __init__(
        self,
        task_id: str,
        raw_table: pd.DataFrame,
        ground_truth: pd.DataFrame,
        budget: int,
        column_descriptions: Dict[str, str],
        *,
        raw_tables: Dict[str, pd.DataFrame] | None = None,
        ground_truth_tables: Dict[str, pd.DataFrame] | None = None,
        column_descriptions_multi: Dict[str, Dict[str, str]] | None = None,
        table_relationships: List[Dict[str, str]] | None = None,
        primary_table: str | None = None,
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

        # Multi-table fields (None for single-table tasks)
        self.primary_table: str | None = primary_table
        self.table_relationships: List[Dict[str, str]] | None = table_relationships
        self.column_descriptions_multi: Dict[str, Dict[str, str]] | None = column_descriptions_multi

        if raw_tables is not None:
            self._raw_tables = raw_tables
            self._ground_truth_tables = ground_truth_tables or {}
            self.tables: Dict[str, pd.DataFrame] = {k: v.copy() for k, v in raw_tables.items()}
            self.prev_tables: Dict[str, pd.DataFrame] = {k: v.copy() for k, v in raw_tables.items()}
        else:
            self._raw_tables = None
            self._ground_truth_tables = None
            self.tables = None
            self.prev_tables = None

    @property
    def is_multi_table(self) -> bool:
        """Whether this episode uses multiple tables."""
        return self.tables is not None

    @property
    def raw_table(self) -> pd.DataFrame:
        """Read-only access to the original messy dataset."""
        return self._raw_table

    @property
    def ground_truth(self) -> pd.DataFrame:
        """Read-only access to the target clean dataset."""
        return self._ground_truth

    @property
    def ground_truth_tables(self) -> Dict[str, pd.DataFrame] | None:
        return self._ground_truth_tables

    def get_table(self, name: str | None) -> pd.DataFrame:
        """Get a table by name. None returns the single-table current_table."""
        if name is None:
            return self.current_table
        if self.tables is None:
            raise ValueError("Not in multi-table mode.")
        if name not in self.tables:
            raise ValueError(f"Unknown table '{name}'. Available: {list(self.tables.keys())}")
        return self.tables[name]

    def set_table(self, name: str | None, df: pd.DataFrame) -> None:
        """Update a table by name."""
        if name is None:
            self.current_table = df
        elif self.tables is not None:
            self.tables[name] = df
            if name == self.primary_table:
                self.current_table = df

    def reset(self) -> None:
        """Restore all mutable fields to initial values without regenerating data."""
        self.current_table = self._raw_table.copy()
        self.prev_table = self._raw_table.copy()
        self.step_count = 0
        self.budget_remaining = self.initial_budget
        self.best_quality_so_far = 0.0
        self.operations_history = []
        if self._raw_tables is not None:
            self.tables = {k: v.copy() for k, v in self._raw_tables.items()}
            self.prev_tables = {k: v.copy() for k, v in self._raw_tables.items()}

    def snapshot(self) -> pd.DataFrame:
        """Return a deep copy of current_table (used before applying an action)."""
        return self.current_table.copy()

    def to_dict(self) -> Dict[str, Any]:
        """Serializable dict of all state fields (DataFrames replaced with shape info)."""
        d = {
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
        if self.is_multi_table:
            d["tables"] = {k: list(v.shape) for k, v in self.tables.items()}
            d["table_relationships"] = self.table_relationships
            d["primary_table"] = self.primary_table
        return d
