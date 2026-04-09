from __future__ import annotations

# Any = accepts any Python type (used for cell values which can be int, str, float, None, etc.)
# Dict = a dictionary with typed keys and values
# List = a list with typed elements
# Optional = means the field can be None
from typing import Any, Dict, List, Optional

# BaseModel = the base class all Pydantic models inherit from
# Field = lets us add metadata like descriptions and constraints to each field
# field_validator = lets us write validation logic for a single field
# model_validator = lets us write validation logic that checks multiple fields together
from pydantic import BaseModel, Field, field_validator, model_validator


class TablePreviewRow(BaseModel):
    """
    Represents a single row from the table preview shown to the agent.

    The agent only ever sees the first 5 rows of the table — not the full dataset.
    This forces the agent to reason from partial information, like a real data analyst
    would when previewing a large file.
    """

    # The original row index from the DataFrame (e.g. 0, 1, 2 ...)
    row_index: int = Field(..., description="Original row index from the DataFrame.")

    # The actual cell values for this row, keyed by column name.
    # Optional[Any] means a cell can hold any value OR be None (representing a missing/null value).
    # Using None instead of NaN is important because JSON has no NaN — None serializes cleanly.
    values: Dict[str, Optional[Any]] = Field(
        ...,
        description="Column name to cell value mapping. None represents a missing (null) value."
    )


class TableObservation(BaseModel):
    """Observation data for a single table in multi-table mode."""
    table_preview: List[TablePreviewRow] = Field(default_factory=list)
    table_schema: Dict[str, str] = Field(default_factory=dict, alias="schema")
    null_counts: Dict[str, int] = Field(default_factory=dict)
    duplicate_count: int = 0
    stats: Dict[str, float] = Field(default_factory=dict)
    distribution: Dict[str, Dict[str, float]] = Field(default_factory=dict)
    column_descriptions: Dict[str, str] = Field(default_factory=dict)

    model_config = {"populate_by_name": True, "serialize_by_alias": True}


class ObservationModel(BaseModel):
    model_config = {"populate_by_name": True, "serialize_by_alias": True}

    """
    The full observation returned to the agent after each reset() or step() call.

    This is everything the agent can see about the current state of the dataset.
    All decision-critical fields (null_counts, duplicate_count, stats, distribution,
    schema) reflect the CURRENT table state so the agent can react immediately
    to changes (e.g. new duplicates created by null-filling).
    table_preview uses the previous step's state for before/after comparison in the UI.
    """

    # The first 5 rows of the table from the PREVIOUS step.
    # Max 10 rows enforced by the validator below.
    # Shown in the UI for before/after comparison.
    table_preview: List[TablePreviewRow] = Field(
        ...,
        description="First 5 rows of the table from the previous step (for before/after display). Max 10 rows."
    )

    # Maps each column name to its inferred data type as a human-readable string.
    # e.g. {"age": "float", "name": "string", "dob": "datetime"}
    table_schema: Dict[str, str] = Field(
        ...,
        alias="schema",
        description="Column name to inferred dtype mapping (e.g. 'float', 'int', 'string', 'datetime'). Reflects current table state.",
    )

    # How many null (missing) values exist per column — current table state.
    # e.g. {"age": 5, "salary": 3} means age currently has 5 missing values
    null_counts: Dict[str, int] = Field(
        ...,
        description="Number of null values per column. Reflects current table state."
    )

    # Total number of fully duplicate rows — current table state.
    duplicate_count: int = Field(
        ...,
        description="Number of fully duplicate rows. Reflects current table state."
    )

    # Basic statistics (mean, std) for each numeric column — current table state.
    # e.g. {"age_mean": 29.4, "age_std": 6.1, "salary_mean": 52000.0}
    stats: Dict[str, float] = Field(
        ...,
        description="Mean and std per numeric column. Reflects current table state. Keys: '{col}_mean', '{col}_std'."
    )

    # Distribution stats per numeric column — quartiles, min, max, skewness.
    # e.g. {"age": {"min": 22, "q1": 33, "median": 44, "q3": 55, "max": 64, "skew": 0.1}}
    distribution: Dict[str, Dict[str, float]] = Field(
        default_factory=dict,
        description="Distribution stats (min, q1, median, q3, max, skew) per numeric column. Reflects current table state."
    )

    # How many actions the agent has taken so far this episode (starts at 0 after reset).
    step_count: int = Field(
        ...,
        description="Number of steps taken so far in the current episode."
    )

    # How many action credits the agent has left. Each action type costs a different amount.
    # Must be >= 0. When it hits 0, the episode ends.
    budget_remaining: int = Field(
        ...,
        description="Remaining action budget. Each action deducts a fixed cost. Episode ends when this reaches 0. Must be >= 0."
    )

    # Which task is currently active (e.g. "task_easy", "task_medium", "task_hard", "task_expert").
    task_id: str = Field(
        ...,
        description="Identifier of the active task (e.g. 'task_easy', 'task_medium', 'task_hard', 'task_expert')."
    )

    # Human-readable hints about each column — what it means, expected type, valid range, etc.
    # e.g. {"age": "Respondent age in years. Should be integer 18-90."}
    # Agents use these to decide HOW to clean each column (e.g. fill age with mean, not constant).
    column_descriptions: Dict[str, str] = Field(
        ...,
        description="Semantic description of each column including expected type, value range, and cleaning hints."
    )

    # Multi-table fields (None for single-table tasks)
    tables: Optional[Dict[str, TableObservation]] = Field(
        default=None,
        description="Per-table observation data in multi-table mode. None for single-table tasks.",
    )
    table_relationships: Optional[List[Dict[str, str]]] = Field(
        default=None,
        description="Foreign key relationships between tables.",
    )

    # Validates that table_preview never exceeds 10 rows.
    # mode="before" means this runs BEFORE Pydantic tries to build the TablePreviewRow objects,
    # so we can reject the raw list early if it's too long.
    @field_validator("table_preview", mode="before")
    @classmethod
    def limit_preview_rows(cls, value: list) -> list:
        # If someone accidentally passes the full table, we cap it at 10 rows instead of crashing.
        if len(value) > 10:
            raise ValueError(f"table_preview must have at most 10 rows, got {len(value)}.")
        return value

    # Validates that budget_remaining is never negative.
    # A negative budget would be a logic bug — the episode should have ended at 0.
    @field_validator("budget_remaining")
    @classmethod
    def budget_must_be_non_negative(cls, value: int) -> int:
        if value < 0:
            raise ValueError(f"budget_remaining must be >= 0, got {value}.")
        return value