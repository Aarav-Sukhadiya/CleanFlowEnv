from __future__ import annotations

from typing import Any, Dict, Literal, Optional

from pydantic import BaseModel, model_validator


class ActionModel(BaseModel):
    """
    Typed action model for the CleanFlowEnv data cleaning environment.

    Represents a single deterministic transformation an agent can apply
    to the current working DataFrame. All field combinations are validated
    at construction time so invalid actions are caught before they reach
    the environment step loop.
    """

    action_type: Literal[
        "fill_null",
        "drop_duplicates",
        "convert_type",
        "normalize",
        "remove_outliers",
        "strip_whitespace",
        "map_values",
        "replace_substring",
        "standardize_format",
        "lookup_fill",
        "validate_foreign_key",
    ]

    column: Optional[str] = None

    # Multi-table support: target table name (None = single-table mode)
    table: Optional[str] = None

    # For lookup_fill / validate_foreign_key: foreign key relationship
    foreign_key_column: Optional[str] = None
    lookup_table: Optional[str] = None
    lookup_key_column: Optional[str] = None
    lookup_value_column: Optional[str] = None

    method: Optional[Literal["mean", "median", "mode", "constant", "forward_fill", "backward_fill", "sequential"]] = None

    target_type: Optional[Literal["int", "float", "datetime", "string"]] = None

    constant_value: Optional[Any] = None

    # For map_values: a dict mapping old values to new values
    # e.g. {"yes": True, "no": False, "1": True, "0": False}
    mapping: Optional[Dict[str, Any]] = None

    # For replace_substring: the substring to find and what to replace it with
    old_value: Optional[str] = None
    new_value: Optional[str] = None

    # For remove_outliers: method and threshold
    outlier_method: Optional[Literal["iqr", "zscore"]] = None
    outlier_threshold: Optional[float] = None

    @model_validator(mode="after")
    def validate_action_fields(self) -> "ActionModel":
        action = self.action_type

        if action == "fill_null":
            if self.column is None:
                raise ValueError("fill_null requires 'column' to be specified.")
            if self.method is None:
                raise ValueError("fill_null requires 'method' to be specified (mean, median, mode, constant, forward_fill, backward_fill).")
            if self.method == "constant" and self.constant_value is None:
                raise ValueError("fill_null with method='constant' requires 'constant_value' to be provided.")

        elif action == "convert_type":
            if self.column is None:
                raise ValueError("convert_type requires 'column' to be specified.")
            if self.target_type is None:
                raise ValueError("convert_type requires 'target_type' to be specified (int, float, datetime, string).")

        elif action in ("normalize", "remove_outliers", "strip_whitespace", "standardize_format"):
            if self.column is None:
                raise ValueError(f"{action} requires 'column' to be specified.")

        elif action == "map_values":
            if self.column is None:
                raise ValueError("map_values requires 'column' to be specified.")
            if not self.mapping:
                raise ValueError("map_values requires 'mapping' dict to be provided.")

        elif action == "replace_substring":
            if self.column is None:
                raise ValueError("replace_substring requires 'column' to be specified.")
            if self.old_value is None:
                raise ValueError("replace_substring requires 'old_value' to be specified.")
            if self.new_value is None:
                raise ValueError("replace_substring requires 'new_value' to be specified.")

        elif action == "lookup_fill":
            if self.table is None:
                raise ValueError("lookup_fill requires 'table' to be specified.")
            if self.column is None:
                raise ValueError("lookup_fill requires 'column' (column to fill) to be specified.")
            if self.foreign_key_column is None:
                raise ValueError("lookup_fill requires 'foreign_key_column' to be specified.")
            if self.lookup_table is None:
                raise ValueError("lookup_fill requires 'lookup_table' to be specified.")
            if self.lookup_key_column is None:
                raise ValueError("lookup_fill requires 'lookup_key_column' to be specified.")
            if self.lookup_value_column is None:
                raise ValueError("lookup_fill requires 'lookup_value_column' to be specified.")

        elif action == "validate_foreign_key":
            if self.table is None:
                raise ValueError("validate_foreign_key requires 'table' to be specified.")
            if self.foreign_key_column is None:
                raise ValueError("validate_foreign_key requires 'foreign_key_column' to be specified.")
            if self.lookup_table is None:
                raise ValueError("validate_foreign_key requires 'lookup_table' to be specified.")
            if self.lookup_key_column is None:
                raise ValueError("validate_foreign_key requires 'lookup_key_column' to be specified.")

        return self