from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from cleanflow_env.models.action import ActionModel
from cleanflow_env.models.observation import ObservationModel


class RuleBasedAgent:
    """
    A deterministic, rule-based agent for CleanFlowEnv.

    Decision priority:
    1. Drop duplicates if any exist
    2. Fill nulls using column descriptions to pick method
    3. Strip whitespace on string columns that mention it
    4. Replace substrings (e.g. "$", ",") on columns that mention it
    5. Map categorical values (e.g. "yes"/"no" → True/False)
    6. Convert types based on column descriptions
    7. Remove outliers on numeric columns
    """

    def __init__(self) -> None:
        self.action_history: List[Dict[str, Optional[str]]] = []

    def _is_done(self, action_type: str, column: Optional[str]) -> bool:
        """Check if this action was already performed."""
        for prev in self.action_history:
            if prev["action_type"] == action_type and prev["column"] == column:
                return True
        return False

    def _record(self, action_type: str, column: Optional[str]) -> None:
        self.action_history.append({"action_type": action_type, "column": column})

    @staticmethod
    def _looks_sequential(col: str, obs: ObservationModel) -> bool:
        """Check if the column values in the preview follow a prefix+number pattern."""
        pattern = re.compile(r"^(.*?)(\d+)$")
        values = []
        for row in obs.table_preview:
            val = row.values.get(col)
            if val is not None and str(val) != "None":
                values.append(str(val))
        if len(values) < 2:
            return False
        prefixes: dict[str, int] = {}
        for val in values:
            m = pattern.match(val)
            if m:
                prefixes[m.group(1)] = prefixes.get(m.group(1), 0) + 1
        if not prefixes:
            return False
        best_count = max(prefixes.values())
        return best_count >= len(values) * 0.5

    def _pick_fill_method(
        self, col: str, descriptions: Dict[str, str], schema: Dict[str, str],
        obs: ObservationModel | None = None,
    ) -> tuple[str, Any]:
        """Use column descriptions to pick the best fill method.

        Returns (method, constant_value) — constant_value is only used when method == "constant".
        """
        desc = descriptions.get(col, "").lower()
        dtype = schema.get(col, "string")

        # Sequential fill for identifiers that follow a prefix+number pattern
        if any(kw in desc for kw in ["sequential", "sequence"]):
            return "sequential", None
        if dtype == "string" and any(
            kw in desc for kw in ["identifier", "name", "id"]
        ):
            if "no cleaning" not in desc and "no action" not in desc:
                # Check the table preview for sequential patterns
                if self._looks_sequential(col, obs):
                    return "sequential", None
                return "constant", "Unknown"

        # Constant "Unknown" for categorical strings
        if any(kw in desc for kw in ["constant", "'unknown'", '"unknown"']):
            return "constant", "Unknown"

        # Forward-fill for date/time columns
        if any(kw in desc for kw in ["forward fill", "forward-fill", "ffill"]):
            return "forward_fill", None
        if dtype in ("datetime", "string") and any(
            kw in desc for kw in ["date", "datetime"]
        ):
            if "missing" in desc or "null" in desc:
                return "forward_fill", None

        # Explicit method hints
        if any(kw in desc for kw in ["mean", "average"]):
            return "mean", None
        if any(kw in desc for kw in ["mode", "category", "categorical"]):
            return "mode", None

        # Default to median — more robust to outliers than mean
        return "median", None

    def act(self, obs: ObservationModel) -> Optional[ActionModel]:
        """Return the next action or None if all useful actions are exhausted."""

        # Multi-table mode: delegate to per-table logic
        if obs.tables:
            return self._act_multi(obs)

        return self._act_single(obs, obs.table_schema, obs.null_counts,
                                obs.duplicate_count, obs.column_descriptions,
                                obs.table_preview, table_name=None)

    def _act_multi(self, obs: ObservationModel) -> Optional[ActionModel]:
        """Handle multi-table action selection."""
        # Priority 0: Validate foreign keys (remove orphan rows early)
        for rel in (obs.table_relationships or []):
            key = f"validate_fk:{rel['from_table']}.{rel['from_column']}"
            if not self._is_done("validate_foreign_key", key):
                self._record("validate_foreign_key", key)
                return ActionModel(
                    action_type="validate_foreign_key",
                    column=rel["from_column"],
                    table=rel["from_table"],
                    foreign_key_column=rel["from_column"],
                    lookup_table=rel["to_table"],
                    lookup_key_column=rel["to_column"],
                )

        # Then run standard cleaning per table
        for table_name, table_obs in obs.tables.items():
            action = self._act_single(
                obs, table_obs.table_schema, table_obs.null_counts,
                table_obs.duplicate_count, table_obs.column_descriptions,
                table_obs.table_preview, table_name=table_name,
            )
            if action is not None:
                return action

        return None

    def _act_single(
        self, obs: ObservationModel,
        schema: Dict[str, str], null_counts: Dict[str, int],
        duplicate_count: int, column_descriptions: Dict[str, str],
        table_preview, table_name: Optional[str] = None,
    ) -> Optional[ActionModel]:
        """Core single-table action selection logic."""

        def _key(col: Optional[str]) -> str:
            """Prefix column with table name for uniqueness in multi-table mode."""
            if table_name and col:
                return f"{table_name}.{col}"
            return col or ""

        def _make(action_type: str, **kwargs) -> ActionModel:
            if table_name:
                kwargs["table"] = table_name
            return ActionModel(action_type=action_type, **kwargs)

        # Priority 1: Drop duplicates whenever they exist.
        if duplicate_count > 0:
            action = _make("drop_duplicates")
            self._record("drop_duplicates", _key(None))
            return action

        # Priority 2: Strip whitespace on columns that mention it
        for col, dtype in schema.items():
            desc = column_descriptions.get(col, "").lower()
            if dtype == "string" and "whitespace" in desc:
                if not self._is_done("strip_whitespace", _key(col)):
                    action = _make("strip_whitespace", column=col)
                    self._record("strip_whitespace", _key(col))
                    return action

        # Priority 3: Replace substrings (currency symbols, commas, year typos)
        # Must run BEFORE fill_null so numeric columns are clean for median/mean
        for col, dtype in schema.items():
            desc = column_descriptions.get(col, "").lower()
            if dtype == "string":
                # Currency: remove $ and , before numeric conversion
                if any(kw in desc for kw in ["'$'", '"$"', "currency", "usd"]) and ("'$'" in desc or '"$"' in desc or ("$" in desc and "string" in desc)):
                    if not self._is_done("replace_substring_dollar", _key(col)):
                        action = _make(
                            "replace_substring",
                            column=col,
                            old_value="$",
                            new_value="",
                        )
                        self._record("replace_substring_dollar", _key(col))
                        return action
                    if not self._is_done("replace_substring_comma", _key(col)):
                        action = _make(
                            "replace_substring",
                            column=col,
                            old_value=",",
                            new_value="",
                        )
                        self._record("replace_substring_comma", _key(col))
                        return action

                # Year typos: 2033 → 2023
                if "2033" in desc and "2023" in desc:
                    if not self._is_done("replace_substring", _key(col)):
                        action = _make(
                            "replace_substring",
                            column=col,
                            old_value="2033",
                            new_value="2023",
                        )
                        self._record("replace_substring", _key(col))
                        return action

        # Priority 4: Convert types based on descriptions
        # Must run BEFORE fill_null so numeric fills (median/mean) work on proper dtypes
        for col, dtype in schema.items():
            desc = column_descriptions.get(col, "").lower()
            if dtype == "string":
                if "datetime" in desc and "no cleaning" not in desc and "no action" not in desc and "no conversion" not in desc and "no type conversion" not in desc:
                    if not self._is_done("convert_type", _key(col)):
                        action = _make(
                            "convert_type",
                            column=col,
                            target_type="datetime",
                        )
                        self._record("convert_type", _key(col))
                        return action
                if any(kw in desc for kw in ["numeric", "float", "price", "amount", "usd", "$"]):
                    if not self._is_done("convert_type", _key(col)):
                        action = _make(
                            "convert_type",
                            column=col,
                            target_type="float",
                        )
                        self._record("convert_type", _key(col))
                        return action

        # Priority 5: Fill nulls
        for col, count in null_counts.items():
            if count > 0 and not self._is_done("fill_null", _key(col)):
                method, constant_value = self._pick_fill_method(
                    col, column_descriptions, schema, obs
                )
                action = _make(
                    "fill_null",
                    column=col,
                    method=method,
                    constant_value=constant_value,
                )
                self._record("fill_null", _key(col))
                return action

        # Priority 6: Map categorical values (boolean-like columns)
        for col, dtype in schema.items():
            desc = column_descriptions.get(col, "").lower()
            if "boolean" in desc or ("map to boolean" in desc) or ("yes" in desc and "no" in desc and "date" not in desc):
                if dtype == "string" and not self._is_done("map_values", _key(col)):
                    action = _make(
                        "map_values",
                        column=col,
                        mapping={
                            "yes": True, "Yes": True, "YES": True,
                            "no": False, "No": False, "NO": False,
                            "true": True, "True": True, "TRUE": True,
                            "false": False, "False": False, "FALSE": False,
                            "1": True, "0": False,
                        },
                    )
                    self._record("map_values", _key(col))
                    return action

        # Priority 7: Remove outliers on numeric columns mentioned in descriptions
        for col, dtype in schema.items():
            desc = column_descriptions.get(col, "").lower()
            if dtype in ("float", "int") and "outlier" in desc:
                if not self._is_done("remove_outliers", _key(col)):
                    action = _make("remove_outliers", column=col)
                    self._record("remove_outliers", _key(col))
                    return action

        return None

    def reset(self) -> None:
        """Clear action history for a new episode."""
        self.action_history = []
