from __future__ import annotations

import re
from typing import Dict, Optional

import pandas as pd

from cleanflow_env.models.action import ActionModel


class InvalidActionError(Exception):
    """Raised when an action cannot be applied to the current table."""

    pass


def detect_sequential_pattern(series: pd.Series) -> tuple[str, str, set[int]] | None:
    """Detect a prefix+number pattern in a string column.

    Returns (prefix, zero_pad_format, existing_numbers) or None if no pattern found.
    Example: "Employee_007" → ("Employee_", "03d", {0, 1, 2, ..., 199})
    """
    non_null = series.dropna().astype(str)
    if len(non_null) < 2:
        return None

    # Match strings like "Prefix_001", "EMP0042", "P003", etc.
    pattern = re.compile(r"^(.*?)(\d+)$")
    prefixes: dict[str, list[int]] = {}
    pad_widths: dict[str, set[int]] = {}

    for val in non_null:
        m = pattern.match(val)
        if m:
            prefix, num_str = m.group(1), m.group(2)
            prefixes.setdefault(prefix, []).append(int(num_str))
            pad_widths.setdefault(prefix, set()).add(len(num_str))

    if not prefixes:
        return None

    # Pick the most common prefix (must cover >50% of non-null values)
    best_prefix = max(prefixes, key=lambda p: len(prefixes[p]))
    if len(prefixes[best_prefix]) < len(non_null) * 0.5:
        return None

    nums = prefixes[best_prefix]
    widths = pad_widths[best_prefix]
    # Use zero-padding if all number parts have the same width
    if len(widths) == 1:
        fmt = f"0{widths.pop()}d"
    else:
        fmt = "d"

    return best_prefix, fmt, set(nums)


def fill_sequential(col: pd.Series) -> pd.Series:
    """Fill nulls in a sequential ID column by filling gaps first, then extending.

    Detects prefix+number patterns (e.g. Employee_001, PROD_0042) and
    fills each null with the missing gap value in the sequence first.
    If there are more nulls than gaps, continues past the max.
    Falls back to 'Unknown' if no sequential pattern is detected.

    Example: [Emp_1, Emp_2, null, Emp_4, Emp_5] → fills null with Emp_3 (gap),
    not Emp_6 (next after max).
    """
    result = col.copy()
    null_mask = result.isna()
    if not null_mask.any():
        return result

    pattern_info = detect_sequential_pattern(col)
    if pattern_info is None:
        return result.fillna("Unknown")

    prefix, fmt, existing_nums = pattern_info

    # Build the pool of fill values: gaps first, then extend past max
    max_num = max(existing_nums)
    full_range = set(range(min(existing_nums), max_num + 1))
    gaps = sorted(full_range - existing_nums)

    # After gaps are exhausted, continue from max+1, max+2, ...
    next_after_max = max_num + 1
    null_indices = result.index[null_mask]

    for idx in null_indices:
        if gaps:
            num = gaps.pop(0)
        else:
            num = next_after_max
            next_after_max += 1
        result.at[idx] = f"{prefix}{num:{fmt}}"

    return result


def fill_null(
    df: pd.DataFrame, column: str, method: str, constant_value=None
) -> pd.DataFrame:
    """Fill null values in the specified column using the given method."""
    result = df.copy()
    if column not in result.columns:
        raise InvalidActionError(f"Column '{column}' not found in table.")

    col = result[column]
    if method == "mean":
        result[column] = col.fillna(col.mean())
    elif method == "median":
        result[column] = col.fillna(col.median())
    elif method == "mode":
        mode_vals = col.mode()
        if len(mode_vals) == 0:
            raise InvalidActionError(
                f"Cannot compute mode for column '{column}' — all values are null."
            )
        result[column] = col.fillna(mode_vals.iloc[0])
    elif method == "constant":
        result[column] = col.fillna(constant_value)
    elif method == "sequential":
        result[column] = fill_sequential(col)
    elif method == "forward_fill":
        result[column] = col.ffill().bfill()  # bfill fallback for leading NaNs
    elif method == "backward_fill":
        result[column] = col.bfill().ffill()  # ffill fallback for trailing NaNs
    else:
        raise InvalidActionError(f"Unknown fill method '{method}'.")
    return result


def drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Drop all fully duplicate rows and reset the index."""
    return df.drop_duplicates().reset_index(drop=True)


def convert_type(df: pd.DataFrame, column: str, target_type: str) -> pd.DataFrame:
    """Convert the specified column to the target data type."""
    result = df.copy()
    if column not in result.columns:
        raise InvalidActionError(f"Column '{column}' not found in table.")

    if target_type == "datetime":
        result[column] = pd.to_datetime(result[column], errors="coerce", format="mixed")
    elif target_type in ("int", "float"):
        result[column] = pd.to_numeric(result[column], errors="coerce")
        if target_type == "int":
            result[column] = result[column].astype("Int64")
    elif target_type == "string":
        result[column] = result[column].astype(str)
    else:
        raise InvalidActionError(f"Unknown target type '{target_type}'.")
    return result


def normalize(df: pd.DataFrame, column: str, method: str = "minmax") -> pd.DataFrame:
    """Normalize the specified numeric column using minmax or zscore."""
    result = df.copy()
    if column not in result.columns:
        raise InvalidActionError(f"Column '{column}' not found in table.")

    col = pd.to_numeric(result[column], errors="coerce")
    if method == "minmax":
        col_min, col_max = col.min(), col.max()
        if col_max == col_min:
            result[column] = 0.0
        else:
            result[column] = (col - col_min) / (col_max - col_min)
    elif method == "zscore":
        col_std = col.std()
        if col_std == 0:
            result[column] = 0.0
        else:
            result[column] = (col - col.mean()) / col_std
    else:
        raise InvalidActionError(f"Unknown normalization method '{method}'.")
    return result


def remove_outliers(
    df: pd.DataFrame, column: str, method: str = "iqr", threshold: float = 1.5
) -> pd.DataFrame:
    """Remove outliers using IQR or Z-score method.

    Methods:
    - iqr (default): Remove values outside Q1 - threshold*IQR .. Q3 + threshold*IQR
    - zscore: Remove values with |z-score| > threshold (default threshold=3.0 for zscore)
    """
    result = df.copy()
    if column not in result.columns:
        raise InvalidActionError(f"Column '{column}' not found in table.")

    col = pd.to_numeric(result[column], errors="coerce")

    if method == "iqr":
        q1 = col.quantile(0.25)
        q3 = col.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - threshold * iqr
        upper = q3 + threshold * iqr
        mask = (col >= lower) & (col <= upper) | col.isna()
    elif method == "zscore":
        mean = col.mean()
        std = col.std()
        if std == 0:
            mask = pd.Series(True, index=result.index)
        else:
            z = (col - mean).abs() / std
            mask = (z <= threshold) | col.isna()
    else:
        raise InvalidActionError(
            f"Unknown outlier method '{method}'. Use 'iqr' or 'zscore'."
        )

    return result[mask].reset_index(drop=True)


def strip_whitespace(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Strip leading and trailing whitespace from string values in a column."""
    result = df.copy()
    if column not in result.columns:
        raise InvalidActionError(f"Column '{column}' not found in table.")

    if result[column].dtype == "object":
        result[column] = result[column].str.strip()
    else:
        # Convert to string, strip, convert back — handles mixed-type columns
        result[column] = result[column].astype(str).str.strip()
    return result


def map_values(
    df: pd.DataFrame, column: str, mapping: dict
) -> pd.DataFrame:
    """Map categorical values in a column using a provided mapping dict.

    Example mapping: {"yes": True, "no": False, "Yes": True, "No": False}
    Values not in the mapping are left unchanged.
    """
    result = df.copy()
    if column not in result.columns:
        raise InvalidActionError(f"Column '{column}' not found in table.")

    if not mapping:
        raise InvalidActionError("map_values requires a non-empty 'mapping' dict.")

    def _map_val(v):
        if pd.isna(v):
            return v
        # Try exact match first, then string representation
        if v in mapping:
            return mapping[v]
        str_v = str(v)
        if str_v in mapping:
            return mapping[str_v]
        return v

    result[column] = result[column].map(_map_val)
    return result


def replace_substring(
    df: pd.DataFrame, column: str, old: str, new: str
) -> pd.DataFrame:
    """Replace occurrences of a substring within string values of a column.

    Example: replace_substring(df, "price", "$", "") to remove dollar signs.
    """
    result = df.copy()
    if column not in result.columns:
        raise InvalidActionError(f"Column '{column}' not found in table.")

    if result[column].dtype != "object":
        result[column] = result[column].astype(str)

    result[column] = result[column].str.replace(old, new, regex=False)
    return result


# Dispatch dict for O(1) action routing
_ACTION_DISPATCH = {
    "fill_null": lambda df, action: fill_null(
        df, action.column, action.method, action.constant_value
    ),
    "drop_duplicates": lambda df, action: drop_duplicates(df),
    "convert_type": lambda df, action: convert_type(
        df, action.column, action.target_type
    ),
    "normalize": lambda df, action: normalize(df, action.column),
    "remove_outliers": lambda df, action: remove_outliers(
        df, action.column, action.outlier_method or "iqr", action.outlier_threshold or 1.5
    ),
    "strip_whitespace": lambda df, action: strip_whitespace(df, action.column),
    "map_values": lambda df, action: map_values(
        df, action.column, action.mapping or {}
    ),
    "replace_substring": lambda df, action: replace_substring(
        df, action.column, action.old_value or "", action.new_value or ""
    ),
}


def lookup_fill(
    tables: Dict[str, pd.DataFrame], action: ActionModel
) -> Dict[str, pd.DataFrame]:
    """Fill nulls in a column by looking up values from another table via FK."""
    source = tables[action.table].copy()
    lookup = tables[action.lookup_table]

    if action.column not in source.columns:
        raise InvalidActionError(f"Column '{action.column}' not in table '{action.table}'.")
    if action.foreign_key_column not in source.columns:
        raise InvalidActionError(f"FK column '{action.foreign_key_column}' not in table '{action.table}'.")
    if action.lookup_key_column not in lookup.columns:
        raise InvalidActionError(f"Key column '{action.lookup_key_column}' not in table '{action.lookup_table}'.")
    if action.lookup_value_column not in lookup.columns:
        raise InvalidActionError(f"Value column '{action.lookup_value_column}' not in table '{action.lookup_table}'.")

    # Build lookup mapping (deduped — take first match)
    lookup_map = lookup.drop_duplicates(subset=[action.lookup_key_column]).set_index(
        action.lookup_key_column
    )[action.lookup_value_column]

    # Fill only null values using the FK relationship
    null_mask = source[action.column].isna()
    fk_values = source.loc[null_mask, action.foreign_key_column]
    filled = fk_values.map(lookup_map)
    source.loc[null_mask, action.column] = filled

    result = dict(tables)
    result[action.table] = source
    return result


def validate_foreign_key(
    tables: Dict[str, pd.DataFrame], action: ActionModel
) -> Dict[str, pd.DataFrame]:
    """Remove rows where the FK value doesn't exist in the reference table."""
    source = tables[action.table].copy()
    lookup = tables[action.lookup_table]

    if action.foreign_key_column not in source.columns:
        raise InvalidActionError(f"FK column '{action.foreign_key_column}' not in table '{action.table}'.")
    if action.lookup_key_column not in lookup.columns:
        raise InvalidActionError(f"Key column '{action.lookup_key_column}' not in table '{action.lookup_table}'.")

    valid_keys = set(lookup[action.lookup_key_column].dropna())
    mask = source[action.foreign_key_column].isin(valid_keys) | source[action.foreign_key_column].isna()
    source = source[mask].reset_index(drop=True)

    result = dict(tables)
    result[action.table] = source
    return result


_MULTI_TABLE_DISPATCH = {
    "lookup_fill": lookup_fill,
    "validate_foreign_key": validate_foreign_key,
}


def apply_action(
    df: pd.DataFrame,
    action: ActionModel,
    tables: Optional[Dict[str, pd.DataFrame]] = None,
) -> pd.DataFrame | Dict[str, pd.DataFrame]:
    """Dispatch an ActionModel to the correct cleaning function.

    For multi-table actions or when action.table is set, operates on the tables
    dict and returns an updated dict. Otherwise operates on a single DataFrame.
    """
    # Multi-table action types
    if action.action_type in _MULTI_TABLE_DISPATCH:
        if tables is None:
            raise InvalidActionError(f"'{action.action_type}' requires multi-table mode.")
        handler = _MULTI_TABLE_DISPATCH[action.action_type]
        try:
            return handler(tables, action)
        except InvalidActionError:
            raise
        except Exception as e:
            raise InvalidActionError(f"Action failed: {e}") from e

    # Single-table action targeting a specific table in multi-table mode
    if action.table is not None and tables is not None:
        target_df = tables.get(action.table)
        if target_df is None:
            raise InvalidActionError(f"Unknown table '{action.table}'.")
        handler = _ACTION_DISPATCH.get(action.action_type)
        if handler is None:
            raise InvalidActionError(f"Unknown action type '{action.action_type}'.")
        try:
            result = dict(tables)
            result[action.table] = handler(target_df, action)
            return result
        except InvalidActionError:
            raise
        except Exception as e:
            raise InvalidActionError(f"Action failed: {e}") from e

    # Standard single-table path
    handler = _ACTION_DISPATCH.get(action.action_type)
    if handler is None:
        raise InvalidActionError(
            f"Unknown action type '{action.action_type}'. "
            f"Valid types: {list(_ACTION_DISPATCH.keys())}"
        )
    try:
        return handler(df, action)
    except InvalidActionError:
        raise
    except Exception as e:
        raise InvalidActionError(f"Action failed: {e}") from e
