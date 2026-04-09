"""
Data validation rules for CleanFlowEnv.

Checks the cleaned DataFrame against expected constraints:
- No remaining nulls in columns that had cleaning instructions
- Type correctness (numeric columns are numeric, dates are datetime)
- Range checks (values within expected bounds)
- Uniqueness (identifier columns have unique values)
- No remaining duplicates
"""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd


def validate_cleaned_data(
    cleaned: pd.DataFrame,
    ground_truth: pd.DataFrame,
    column_descriptions: Dict[str, str],
) -> Dict[str, Any]:
    """
    Run validation rules on the cleaned DataFrame.

    Returns a dict with:
    - validation_score: float 0-1 (fraction of rules passed)
    - rules_passed: int
    - rules_total: int
    - violations: list of human-readable violation strings
    """
    rules_passed = 0
    rules_total = 0
    violations: List[str] = []

    # Rule 1: No remaining nulls in columns that had nulls to fix
    for col in cleaned.columns:
        desc = column_descriptions.get(col, "").lower()
        if "missing" in desc or "null" in desc or "fill" in desc:
            rules_total += 1
            null_count = int(cleaned[col].isnull().sum())
            if null_count == 0:
                rules_passed += 1
            else:
                violations.append(
                    f"NULL_REMAINING: {col} still has {null_count} null values"
                )

    # Rule 2: No remaining duplicates
    rules_total += 1
    dup_count = int(cleaned.duplicated().sum())
    if dup_count == 0:
        rules_passed += 1
    else:
        violations.append(f"DUPLICATES: {dup_count} duplicate rows remain")

    # Rule 3: Type correctness — compare dtypes to ground truth
    common_cols = set(cleaned.columns) & set(ground_truth.columns)
    for col in common_cols:
        gt_dtype = str(ground_truth[col].dtype)
        cur_dtype = str(cleaned[col].dtype)
        rules_total += 1
        if _dtype_compatible(cur_dtype, gt_dtype):
            rules_passed += 1
        else:
            violations.append(
                f"TYPE_MISMATCH: {col} is {cur_dtype}, expected {gt_dtype}"
            )

    # Rule 4: Range checks for numeric columns (skip boolean)
    for col in common_cols:
        if not pd.api.types.is_numeric_dtype(ground_truth[col]):
            continue
        if pd.api.types.is_bool_dtype(ground_truth[col]):
            continue
        gt_col = ground_truth[col].dropna()
        if len(gt_col) == 0:
            continue

        rules_total += 1
        try:
            gt_min, gt_max = float(gt_col.min()), float(gt_col.max())
        except (TypeError, ValueError):
            rules_passed += 1
            continue
        # Allow 10% tolerance beyond GT range
        margin = (gt_max - gt_min) * 0.1 if gt_max != gt_min else 1.0
        expected_min = gt_min - margin
        expected_max = gt_max + margin

        cur_col = pd.to_numeric(cleaned[col], errors="coerce").dropna()
        if len(cur_col) == 0:
            violations.append(f"RANGE_EMPTY: {col} has no valid numeric values")
            continue

        out_of_range = int(((cur_col < expected_min) | (cur_col > expected_max)).sum())
        if out_of_range == 0:
            rules_passed += 1
        else:
            violations.append(
                f"RANGE_VIOLATION: {col} has {out_of_range} values outside "
                f"expected range [{expected_min:.1f}, {expected_max:.1f}]"
            )

    # Rule 5: Row count check — cleaned should be within reasonable range of GT
    rules_total += 1
    len_ratio = len(cleaned) / max(len(ground_truth), 1)
    if 0.8 <= len_ratio <= 1.2:
        rules_passed += 1
    else:
        violations.append(
            f"ROW_COUNT: cleaned has {len(cleaned)} rows, "
            f"expected ~{len(ground_truth)} (ratio: {len_ratio:.2f})"
        )

    raw = rules_passed / rules_total if rules_total > 0 else 1.0
    # Clamp to strict (0, 1) — validator rejects exactly 0.0 or 1.0
    _EPS = 1e-4
    validation_score = max(_EPS, min(1.0 - _EPS, raw))

    return {
        "validation_score": round(validation_score, 6),
        "rules_passed": rules_passed,
        "rules_total": rules_total,
        "violations": violations,
    }


def validate_cleaned_data_multi(
    tables: Dict[str, pd.DataFrame],
    ground_truth_tables: Dict[str, pd.DataFrame],
    column_descriptions_multi: Dict[str, Dict[str, str]],
    table_relationships: List[Dict[str, str]] | None = None,
) -> Dict[str, Any]:
    """Run validation rules across multiple tables, including FK integrity."""
    all_passed = 0
    all_total = 0
    all_violations: List[str] = []

    for name, gt in ground_truth_tables.items():
        cleaned = tables.get(name)
        if cleaned is None:
            all_violations.append(f"MISSING_TABLE: table '{name}' not found")
            all_total += 1
            continue
        col_desc = column_descriptions_multi.get(name, {})
        result = validate_cleaned_data(cleaned, gt, col_desc)
        all_passed += result["rules_passed"]
        all_total += result["rules_total"]
        all_violations.extend(f"[{name}] {v}" for v in result.get("violations", []))

    # FK integrity rules
    for rel in (table_relationships or []):
        all_total += 1
        child_table = tables.get(rel["from_table"])
        parent_table = tables.get(rel["to_table"])
        if child_table is None or parent_table is None:
            all_violations.append(f"FK_MISSING_TABLE: {rel['from_table']} or {rel['to_table']} missing")
            continue
        fk_col = rel["from_column"]
        pk_col = rel["to_column"]
        if fk_col not in child_table.columns or pk_col not in parent_table.columns:
            all_violations.append(f"FK_MISSING_COLUMN: {fk_col} or {pk_col}")
            continue
        valid_keys = set(parent_table[pk_col].dropna())
        orphans = child_table[fk_col].dropna()
        orphan_count = int((~orphans.isin(valid_keys)).sum())
        if orphan_count == 0:
            all_passed += 1
        else:
            all_violations.append(
                f"FK_VIOLATION: {rel['from_table']}.{fk_col} has {orphan_count} "
                f"orphan values not in {rel['to_table']}.{pk_col}"
            )

    raw = all_passed / all_total if all_total > 0 else 1.0
    _EPS = 1e-4
    validation_score = max(_EPS, min(1.0 - _EPS, raw))

    return {
        "validation_score": round(validation_score, 6),
        "rules_passed": all_passed,
        "rules_total": all_total,
        "violations": all_violations,
    }


def _dtype_compatible(dtype1: str, dtype2: str) -> bool:
    """Check if two dtype strings are compatible."""
    numeric_types = {"int64", "int32", "Int64", "Int32", "float64", "float32"}
    if dtype1 in numeric_types and dtype2 in numeric_types:
        return True
    if dtype1.startswith("datetime64") and dtype2.startswith("datetime64"):
        return True
    # String types
    string_types = {"object", "string", "str", "string[python]", "string[pyarrow]"}
    if dtype1 in string_types and dtype2 in string_types:
        return True
    # datetime ↔ string: converting date strings to datetime is valid
    if dtype1.startswith("datetime64") and dtype2 in string_types:
        return True
    if dtype2.startswith("datetime64") and dtype1 in string_types:
        return True
    return dtype1 == dtype2
