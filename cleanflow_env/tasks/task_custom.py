"""
Custom dataset task generator.
Auto-detects data quality issues and generates a ground truth from user-uploaded CSV.
Supports difficulty levels: easy, medium, hard — controlling which issues are fixed.
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd


def _is_date_column(series: pd.Series) -> bool:
    """Heuristic: check if a string column looks like dates."""
    if "datetime" in str(series.dtype):
        return True
    sample = series.dropna().head(20)
    if len(sample) == 0:
        return False
    try:
        parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
        return parsed.notna().sum() > len(sample) * 0.5
    except Exception:
        # Fallback for older pandas without format="mixed"
        try:
            parsed = pd.to_datetime(sample, errors="coerce")
            return parsed.notna().sum() > len(sample) * 0.5
        except Exception:
            return False


def _is_identifier_column(col_name: str, series: pd.Series) -> bool:
    """Heuristic: check if a column is an identifier (name, id, etc.)."""
    name_lower = col_name.lower()
    if any(kw in name_lower for kw in ["name", "_id", "identifier", "email", "code"]):
        return True
    # High cardinality string columns are likely identifiers
    sample = series.dropna()
    if len(sample) > 0:
        ratio = sample.nunique() / len(sample)
        if ratio > 0.8:
            return True
    return False


def analyze_dataset(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyze a DataFrame and report all detected data quality issues.
    Returns a dict describing nulls, duplicates, type issues, outliers, etc.
    """
    issues: List[str] = []
    details: Dict[str, Any] = {
        "rows": len(df),
        "columns": len(df.columns),
        "null_columns": {},
        "duplicate_count": 0,
        "type_issues": {},
        "outlier_columns": [],
        "whitespace_columns": [],
        "categorical_columns": [],
        "issues_summary": [],
    }

    # Nulls
    for col in df.columns:
        null_count = int(df[col].isnull().sum())
        if null_count > 0:
            details["null_columns"][col] = null_count
            issues.append(f"{col}: {null_count} missing values")

    # Duplicates
    dup_count = int(df.duplicated().sum())
    details["duplicate_count"] = dup_count
    if dup_count > 0:
        issues.append(f"{dup_count} duplicate rows")

    # Type issues: columns that look numeric but are stored as strings
    str_columns = [c for c in df.columns if pd.api.types.is_string_dtype(df[c])]
    for col in str_columns:
        sample = df[col].dropna().head(20)
        if len(sample) == 0:
            continue

        # Check for whitespace
        stripped = sample.str.strip()
        if not (sample == stripped).all():
            details["whitespace_columns"].append(col)
            issues.append(f"{col}: has leading/trailing whitespace")

        # Check for categorical boolean-like values
        unique_lower = set(sample.str.lower().unique())
        bool_vals = {"yes", "no", "true", "false", "1", "0"}
        if unique_lower & bool_vals and len(unique_lower) <= 10:
            details["categorical_columns"].append(col)
            issues.append(f"{col}: looks like boolean/categorical with mixed representations")

        numeric_count = pd.to_numeric(sample, errors="coerce").notna().sum()
        if numeric_count > len(sample) * 0.5:
            details["type_issues"][col] = "looks numeric but stored as string"
            issues.append(f"{col}: looks numeric but stored as string")

        # Check for date-like strings
        try:
            parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
            if parsed.notna().sum() > len(sample) * 0.5:
                details["type_issues"][col] = "looks like dates stored as string"
                issues.append(f"{col}: looks like dates stored as string")
        except Exception:
            pass

    # Outliers in numeric columns (IQR x 1.5)
    for col in df.select_dtypes(include=["number"]).columns:
        s = df[col].dropna()
        if len(s) < 10:
            continue
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            continue
        outlier_count = int(((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum())
        if outlier_count > 0:
            details["outlier_columns"].append({"column": col, "count": outlier_count})
            issues.append(f"{col}: {outlier_count} outliers detected (IQR x 1.5)")

    details["issues_summary"] = issues
    return details


def auto_generate_ground_truth(
    df: pd.DataFrame, difficulty: str = "hard",
    date_cols: set | None = None,
) -> pd.DataFrame:
    """
    Automatically generate a cleaned ground truth from a messy DataFrame.

    Difficulty controls how many issue types are addressed:
    - easy:   fill nulls + drop duplicates only
    - medium: easy + convert types + strip whitespace
    - hard:   medium + remove outliers (full cleaning)
    """
    gt = df.copy()

    # --- EASY: dedup → fill → dedup → convert dates ---

    # Use pre-detected date columns if provided, otherwise detect here.
    if date_cols is None:
        date_cols = {
            col for col in gt.columns
            if pd.api.types.is_string_dtype(gt[col]) and _is_date_column(gt[col])
        }

    # First dedup: remove exact duplicates before filling.
    gt = gt.drop_duplicates().reset_index(drop=True)

    for col in gt.columns:
        if gt[col].isnull().sum() == 0:
            continue
        if gt[col].dtype in ("float64", "float32", "int64", "int32"):
            gt[col] = gt[col].fillna(gt[col].median())
        elif col in date_cols:
            gt[col] = gt[col].ffill().bfill()
        elif _is_identifier_column(col, gt[col]):
            gt[col] = gt[col].fillna("Unknown")
        else:
            gt[col] = gt[col].fillna("Unknown")

    # Second dedup: null-filling can create NEW duplicates when rows that
    # differed only by NaN vs a value become identical after fill.
    gt = gt.drop_duplicates().reset_index(drop=True)

    # Convert pre-detected date columns to datetime (all difficulty levels).
    for col in date_cols:
        if col not in gt.columns:
            continue
        if not pd.api.types.is_string_dtype(gt[col]):
            continue
        try:
            gt[col] = pd.to_datetime(gt[col], errors="coerce", format="mixed")
        except Exception:
            try:
                gt[col] = pd.to_datetime(gt[col], errors="coerce")
            except Exception:
                pass

    if difficulty == "easy":
        return gt.reset_index(drop=True)

    # --- MEDIUM: + type conversion + strip whitespace + categorical mapping ---
    str_cols = [c for c in gt.columns if pd.api.types.is_string_dtype(gt[c])]
    for col in str_cols:
        # Strip whitespace
        try:
            gt[col] = gt[col].astype(str).str.strip()
        except Exception:
            pass

        sample = gt[col].dropna().head(20)
        if len(sample) == 0:
            continue

        # Map boolean-like values
        try:
            unique_lower = set(sample.astype(str).str.lower().unique())
            bool_vals = {"yes", "no", "true", "false", "1", "0"}
            if unique_lower & bool_vals and len(unique_lower) <= 10:
                bool_map = {
                    "yes": True, "Yes": True, "YES": True,
                    "no": False, "No": False, "NO": False,
                    "true": True, "True": True, "TRUE": True,
                    "false": False, "False": False, "FALSE": False,
                    "1": True, "0": False,
                }
                gt[col] = gt[col].map(lambda v: bool_map.get(str(v), bool_map.get(v, v)) if pd.notna(v) else v)
                continue
        except Exception:
            pass

        numeric_count = pd.to_numeric(sample, errors="coerce").notna().sum()
        if numeric_count > len(sample) * 0.7:
            gt[col] = pd.to_numeric(gt[col], errors="coerce")
            continue

        # Try datetime
        try:
            parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
            if parsed.notna().sum() > len(sample) * 0.7:
                gt[col] = pd.to_datetime(gt[col], errors="coerce", format="mixed")
        except Exception:
            pass

    if difficulty == "medium":
        return gt.reset_index(drop=True)

    # --- HARD: + remove outliers ---
    for col in gt.select_dtypes(include=["number"]).columns:
        s = gt[col].dropna()
        if len(s) < 10:
            continue
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            continue
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        mask = (gt[col] >= lower) & (gt[col] <= upper) | gt[col].isna()
        gt = gt[mask]

    gt = gt.reset_index(drop=True)
    return gt


def auto_generate_descriptions(
    df: pd.DataFrame, difficulty: str = "hard",
    date_cols: set | None = None,
) -> Dict[str, str]:
    """Generate human-readable column descriptions from data inspection.

    Descriptions include cleaning hints appropriate to the difficulty level.
    """
    descriptions = {}
    for col in df.columns:
        parts = []
        dtype = str(df[col].dtype)
        null_count = int(df[col].isnull().sum())

        if "int" in dtype or "float" in dtype:
            non_null = df[col].dropna()
            if len(non_null) > 0:
                parts.append(
                    f"Numeric column, range {non_null.min():.1f} to {non_null.max():.1f}"
                )
            else:
                parts.append("Numeric column (all values missing)")
            if null_count > 0:
                parts.append(f"{null_count} missing values — fill with median")
            if difficulty == "hard":
                # Check for outliers
                if len(non_null) >= 10:
                    q1, q3 = non_null.quantile(0.25), non_null.quantile(0.75)
                    iqr = q3 - q1
                    if iqr > 0:
                        n_out = int(((non_null < q1 - 1.5 * iqr) | (non_null > q3 + 1.5 * iqr)).sum())
                        if n_out > 0:
                            parts.append(f"Contains {n_out} outliers — remove using IQR method")
        elif "datetime" in dtype:
            parts.append("Datetime column")
        elif pd.api.types.is_string_dtype(df[col]):
            n_unique = df[col].nunique()
            sample = df[col].dropna().head(5).tolist()
            sample20 = df[col].dropna().head(20)

            # Check for whitespace
            has_whitespace = False
            if len(sample20) > 0:
                try:
                    stripped = sample20.astype(str).str.strip()
                    has_whitespace = not (sample20.astype(str) == stripped).all()
                except Exception:
                    pass

            if has_whitespace and difficulty in ("medium", "hard"):
                parts.append("Has trailing whitespace — strip it")

            # Check if boolean-like
            unique_lower = set()
            if len(sample20) > 0:
                try:
                    unique_lower = set(sample20.astype(str).str.lower().unique())
                except Exception:
                    pass
            bool_vals = {"yes", "no", "true", "false", "1", "0"}
            if unique_lower & bool_vals and len(unique_lower) <= 10:
                if difficulty in ("medium", "hard"):
                    parts.append("Should be boolean. Currently mixed ('yes'/'no', 1/0, True/False) — map to boolean")
                else:
                    parts.append(f"Categorical with {n_unique} unique values")
            # Check if it looks numeric
            elif len(sample20) > 0:
                numeric_test = pd.to_numeric(sample20, errors="coerce")
                if numeric_test.notna().sum() > 10:
                    if difficulty in ("medium", "hard"):
                        parts.append("Stored as string but looks numeric — convert to float")
                    else:
                        parts.append(f"String column ({n_unique} unique values)")
                elif n_unique < 20:
                    parts.append(f"Categorical with {n_unique} unique values")
                else:
                    parts.append(f"String column ({n_unique} unique values)")
            else:
                parts.append(f"String column ({n_unique} unique values)")

            # Check for date-like strings (all difficulty levels — dates as strings
            # are always a quality issue and the GT converts them).
            # Use shared date_cols when available to stay in sync with the GT.
            col_is_date = col in date_cols if date_cols is not None else False
            if not col_is_date and len(sample20) > 0:
                try:
                    parsed = pd.to_datetime(sample20, errors="coerce", format="mixed")
                    col_is_date = parsed.notna().sum() > len(sample20) * 0.5
                except Exception:
                    pass
            if col_is_date:
                parts.append("Looks like dates stored as string — convert to datetime")

            if null_count > 0:
                if col_is_date:
                    parts.append(f"{null_count} missing values — use forward fill")
                elif _is_identifier_column(col, df[col]):
                    parts.append(f"{null_count} missing values — fill with constant 'Unknown'")
                else:
                    parts.append(f"{null_count} missing values — fill with constant 'Unknown'")

            if sample:
                parts.append(f"Sample: {sample[:3]}")
        elif dtype == "bool":
            parts.append("Boolean column")
        else:
            parts.append(f"Column type: {dtype}")

        descriptions[col] = ". ".join(parts) + "."

    return descriptions


def generate_custom_task(
    raw_df: pd.DataFrame,
    ground_truth_df: pd.DataFrame | None = None,
    budget: int = 20,
    difficulty: str = "hard",
) -> Tuple[pd.DataFrame, pd.DataFrame, int, Dict[str, str]]:
    """
    Build a task from user-uploaded data.

    If no ground truth is provided, one is auto-generated based on difficulty.
    difficulty: "easy", "medium", or "hard"
    """
    # Detect date columns once — shared between GT and descriptions so they
    # always agree on which columns to convert to datetime.
    date_cols = {
        col for col in raw_df.columns
        if pd.api.types.is_string_dtype(raw_df[col]) and _is_date_column(raw_df[col])
    }

    if ground_truth_df is None:
        ground_truth_df = auto_generate_ground_truth(
            raw_df, difficulty=difficulty, date_cols=date_cols
        )

    column_descriptions = auto_generate_descriptions(
        raw_df, difficulty=difficulty, date_cols=date_cols
    )

    return raw_df.copy(), ground_truth_df, budget, column_descriptions
