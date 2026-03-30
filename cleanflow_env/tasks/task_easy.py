from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd


def generate_easy_task() -> Tuple[pd.DataFrame, pd.DataFrame, int, Dict[str, str]]:
    """
    Generate Task 1: Basic Cleaning (Easy).

    Dataset: Employee survey data, 200 rows, 5 columns.
    Issues: 15 missing age values, 8 missing salary values, 12 duplicate rows.
    Ground truth: age nulls filled with median, salary nulls filled with median, duplicates removed.
    """
    np.random.seed(42)

    n_rows = 200
    departments = ["Engineering", "Marketing", "Sales", "HR", "Finance", "Operations"]

    # Generate clean base data
    names = [f"Employee_{i:03d}" for i in range(n_rows)]
    ages = np.random.randint(22, 65, size=n_rows).astype(float)
    salaries = np.round(np.random.uniform(30000, 120000, size=n_rows), 2)
    depts = np.random.choice(departments, size=n_rows)
    start_dates = pd.date_range("2015-01-01", periods=n_rows, freq="5D").strftime("%Y-%m-%d").tolist()

    raw = pd.DataFrame({
        "name": names,
        "age": ages,
        "salary": salaries,
        "department": depts,
        "start_date": start_dates,
    })

    # Inject 15 nulls in age
    age_null_idx = np.random.choice(n_rows, size=15, replace=False)
    raw.loc[age_null_idx, "age"] = np.nan

    # Inject 8 nulls in salary
    salary_null_idx = np.random.choice(n_rows, size=8, replace=False)
    raw.loc[salary_null_idx, "salary"] = np.nan

    # Inject 10 nulls in name (sequential identifier — should fill with sequential pattern)
    name_null_idx = np.random.choice(n_rows, size=10, replace=False)
    raw.loc[name_null_idx, "name"] = None

    # Inject 7 nulls in department (categorical — should fill with "Unknown")
    dept_null_idx = np.random.choice(n_rows, size=7, replace=False)
    raw.loc[dept_null_idx, "department"] = None

    # Inject 5 nulls in start_date (date — should forward-fill)
    date_null_idx = np.random.choice(n_rows, size=5, replace=False)
    raw.loc[date_null_idx, "start_date"] = None

    # Inject 12 duplicate rows
    dup_idx = np.random.choice(n_rows, size=12, replace=False)
    duplicates = raw.iloc[dup_idx].copy()
    raw = pd.concat([raw, duplicates], ignore_index=True)

    # Build ground truth from raw — dedup → fill → dedup.
    gt = raw.copy()
    gt = gt.drop_duplicates().reset_index(drop=True)
    # Numeric: fill with median
    gt["age"] = gt["age"].fillna(gt["age"].median())
    gt["age"] = gt["age"].round(1)
    gt["salary"] = gt["salary"].fillna(gt["salary"].median())
    gt["salary"] = gt["salary"].round(2)
    # Sequential identifiers: fill with next values in the pattern
    from cleanflow_env.env.actions import fill_sequential
    gt["name"] = fill_sequential(gt["name"])
    # Categorical: fill with "Unknown"
    gt["department"] = gt["department"].fillna("Unknown")
    # Dates: forward-fill, bfill for leading NaN, then convert to datetime
    gt["start_date"] = gt["start_date"].ffill().bfill()
    gt["start_date"] = pd.to_datetime(gt["start_date"], errors="coerce", format="mixed")
    # Second dedup: filling can create new duplicates
    gt = gt.drop_duplicates().reset_index(drop=True)

    column_descriptions = {
        "name": "Employee identifier. String, sequential pattern (Employee_000, Employee_001, ...). 10 missing values — fill with sequential method.",
        "age": "Employee age in years. Numeric, expected range 18-90. Missing values should use median.",
        "salary": "Annual salary in USD. Numeric, expected range 30000-120000. Missing values should use median.",
        "department": "Department name. Categorical string. 7 missing values — fill with constant 'Unknown'.",
        "start_date": "Employment start date in YYYY-MM-DD. Should be datetime. 5 missing values — use forward fill then convert to datetime.",
    }

    return raw, gt, 20, column_descriptions
