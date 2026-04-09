from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd


def generate_hard_task() -> Tuple[pd.DataFrame, pd.DataFrame, int, Dict[str, str]]:
    """
    Generate Task 3: Advanced Cleaning (Hard).

    Dataset: Medical trial records, 400 rows, 8 columns.
    Issues: outliers in blood_pressure/cholesterol, mixed patient_id formats,
    subtle year typos in visit_date.
    """
    np.random.seed(42)

    n_rows = 400
    treatments = ["Drug_A", "Drug_B", "Placebo", "Drug_C"]
    outcomes = ["Improved", "No Change", "Worsened", "Remission"]

    # --- Clean base data ---
    ages = np.random.randint(25, 75, size=n_rows).astype(float)
    blood_pressure = np.round(np.random.normal(120, 15, size=n_rows), 1)
    cholesterol = np.round(np.random.normal(200, 30, size=n_rows), 1)
    treatment = np.random.choice(treatments, size=n_rows)
    outcome = np.random.choice(outcomes, size=n_rows)
    dosage = np.round(np.random.uniform(5, 100, size=n_rows), 1)

    # Patient IDs — clean format P001, P002, ...
    patient_ids_clean = [f"P{i:03d}" for i in range(n_rows)]

    # Visit dates — clean
    visit_dates_clean = pd.date_range("2023-01-01", periods=n_rows, freq="6h")

    # --- Build messy raw ---

    # patient_id: mixed formats
    patient_ids_messy = []
    for i in range(n_rows):
        fmt = i % 3
        if fmt == 0:
            patient_ids_messy.append(f"P{i:03d}")
        elif fmt == 1:
            patient_ids_messy.append(str(i))
        else:
            patient_ids_messy.append(f"{i:03d}")

    # visit_date: 20 rows have year 2033 instead of 2023
    visit_dates_messy = visit_dates_clean.copy()
    typo_idx = np.random.choice(n_rows, size=20, replace=False)
    visit_dates_str = []
    for i, d in enumerate(visit_dates_messy):
        if i in typo_idx:
            visit_dates_str.append(d.strftime("%Y-%m-%d %H:%M:%S").replace("2023", "2033"))
        else:
            visit_dates_str.append(d.strftime("%Y-%m-%d %H:%M:%S"))

    # blood_pressure outliers: inject values at Q3 + 2*IQR
    bp_series = pd.Series(blood_pressure)
    bp_q1, bp_q3 = bp_series.quantile(0.25), bp_series.quantile(0.75)
    bp_iqr = bp_q3 - bp_q1
    bp_outlier_idx = np.random.choice(n_rows, size=15, replace=False)
    blood_pressure_messy = blood_pressure.copy()
    for idx in bp_outlier_idx:
        blood_pressure_messy[idx] = round(bp_q3 + 2 * bp_iqr + np.random.uniform(0, 10), 1)

    # cholesterol outliers
    ch_series = pd.Series(cholesterol)
    ch_q1, ch_q3 = ch_series.quantile(0.25), ch_series.quantile(0.75)
    ch_iqr = ch_q3 - ch_q1
    ch_outlier_idx = np.random.choice(n_rows, size=10, replace=False)
    cholesterol_messy = cholesterol.copy()
    for idx in ch_outlier_idx:
        cholesterol_messy[idx] = round(ch_q3 + 2 * ch_iqr + np.random.uniform(0, 15), 1)

    raw = pd.DataFrame({
        "patient_id": patient_ids_messy,
        "age": ages,
        "blood_pressure": blood_pressure_messy,
        "cholesterol": cholesterol_messy,
        "visit_date": visit_dates_str,
        "treatment": treatment,
        "outcome": outcome,
        "dosage": dosage,
    })

    # --- Ground truth ---
    # Start from the messy data (same as what the agent sees) so IQR bounds match
    gt = raw.copy()
    gt["patient_id"] = patient_ids_clean
    gt["visit_date"] = visit_dates_clean

    # Remove outliers using IQR x 1.5 on the messy values (same as agent will compute)
    for col in ["blood_pressure", "cholesterol"]:
        s = pd.to_numeric(gt[col], errors="coerce")
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        mask = (s >= q1 - 1.5 * iqr) & (s <= q3 + 1.5 * iqr) | s.isna()
        gt = gt[mask]
    gt = gt.reset_index(drop=True)

    column_descriptions = {
        "patient_id": "Patient identifier. String, expected pattern 'P001'. Currently mixed (e.g. plain numbers, zero-padded). No type conversion needed — this is not a date.",
        "age": "Patient age in years. Numeric integer, range 18-90. No cleaning needed.",
        "blood_pressure": "Systolic blood pressure mmHg. Numeric, expected range 80-180. Contains outliers that should be removed using IQR method.",
        "cholesterol": "Total cholesterol mg/dL. Numeric, expected range 120-300. Contains outliers that should be removed using IQR method.",
        "visit_date": "Date of clinical visit. Should be datetime. Some entries have year typo (2033 instead of 2023). Replace 2033 with 2023 then convert to datetime.",
        "treatment": "Treatment group assignment. Categorical string, no cleaning needed.",
        "outcome": "Treatment outcome. Categorical string, no cleaning needed.",
        "dosage": "Drug dosage in mg. Numeric float, no cleaning needed.",
    }

    return raw, gt, 20, column_descriptions
