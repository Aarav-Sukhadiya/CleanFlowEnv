from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd


def generate_expert_task() -> Tuple[pd.DataFrame, pd.DataFrame, int, Dict[str, str]]:
    """
    Generate Task 4: Budget-Constrained Cleaning (Expert).

    Dataset: E-commerce product catalog, 500 rows, 10 columns.
    Combines all issues from Tasks 1-3 plus 5 irrelevant distractor columns.
    Budget: 15 (tight).
    """
    np.random.seed(42)

    n_rows = 500

    # --- Core columns (need cleaning) ---
    product_ids = [f"PROD_{i:04d}" for i in range(n_rows)]

    # Price: stored as string with $ (like medium task)
    prices_clean = np.round(np.random.uniform(5, 500, size=n_rows), 2)
    prices_messy = [f"${p:,.2f}" for p in prices_clean]

    # Stock quantity: has nulls (like easy task)
    stock_clean = np.random.randint(0, 1000, size=n_rows).astype(float)
    stock_messy = stock_clean.copy()
    stock_null_idx = np.random.choice(n_rows, size=20, replace=False)
    stock_messy[stock_null_idx] = np.nan

    # Rating: has outliers (like hard task)
    ratings_clean = np.round(np.random.normal(3.5, 0.8, size=n_rows), 2)
    ratings_clean = np.clip(ratings_clean, 1.0, 5.0)
    ratings_messy = ratings_clean.copy()
    r_series = pd.Series(ratings_clean)
    r_q1, r_q3 = r_series.quantile(0.25), r_series.quantile(0.75)
    r_iqr = r_q3 - r_q1
    outlier_idx = np.random.choice(n_rows, size=12, replace=False)
    for idx in outlier_idx:
        ratings_messy[idx] = round(r_q3 + 2 * r_iqr + np.random.uniform(0, 1), 2)

    # Category: has duplicates (easy task)
    categories = np.random.choice(
        ["Electronics", "Clothing", "Home", "Books", "Sports", "Food", "Toys"], size=n_rows
    )

    # Date added: mixed formats (medium task)
    base_dates = pd.date_range("2022-01-01", periods=n_rows, freq="D")
    date_formats = ["%d-%b-%Y", "%Y/%m/%d", "%b %d %Y"]
    dates_messy = []
    for i, d in enumerate(base_dates):
        dates_messy.append(d.strftime(date_formats[i % len(date_formats)]))

    # --- 5 distractor columns (already clean — normalizing them wastes budget) ---
    sku = [f"SKU-{np.random.randint(10000, 99999)}" for _ in range(n_rows)]
    weight = np.round(np.random.uniform(0.1, 50.0, size=n_rows), 2)
    warehouse = np.random.choice(["WH-East", "WH-West", "WH-Central"], size=n_rows)
    supplier_code = [f"SUP_{np.random.randint(100, 999)}" for _ in range(n_rows)]
    barcode = [f"{np.random.randint(1000000000000, 9999999999999)}" for _ in range(n_rows)]

    raw = pd.DataFrame({
        "product_id": product_ids,
        "price": prices_messy,
        "stock_quantity": stock_messy,
        "rating": ratings_messy,
        "category": categories,
        "date_added": dates_messy,
        "sku": sku,
        "weight_kg": weight,
        "warehouse": warehouse,
        "supplier_code": supplier_code,
        "barcode": barcode,
    })

    # Inject 10 duplicate rows
    dup_idx = np.random.choice(n_rows, size=10, replace=False)
    duplicates = raw.iloc[dup_idx].copy()
    raw = pd.concat([raw, duplicates], ignore_index=True)

    # --- Ground truth: start from raw (messy) so IQR bounds and median match agent ---
    gt = raw.copy()

    # Remove duplicates first
    gt = gt.drop_duplicates().reset_index(drop=True)

    # Fix price: remove '$' and ',' then convert to float
    gt["price"] = pd.to_numeric(
        gt["price"].astype(str).str.replace(r"[$,]", "", regex=True), errors="coerce"
    )

    # Fill stock nulls with median (computed from messy data, same as agent will see)
    gt["stock_quantity"] = pd.to_numeric(gt["stock_quantity"], errors="coerce")
    gt["stock_quantity"] = gt["stock_quantity"].fillna(gt["stock_quantity"].median())

    # Second dedup: filling can create new duplicates
    gt = gt.drop_duplicates().reset_index(drop=True)

    # Remove rating outliers using IQR x 1.5 on the messy ratings (same as agent will compute)
    r_col = pd.to_numeric(gt["rating"], errors="coerce")
    q1, q3 = r_col.quantile(0.25), r_col.quantile(0.75)
    iqr = q3 - q1
    mask = (r_col >= q1 - 1.5 * iqr) & (r_col <= q3 + 1.5 * iqr) | r_col.isna()
    gt = gt[mask].reset_index(drop=True)

    # Convert date_added to datetime
    gt["date_added"] = pd.to_datetime(gt["date_added"], format="mixed", errors="coerce")

    column_descriptions = {
        "product_id": "Unique product identifier. String, no cleaning needed.",
        "price": "Product price in USD. Should be numeric float. Currently stored as string with '$' and ',' characters. Remove '$' and ',' then convert to float.",
        "stock_quantity": "Units in stock. Numeric integer. Has missing values — use median to fill.",
        "rating": "Customer rating 1.0-5.0. Numeric float. Contains outliers that should be removed using IQR method.",
        "category": "Product category. Categorical string. Table may have duplicate rows to remove.",
        "date_added": "Date product was added to catalog. Should be datetime. Currently mixed format strings.",
        "sku": "Stock keeping unit code. String, already clean — no action needed.",
        "weight_kg": "Product weight in kilograms. Numeric float, already clean — no action needed.",
        "warehouse": "Assigned warehouse. Categorical string, already clean — no action needed.",
        "supplier_code": "Supplier identifier. String, already clean — no action needed.",
        "barcode": "Product barcode. String, already clean — no action needed.",
    }

    return raw, gt, 15, column_descriptions
