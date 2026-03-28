from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd


def generate_medium_task() -> Tuple[pd.DataFrame, pd.DataFrame, int, Dict[str, str]]:
    """
    Generate Task 2: Schema Normalization (Medium).

    Dataset: Customer transaction records, 300 rows, 6 columns.
    Issues: mixed date formats, amount as string with currency symbols,
    mixed boolean is_active, trailing whitespace in country_code.
    """
    np.random.seed(42)

    n_rows = 300

    customer_ids = [f"CUST_{i:04d}" for i in range(n_rows)]

    # Amount as float (ground truth), then corrupt to string with $ and ,
    amounts_clean = np.round(np.random.uniform(10, 5000, size=n_rows), 2)
    amounts_messy = []
    for a in amounts_clean:
        amounts_messy.append(f"${a:,.2f}")

    # Transaction dates — clean as datetime, messy as mixed format strings
    # Use daily freq so no time component is lost when formatting to date-only strings
    base_dates = pd.date_range("2022-01-01", periods=n_rows, freq="D")
    date_formats = ["%d-%b-%Y", "%Y/%m/%d", "%b %d %Y"]
    dates_messy = []
    for i, d in enumerate(base_dates):
        fmt = date_formats[i % len(date_formats)]
        dates_messy.append(d.strftime(fmt))

    categories = np.random.choice(
        ["Electronics", "Clothing", "Food", "Books", "Home", "Sports"], size=n_rows
    )

    # Country codes with trailing whitespace on ~30%
    countries_clean = np.random.choice(["US", "UK", "CA", "DE", "FR", "JP", "AU"], size=n_rows)
    countries_messy = []
    for i, c in enumerate(countries_clean):
        if i % 3 == 0:
            countries_messy.append(c + "  ")
        else:
            countries_messy.append(c)

    # is_active — mixed booleans
    is_active_clean = np.random.choice([True, False], size=n_rows)
    is_active_messy = []
    for i, val in enumerate(is_active_clean):
        choice = i % 4
        if choice == 0:
            is_active_messy.append("yes" if val else "no")
        elif choice == 1:
            is_active_messy.append(1 if val else 0)
        elif choice == 2:
            is_active_messy.append("Yes" if val else "No")
        else:
            is_active_messy.append(True if val else False)

    raw = pd.DataFrame({
        "customer_id": customer_ids,
        "amount": amounts_messy,
        "transaction_date": dates_messy,
        "category": categories,
        "country_code": countries_messy,
        "is_active": is_active_messy,
    })

    # Ground truth
    gt = pd.DataFrame({
        "customer_id": customer_ids,
        "amount": amounts_clean,
        "transaction_date": base_dates,
        "category": categories,
        "country_code": list(countries_clean),
        "is_active": list(is_active_clean),
    })

    column_descriptions = {
        "customer_id": "Unique customer identifier. String, no cleaning needed.",
        "amount": "Transaction amount in USD. Should be numeric float. Currently stored as string with '$' and ',' characters. Remove '$' and ',' then convert to float.",
        "transaction_date": "Date of transaction. Should be datetime. Currently stored as mixed-format strings.",
        "category": "Product category. Categorical string, no cleaning needed.",
        "country_code": "Two-letter country code. String, has trailing whitespace that should be stripped.",
        "is_active": "Whether customer account is active. Should be boolean. Currently mixed ('yes'/'no', 1/0, True/False). Map to boolean values.",
    }

    return raw, gt, 20, column_descriptions
