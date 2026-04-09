from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def generate_multi_task() -> Tuple[
    Dict[str, pd.DataFrame],
    Dict[str, pd.DataFrame],
    int,
    Dict[str, Dict[str, str]],
    List[Dict[str, str]],
    str,
]:
    """
    Generate Task: Multi-Table Cleaning (Expert+).

    Two linked tables — customers (100 rows) and orders (300 rows).
    orders.customer_id references customers.id.

    Issues in customers:
    - 8 null emails, 5 null cities, whitespace in names, 5 duplicate rows

    Issues in orders:
    - 12 null amounts, 6 null statuses, 10 orphan customer_ids,
      "$" in amount strings, 8 duplicate rows, mixed date formats

    Returns 6-tuple: (raw_tables, gt_tables, budget, col_desc_multi,
                       relationships, primary_table)
    """
    np.random.seed(42)

    # ---- Customers table ----
    n_customers = 100
    cities = ["New York", "Los Angeles", "Chicago", "Houston", "Phoenix",
              "Philadelphia", "San Antonio", "San Diego", "Dallas", "Austin"]

    cust_ids = [f"CUST_{i:04d}" for i in range(n_customers)]
    cust_names = [f"Customer {i}" for i in range(n_customers)]
    cust_emails = [f"customer{i}@example.com" for i in range(n_customers)]
    cust_cities = np.random.choice(cities, size=n_customers).tolist()
    cust_signup = pd.date_range("2020-01-01", periods=n_customers, freq="7D").strftime("%Y-%m-%d").tolist()

    customers_raw = pd.DataFrame({
        "id": cust_ids,
        "name": cust_names,
        "email": cust_emails,
        "city": cust_cities,
        "signup_date": cust_signup,
    })

    # Inject whitespace in names
    ws_idx = np.random.choice(n_customers, size=15, replace=False)
    for i in ws_idx:
        customers_raw.at[i, "name"] = f"  {customers_raw.at[i, 'name']}  "

    # Inject null emails
    email_null_idx = np.random.choice(n_customers, size=8, replace=False)
    customers_raw.loc[email_null_idx, "email"] = None

    # Inject null cities
    city_null_idx = np.random.choice(n_customers, size=5, replace=False)
    customers_raw.loc[city_null_idx, "city"] = None

    # Inject 5 duplicate rows
    dup_idx = np.random.choice(n_customers, size=5, replace=False)
    customers_raw = pd.concat([customers_raw, customers_raw.iloc[dup_idx]], ignore_index=True)

    # ---- Orders table ----
    n_orders = 300
    products = ["Widget A", "Widget B", "Gadget X", "Gadget Y", "Tool Z",
                "Part Alpha", "Part Beta", "Module C", "Module D", "Kit E"]
    statuses = ["completed", "pending", "shipped", "cancelled"]

    order_ids = [f"ORD_{i:05d}" for i in range(n_orders)]
    # Most customer_ids are valid, 10 will be orphans
    valid_cust_ids = np.random.choice(cust_ids, size=n_orders - 10).tolist()
    orphan_ids = [f"CUST_{9000 + i:04d}" for i in range(10)]
    all_cust_refs = valid_cust_ids + orphan_ids
    np.random.shuffle(all_cust_refs)

    amounts = np.round(np.random.uniform(10, 5000, size=n_orders), 2)
    order_products = np.random.choice(products, size=n_orders).tolist()
    order_statuses = np.random.choice(statuses, size=n_orders).tolist()
    order_dates = pd.date_range("2022-01-01", periods=n_orders, freq="1D").strftime("%Y-%m-%d").tolist()

    orders_raw = pd.DataFrame({
        "order_id": order_ids,
        "customer_id": all_cust_refs,
        "product": order_products,
        "amount": [f"${a:.2f}" for a in amounts],  # "$123.45" format — needs cleaning
        "status": order_statuses,
        "order_date": order_dates,
    })

    # Inject null amounts
    amt_null_idx = np.random.choice(n_orders, size=12, replace=False)
    orders_raw.loc[amt_null_idx, "amount"] = None

    # Inject null statuses
    stat_null_idx = np.random.choice(n_orders, size=6, replace=False)
    orders_raw.loc[stat_null_idx, "status"] = None

    # Inject 8 duplicate rows
    dup_idx = np.random.choice(n_orders, size=8, replace=False)
    orders_raw = pd.concat([orders_raw, orders_raw.iloc[dup_idx]], ignore_index=True)

    # ---- Ground truth ----
    # Customers GT
    gt_cust = customers_raw.copy()
    gt_cust = gt_cust.drop_duplicates().reset_index(drop=True)
    gt_cust["name"] = gt_cust["name"].str.strip()
    gt_cust["email"] = gt_cust["email"].fillna("unknown@example.com")
    gt_cust["city"] = gt_cust["city"].fillna(gt_cust["city"].mode().iloc[0])
    gt_cust["signup_date"] = pd.to_datetime(gt_cust["signup_date"], format="mixed")

    # Orders GT
    gt_orders = orders_raw.copy()
    gt_orders = gt_orders.drop_duplicates().reset_index(drop=True)
    # Remove "$" from amount and convert to float
    gt_orders["amount"] = gt_orders["amount"].str.replace("$", "", regex=False)
    gt_orders["amount"] = pd.to_numeric(gt_orders["amount"], errors="coerce")
    # Fill null amounts with median
    gt_orders["amount"] = gt_orders["amount"].fillna(gt_orders["amount"].median())
    # Fill null statuses
    gt_orders["status"] = gt_orders["status"].fillna("pending")
    # Remove orphan customer_ids
    valid_keys = set(gt_cust["id"])
    gt_orders = gt_orders[gt_orders["customer_id"].isin(valid_keys)].reset_index(drop=True)
    # Convert dates
    gt_orders["order_date"] = pd.to_datetime(gt_orders["order_date"], format="mixed")
    # Second dedup after cleaning
    gt_orders = gt_orders.drop_duplicates().reset_index(drop=True)

    raw_tables = {"orders": orders_raw, "customers": customers_raw}
    gt_tables = {"orders": gt_orders, "customers": gt_cust}

    relationships = [
        {
            "from_table": "orders",
            "from_column": "customer_id",
            "to_table": "customers",
            "to_column": "id",
        }
    ]

    column_descriptions = {
        "customers": {
            "id": "Customer identifier. String, sequential pattern (CUST_0000). No cleaning needed.",
            "name": "Customer name. String. Has leading/trailing whitespace — strip whitespace.",
            "email": "Customer email. String. 8 missing values — fill with constant 'unknown@example.com'.",
            "city": "City name. Categorical string. 5 missing values — fill with mode.",
            "signup_date": "Signup date in YYYY-MM-DD. Should be datetime. Convert to datetime.",
        },
        "orders": {
            "order_id": "Order identifier. String, sequential pattern (ORD_00000). No cleaning needed.",
            "customer_id": "Foreign key to customers.id. Some values reference non-existent customers — validate foreign key.",
            "product": "Product name. Categorical string. No cleaning needed.",
            "amount": "Order amount in USD. Contains '$' prefix — remove '$' then convert to float. 12 missing values — fill with median.",
            "status": "Order status. Categorical string. 6 missing values — fill with constant 'pending'.",
            "order_date": "Order date in YYYY-MM-DD. Should be datetime. Convert to datetime.",
        },
    }

    return raw_tables, gt_tables, 25, column_descriptions, relationships, "orders"
