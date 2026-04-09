from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd


def generate_messy_contacts_task() -> Tuple[pd.DataFrame, pd.DataFrame, int, Dict[str, str]]:
    """
    Generate Task 6: Messy Contacts (Medium-Hard).

    Dataset: Contact directory, 250 rows, 7 columns.
    Issues: whitespace in names, mixed phone formats, duplicate entries,
    nulls in email/phone, messy department values, salary with $ prefix.
    """
    np.random.seed(123)

    n_rows = 250
    departments = ["Engineering", "Marketing", "Sales", "HR", "Finance", "Operations", "Legal"]
    cities = ["New York", "San Francisco", "Chicago", "Austin", "Seattle", "Boston", "Denver"]

    # --- Clean base data ---
    first_names = [
        "Alice", "Bob", "Carol", "Dave", "Eve", "Frank", "Grace", "Hank",
        "Iris", "Jack", "Karen", "Leo", "Mia", "Nick", "Olivia", "Paul",
        "Quinn", "Rosa", "Sam", "Tina", "Uma", "Vic", "Wendy", "Xander", "Yara", "Zane",
    ]
    last_names = [
        "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller",
        "Davis", "Rodriguez", "Martinez", "Anderson", "Taylor", "Thomas", "Jackson",
    ]

    names_clean = [
        f"{first_names[i % len(first_names)]} {last_names[i % len(last_names)]}"
        for i in range(n_rows)
    ]

    # Phone numbers — clean format: (555) 123-4567
    phones_clean = [f"(555) {np.random.randint(100, 999)}-{np.random.randint(1000, 9999)}" for _ in range(n_rows)]

    emails_clean = [f"{names_clean[i].lower().replace(' ', '.')}@company.com" for i in range(n_rows)]
    depts_clean = np.random.choice(departments, size=n_rows)
    cities_clean = np.random.choice(cities, size=n_rows)
    salaries_clean = np.round(np.random.uniform(45000, 150000, size=n_rows), 2)
    hire_dates_clean = pd.date_range("2018-01-01", periods=n_rows, freq="3D")

    # --- Build messy raw ---

    # Names with whitespace issues
    names_messy = []
    for i, name in enumerate(names_clean):
        if i % 5 == 0:
            names_messy.append(f"  {name}  ")
        elif i % 7 == 0:
            names_messy.append(f"{name}   ")
        else:
            names_messy.append(name)

    # Phones — mixed formats
    phones_messy = []
    for i, phone in enumerate(phones_clean):
        # Extract digits
        digits = phone.replace("(", "").replace(")", "").replace(" ", "").replace("-", "")
        fmt = i % 4
        if fmt == 0:
            phones_messy.append(phone)  # (555) 123-4567
        elif fmt == 1:
            phones_messy.append(f"{digits[:3]}-{digits[3:6]}-{digits[6:]}")  # 555-123-4567
        elif fmt == 2:
            phones_messy.append(f"{digits[:3]}.{digits[3:6]}.{digits[6:]}")  # 555.123.4567
        else:
            phones_messy.append(digits)  # 5551234567

    # Inject 12 nulls in email
    email_list = list(emails_clean)
    email_null_idx = np.random.choice(n_rows, size=12, replace=False)
    for idx in email_null_idx:
        email_list[idx] = None

    # Inject 8 nulls in phone
    phone_null_idx = np.random.choice(n_rows, size=8, replace=False)
    for idx in phone_null_idx:
        phones_messy[idx] = None

    # Inject 6 nulls in salary
    salaries_messy = [f"${s:,.2f}" for s in salaries_clean]
    salary_null_idx = np.random.choice(n_rows, size=6, replace=False)
    for idx in salary_null_idx:
        salaries_messy[idx] = None

    # Departments with case issues
    depts_messy = []
    for i, d in enumerate(depts_clean):
        if i % 6 == 0:
            depts_messy.append(d.upper())
        elif i % 6 == 1:
            depts_messy.append(d.lower())
        else:
            depts_messy.append(d)

    # Hire dates as mixed format strings
    date_formats = ["%Y-%m-%d", "%m/%d/%Y", "%d-%b-%Y"]
    dates_messy = []
    for i, d in enumerate(hire_dates_clean):
        fmt = date_formats[i % len(date_formats)]
        dates_messy.append(d.strftime(fmt))

    raw = pd.DataFrame({
        "name": names_messy,
        "email": email_list,
        "phone": phones_messy,
        "department": depts_messy,
        "city": list(cities_clean),
        "salary": salaries_messy,
        "hire_date": dates_messy,
    })

    # Inject 10 duplicate rows
    dup_idx = np.random.choice(n_rows, size=10, replace=False)
    duplicates = raw.iloc[dup_idx].copy()
    raw = pd.concat([raw, duplicates], ignore_index=True)

    # --- Ground truth ---
    gt = raw.copy()
    gt = gt.drop_duplicates().reset_index(drop=True)

    # Strip whitespace from names
    gt["name"] = gt["name"].str.strip()

    # Fill email nulls with constant
    gt["email"] = gt["email"].fillna("unknown@company.com")

    # Fill phone nulls with constant
    gt["phone"] = gt["phone"].fillna("Unknown")

    # Salary: remove $ and , → float, fill nulls with median
    gt["salary"] = gt["salary"].str.replace("$", "", regex=False).str.replace(",", "", regex=False)
    gt["salary"] = pd.to_numeric(gt["salary"], errors="coerce")
    gt["salary"] = gt["salary"].fillna(gt["salary"].median())

    # Convert hire_date to datetime
    gt["hire_date"] = pd.to_datetime(gt["hire_date"], format="mixed", errors="coerce")

    # Dedup again after fills
    gt = gt.drop_duplicates().reset_index(drop=True)

    column_descriptions = {
        "name": "Contact full name. String. Has leading/trailing whitespace that should be stripped.",
        "email": "Email address. String. 12 missing values — fill with constant 'unknown@company.com'.",
        "phone": "Phone number. String with mixed formats. 8 missing values — fill with constant 'Unknown'. No type conversion needed — this is not a date.",
        "department": "Department name. Categorical string, no cleaning needed.",
        "city": "City name. Categorical string, no cleaning needed.",
        "salary": "Annual salary in USD. Currently stored as string with '$' and ',' characters. Remove '$' and ',' then convert to float. 6 missing values — fill with median after conversion.",
        "hire_date": "Date of hire. Should be datetime. Currently stored as mixed-format strings.",
    }

    return raw, gt, 20, column_descriptions
