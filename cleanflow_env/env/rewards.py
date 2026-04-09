from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd

from cleanflow_env.models.reward import RewardModel


def compute_quality(current: pd.DataFrame, ground_truth: pd.DataFrame) -> Dict[str, float]:
    """
    Compute quality metrics comparing current_table against ground_truth.

    Returns dict with correctness, completeness, schema_accuracy, and overall.

    Design notes:
    - correctness: cell-level matching, with row-count penalty applied smoothly
      instead of treating extra/missing rows as all-wrong columns.
    - completeness: uses a sharper curve so partial cleaning is clearly
      distinguished from "almost done".
    - schema_accuracy: fraction of columns with matching dtype.
    - overall: weighted combination used as the per-step quality signal.
    """
    _EPS = 1e-4
    if current.empty or ground_truth.empty:
        return {
            "correctness": _EPS,
            "completeness": _EPS,
            "schema_accuracy": _EPS,
            "overall": _EPS,
        }

    # Align on common columns
    common_cols = list(set(current.columns) & set(ground_truth.columns))
    if not common_cols:
        return {
            "correctness": _EPS,
            "completeness": _EPS,
            "schema_accuracy": _EPS,
            "overall": _EPS,
        }

    cur = current[common_cols]
    gt = ground_truth[common_cols]

    # --- Correctness: column-by-column sorted comparison ---
    # Comparing sorted values per column eliminates row-alignment issues caused by
    # outlier removal, dedup, or any operation that changes row count or order.
    # Row-count penalty is applied separately via row_ratio.
    min_rows = min(len(cur), len(gt))
    max_rows = max(len(cur), len(gt))

    if max_rows == 0:
        correctness = 0.0
    else:
        matching = 0
        total_cells = min_rows * len(common_cols)

        for col in common_cols:
            c_col = cur[col]
            g_col = gt[col]

            c_num = pd.to_numeric(c_col, errors="coerce")
            g_num = pd.to_numeric(g_col, errors="coerce")

            # Use numeric path only when the majority of values are numeric
            c_numeric_ratio = c_num.notna().sum() / max(len(c_col), 1)
            g_numeric_ratio = g_num.notna().sum() / max(len(g_col), 1)

            if c_numeric_ratio > 0.5 and g_numeric_ratio > 0.5:
                # Sort non-null numerics independently, then compare with tolerance
                c_sorted = sorted(c_num.dropna().tolist())
                g_sorted = sorted(g_num.dropna().tolist())
                c_null_count = int(c_num.isna().sum())
                g_null_count = int(g_num.isna().sum())

                num_matches = sum(
                    abs(a - b) < 1e-6
                    for a, b in zip(c_sorted[:min_rows], g_sorted[:min_rows])
                )
                null_matches = min(c_null_count, g_null_count)
                col_matches = num_matches + null_matches
            else:
                # String/categorical/datetime: sort lexicographically
                c_vals = _sorted_col_vals(c_col)
                g_vals = _sorted_col_vals(g_col)
                col_matches = sum(a == b for a, b in zip(c_vals[:min_rows], g_vals[:min_rows]))

            matching += min(int(col_matches), min_rows)

        # Cell accuracy among the shorter table's rows
        cell_accuracy = matching / total_cells if total_cells > 0 else 0.0

        # Row-count penalty: penalises having too many or too few rows vs GT.
        # If agent has 371 rows and GT has 371 → ratio 1.0, no penalty.
        # If agent has 510 rows (dups not removed) and GT has 488 → ratio 488/510 ≈ 0.957.
        row_ratio = min_rows / max_rows
        correctness = cell_accuracy * row_ratio

    # --- Completeness: sharper curve ---
    # null_fraction and dup_fraction each independently reduce completeness
    total_cells = cur.shape[0] * cur.shape[1] if cur.size > 0 else 1
    null_fraction = cur.isnull().sum().sum() / total_cells
    dup_fraction = cur.duplicated().sum() / max(len(cur), 1)

    # Use quadratic penalty so partial cleaning is distinguishable from near-done
    # 0% nulls → 1.0, 5% nulls → 0.75, 20% nulls → 0.0
    null_score = max(0.0, 1.0 - (null_fraction / 0.20)) ** 2
    dup_score = max(0.0, 1.0 - (dup_fraction / 0.20)) ** 2
    completeness = null_score * 0.6 + dup_score * 0.4

    # --- Schema accuracy: fraction of columns with matching dtype ---
    schema_matches = 0
    for col in common_cols:
        cur_dtype = str(cur[col].dtype)
        gt_dtype = str(gt[col].dtype)
        if cur_dtype == gt_dtype:
            schema_matches += 1
        elif _dtype_compatible(cur_dtype, gt_dtype):
            schema_matches += 1
    schema_accuracy = schema_matches / len(common_cols) if common_cols else 0.0

    overall = 0.6 * correctness + 0.3 * completeness + 0.1 * schema_accuracy

    # Clamp all to strict (0, 1) — validator rejects exactly 0.0 or 1.0
    import math
    def _clamp(v: float) -> float:
        if not isinstance(v, (int, float)) or not math.isfinite(v):
            return _EPS
        return max(_EPS, min(1.0 - _EPS, float(v)))

    return {
        "correctness": round(_clamp(correctness), 6),
        "completeness": round(_clamp(completeness), 6),
        "schema_accuracy": round(_clamp(schema_accuracy), 6),
        "overall": round(_clamp(overall), 6),
    }


def _sorted_col_vals(series: pd.Series) -> list:
    """Return sorted string representations of a column's values, nulls as '__NULL__'."""
    return sorted(series.fillna("__NULL__").astype(str).tolist())


def _dtype_compatible(dtype1: str, dtype2: str) -> bool:
    """Check if two dtype strings are compatible (e.g. int64 vs Int64, float64 vs float32)."""
    numeric_types = {"int64", "int32", "Int64", "Int32", "float64", "float32"}
    if dtype1 in numeric_types and dtype2 in numeric_types:
        return True
    datetime_types = {"datetime64[ns]", "datetime64[us]", "datetime64[ms]"}
    if dtype1 in datetime_types and dtype2 in datetime_types:
        return True
    # String types
    string_types = {"object", "string", "str", "string[python]", "string[pyarrow]"}
    if dtype1 in string_types and dtype2 in string_types:
        return True
    return False


def compute_fk_integrity(
    tables: Dict[str, pd.DataFrame],
    relationships: List[Dict[str, Any]],
) -> float:
    """Compute foreign key integrity score across table relationships."""
    if not relationships:
        return 1.0
    scores = []
    for rel in relationships:
        child = tables.get(rel["from_table"])
        parent = tables.get(rel["to_table"])
        if child is None or parent is None:
            scores.append(0.0)
            continue
        fk_col = rel["from_column"]
        pk_col = rel["to_column"]
        if fk_col not in child.columns or pk_col not in parent.columns:
            scores.append(0.0)
            continue
        valid_keys = set(parent[pk_col].dropna())
        child_keys = child[fk_col].dropna()
        if len(child_keys) == 0:
            scores.append(1.0)
            continue
        matched = child_keys.isin(valid_keys).sum()
        scores.append(matched / len(child_keys))
    return sum(scores) / len(scores) if scores else 1.0


def compute_quality_multi(
    tables: Dict[str, pd.DataFrame],
    ground_truth_tables: Dict[str, pd.DataFrame],
    relationships: List[Dict[str, Any]] | None = None,
) -> Dict[str, float]:
    """Compute quality metrics across multiple tables.

    Averages per-table quality and adds FK integrity as a component.
    """
    _EPS = 1e-4
    per_table_quality = []
    for name in ground_truth_tables:
        current = tables.get(name)
        gt = ground_truth_tables[name]
        if current is None:
            per_table_quality.append({
                "correctness": _EPS, "completeness": _EPS,
                "schema_accuracy": _EPS, "overall": _EPS,
            })
        else:
            per_table_quality.append(compute_quality(current, gt))

    # Average across tables
    avg = {}
    for key in ("correctness", "completeness", "schema_accuracy", "overall"):
        avg[key] = sum(q[key] for q in per_table_quality) / len(per_table_quality)

    # FK integrity (5% weight, taken from correctness)
    fk_score = compute_fk_integrity(tables, relationships or [])
    import math
    def _clamp(v: float) -> float:
        if not isinstance(v, (int, float)) or not math.isfinite(v):
            return _EPS
        return max(_EPS, min(1.0 - _EPS, float(v)))

    # Adjust overall to include FK integrity
    base_overall = avg["overall"]
    avg["overall"] = round(_clamp(0.90 * base_overall + 0.10 * fk_score), 6)
    avg["fk_integrity"] = round(_clamp(fk_score), 6)
    for key in ("correctness", "completeness", "schema_accuracy"):
        avg[key] = round(_clamp(avg[key]), 6)

    return avg


def compute_reward(
    current: pd.DataFrame,
    ground_truth: pd.DataFrame,
    best_quality_so_far: float,
    action_dict: dict,
    budget_cost: int,
    initial_budget: int,
    operations_history: List[Dict[str, Any]],
    done: bool = False,
    invalid: bool = False,
) -> tuple[RewardModel, float]:
    """
    Compute the reward for a single step.

    Reward formula (calibrated):
        reward = quality_delta * REWARD_SCALE - penalty - normalized_cost

    Where:
    - quality_delta is the improvement over best_quality_so_far (0 if none)
    - REWARD_SCALE amplifies the quality signal so improvements outweigh costs
    - penalty is a flat penalty for invalid/harmful/redundant actions
    - normalized_cost = budget_cost / initial_budget (so cost is always 0-1 range)

    Returns (RewardModel, new_best_quality_so_far).
    """
    REWARD_SCALE = 10.0

    quality = compute_quality(current, ground_truth)
    overall = quality["overall"]

    # Quality delta using high-water mark
    quality_delta = max(0.0, overall - best_quality_so_far)
    new_best = max(best_quality_so_far, overall)

    # Normalize budget cost to 0-1 range
    normalized_cost = budget_cost / max(initial_budget, 1)

    # Penalty calculation
    penalty = 0.0
    if invalid:
        penalty = 0.5
    elif overall < best_quality_so_far - 0.001:
        # Harmful action — quality dropped
        penalty = 0.3 + (best_quality_so_far - overall)  # proportional to damage
    elif _is_redundant(action_dict, operations_history):
        penalty = 0.1

    reward_model = RewardModel.from_step(
        quality_delta=round(quality_delta * REWARD_SCALE, 6),
        penalty=round(penalty, 6),
        budget_cost=round(normalized_cost, 6),
        cumulative_quality=overall,
        done=done,
        info={
            "action": action_dict,
            "quality_breakdown": quality,
            "raw_budget_cost": budget_cost,
        },
    )

    return reward_model, new_best


def _is_redundant(action_dict: dict, history: List[Dict[str, Any]]) -> bool:
    """Check if the same action_type + column was already applied.

    Exception: drop_duplicates is only redundant if it was the immediately
    previous action. A second dedup after null-filling is legitimate because
    filling can create new duplicates.
    """
    action_type = action_dict.get("action_type")
    column = action_dict.get("column")

    if action_type == "drop_duplicates":
        # Only redundant if the very last action was also drop_duplicates
        if history and history[-1].get("action_type") == "drop_duplicates":
            return True
        return False

    for prev in history:
        if prev.get("action_type") == action_type and prev.get("column") == column:
            return True
    return False
