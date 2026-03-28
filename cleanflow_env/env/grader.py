from __future__ import annotations

from typing import Any, Dict, List

from cleanflow_env.env.rewards import compute_quality
from cleanflow_env.env.state import EnvironmentState
from cleanflow_env.env.validation import validate_cleaned_data


def final_score(state: EnvironmentState) -> Dict[str, float]:
    """
    Compute the final grading score for a completed episode.

    The final score extends the per-step quality signal with episode-level
    metrics that only make sense at termination:

    Score = 0.40 * quality_overall       (same metric used for per-step reward)
          + 0.20 * validation            (data validation rules)
          + 0.15 * efficiency            (budget utilization)
          + 0.10 * action_quality        (redundancy penalty)
          + 0.15 * schema_accuracy       (type correctness)

    This ensures the final score is aligned with (but richer than) the
    per-step signal — an agent optimizing per-step reward also pushes
    toward a higher final score.
    """
    quality = compute_quality(state.current_table, state.ground_truth)

    # Per-step quality components (already computed and used during episode)
    quality_overall = quality["overall"]
    correctness = quality["correctness"]
    completeness = quality["completeness"]
    schema_accuracy = quality["schema_accuracy"]

    # Efficiency: 1 - (budget_used / total_budget)
    budget_used = state.initial_budget - state.budget_remaining
    if state.initial_budget > 0:
        efficiency = 1.0 - (budget_used / state.initial_budget)
    else:
        efficiency = 1.0

    # Action quality: 1.0 - (redundant_actions / total_actions)
    history = state.operations_history
    if len(history) == 0:
        action_quality = 1.0
    else:
        seen = set()
        redundant = 0
        for op in history:
            key = (op.get("action_type"), op.get("column"))
            if key in seen:
                redundant += 1
            seen.add(key)
        action_quality = 1.0 - (redundant / len(history))

    # Validation: fraction of data validation rules passed
    validation_result = validate_cleaned_data(
        state.current_table, state.ground_truth, state.column_descriptions
    )
    validation = validation_result["validation_score"]

    # Clamp all to [0, 1]
    quality_overall = max(0.0, min(1.0, quality_overall))
    validation = max(0.0, min(1.0, validation))
    efficiency = max(0.0, min(1.0, efficiency))
    action_quality = max(0.0, min(1.0, action_quality))
    schema_accuracy = max(0.0, min(1.0, schema_accuracy))

    score = (
        0.40 * quality_overall
        + 0.20 * validation
        + 0.15 * efficiency
        + 0.10 * action_quality
        + 0.15 * schema_accuracy
    )
    score = max(0.0, min(1.0, score))

    return {
        "score": round(score, 6),
        "correctness": round(correctness, 6),
        "completeness": round(completeness, 6),
        "schema_accuracy": round(schema_accuracy, 6),
        "quality_overall": round(quality_overall, 6),
        "efficiency": round(efficiency, 6),
        "action_quality": round(action_quality, 6),
        "validation": round(validation, 6),
        "validation_details": validation_result,
    }


def score_breakdown_report(state: EnvironmentState) -> str:
    """Return a human-readable grading report."""
    result = final_score(state)
    budget_used = state.initial_budget - state.budget_remaining

    # Validation details
    v = result.get("validation_details", {})
    violations = v.get("violations", [])
    v_lines = ""
    if violations:
        v_lines = "\nValidation violations:\n" + "\n".join(
            f"  - {viol}" for viol in violations
        ) + "\n"

    return (
        f"=== CleanFlowEnv Grader Report ===\n"
        f"Task: {state.task_id}\n"
        f"Steps used: {state.step_count} / 20\n"
        f"Budget used: {budget_used} / {state.initial_budget}\n"
        f"\n"
        f"--- Per-step quality (same as reward signal) ---\n"
        f"  Correctness:      {result['correctness']:.4f}\n"
        f"  Completeness:     {result['completeness']:.4f}\n"
        f"  Schema Accuracy:  {result['schema_accuracy']:.4f}\n"
        f"  Quality Overall:  {result['quality_overall']:.4f}  (0.6*corr + 0.3*comp + 0.1*schema)\n"
        f"\n"
        f"--- Final score components ---\n"
        f"  Quality Overall:  {result['quality_overall']:.4f}  (weight: 40%)\n"
        f"  Validation:       {result['validation']:.4f}  (weight: 20%)\n"
        f"  Efficiency:       {result['efficiency']:.4f}  (weight: 15%)\n"
        f"  Action Quality:   {result['action_quality']:.4f}  (weight: 10%)\n"
        f"  Schema Accuracy:  {result['schema_accuracy']:.4f}  (weight: 15%)\n"
        f"  Rules passed: {v.get('rules_passed', 0)} / {v.get('rules_total', 0)}\n"
        f"{v_lines}"
        f"\n"
        f"FINAL SCORE: {result['score']:.4f}\n"
        f"=================================="
    )
