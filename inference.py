"""
Inference script for CleanFlowEnv — OpenEnv Hackathon submission.

Uses the HuggingFace Inference API to run an LLM-based data cleaning agent
against all tasks in the environment.

Required environment variables:
  API_BASE_URL  — LLM endpoint (default: https://router.huggingface.co/v1)
  MODEL_NAME    — model to use (default: Qwen/Qwen2.5-72B-Instruct)
  HF_TOKEN      — Hugging Face token for authentication
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

import requests

# ---------------------------------------------------------------------------
# Configuration from environment
# ---------------------------------------------------------------------------
API_BASE_URL = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.environ.get("HF_TOKEN", "")
ENV_BASE_URL = os.environ.get("ENV_BASE_URL", "http://localhost:7860")

# ---------------------------------------------------------------------------
# Action schema (passed to the LLM so it knows what it can do)
# ---------------------------------------------------------------------------
ACTION_SCHEMA = """You are a data cleaning agent for CleanFlowEnv. You must return ONE JSON action per turn.

Available action types and their required fields:

1. fill_null — Fill missing values in a column (cost: 1)
   Required: action_type, column, method
   method options: "mean", "median", "mode", "constant", "forward_fill", "backward_fill"
   If method is "constant", also provide "constant_value".

2. drop_duplicates — Remove duplicate rows (cost: 1)
   Required: action_type (no column needed)

3. strip_whitespace — Strip leading/trailing whitespace from string column (cost: 1)
   Required: action_type, column

4. replace_substring — Replace a substring in string values (cost: 1)
   Required: action_type, column, old_value, new_value
   Example: remove "$" from prices: {"action_type": "replace_substring", "column": "price", "old_value": "$", "new_value": ""}

5. convert_type — Convert a column's data type (cost: 2)
   Required: action_type, column, target_type
   target_type options: "int", "float", "datetime", "string"

6. map_values — Map categorical values using a dict (cost: 2)
   Required: action_type, column, mapping
   Example: {"action_type": "map_values", "column": "is_active", "mapping": {"yes": true, "no": false, "Yes": true, "No": false}}

7. normalize — Normalize a numeric column (cost: 2)
   Required: action_type, column (uses min-max normalization)

8. remove_outliers — Remove outlier rows using IQR x 1.5 (cost: 3)
   Required: action_type, column

Example actions:
  {"action_type": "fill_null", "column": "age", "method": "mean"}
  {"action_type": "drop_duplicates"}
  {"action_type": "strip_whitespace", "column": "country_code"}
  {"action_type": "replace_substring", "column": "price", "old_value": "$", "new_value": ""}
  {"action_type": "map_values", "column": "is_active", "mapping": {"yes": true, "no": false}}
  {"action_type": "convert_type", "column": "price", "target_type": "float"}
  {"action_type": "remove_outliers", "column": "salary"}
"""


def build_prompt(obs: Dict[str, Any], action_history: List[Dict]) -> str:
    """Build the LLM prompt from the current observation."""
    # Summarize key info compactly to fit in context
    null_info = {k: v for k, v in obs.get("null_counts", {}).items() if v > 0}
    schema = obs.get("schema", {})
    budget = obs.get("budget_remaining", "unknown")
    step = obs.get("step_count", 0)
    dupes = obs.get("duplicate_count", 0)
    col_desc = obs.get("column_descriptions", {})

    # Format table preview
    preview_rows = obs.get("table_preview", [])
    preview_str = ""
    if preview_rows:
        cols = list(preview_rows[0].get("values", {}).keys()) if preview_rows else []
        preview_str = f"Columns: {cols}\n"
        for row in preview_rows[:3]:
            preview_str += f"  Row {row.get('row_index', '?')}: {row.get('values', {})}\n"

    history_str = ""
    if action_history:
        history_str = "Actions taken so far:\n"
        for i, a in enumerate(action_history):
            history_str += f"  {i+1}. {json.dumps(a)}\n"

    prompt = f"""{ACTION_SCHEMA}

Current observation:
- Step: {step}
- Budget remaining: {budget}
- Duplicate rows: {dupes}
- Columns with nulls: {json.dumps(null_info) if null_info else "none"}
- Column types: {json.dumps(schema)}
- Column descriptions: {json.dumps(col_desc, default=str)}

Table preview (first 3 rows):
{preview_str}
{history_str}

Strategy:
1. First drop duplicates if any exist (cost 1)
2. Fill nulls in columns that have missing values (cost 1 each)
3. Convert types where column descriptions suggest type mismatch (cost 2 each)
4. Remove outliers in numeric columns if budget allows (cost 3 each)
5. Stop when budget is low or all issues are fixed

Based on the observation above, decide the single best next action. If all issues are resolved or budget is too low for useful actions, return {{"action_type": "drop_duplicates"}} as a low-cost finishing move.

Return ONLY a valid JSON object with the action. No explanation, no markdown."""

    return prompt


def parse_action(response_text: str) -> Optional[Dict[str, Any]]:
    """Extract a JSON action from the LLM response."""
    text = response_text.strip()

    # Strip markdown code fences if present
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to find JSON in the response
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                return json.loads(text[start:end])
            except json.JSONDecodeError:
                return None
    return None


def run_episode(task_id: str) -> Dict[str, Any]:
    """Run a full episode for one task using the LLM agent."""
    # Reset environment
    resp = requests.post(f"{ENV_BASE_URL}/reset", json={"task_id": task_id})
    resp.raise_for_status()
    obs = resp.json()

    done = False
    steps = 0
    action_history: List[Dict] = []
    max_steps = 30  # safety limit

    while not done and steps < max_steps:
        prompt = build_prompt(obs, action_history)

        try:
            llm_resp = requests.post(
                f"{API_BASE_URL}/chat/completions",
                headers={"Authorization": f"Bearer {HF_TOKEN}",
                         "Content-Type": "application/json"},
                json={
                    "model": MODEL_NAME,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 256,
                    "temperature": 0.1,
                },
                timeout=30,
            )
            llm_resp.raise_for_status()
            response_text = llm_resp.json()["choices"][0]["message"]["content"] or ""
        except Exception as e:
            print(f"  [{task_id}] LLM error at step {steps}: {e}")
            break

        action = parse_action(response_text)
        if action is None:
            print(f"  [{task_id}] Failed to parse action from LLM response, stopping.")
            break

        # Validate action_type
        valid_types = {"fill_null", "drop_duplicates", "convert_type", "normalize", "remove_outliers", "strip_whitespace", "map_values", "replace_substring"}
        if action.get("action_type") not in valid_types:
            print(f"  [{task_id}] Invalid action type: {action.get('action_type')}, stopping.")
            break

        try:
            result = requests.post(
                f"{ENV_BASE_URL}/step", json={"action": action}
            )
            result.raise_for_status()
            step_data = result.json()
            obs = step_data["observation"]
            done = step_data["reward"]["done"]
            action_history.append(action)
            steps += 1
        except Exception as e:
            print(f"  [{task_id}] Step error at step {steps}: {e}")
            break

    # Get final score
    try:
        score_resp = requests.get(f"{ENV_BASE_URL}/grader")
        score_resp.raise_for_status()
        grader = score_resp.json()
        return {
            "task_id": task_id,
            "score": grader["score"],
            "steps": steps,
            "breakdown": grader,
        }
    except Exception as e:
        return {"task_id": task_id, "score": None, "steps": steps, "error": str(e)}


def main():
    """Run inference on all tasks."""
    # Get available tasks
    try:
        tasks_resp = requests.get(f"{ENV_BASE_URL}/tasks")
        tasks_resp.raise_for_status()
        task_list = [t["id"] for t in tasks_resp.json()["tasks"]]
    except Exception:
        task_list = ["task_easy", "task_medium", "task_hard", "task_expert"]

    print(f"Running LLM inference on {len(task_list)} tasks...")
    print(f"Model: {MODEL_NAME}")
    print(f"API: {API_BASE_URL}")
    print()

    results = {}
    scores = []

    for task_id in task_list:
        print(f"  Running {task_id}...")
        result = run_episode(task_id)
        results[task_id] = result
        if result.get("score") is not None:
            scores.append(result["score"])
            print(f"    Score: {result['score']:.3f} ({result['steps']} steps)")
        else:
            print(f"    ERROR: {result.get('error')}")

    avg = sum(scores) / len(scores) if scores else 0.0

    print(f"\n{'='*40}")
    print(f"Average score: {avg:.3f}")
    print(f"{'='*40}")

    return {"results": results, "average_score": round(avg, 6)}


if __name__ == "__main__":
    main()
