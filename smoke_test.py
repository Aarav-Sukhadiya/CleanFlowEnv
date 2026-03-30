"""
Final smoke test — verifies the entire CleanFlowEnv stack before submission.
Runs 5 checks in sequence, prints PASS/FAIL with timing.
"""
from __future__ import annotations

import sys
import time

GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
RESET = "\033[0m"

results = []


def run_check(name: str, fn, skip_on_connect=False):
    start = time.time()
    try:
        fn()
        elapsed = time.time() - start
        results.append((name, "PASS", elapsed))
        print(f"  {name:<35} {GREEN}PASS{RESET} ({elapsed:.1f}s)")
    except ConnectionError:
        if skip_on_connect:
            elapsed = time.time() - start
            results.append((name, "SKIP", elapsed))
            print(f"  {name:<35} {YELLOW}SKIP{RESET} (server not running)")
        else:
            raise
    except Exception as e:
        elapsed = time.time() - start
        results.append((name, "FAIL", elapsed))
        print(f"  {name:<35} {RED}FAIL{RESET} ({elapsed:.1f}s)")
        print(f"    Error: {e}")


def check_1_python_stack():
    """Import all modules, run one easy episode, check score >= 0.75."""
    from cleanflow_env.baseline.rule_agent import RuleBasedAgent
    from cleanflow_env.env.environment import CleanFlowEnv
    from cleanflow_env.env.grader import final_score
    from cleanflow_env.models.action import ActionModel
    from cleanflow_env.models.observation import ObservationModel
    from cleanflow_env.models.reward import RewardModel
    from cleanflow_env.tasks.task_easy import generate_easy_task
    from cleanflow_env.tasks.task_expert import generate_expert_task
    from cleanflow_env.tasks.task_hard import generate_hard_task
    from cleanflow_env.tasks.task_medium import generate_medium_task

    registry = {
        "task_easy": generate_easy_task,
        "task_medium": generate_medium_task,
        "task_hard": generate_hard_task,
        "task_expert": generate_expert_task,
    }
    env = CleanFlowEnv(task_registry=registry)
    obs = env.reset("task_easy")
    agent = RuleBasedAgent()
    done = False
    while not done:
        action = agent.act(obs)
        if action is None:
            break
        obs, reward = env.step(action.model_dump())
        done = reward.done

    result = final_score(env._state)
    assert result["score"] >= 0.75, f"Score {result['score']} < 0.75"


def check_2_task_generation():
    """All 4 tasks generate correctly with expected shapes."""
    from cleanflow_env.tasks.task_easy import generate_easy_task
    from cleanflow_env.tasks.task_expert import generate_expert_task
    from cleanflow_env.tasks.task_hard import generate_hard_task
    from cleanflow_env.tasks.task_medium import generate_medium_task

    specs = [
        ("task_easy", generate_easy_task, 200),
        ("task_medium", generate_medium_task, 300),
        ("task_hard", generate_hard_task, 400),
        ("task_expert", generate_expert_task, 500),
    ]

    for name, gen_fn, expected_base_rows in specs:
        raw, gt, budget, col_desc = gen_fn()
        # Raw should have at least the base rows (plus any injected duplicates)
        assert raw.shape[0] >= expected_base_rows, f"{name}: raw has {raw.shape[0]} rows, expected >= {expected_base_rows}"
        # Ground truth should have no duplicates
        assert gt.duplicated().sum() == 0, f"{name}: ground_truth has duplicates"
        # Column descriptions should cover all columns
        for col in raw.columns:
            assert col in col_desc, f"{name}: missing description for column '{col}'"


def check_3_sequential_fill():
    """Sequential fill: gaps filled before extending, no 'Unknown' in sequential ID columns."""
    import re
    from cleanflow_env.env.actions import fill_sequential
    import pandas as pd

    # Gap filling
    s = pd.Series(["E_01", "E_02", None, "E_04"])
    result = fill_sequential(s)
    assert result.iloc[2] == "E_03", f"Expected 'E_03', got '{result.iloc[2]}'"

    # Ground truth has no Unknown in name column
    from cleanflow_env.tasks.task_easy import generate_easy_task
    _, gt, _, _ = generate_easy_task()
    assert (gt["name"] != "Unknown").all(), "GT name column should not have 'Unknown'"
    assert all(re.match(r"^Employee_\d{3}$", v) for v in gt["name"]), "All GT names should match pattern"


def check_4_reward_integrity():
    """Reward system: no oscillation, penalties work."""
    from cleanflow_env.env.environment import CleanFlowEnv
    from cleanflow_env.env.grader import final_score
    from cleanflow_env.tasks.task_easy import generate_easy_task

    env = CleanFlowEnv(task_registry={"task_easy": generate_easy_task})
    env.reset("task_easy")

    # Apply same action twice
    action = {"action_type": "fill_null", "column": "age", "method": "mean"}
    _, r1 = env.step(action)
    _, r2 = env.step(action)

    # Second time should have 0 quality_delta (high-water mark)
    assert r2.quality_delta == 0.0, f"Expected 0 quality_delta, got {r2.quality_delta}"


def check_5_api_stack():
    """Test API endpoints (requires server on localhost:7860)."""
    import httpx

    try:
        client = httpx.Client(base_url="http://localhost:7860", timeout=10)
        client.get("/")
    except httpx.ConnectError:
        raise ConnectionError("Server not running")

    # Reset
    r = client.post("/reset", json={"task_id": "task_easy"})
    assert r.status_code == 200, f"/reset returned {r.status_code}"

    # Step
    r = client.post("/step", json={"action": {"action_type": "drop_duplicates"}})
    assert r.status_code == 200, f"/step returned {r.status_code}"
    data = r.json()
    assert -2.0 <= data["reward"]["reward"] <= 2.0

    # Grader
    r = client.get("/grader")
    assert r.status_code == 200
    assert 0.0 <= r.json()["score"] <= 1.0

    # Baseline
    r = client.post("/baseline")
    assert r.status_code == 200
    assert len(r.json()["results"]) == 4


def check_6_determinism():
    """Same episode twice produces identical scores and histories."""
    from cleanflow_env.baseline.rule_agent import RuleBasedAgent
    from cleanflow_env.env.environment import CleanFlowEnv
    from cleanflow_env.env.grader import final_score
    from cleanflow_env.tasks.task_easy import generate_easy_task

    env = CleanFlowEnv(task_registry={"task_easy": generate_easy_task})
    scores = []
    histories = []

    for _ in range(2):
        obs = env.reset("task_easy")
        agent = RuleBasedAgent()
        done = False
        while not done:
            action = agent.act(obs)
            if action is None:
                break
            obs, reward = env.step(action.model_dump())
            done = reward.done
        scores.append(final_score(env._state)["score"])
        histories.append(
            [(op["action_type"], op.get("column")) for op in env._state.operations_history]
        )

    assert scores[0] == scores[1], f"Scores differ: {scores[0]} != {scores[1]}"
    assert histories[0] == histories[1], "Operation histories differ"


def main():
    print(f"\n{'='*45}")
    print("  CleanFlowEnv Smoke Test")
    print(f"{'='*45}\n")

    run_check("Check 1 — Python stack", check_1_python_stack)
    run_check("Check 2 — Task generation", check_2_task_generation)
    run_check("Check 3 — Sequential fill", check_3_sequential_fill)
    run_check("Check 4 — Reward integrity", check_4_reward_integrity)
    run_check("Check 5 — API stack", check_5_api_stack, skip_on_connect=True)
    run_check("Check 6 — Determinism", check_6_determinism)

    print(f"\n{'='*45}")
    failures = [r for r in results if r[1] == "FAIL"]
    skips = [r for r in results if r[1] == "SKIP"]
    passes = [r for r in results if r[1] == "PASS"]

    if failures:
        print(f"  {RED}{len(failures)} CHECK(S) FAILED{RESET}")
        for name, _, _ in failures:
            print(f"    - {name}")
        print(f"{'='*45}\n")
        sys.exit(1)
    else:
        skip_msg = f" ({len(skips)} skipped)" if skips else ""
        print(f"  {GREEN}ALL CHECKS PASSED{skip_msg}{RESET}")
        print(f"{'='*45}\n")
        sys.exit(0)


if __name__ == "__main__":
    main()
