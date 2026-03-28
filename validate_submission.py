"""
Pre-submission validation script for CleanFlowEnv.
Checks both direct Python stack and HTTP API (if server running).
"""
from __future__ import annotations

import sys
import time

# ANSI colors
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
RESET = "\033[0m"

passed = 0
failed = 0
skipped = 0
results = []


def check(name: str, fn):
    global passed, failed, skipped
    start = time.time()
    try:
        fn()
        elapsed = time.time() - start
        results.append((name, "PASS", elapsed))
        passed += 1
        print(f"  {GREEN}PASS{RESET}  {name} ({elapsed:.2f}s)")
    except Exception as e:
        elapsed = time.time() - start
        results.append((name, "FAIL", elapsed, str(e)))
        failed += 1
        print(f"  {RED}FAIL{RESET}  {name} ({elapsed:.2f}s) — {e}")


def check_skip(name: str, fn):
    global skipped
    start = time.time()
    try:
        fn()
        elapsed = time.time() - start
        results.append((name, "PASS", elapsed))
        global passed
        passed += 1
        print(f"  {GREEN}PASS{RESET}  {name} ({elapsed:.2f}s)")
    except ConnectionError:
        elapsed = time.time() - start
        results.append((name, "SKIP", elapsed))
        skipped += 1
        print(f"  {YELLOW}SKIP{RESET}  {name} — server not running")
    except Exception as e:
        elapsed = time.time() - start
        results.append((name, "FAIL", elapsed, str(e)))
        global failed
        failed += 1
        print(f"  {RED}FAIL{RESET}  {name} ({elapsed:.2f}s) — {e}")


def main():
    print("\n=== CleanFlowEnv Submission Validator ===\n")

    # --- Environment checks (direct Python) ---
    print("[Python Stack]")

    def check_instantiate():
        from cleanflow_env.env.environment import CleanFlowEnv
        from cleanflow_env.tasks.task_easy import generate_easy_task
        env = CleanFlowEnv(task_registry={"task_easy": generate_easy_task})
        assert env is not None

    check("CleanFlowEnv instantiates", check_instantiate)

    def check_reset_all_tasks():
        from cleanflow_env.env.environment import CleanFlowEnv
        from cleanflow_env.tasks.task_easy import generate_easy_task
        from cleanflow_env.tasks.task_medium import generate_medium_task
        from cleanflow_env.tasks.task_hard import generate_hard_task
        from cleanflow_env.tasks.task_expert import generate_expert_task
        registry = {
            "task_easy": generate_easy_task,
            "task_medium": generate_medium_task,
            "task_hard": generate_hard_task,
            "task_expert": generate_expert_task,
        }
        env = CleanFlowEnv(task_registry=registry)
        for task_id in registry:
            obs = env.reset(task_id)
            assert obs.task_id == task_id

    check("reset() returns valid obs for all 4 tasks", check_reset_all_tasks)

    def check_step():
        from cleanflow_env.env.environment import CleanFlowEnv
        from cleanflow_env.tasks.task_easy import generate_easy_task
        env = CleanFlowEnv(task_registry={"task_easy": generate_easy_task})
        env.reset("task_easy")
        obs, reward = env.step({"action_type": "drop_duplicates"})
        assert obs is not None
        assert reward is not None

    check("step() returns valid obs + reward", check_step)

    def check_state():
        from cleanflow_env.env.environment import CleanFlowEnv
        from cleanflow_env.tasks.task_easy import generate_easy_task
        env = CleanFlowEnv(task_registry={"task_easy": generate_easy_task})
        env.reset("task_easy")
        state = env.state()
        assert isinstance(state, dict)
        assert "task_id" in state

    check("state() returns dict", check_state)

    def check_grader():
        from cleanflow_env.env.environment import CleanFlowEnv
        from cleanflow_env.env.grader import final_score
        from cleanflow_env.tasks.task_easy import generate_easy_task
        env = CleanFlowEnv(task_registry={"task_easy": generate_easy_task})
        env.reset("task_easy")
        result = final_score(env._state)
        assert 0.0 <= result["score"] <= 1.0

    check("grader returns score in [0.0, 1.0]", check_grader)

    def check_determinism():
        from cleanflow_env.baseline.rule_agent import RuleBasedAgent
        from cleanflow_env.env.environment import CleanFlowEnv
        from cleanflow_env.env.grader import final_score
        from cleanflow_env.tasks.task_easy import generate_easy_task
        env = CleanFlowEnv(task_registry={"task_easy": generate_easy_task})
        scores = []
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
        assert scores[0] == scores[1], f"Non-deterministic: {scores}"

    check("Deterministic: same episode → same score", check_determinism)

    # --- Spec checks ---
    print("\n[Spec Compliance]")

    def check_openenv_yaml():
        import yaml
        with open("openenv.yaml") as f:
            data = yaml.safe_load(f)
        required = ["name", "version", "tags", "endpoints", "tasks", "score_range", "deterministic"]
        for key in required:
            assert key in data, f"Missing key: {key}"
        assert len(data["tasks"]) >= 3

    try:
        import yaml
        check("openenv.yaml has all required keys", check_openenv_yaml)
    except ImportError:
        # Fallback: just check file exists and is non-empty
        def check_openenv_exists():
            import os
            assert os.path.getsize("openenv.yaml") > 0
        check("openenv.yaml exists and non-empty", check_openenv_exists)

    def check_action_schema():
        from cleanflow_env.models.action import ActionModel
        schema = ActionModel.model_json_schema()
        assert len(schema) > 0

    check("ActionModel.model_json_schema() non-empty", check_action_schema)

    def check_observation_schema():
        from cleanflow_env.models.observation import ObservationModel
        schema = ObservationModel.model_json_schema()
        assert len(schema) > 0

    check("ObservationModel.model_json_schema() non-empty", check_observation_schema)

    def check_dockerfile():
        import os
        assert os.path.getsize("Dockerfile") > 0

    check("Dockerfile exists and non-empty", check_dockerfile)

    def check_readme():
        with open("README.md") as f:
            content = f.read()
        assert "openenv" in content.lower()
        assert len(content) > 500

    check("README.md exists with content", check_readme)

    # --- API checks (skip if server not running) ---
    print("\n[API Stack — requires server on localhost:7860]")

    def make_api_check(method, path, json_data=None, validate=None):
        def _check():
            import httpx
            try:
                client = httpx.Client(base_url="http://localhost:7860", timeout=10)
                if method == "GET":
                    r = client.get(path)
                else:
                    r = client.post(path, json=json_data)
                assert r.status_code == 200, f"Status {r.status_code}: {r.text[:200]}"
                if validate:
                    validate(r.json())
            except httpx.ConnectError:
                raise ConnectionError("Server not running")
        return _check

    check_skip("GET / returns 200", make_api_check("GET", "/"))
    check_skip("POST /reset returns 200", make_api_check("POST", "/reset", {"task_id": "task_easy"}))
    check_skip("POST /step returns 200", make_api_check(
        "POST", "/step", {"action": {"action_type": "drop_duplicates"}}
    ))
    check_skip("GET /grader returns score", make_api_check(
        "GET", "/grader", validate=lambda j: 0.0 <= j["score"] <= 1.0
    ))
    check_skip("GET /tasks returns 4 tasks", make_api_check(
        "GET", "/tasks", validate=lambda j: len(j["tasks"]) == 4
    ))
    check_skip("POST /baseline completes", make_api_check(
        "POST", "/baseline", validate=lambda j: len(j["results"]) == 4
    ))

    # --- Summary ---
    total = passed + failed
    print(f"\n{'='*42}")
    print(f"  Total: {passed}/{total} checks passed", end="")
    if skipped:
        print(f" ({skipped} skipped)", end="")
    print()

    if failed == 0:
        print(f"  {GREEN}Ready to submit{RESET}")
    else:
        print(f"  {RED}{failed} check(s) failed{RESET}")

    print(f"{'='*42}\n")
    sys.exit(1 if failed > 0 else 0)


if __name__ == "__main__":
    main()
