"""
Baseline inference script for CleanFlowEnv.

Rule-based agent — works without API key, fully reproducible.
"""
from __future__ import annotations

from typing import Any, Dict

from cleanflow_env.baseline.rule_agent import RuleBasedAgent
from cleanflow_env.env.environment import CleanFlowEnv, build_observation
from cleanflow_env.env.grader import final_score
from cleanflow_env.models.observation import ObservationModel


def run_episode(
    env: CleanFlowEnv, agent: RuleBasedAgent, task_id: str
) -> Dict[str, Any]:
    """
    Run a full episode with the given agent.

    Returns dict with score, steps, budget_used, breakdown.
    """
    obs = env.reset(task_id)
    agent.reset()
    done = False
    steps = 0

    while not done:
        action = agent.act(obs)
        if action is None:
            break
        obs, reward = env.step(action.model_dump())
        done = reward.done
        steps += 1

    state = env._state
    result = final_score(state)
    budget_used = state.initial_budget - state.budget_remaining

    return {
        "score": result["score"],
        "steps": steps,
        "budget_used": budget_used,
        "breakdown": result,
    }


def run_baseline_all(env: CleanFlowEnv) -> Dict[str, Any]:
    """Run the rule-based baseline on all tasks and return results."""
    from datetime import datetime, timezone

    agent = RuleBasedAgent()
    tasks = list(env.task_registry.keys())
    results = {}
    scores = []

    for task_id in tasks:
        try:
            result = run_episode(env, agent, task_id)
            results[task_id] = result
            scores.append(result["score"])
        except Exception as e:
            results[task_id] = {"score": None, "error": str(e)}

    valid_scores = [s for s in scores if s is not None]
    avg = sum(valid_scores) / len(valid_scores) if valid_scores else 0.0

    return {
        "results": results,
        "average_score": round(avg, 6),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }



if __name__ == "__main__":
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
    results = run_baseline_all(env)

    print("\n=== CleanFlowEnv Baseline Results ===")
    for task_id, res in results["results"].items():
        if res.get("score") is not None:
            print(f"  {task_id}: {res['score']:.3f} ({res['steps']} steps, {res['budget_used']} budget)")
        else:
            print(f"  {task_id}: ERROR - {res.get('error')}")
    print(f"\n  Average: {results['average_score']:.3f}")
    print("=====================================")
