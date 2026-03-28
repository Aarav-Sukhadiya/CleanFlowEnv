"""
End-to-end simulation of CleanFlowEnv without the HTTP API layer.
Direct Python calls only — useful for debugging and validation.
"""
from __future__ import annotations

from cleanflow_env.baseline.rule_agent import RuleBasedAgent
from cleanflow_env.env.environment import CleanFlowEnv
from cleanflow_env.env.grader import final_score, score_breakdown_report
from cleanflow_env.tasks.task_easy import generate_easy_task
from cleanflow_env.tasks.task_expert import generate_expert_task
from cleanflow_env.tasks.task_hard import generate_hard_task
from cleanflow_env.tasks.task_medium import generate_medium_task

TASK_REGISTRY = {
    "task_easy": generate_easy_task,
    "task_medium": generate_medium_task,
    "task_hard": generate_hard_task,
    "task_expert": generate_expert_task,
}


def run_simulation(task_id: str, env: CleanFlowEnv) -> None:
    print(f"\n{'='*60}")
    print(f"  Task: {task_id}")
    print(f"{'='*60}")

    obs = env.reset(task_id)
    agent = RuleBasedAgent()

    # Print initial state
    total_nulls = sum(obs.null_counts.values())
    print(f"  Initial nulls: {total_nulls}  |  Duplicates: {obs.duplicate_count}  |  Budget: {obs.budget_remaining}")
    print(f"  Columns: {list(obs.table_schema.values()) if hasattr(obs, 'table_schema') else 'N/A'}")
    print()
    print(f"  {'Step':<6} {'Action':<20} {'Column':<18} {'Reward':>8} {'Quality':>8} {'Budget':>7}")
    print(f"  {'-'*6} {'-'*20} {'-'*18} {'-'*8} {'-'*8} {'-'*7}")

    done = False
    step = 0
    while not done:
        action = agent.act(obs)
        if action is None:
            print(f"  Agent exhausted all useful actions at step {step}.")
            break

        obs, reward = env.step(action.model_dump())
        done = reward.done
        step += 1

        col_str = action.column or "-"
        print(
            f"  {step:<6} {action.action_type:<20} {col_str:<18} "
            f"{reward.reward:>8.3f} {reward.cumulative_quality:>8.3f} {obs.budget_remaining:>7}"
        )

    print()
    print(score_breakdown_report(env._state))


def main():
    env = CleanFlowEnv(task_registry=TASK_REGISTRY)

    summary = []
    for task_id in ["task_easy", "task_medium", "task_hard", "task_expert"]:
        run_simulation(task_id, env)
        result = final_score(env._state)
        summary.append((task_id, result["score"]))

    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    for task_id, score in summary:
        print(f"  {task_id:<20} {score:.3f}")
    avg = sum(s for _, s in summary) / len(summary)
    print(f"  {'Average':<20} {avg:.3f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
