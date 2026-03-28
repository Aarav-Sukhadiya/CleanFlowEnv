"""Integration tests — full episode end-to-end using direct Python calls."""
import pytest

from cleanflow_env.baseline.rule_agent import RuleBasedAgent
from cleanflow_env.env.environment import CleanFlowEnv
from cleanflow_env.env.grader import final_score
from cleanflow_env.models.action import ActionModel
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


@pytest.fixture
def env():
    return CleanFlowEnv(task_registry=TASK_REGISTRY)


class TestFullEpisodeEasyTask:
    """End-to-end test: run a full easy episode with the rule-based agent."""

    def test_full_episode(self, env):
        """Rule-based agent should score >= 0.80 on task_easy."""
        obs = env.reset("task_easy")
        agent = RuleBasedAgent()

        # Initial state should have nulls and duplicates
        assert sum(obs.null_counts.values()) > 0, "Expected nulls in initial observation"
        assert obs.duplicate_count > 0, "Expected duplicates in initial observation"

        done = False
        steps = 0
        while not done and steps < 20:
            action = agent.act(obs)
            if action is None:
                break
            obs, reward = env.step(action.model_dump())
            done = reward.done
            steps += 1

        # Episode should complete before 20 steps
        assert steps < 20, "Agent should finish before max steps"

        # Score should be >= 0.80
        result = final_score(env._state)
        assert result["score"] >= 0.80, f"Expected score >= 0.80, got {result['score']}"

        # Operations history should be non-empty
        assert len(env._state.operations_history) > 0

        # Budget should still be >= 0
        assert env._state.budget_remaining >= 0


class TestAllTasksComplete:
    """Verify all 4 tasks can run to completion without errors."""

    @pytest.mark.parametrize("task_id", ["task_easy", "task_medium", "task_hard", "task_expert"])
    def test_task_completes(self, env, task_id):
        """Each task should complete without errors."""
        obs = env.reset(task_id)
        agent = RuleBasedAgent()

        done = False
        steps = 0
        while not done and steps < 20:
            action = agent.act(obs)
            if action is None:
                break
            obs, reward = env.step(action.model_dump())
            done = reward.done
            steps += 1

        result = final_score(env._state)
        assert 0.0 <= result["score"] <= 1.0, f"Score out of range: {result['score']}"


class TestRewardNoOscillation:
    """Verify the high-water mark prevents reward oscillation."""

    def test_same_action_twice_no_double_reward(self, env):
        """Applying the same action twice should give 0 quality_delta on the second."""
        env.reset("task_easy")
        action = {"action_type": "fill_null", "column": "age", "method": "mean"}

        _, reward1 = env.step(action)
        _, reward2 = env.step(action)

        # Second time: same action, no new improvement — quality_delta should be 0
        assert reward2.quality_delta == 0.0, (
            f"Expected 0 quality_delta on redundant action, got {reward2.quality_delta}"
        )


class TestBudgetExhaustion:
    """Verify budget exhaustion terminates the episode correctly."""

    def test_budget_runs_out(self, env):
        """Exhausting the budget should end the episode with done=True."""
        env.reset("task_easy")

        # Burn budget quickly with expensive actions
        done = False
        steps = 0
        while not done and steps < 30:
            # Use remove_outliers (cost=3) to burn budget fast
            action = {
                "action_type": "remove_outliers",
                "column": "age",
            }
            _, reward = env.step(action)
            done = reward.done
            steps += 1

        # Episode should have ended
        assert done, "Episode should terminate when budget is exhausted"
        assert env._state.budget_remaining >= 0, "Budget should never go negative"


class TestStepBeforeResetRaises:
    """Verify that calling step() before reset() raises RuntimeError."""

    def test_step_before_reset(self):
        """step() without reset() should raise RuntimeError."""
        fresh_env = CleanFlowEnv(task_registry=TASK_REGISTRY)
        with pytest.raises(RuntimeError):
            fresh_env.step({"action_type": "drop_duplicates"})


class TestDeterminism:
    """Verify that the environment is fully deterministic."""

    def test_same_episode_twice_same_score(self, env):
        """Running the same task twice should produce identical scores."""
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
            result = final_score(env._state)
            scores.append(result["score"])
            histories.append(
                [(op["action_type"], op.get("column")) for op in env._state.operations_history]
            )

        assert scores[0] == scores[1], f"Scores differ: {scores[0]} vs {scores[1]}"
        assert histories[0] == histories[1], "Operation histories differ between runs"
