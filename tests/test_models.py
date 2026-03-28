"""Unit tests for ActionModel, ObservationModel, and RewardModel."""
import pytest
from pydantic import ValidationError

from cleanflow_env.models.action import ActionModel
from cleanflow_env.models.observation import ObservationModel, TablePreviewRow
from cleanflow_env.models.reward import RewardModel


# --- Fixtures ---


@pytest.fixture
def valid_observation_data():
    return {
        "table_preview": [
            TablePreviewRow(row_index=0, values={"age": 25, "name": "Alice"}),
            TablePreviewRow(row_index=1, values={"age": None, "name": "Bob"}),
        ],
        "table_schema": {"age": "float", "name": "string"},
        "null_counts": {"age": 1, "name": 0},
        "duplicate_count": 0,
        "stats": {"age_mean": 25.0, "age_std": 0.0},
        "step_count": 0,
        "budget_remaining": 20,
        "task_id": "task_easy",
        "column_descriptions": {"age": "Age in years", "name": "Full name"},
    }


# --- TestActionModel ---


class TestActionModel:
    """Tests for ActionModel validation."""

    def test_valid_fill_null(self):
        """Valid fill_null with column and method should pass."""
        action = ActionModel(action_type="fill_null", column="age", method="mean")
        assert action.action_type == "fill_null"
        assert action.column == "age"
        assert action.method == "mean"

    def test_fill_null_without_column_raises(self):
        """fill_null requires column."""
        with pytest.raises(ValidationError):
            ActionModel(action_type="fill_null", method="mean")

    def test_fill_null_without_method_raises(self):
        """fill_null requires method."""
        with pytest.raises(ValidationError):
            ActionModel(action_type="fill_null", column="age")

    def test_fill_null_constant_without_value_raises(self):
        """fill_null with method=constant requires constant_value."""
        with pytest.raises(ValidationError):
            ActionModel(
                action_type="fill_null", column="age", method="constant"
            )

    def test_fill_null_constant_with_value_passes(self):
        """fill_null with method=constant and constant_value should pass."""
        action = ActionModel(
            action_type="fill_null",
            column="age",
            method="constant",
            constant_value=0,
        )
        assert action.constant_value == 0

    def test_convert_type_valid(self):
        """Valid convert_type with column and target_type should pass."""
        action = ActionModel(
            action_type="convert_type", column="dob", target_type="datetime"
        )
        assert action.target_type == "datetime"

    def test_convert_type_without_target_type_raises(self):
        """convert_type requires target_type."""
        with pytest.raises(ValidationError):
            ActionModel(action_type="convert_type", column="dob")

    def test_convert_type_without_column_raises(self):
        """convert_type requires column."""
        with pytest.raises(ValidationError):
            ActionModel(action_type="convert_type", target_type="datetime")

    def test_normalize_requires_column(self):
        """normalize requires column."""
        with pytest.raises(ValidationError):
            ActionModel(action_type="normalize")

    def test_remove_outliers_requires_column(self):
        """remove_outliers requires column."""
        with pytest.raises(ValidationError):
            ActionModel(action_type="remove_outliers")

    def test_drop_duplicates_no_column_needed(self):
        """drop_duplicates should work without column."""
        action = ActionModel(action_type="drop_duplicates")
        assert action.action_type == "drop_duplicates"
        assert action.column is None

    def test_invalid_action_type_raises(self):
        """Unknown action_type should be rejected by Literal."""
        with pytest.raises(ValidationError):
            ActionModel(action_type="unknown_action")

    def test_invalid_method_raises(self):
        """Unknown fill method should be rejected by Literal."""
        with pytest.raises(ValidationError):
            ActionModel(action_type="fill_null", column="age", method="invalid")


# --- TestObservationModel ---


class TestObservationModel:
    """Tests for ObservationModel validation."""

    def test_valid_construction(self, valid_observation_data):
        """Full valid ObservationModel should construct correctly."""
        obs = ObservationModel(**valid_observation_data)
        assert obs.step_count == 0
        assert obs.budget_remaining == 20
        assert obs.task_id == "task_easy"

    def test_preview_exceeds_max_raises(self, valid_observation_data):
        """table_preview with more than 10 rows should raise."""
        valid_observation_data["table_preview"] = [
            TablePreviewRow(row_index=i, values={"a": i}) for i in range(11)
        ]
        with pytest.raises(ValidationError):
            ObservationModel(**valid_observation_data)

    def test_negative_budget_raises(self, valid_observation_data):
        """budget_remaining below 0 should raise."""
        valid_observation_data["budget_remaining"] = -1
        with pytest.raises(ValidationError):
            ObservationModel(**valid_observation_data)

    def test_empty_stats_valid(self, valid_observation_data):
        """Empty stats and null_counts should be valid."""
        valid_observation_data["stats"] = {}
        valid_observation_data["null_counts"] = {}
        obs = ObservationModel(**valid_observation_data)
        assert obs.stats == {}

    def test_schema_serializes_with_alias(self, valid_observation_data):
        """table_schema should serialize as 'schema' in JSON output."""
        obs = ObservationModel(**valid_observation_data)
        dumped = obs.model_dump()
        assert "schema" in dumped
        assert "table_schema" not in dumped

    def test_zero_budget_valid(self, valid_observation_data):
        """budget_remaining of exactly 0 should be valid."""
        valid_observation_data["budget_remaining"] = 0
        obs = ObservationModel(**valid_observation_data)
        assert obs.budget_remaining == 0


# --- TestRewardModel ---


class TestRewardModel:
    """Tests for RewardModel construction and validation."""

    def test_from_step_constructs_correctly(self):
        """from_step() should compute reward = delta - penalty - cost."""
        reward = RewardModel.from_step(
            quality_delta=0.1,
            penalty=0.0,
            budget_cost=1.0,
            cumulative_quality=0.5,
            done=False,
            info={"action": "fill_null"},
        )
        assert abs(reward.reward - (0.1 - 0.0 - 1.0)) < 1e-6
        assert reward.cumulative_quality == 0.5
        assert reward.done is False

    def test_reward_equals_formula(self):
        """reward should equal quality_delta - penalty - budget_cost."""
        reward = RewardModel.from_step(
            quality_delta=0.15,
            penalty=0.05,
            budget_cost=2.0,
            cumulative_quality=0.6,
            done=False,
        )
        expected = 0.15 - 0.05 - 2.0
        assert abs(reward.reward - expected) < 1e-6

    def test_done_true_accepted(self):
        """done=True should be accepted."""
        reward = RewardModel.from_step(
            quality_delta=0.0,
            penalty=0.5,
            budget_cost=0.0,
            cumulative_quality=0.3,
            done=True,
            info={"reason": "budget_exhausted"},
        )
        assert reward.done is True

    def test_info_accepts_arbitrary_dict(self):
        """info should accept any dict."""
        reward = RewardModel.from_step(
            quality_delta=0.0,
            penalty=0.0,
            budget_cost=1.0,
            cumulative_quality=0.0,
            done=False,
            info={"nested": {"key": [1, 2, 3]}, "flag": True},
        )
        assert reward.info["nested"]["key"] == [1, 2, 3]

    def test_default_info_is_empty_dict(self):
        """info should default to empty dict if not provided."""
        reward = RewardModel.from_step(
            quality_delta=0.0,
            penalty=0.0,
            budget_cost=0.0,
            cumulative_quality=0.0,
            done=False,
        )
        assert reward.info == {}
