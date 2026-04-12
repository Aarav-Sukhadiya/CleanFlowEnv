"""Unit tests for data cleaning functions and reward computation."""
import numpy as np
import pandas as pd
import pytest

from cleanflow_env.env.actions import (
    InvalidActionError,
    apply_action,
    convert_type,
    drop_duplicates,
    fill_null,
    fill_sequential,
    normalize,
    remove_outliers,
)
from cleanflow_env.env.rewards import compute_quality
from cleanflow_env.models.action import ActionModel


# --- Test fill_null ---


class TestFillNull:
    """Tests for the fill_null function."""

    def test_mean_fill(self):
        """Mean fill should replace nulls with column mean."""
        df = pd.DataFrame({"x": [10.0, 20.0, np.nan, 40.0]})
        result = fill_null(df, "x", "mean")
        # Mean of [10, 20, 40] = 23.333...
        assert not result["x"].isna().any()
        assert abs(result.loc[2, "x"] - 23.333333) < 0.01

    def test_median_fill(self):
        """Median fill should replace nulls with column median."""
        df = pd.DataFrame({"x": [10.0, 20.0, np.nan, 100.0]})
        result = fill_null(df, "x", "median")
        # Median of [10, 20, 100] = 20.0
        assert result.loc[2, "x"] == 20.0

    def test_mode_fill(self):
        """Mode fill should replace nulls with most common value."""
        df = pd.DataFrame({"x": ["a", "b", "a", None, "a"]})
        result = fill_null(df, "x", "mode")
        assert result.loc[3, "x"] == "a"

    def test_constant_fill(self):
        """Constant fill should replace nulls with the given value."""
        df = pd.DataFrame({"x": [1.0, np.nan, 3.0]})
        result = fill_null(df, "x", "constant", constant_value=0)
        assert result.loc[1, "x"] == 0

    def test_forward_fill(self):
        """Forward fill should propagate last valid value forward."""
        df = pd.DataFrame({"x": [1.0, np.nan, np.nan, 4.0]})
        result = fill_null(df, "x", "forward_fill")
        assert result.loc[1, "x"] == 1.0
        assert result.loc[2, "x"] == 1.0

    def test_sequential_fill_gaps(self):
        """Sequential fill should fill gaps before extending past max."""
        df = pd.DataFrame({"x": ["Emp_1", "Emp_2", None, "Emp_4", "Emp_5"]})
        result = fill_null(df, "x", "sequential")
        assert result.loc[2, "x"] == "Emp_3"

    def test_sequential_fill_zero_padded(self):
        """Sequential fill should preserve zero-padding format."""
        df = pd.DataFrame({"x": ["Employee_000", "Employee_001", None, "Employee_003"]})
        result = fill_null(df, "x", "sequential")
        assert result.loc[2, "x"] == "Employee_002"

    def test_sequential_fill_extends_past_max(self):
        """When more nulls than gaps, should extend past max."""
        df = pd.DataFrame({"x": ["A_1", None, None, "A_3"]})
        result = fill_null(df, "x", "sequential")
        assert result.loc[1, "x"] == "A_2"  # gap
        assert result.loc[2, "x"] == "A_4"  # extension

    def test_sequential_fill_no_pattern_falls_back(self):
        """Non-sequential column should fall back to Unknown."""
        df = pd.DataFrame({"x": ["apple", "banana", None, "cherry"]})
        result = fill_null(df, "x", "sequential")
        assert result.loc[2, "x"] == "Unknown"

    def test_unknown_method_raises(self):
        """Unknown fill method should raise InvalidActionError."""
        df = pd.DataFrame({"x": [1.0, np.nan]})
        with pytest.raises(InvalidActionError):
            fill_null(df, "x", "unknown_method")

    def test_missing_column_raises(self):
        """Non-existent column should raise InvalidActionError."""
        df = pd.DataFrame({"x": [1.0]})
        with pytest.raises(InvalidActionError):
            fill_null(df, "nonexistent", "mean")


# --- Test fill_sequential ---


class TestFillSequential:
    """Tests for the fill_sequential function (gap-aware sequential ID filling)."""

    def test_single_gap_filled(self):
        """A single gap should be filled with the missing number."""
        s = pd.Series(["P_1", "P_2", None, "P_4", "P_5"])
        result = fill_sequential(s)
        assert result.iloc[2] == "P_3"

    def test_multiple_gaps(self):
        """Multiple gaps should all be filled."""
        s = pd.Series(["X_1", None, "X_3", None, "X_5"])
        result = fill_sequential(s)
        assert result.iloc[1] == "X_2"
        assert result.iloc[3] == "X_4"

    def test_gaps_before_extension(self):
        """Gaps should be filled before extending past max."""
        s = pd.Series(["R_1", None, None, None, "R_3"])
        result = fill_sequential(s)
        values = result.tolist()
        assert "R_2" in values  # gap
        assert "R_4" in values  # extension
        assert "R_5" in values  # extension

    def test_zero_padding_preserved(self):
        """Zero-padded formats like 001, 002 should be preserved."""
        s = pd.Series(["EMP_001", None, "EMP_003", "EMP_004"])
        result = fill_sequential(s)
        assert result.iloc[1] == "EMP_002"

    def test_no_nulls_returns_unchanged(self):
        """No nulls should return the series unchanged."""
        s = pd.Series(["A_1", "A_2", "A_3"])
        result = fill_sequential(s)
        assert result.tolist() == ["A_1", "A_2", "A_3"]

    def test_no_pattern_falls_back_to_unknown(self):
        """Non-sequential data should fall back to 'Unknown'."""
        s = pd.Series(["hello", "world", None])
        result = fill_sequential(s)
        assert result.iloc[2] == "Unknown"

    def test_single_non_null_falls_back(self):
        """Only one non-null value is not enough to detect a pattern."""
        s = pd.Series(["A_1", None, None])
        result = fill_sequential(s)
        # Only 1 non-null — can't confirm pattern, but prefix matches >50%
        # so it should still work
        assert not result.isna().any()


# --- Test drop_duplicates ---


class TestDropDuplicates:
    """Tests for the drop_duplicates function."""

    def test_removes_exact_duplicates(self):
        """Should remove fully duplicate rows."""
        df = pd.DataFrame({"a": [1, 2, 1], "b": [10, 20, 10]})
        result = drop_duplicates(df)
        assert len(result) == 2

    def test_index_reset_after_drop(self):
        """Index should be contiguous after dropping duplicates."""
        df = pd.DataFrame({"a": [1, 2, 1, 3, 2], "b": [10, 20, 10, 30, 20]})
        result = drop_duplicates(df)
        assert list(result.index) == list(range(len(result)))

    def test_no_duplicates_unchanged(self):
        """Table with no duplicates should remain unchanged."""
        df = pd.DataFrame({"a": [1, 2, 3]})
        result = drop_duplicates(df)
        assert len(result) == 3


# --- Test convert_type ---


class TestConvertType:
    """Tests for the convert_type function."""

    def test_string_to_int(self):
        """String '123' should convert to int."""
        df = pd.DataFrame({"x": ["1", "2", "3"]})
        result = convert_type(df, "x", "int")
        assert result["x"].dtype.name == "Int64"
        assert result.loc[0, "x"] == 1

    def test_mixed_dates_to_datetime(self):
        """Mixed date format strings should convert to datetime."""
        df = pd.DataFrame({"d": ["2023-01-01", "01-Jan-2023", "Jan 1 2023"]})
        result = convert_type(df, "d", "datetime")
        assert pd.api.types.is_datetime64_any_dtype(result["d"])

    def test_invalid_string_to_float_coerces_nan(self):
        """Non-numeric string should become NaN with errors=coerce."""
        df = pd.DataFrame({"x": ["1.5", "abc", "3.0"]})
        result = convert_type(df, "x", "float")
        assert pd.isna(result.loc[1, "x"])
        assert result.loc[0, "x"] == 1.5

    def test_unknown_target_type_raises(self):
        """Unknown target type should raise InvalidActionError."""
        df = pd.DataFrame({"x": ["1"]})
        with pytest.raises(InvalidActionError):
            convert_type(df, "x", "boolean")


# --- Test normalize ---


class TestNormalize:
    """Tests for the normalize function."""

    def test_minmax_scales_to_01(self):
        """Minmax normalization should scale values to [0, 1]."""
        df = pd.DataFrame({"x": [0.0, 50.0, 100.0]})
        result = normalize(df, "x", "minmax")
        assert result.loc[0, "x"] == 0.0
        assert result.loc[1, "x"] == 0.5
        assert result.loc[2, "x"] == 1.0

    def test_zscore_centers_data(self):
        """Zscore normalization should have mean ~0."""
        df = pd.DataFrame({"x": [10.0, 20.0, 30.0, 40.0, 50.0]})
        result = normalize(df, "x", "zscore")
        assert abs(result["x"].mean()) < 1e-10

    def test_constant_column_returns_zeros(self):
        """Column with zero variance should return all zeros."""
        df = pd.DataFrame({"x": [5.0, 5.0, 5.0]})
        result = normalize(df, "x", "minmax")
        assert (result["x"] == 0.0).all()


# --- Test remove_outliers ---


class TestRemoveOutliers:
    """Tests for the remove_outliers function (IQR x 1.5)."""

    def test_injected_outlier_removed(self):
        """Value at Q3 + 2*IQR should be removed."""
        np.random.seed(42)
        data = np.random.normal(100, 10, size=50)
        q1, q3 = np.percentile(data, [25, 75])
        iqr = q3 - q1
        # Inject a clear outlier
        data = np.append(data, q3 + 2 * iqr + 10)
        df = pd.DataFrame({"x": data})
        result = remove_outliers(df, "x")
        assert len(result) < len(df)

    def test_non_outliers_preserved(self):
        """Values within IQR bounds should not be removed."""
        df = pd.DataFrame({"x": [10.0, 20.0, 30.0, 40.0, 50.0]})
        result = remove_outliers(df, "x")
        assert len(result) == 5

    def test_index_reset_after_removal(self):
        """Index should be contiguous after outlier removal."""
        data = [10.0, 20.0, 30.0, 40.0, 1000.0]
        df = pd.DataFrame({"x": data})
        result = remove_outliers(df, "x")
        assert list(result.index) == list(range(len(result)))

    def test_zscore_removes_extreme(self):
        """Z-score method should remove values > threshold std devs from mean."""
        np.random.seed(42)
        data = np.random.normal(100, 10, size=100).tolist()
        data.append(200.0)  # ~10 std devs away
        df = pd.DataFrame({"x": data})
        result = remove_outliers(df, "x", method="zscore", threshold=3.0)
        assert len(result) < len(df)
        assert 200.0 not in result["x"].values

    def test_zscore_preserves_normal(self):
        """Z-score method should preserve values within threshold."""
        df = pd.DataFrame({"x": [10.0, 20.0, 30.0, 40.0, 50.0]})
        result = remove_outliers(df, "x", method="zscore", threshold=3.0)
        assert len(result) == 5

    def test_unknown_outlier_method_raises(self):
        """Unknown outlier method should raise InvalidActionError."""
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
        with pytest.raises(InvalidActionError):
            remove_outliers(df, "x", method="unknown")


# --- Test apply_action dispatcher ---


class TestApplyAction:
    """Tests for the apply_action dispatch function."""

    def test_dispatches_fill_null(self):
        """apply_action should route fill_null correctly."""
        df = pd.DataFrame({"x": [1.0, np.nan, 3.0]})
        action = ActionModel(action_type="fill_null", column="x", method="mean")
        result = apply_action(df, action)
        assert not result["x"].isna().any()

    def test_dispatches_drop_duplicates(self):
        """apply_action should route drop_duplicates correctly."""
        df = pd.DataFrame({"x": [1, 1, 2]})
        action = ActionModel(action_type="drop_duplicates")
        result = apply_action(df, action)
        assert len(result) == 2


# --- Test compute_quality ---


class TestComputeQuality:
    """Tests for the compute_quality function."""

    def test_perfect_match_returns_high_quality(self):
        """Identical tables should score ~1.0 overall."""
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
        quality = compute_quality(df, df)
        assert quality["overall"] >= 0.99

    def test_all_nulls_low_completeness(self):
        """Table full of nulls should have low completeness."""
        gt = pd.DataFrame({"a": [1, 2, 3]})
        cur = pd.DataFrame({"a": [np.nan, np.nan, np.nan]})
        quality = compute_quality(cur, gt)
        assert quality["completeness"] < 0.5

    def test_empty_df_returns_zero(self):
        """Empty dataframe should return ~0 (clamped to epsilon) for all metrics."""
        quality = compute_quality(pd.DataFrame(), pd.DataFrame({"a": [1]}))
        # Scores are clamped to strict (0, 1) — epsilon (1e-4) is returned instead of 0.0
        assert quality["overall"] < 0.01

    def test_dtype_mismatch_lowers_schema_accuracy(self):
        """Different dtypes should lower schema_accuracy."""
        gt = pd.DataFrame({"a": [1, 2, 3]})  # int64
        cur = pd.DataFrame({"a": ["1", "2", "3"]})  # object
        quality = compute_quality(cur, gt)
        assert quality["schema_accuracy"] < 1.0


# --- Test validation ---


class TestValidation:
    """Tests for the data validation rules."""

    def test_clean_data_passes_all(self):
        """Perfectly cleaned data should pass all validation rules."""
        from cleanflow_env.env.validation import validate_cleaned_data

        df = pd.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
        descs = {"a": "Numeric. 2 missing values — fill with median.", "b": "Numeric."}
        result = validate_cleaned_data(df, df, descs)
        # Score clamped to strict (0, 1) — perfect validation returns 0.9999
        assert result["validation_score"] > 0.99
        assert len(result["violations"]) == 0

    def test_remaining_nulls_flagged(self):
        """Remaining nulls in a column that was supposed to be filled should be flagged."""
        from cleanflow_env.env.validation import validate_cleaned_data

        cleaned = pd.DataFrame({"a": [1, np.nan, 3]})
        gt = pd.DataFrame({"a": [1, 2, 3]})
        descs = {"a": "Numeric. Has missing values — fill with median."}
        result = validate_cleaned_data(cleaned, gt, descs)
        assert result["validation_score"] < 1.0
        assert any("NULL_REMAINING" in v for v in result["violations"])

    def test_type_mismatch_flagged(self):
        """Wrong dtype should be flagged as a violation."""
        from cleanflow_env.env.validation import validate_cleaned_data

        cleaned = pd.DataFrame({"a": ["1", "2", "3"]})
        gt = pd.DataFrame({"a": [1, 2, 3]})
        descs = {"a": "Numeric."}
        result = validate_cleaned_data(cleaned, gt, descs)
        assert any("TYPE_MISMATCH" in v for v in result["violations"])

    def test_range_violation_flagged(self):
        """Values far outside GT range should be flagged."""
        from cleanflow_env.env.validation import validate_cleaned_data

        cleaned = pd.DataFrame({"a": [1.0, 2.0, 999.0]})
        gt = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
        descs = {"a": "Numeric."}
        result = validate_cleaned_data(cleaned, gt, descs)
        assert any("RANGE_VIOLATION" in v for v in result["violations"])


# --- Test distribution in observation ---


class TestDistribution:
    """Tests for the distribution stats in observations."""

    def test_observation_has_distribution(self):
        """Observation should include distribution stats for numeric columns."""
        from cleanflow_env.env.environment import CleanFlowEnv
        from cleanflow_env.tasks.task_easy import generate_easy_task

        env = CleanFlowEnv(task_registry={"task_easy": generate_easy_task})
        obs = env.reset("task_easy")
        assert "age" in obs.distribution
        assert "median" in obs.distribution["age"]
        assert "skew" in obs.distribution["age"]
        assert "q1" in obs.distribution["age"]
        assert "q3" in obs.distribution["age"]
