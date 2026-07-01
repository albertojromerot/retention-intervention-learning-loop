"""Tests for feature engineering and temporal split logic."""

import pandas as pd
import pytest

from src.config import (
    CUSTOMER_ID_COLUMN,
    SNAPSHOT_MONTH_COLUMN,
    TARGET_COLUMN,
    get_profile,
)
from src.feature_engineering import (
    CATEGORICAL_FEATURE_COLUMNS,
    FEATURE_COLUMNS,
    NUMERIC_FEATURE_COLUMNS,
    SPLIT_COLUMN,
    build_feature_table,
    summarise_feature_table,
)
from src.synthetic_data import generate_customer_month_panel


def _build_compact_feature_table() -> pd.DataFrame:
    """Generate a compact feature table for fast tests."""
    panel = generate_customer_month_panel(profile=get_profile("compact"))
    result = build_feature_table(panel)
    return result.feature_table


def test_feature_table_has_expected_shape() -> None:
    """The feature table should preserve the synthetic panel row count."""
    profile = get_profile("compact")
    feature_table = _build_compact_feature_table()

    assert feature_table.shape[0] == profile.n_customers * profile.n_months
    assert feature_table[CUSTOMER_ID_COLUMN].nunique() == profile.n_customers
    assert feature_table[SNAPSHOT_MONTH_COLUMN].nunique() == profile.n_months


def test_feature_columns_are_present() -> None:
    """All configured feature columns should exist in the feature table."""
    feature_table = _build_compact_feature_table()

    required_columns = {
        CUSTOMER_ID_COLUMN,
        SNAPSHOT_MONTH_COLUMN,
        SPLIT_COLUMN,
        TARGET_COLUMN,
        *FEATURE_COLUMNS,
    }

    assert required_columns.issubset(set(feature_table.columns))


def test_feature_metadata_has_no_overlap() -> None:
    """Numeric and categorical feature lists should not overlap."""
    overlap = set(NUMERIC_FEATURE_COLUMNS).intersection(CATEGORICAL_FEATURE_COLUMNS)

    assert overlap == set()


def test_temporal_split_contains_train_validation_and_test() -> None:
    """The feature table should contain all three temporal splits."""
    feature_table = _build_compact_feature_table()

    observed_splits = set(feature_table[SPLIT_COLUMN].unique())

    assert observed_splits == {"train", "validation", "test"}


def test_temporal_split_is_ordered_by_snapshot_month() -> None:
    """Train months should come before validation months and test months."""
    feature_table = _build_compact_feature_table()

    month_order = {
        month: index
        for index, month in enumerate(sorted(feature_table[SNAPSHOT_MONTH_COLUMN].unique()))
    }

    split_month_positions = (
        feature_table[[SNAPSHOT_MONTH_COLUMN, SPLIT_COLUMN]]
        .drop_duplicates()
        .assign(month_position=lambda df: df[SNAPSHOT_MONTH_COLUMN].map(month_order))
    )

    max_train_month = split_month_positions.loc[
        split_month_positions[SPLIT_COLUMN].eq("train"),
        "month_position",
    ].max()

    min_validation_month = split_month_positions.loc[
        split_month_positions[SPLIT_COLUMN].eq("validation"),
        "month_position",
    ].min()

    max_validation_month = split_month_positions.loc[
        split_month_positions[SPLIT_COLUMN].eq("validation"),
        "month_position",
    ].max()

    min_test_month = split_month_positions.loc[
        split_month_positions[SPLIT_COLUMN].eq("test"),
        "month_position",
    ].min()

    assert max_train_month < min_validation_month
    assert max_validation_month < min_test_month


def test_customer_month_key_remains_unique() -> None:
    """Feature engineering should not duplicate customer-month rows."""
    feature_table = _build_compact_feature_table()

    duplicated_keys = feature_table.duplicated(
        subset=[CUSTOMER_ID_COLUMN, SNAPSHOT_MONTH_COLUMN]
    ).sum()

    assert duplicated_keys == 0


def test_engineered_score_columns_are_bounded() -> None:
    """Engineered score columns should remain between 0 and 100."""
    feature_table = _build_compact_feature_table()

    bounded_columns = [
        "relationship_depth_score",
        "credit_stress_score",
        "relationship_health_score",
        "value_at_risk_index",
    ]

    for column in bounded_columns:
        assert feature_table[column].between(0, 100).all(), column


def test_engineered_flag_columns_are_binary() -> None:
    """Engineered flag columns should contain only 0 and 1."""
    feature_table = _build_compact_feature_table()

    binary_columns = [
        "digital_inactivity_flag",
        "engagement_decline_flag",
        "high_service_friction_flag",
        "low_satisfaction_flag",
    ]

    for column in binary_columns:
        assert set(feature_table[column].unique()).issubset({0, 1}), column


def test_target_is_preserved_after_feature_engineering() -> None:
    """The target should remain binary and within the expected synthetic rate."""
    feature_table = _build_compact_feature_table()

    target_values = set(feature_table[TARGET_COLUMN].unique())
    target_rate = feature_table[TARGET_COLUMN].mean()

    assert target_values.issubset({0, 1})
    assert 0.05 <= target_rate <= 0.15


def test_no_missing_values_after_feature_engineering() -> None:
    """The engineered feature table should not contain missing values."""
    feature_table = _build_compact_feature_table()

    assert feature_table.isna().sum().sum() == 0


def test_feature_summary_matches_table() -> None:
    """The feature summary helper should match the generated table."""
    feature_table = _build_compact_feature_table()
    summary = summarise_feature_table(feature_table)

    assert summary["rows"] == len(feature_table)
    assert summary["customers"] == feature_table[CUSTOMER_ID_COLUMN].nunique()
    assert summary["months"] == feature_table[SNAPSHOT_MONTH_COLUMN].nunique()
    assert summary["columns"] == feature_table.shape[1]
    assert summary["features"] == len(FEATURE_COLUMNS)
    assert summary["numeric_features"] == len(NUMERIC_FEATURE_COLUMNS)
    assert summary["categorical_features"] == len(CATEGORICAL_FEATURE_COLUMNS)
    assert summary["missing_values"] == 0
    assert 0.05 <= summary["target_rate"] <= 0.15


def test_missing_required_columns_raise_error() -> None:
    """Feature engineering should fail clearly when required columns are absent."""
    panel = generate_customer_month_panel(profile=get_profile("compact"))
    broken_panel = panel.drop(columns=["active_products"])

    with pytest.raises(ValueError, match="Missing required columns"):
        build_feature_table(broken_panel)