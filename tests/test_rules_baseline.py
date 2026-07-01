"""Tests for the rules-based baseline module."""

import pandas as pd
import pytest

from src.config import (
    CUSTOMER_ID_COLUMN,
    RULES_BASELINE_VERSION,
    SNAPSHOT_MONTH_COLUMN,
    TARGET_COLUMN,
    get_profile,
)
from src.feature_engineering import SPLIT_COLUMN, build_feature_table
from src.rules_baseline import (
    RULES_DECILE_COLUMN,
    RULES_SCORE_COLUMN,
    evaluate_rules_baseline,
    run_rules_baseline,
    score_with_rules_baseline,
)
from src.synthetic_data import generate_customer_month_panel


def _build_compact_feature_table() -> pd.DataFrame:
    """Generate a compact engineered feature table for fast tests."""
    panel = generate_customer_month_panel(profile=get_profile("compact"))
    return build_feature_table(panel).feature_table


def test_rules_baseline_preserves_row_count() -> None:
    """Rules scoring should preserve the feature-table row count."""
    feature_table = _build_compact_feature_table()
    scored_table = score_with_rules_baseline(feature_table)

    assert len(scored_table) == len(feature_table)


def test_rules_baseline_required_columns_exist() -> None:
    """Rules output should contain expected identifiers, target, score, and decile."""
    feature_table = _build_compact_feature_table()
    scored_table = score_with_rules_baseline(feature_table)

    required_columns = {
        CUSTOMER_ID_COLUMN,
        SNAPSHOT_MONTH_COLUMN,
        SPLIT_COLUMN,
        TARGET_COLUMN,
        "relationship_value_index",
        "value_at_risk_index",
        "relationship_health_score",
        RULES_SCORE_COLUMN,
        RULES_DECILE_COLUMN,
    }

    assert required_columns.issubset(set(scored_table.columns))


def test_rules_score_is_between_zero_and_one() -> None:
    """Rules risk score should be scaled between 0 and 1."""
    feature_table = _build_compact_feature_table()
    scored_table = score_with_rules_baseline(feature_table)

    assert scored_table[RULES_SCORE_COLUMN].between(0, 1).all()


def test_rules_decile_is_between_one_and_ten() -> None:
    """Rules risk decile should be in the 1 to 10 range."""
    feature_table = _build_compact_feature_table()
    scored_table = score_with_rules_baseline(feature_table)

    assert scored_table[RULES_DECILE_COLUMN].between(1, 10).all()


def test_customer_month_key_remains_unique_after_rules_scoring() -> None:
    """Rules scoring should not duplicate customer-month rows."""
    feature_table = _build_compact_feature_table()
    scored_table = score_with_rules_baseline(feature_table)

    duplicated_keys = scored_table.duplicated(
        subset=[CUSTOMER_ID_COLUMN, SNAPSHOT_MONTH_COLUMN]
    ).sum()

    assert duplicated_keys == 0


def test_rules_baseline_metrics_are_generated() -> None:
    """Rules baseline should generate validation and test metrics."""
    feature_table = _build_compact_feature_table()
    result = run_rules_baseline(feature_table)

    assert not result.metrics.empty
    assert set(result.metrics["split"]) == {"validation", "test"}
    assert set(result.metrics["model_version"]) == {RULES_BASELINE_VERSION}


def test_rules_baseline_metrics_have_expected_columns() -> None:
    """Rules metrics should contain classification and ranking metrics."""
    feature_table = _build_compact_feature_table()
    result = run_rules_baseline(feature_table)

    required_columns = {
        "model_version",
        "split",
        "rows",
        "target_rate",
        "roc_auc",
        "pr_auc",
        "brier_score",
        "precision_at_100",
        "lift_at_100",
        "precision_at_500",
        "lift_at_500",
        "precision_at_1000",
        "lift_at_1000",
    }

    assert required_columns.issubset(set(result.metrics.columns))


def test_rules_baseline_metrics_are_reasonable() -> None:
    """Rules metrics should be finite and within valid metric ranges."""
    feature_table = _build_compact_feature_table()
    result = run_rules_baseline(feature_table)

    for column in ["target_rate", "roc_auc", "pr_auc", "brier_score"]:
        assert result.metrics[column].notna().all(), column

    assert result.metrics["target_rate"].between(0.05, 0.15).all()
    assert result.metrics["roc_auc"].between(0.5, 1.0).all()
    assert result.metrics["pr_auc"].between(0.0, 1.0).all()
    assert result.metrics["brier_score"].between(0.0, 1.0).all()


def test_rules_baseline_lift_is_above_random_for_top_100() -> None:
    """The transparent rules baseline should improve over random ranking at top 100."""
    feature_table = _build_compact_feature_table()
    result = run_rules_baseline(feature_table)

    assert (result.metrics["lift_at_100"] > 1.0).all()


def test_evaluate_rules_baseline_can_use_custom_k_values() -> None:
    """Rules evaluation should support custom precision/lift cut-offs."""
    feature_table = _build_compact_feature_table()
    scored_table = score_with_rules_baseline(feature_table)

    metrics = evaluate_rules_baseline(
        scored_table=scored_table,
        evaluation_splits=("validation",),
        k_values=(50,),
    )

    assert "precision_at_50" in metrics.columns
    assert "lift_at_50" in metrics.columns
    assert len(metrics) == 1
    assert metrics.loc[0, "split"] == "validation"


def test_missing_required_rules_columns_raise_error() -> None:
    """Rules scoring should fail clearly if required columns are missing."""
    feature_table = _build_compact_feature_table()
    broken_table = feature_table.drop(columns=["relationship_health_score"])

    with pytest.raises(ValueError, match="Missing required columns"):
        score_with_rules_baseline(broken_table)