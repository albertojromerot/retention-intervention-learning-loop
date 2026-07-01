"""Tests for ML model v1 training, scoring, and evaluation."""

import pandas as pd
import pytest

from src.config import (
    CUSTOMER_ID_COLUMN,
    MODEL_V1_VERSION,
    SNAPSHOT_MONTH_COLUMN,
    TARGET_COLUMN,
    get_profile,
)
from src.feature_engineering import FEATURE_COLUMNS, SPLIT_COLUMN, build_feature_table
from src.model_training import (
    MODEL_V1_DECILE_COLUMN,
    MODEL_V1_SCORE_COLUMN,
    build_model_v1_pipeline,
    evaluate_model_scores,
    run_model_v1,
    score_model_v1,
    train_model_v1,
)
from src.synthetic_data import generate_customer_month_panel


@pytest.fixture(scope="module")
def compact_feature_table() -> pd.DataFrame:
    """Build a compact feature table once for this test module."""
    panel = generate_customer_month_panel(profile=get_profile("compact"))
    return build_feature_table(panel).feature_table


@pytest.fixture(scope="module")
def compact_model_result(compact_feature_table: pd.DataFrame):
    """Train and evaluate ML model v1 once for this test module."""
    return run_model_v1(compact_feature_table)


def test_model_pipeline_can_be_built() -> None:
    """The model pipeline should contain preprocessing and classifier steps."""
    model = build_model_v1_pipeline()

    assert "preprocessor" in model.named_steps
    assert "classifier" in model.named_steps


def test_train_model_v1_returns_fitted_pipeline(
    compact_feature_table: pd.DataFrame,
) -> None:
    """Training should return a fitted sklearn pipeline."""
    model = train_model_v1(compact_feature_table)

    assert hasattr(model, "predict_proba")


def test_model_v1_scoring_preserves_row_count(
    compact_feature_table: pd.DataFrame,
) -> None:
    """Scoring should preserve the feature-table row count."""
    model = train_model_v1(compact_feature_table)
    scored_table = score_model_v1(model, compact_feature_table)

    assert len(scored_table) == len(compact_feature_table)


def test_model_v1_scored_table_has_required_columns(
    compact_model_result,
) -> None:
    """Model v1 scored output should contain expected columns."""
    scored_table = compact_model_result.scored_table

    required_columns = {
        CUSTOMER_ID_COLUMN,
        SNAPSHOT_MONTH_COLUMN,
        SPLIT_COLUMN,
        TARGET_COLUMN,
        "relationship_value_index",
        "value_at_risk_index",
        "relationship_health_score",
        MODEL_V1_SCORE_COLUMN,
        MODEL_V1_DECILE_COLUMN,
    }

    assert required_columns.issubset(set(scored_table.columns))


def test_model_v1_scores_are_probabilities(compact_model_result) -> None:
    """Model v1 risk scores should be between 0 and 1."""
    scored_table = compact_model_result.scored_table

    assert scored_table[MODEL_V1_SCORE_COLUMN].between(0, 1).all()


def test_model_v1_deciles_are_between_one_and_ten(compact_model_result) -> None:
    """Model v1 deciles should be in the 1 to 10 range."""
    scored_table = compact_model_result.scored_table

    assert scored_table[MODEL_V1_DECILE_COLUMN].between(1, 10).all()


def test_customer_month_key_remains_unique_after_model_scoring(
    compact_model_result,
) -> None:
    """Model scoring should not duplicate customer-month rows."""
    scored_table = compact_model_result.scored_table

    duplicated_keys = scored_table.duplicated(
        subset=[CUSTOMER_ID_COLUMN, SNAPSHOT_MONTH_COLUMN]
    ).sum()

    assert duplicated_keys == 0


def test_model_v1_metrics_are_generated(compact_model_result) -> None:
    """Model v1 should generate validation and test metrics."""
    metrics = compact_model_result.metrics

    assert not metrics.empty
    assert set(metrics["split"]) == {"validation", "test"}
    assert set(metrics["model_version"]) == {MODEL_V1_VERSION}


def test_model_v1_metrics_have_expected_columns(compact_model_result) -> None:
    """Model metrics should contain classification and ranking metrics."""
    metrics = compact_model_result.metrics

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

    assert required_columns.issubset(set(metrics.columns))


def test_model_v1_metrics_are_reasonable(compact_model_result) -> None:
    """Model metrics should be finite and within valid metric ranges."""
    metrics = compact_model_result.metrics

    for column in ["target_rate", "roc_auc", "pr_auc", "brier_score"]:
        assert metrics[column].notna().all(), column

    assert metrics["target_rate"].between(0.05, 0.15).all()
    assert metrics["roc_auc"].between(0.5, 1.0).all()
    assert metrics["pr_auc"].between(0.0, 1.0).all()
    assert metrics["brier_score"].between(0.0, 1.0).all()


def test_model_v1_lift_is_above_random_for_top_100(compact_model_result) -> None:
    """Model v1 should improve over random ranking at top 100."""
    metrics = compact_model_result.metrics

    assert (metrics["lift_at_100"] > 1.0).all()


def test_evaluate_model_scores_can_use_custom_k_values(
    compact_model_result,
) -> None:
    """Model evaluation should support custom precision/lift cut-offs."""
    scored_table = compact_model_result.scored_table

    metrics = evaluate_model_scores(
        scored_table=scored_table,
        score_column=MODEL_V1_SCORE_COLUMN,
        model_version=MODEL_V1_VERSION,
        evaluation_splits=("validation",),
        k_values=(50,),
    )

    assert "precision_at_50" in metrics.columns
    assert "lift_at_50" in metrics.columns
    assert len(metrics) == 1
    assert metrics.loc[0, "split"] == "validation"


def test_missing_required_model_columns_raise_error(
    compact_feature_table: pd.DataFrame,
) -> None:
    """Training should fail clearly if required feature columns are missing."""
    broken_table = compact_feature_table.drop(columns=[FEATURE_COLUMNS[0]])

    with pytest.raises(ValueError, match="Missing required columns"):
        train_model_v1(broken_table)