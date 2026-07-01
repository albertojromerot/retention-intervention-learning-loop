"""Tests for treatment-aware model v2 learning."""

import pandas as pd
import pytest

from src.config import (
    CUSTOMER_ID_COLUMN,
    MODEL_V2_VERSION,
    SNAPSHOT_MONTH_COLUMN,
    get_profile,
)
from src.feature_engineering import FEATURE_COLUMNS, build_feature_table
from src.intervention_policy import (
    ASSIGNED_GROUP_COLUMN,
    RECOMMENDED_ACTION_COLUMN,
    create_intervention_list,
)
from src.model_training import MODEL_V1_SCORE_COLUMN, run_model_v1
from src.model_v2_learning import (
    MODEL_V2_DECILE_COLUMN,
    MODEL_V2_FEATURE_COLUMNS,
    MODEL_V2_SCORE_COLUMN,
    TREATMENT_LEARNING_SPLIT_COLUMN,
    build_model_v2_pipeline,
    build_treatment_learning_table,
    evaluate_model_v2,
    run_model_v2,
    score_model_v2,
    train_model_v2,
)
from src.synthetic_data import generate_customer_month_panel
from src.treatment_log import (
    CONTACT_ATTEMPTED_COLUMN,
    CONTACT_SUCCESS_COLUMN,
    INTERVENTION_COST_COLUMN,
    POST_INTERVENTION_TARGET_COLUMN,
    TREATMENT_COMPLETED_COLUMN,
    generate_treatment_log,
)


@pytest.fixture(scope="module")
def compact_v2_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate compact feature table and treatment log once for this module."""
    panel = generate_customer_month_panel(profile=get_profile("compact"))
    feature_table = build_feature_table(panel).feature_table
    model_v1_result = run_model_v1(feature_table)

    policy_result = create_intervention_list(
        scored_table=model_v1_result.scored_table,
        feature_table=feature_table,
        profile_name="compact",
    )

    treatment_log = generate_treatment_log(
        policy_result.intervention_list
    ).treatment_log

    return feature_table, treatment_log


@pytest.fixture(scope="module")
def compact_learning_table(
    compact_v2_inputs: tuple[pd.DataFrame, pd.DataFrame],
) -> pd.DataFrame:
    """Build compact treatment-learning table once for this module."""
    feature_table, treatment_log = compact_v2_inputs
    return build_treatment_learning_table(
        feature_table=feature_table,
        treatment_log=treatment_log,
    )


@pytest.fixture(scope="module")
def compact_model_v2_result(
    compact_v2_inputs: tuple[pd.DataFrame, pd.DataFrame],
):
    """Train and evaluate model v2 once for this module."""
    feature_table, treatment_log = compact_v2_inputs
    return run_model_v2(
        feature_table=feature_table,
        treatment_log=treatment_log,
    )


def test_treatment_learning_table_has_expected_row_count(
    compact_learning_table: pd.DataFrame,
) -> None:
    """Compact treatment-learning table should preserve intervention capacity."""
    assert len(compact_learning_table) == 250


def test_treatment_learning_table_required_columns_exist(
    compact_learning_table: pd.DataFrame,
) -> None:
    """Learning table should contain treatment, feature and target columns."""
    required_columns = {
        "treatment_log_id",
        "policy_rank",
        CUSTOMER_ID_COLUMN,
        SNAPSHOT_MONTH_COLUMN,
        TREATMENT_LEARNING_SPLIT_COLUMN,
        ASSIGNED_GROUP_COLUMN,
        RECOMMENDED_ACTION_COLUMN,
        "contact_channel",
        MODEL_V1_SCORE_COLUMN,
        CONTACT_ATTEMPTED_COLUMN,
        CONTACT_SUCCESS_COLUMN,
        TREATMENT_COMPLETED_COLUMN,
        INTERVENTION_COST_COLUMN,
        POST_INTERVENTION_TARGET_COLUMN,
        "assigned_treatment_flag",
        *FEATURE_COLUMNS,
    }

    assert required_columns.issubset(set(compact_learning_table.columns))


def test_treatment_learning_table_has_expected_splits(
    compact_learning_table: pd.DataFrame,
) -> None:
    """Model v2 learning table should use 60/20/20 train/validation/test split."""
    split_counts = compact_learning_table[TREATMENT_LEARNING_SPLIT_COLUMN].value_counts()

    assert split_counts["train"] == 150
    assert split_counts["validation"] == 50
    assert split_counts["test"] == 50


def test_treatment_learning_table_has_no_unassigned_rows(
    compact_learning_table: pd.DataFrame,
) -> None:
    """Every learning row should be assigned to train, validation or test."""
    assert set(compact_learning_table[TREATMENT_LEARNING_SPLIT_COLUMN].unique()) == {
        "train",
        "validation",
        "test",
    }


def test_assigned_treatment_flag_matches_group(
    compact_learning_table: pd.DataFrame,
) -> None:
    """Assigned treatment flag should equal 1 for treatment and 0 for control."""
    expected_flag = compact_learning_table[ASSIGNED_GROUP_COLUMN].eq("treatment").astype("int8")

    assert compact_learning_table["assigned_treatment_flag"].equals(expected_flag)


def test_learning_table_customer_month_key_is_unique(
    compact_learning_table: pd.DataFrame,
) -> None:
    """Learning table should not duplicate customer-month rows."""
    duplicated_keys = compact_learning_table.duplicated(
        subset=[CUSTOMER_ID_COLUMN, SNAPSHOT_MONTH_COLUMN]
    ).sum()

    assert duplicated_keys == 0


def test_model_v2_pipeline_can_be_built() -> None:
    """Model v2 pipeline should contain preprocessing and classifier steps."""
    model = build_model_v2_pipeline()

    assert "preprocessor" in model.named_steps
    assert "classifier" in model.named_steps


def test_train_model_v2_returns_fitted_pipeline(
    compact_learning_table: pd.DataFrame,
) -> None:
    """Training model v2 should return a fitted sklearn pipeline."""
    model = train_model_v2(compact_learning_table)

    assert hasattr(model, "predict_proba")


def test_model_v2_feature_columns_are_available(
    compact_learning_table: pd.DataFrame,
) -> None:
    """All configured model v2 features should exist in the learning table."""
    assert set(MODEL_V2_FEATURE_COLUMNS).issubset(set(compact_learning_table.columns))


def test_score_model_v2_preserves_row_count(
    compact_learning_table: pd.DataFrame,
) -> None:
    """Scoring model v2 should preserve learning-table row count."""
    model = train_model_v2(compact_learning_table)
    scored_table = score_model_v2(model, compact_learning_table)

    assert len(scored_table) == len(compact_learning_table)


def test_model_v2_scored_table_required_columns_exist(
    compact_model_v2_result,
) -> None:
    """Model v2 scored output should contain required audit and score columns."""
    scored_table = compact_model_v2_result.scored_table

    required_columns = {
        "treatment_log_id",
        "policy_rank",
        CUSTOMER_ID_COLUMN,
        SNAPSHOT_MONTH_COLUMN,
        TREATMENT_LEARNING_SPLIT_COLUMN,
        ASSIGNED_GROUP_COLUMN,
        RECOMMENDED_ACTION_COLUMN,
        "contact_channel",
        POST_INTERVENTION_TARGET_COLUMN,
        MODEL_V1_SCORE_COLUMN,
        CONTACT_ATTEMPTED_COLUMN,
        CONTACT_SUCCESS_COLUMN,
        TREATMENT_COMPLETED_COLUMN,
        INTERVENTION_COST_COLUMN,
        MODEL_V2_SCORE_COLUMN,
        MODEL_V2_DECILE_COLUMN,
    }

    assert required_columns.issubset(set(scored_table.columns))


def test_model_v2_scores_are_probabilities(compact_model_v2_result) -> None:
    """Model v2 scores should be valid probabilities."""
    scored_table = compact_model_v2_result.scored_table

    assert scored_table[MODEL_V2_SCORE_COLUMN].between(0, 1).all()


def test_model_v2_deciles_are_between_one_and_ten(compact_model_v2_result) -> None:
    """Model v2 risk deciles should be in the 1 to 10 range."""
    scored_table = compact_model_v2_result.scored_table

    assert scored_table[MODEL_V2_DECILE_COLUMN].between(1, 10).all()


def test_model_v2_metrics_are_generated(compact_model_v2_result) -> None:
    """Model v2 should generate validation and test metrics."""
    metrics = compact_model_v2_result.metrics

    assert not metrics.empty
    assert set(metrics["split"]) == {"validation", "test"}
    assert set(metrics["model_version"]) == {MODEL_V2_VERSION}


def test_model_v2_metrics_have_expected_columns(compact_model_v2_result) -> None:
    """Model v2 metrics should include classification and ranking metrics."""
    metrics = compact_model_v2_result.metrics

    required_columns = {
        "model_version",
        "split",
        "rows",
        "target_rate",
        "roc_auc",
        "pr_auc",
        "brier_score",
        "precision_at_50",
        "lift_at_50",
        "precision_at_100",
        "lift_at_100",
        "precision_at_200",
        "lift_at_200",
    }

    assert required_columns.issubset(set(metrics.columns))


def test_model_v2_metrics_are_in_valid_ranges(compact_model_v2_result) -> None:
    """Model v2 metrics should be finite and in valid metric ranges."""
    metrics = compact_model_v2_result.metrics

    for column in ["target_rate", "roc_auc", "pr_auc", "brier_score"]:
        assert metrics[column].notna().all(), column

    assert metrics["target_rate"].between(0.0, 1.0).all()
    assert metrics["roc_auc"].between(0.0, 1.0).all()
    assert metrics["pr_auc"].between(0.0, 1.0).all()
    assert metrics["brier_score"].between(0.0, 1.0).all()


def test_evaluate_model_v2_can_use_custom_k_values(compact_model_v2_result) -> None:
    """Model v2 evaluation should support custom precision/lift cut-offs."""
    scored_table = compact_model_v2_result.scored_table

    metrics = evaluate_model_v2(
        scored_table=scored_table,
        evaluation_splits=("validation",),
        k_values=(25,),
    )

    assert "precision_at_25" in metrics.columns
    assert "lift_at_25" in metrics.columns
    assert len(metrics) == 1
    assert metrics.loc[0, "split"] == "validation"


def test_missing_feature_columns_raise_error(
    compact_v2_inputs: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    """Model v2 learning table should fail clearly if features are missing."""
    feature_table, treatment_log = compact_v2_inputs
    broken_feature_table = feature_table.drop(columns=[FEATURE_COLUMNS[0]])

    with pytest.raises(ValueError, match="Missing required feature-table columns"):
        build_treatment_learning_table(
            feature_table=broken_feature_table,
            treatment_log=treatment_log,
        )


def test_missing_treatment_log_columns_raise_error(
    compact_v2_inputs: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    """Model v2 learning table should fail clearly if treatment-log columns are missing."""
    feature_table, treatment_log = compact_v2_inputs
    broken_treatment_log = treatment_log.drop(columns=[POST_INTERVENTION_TARGET_COLUMN])

    with pytest.raises(ValueError, match="Missing required treatment-log columns"):
        build_treatment_learning_table(
            feature_table=feature_table,
            treatment_log=broken_treatment_log,
        )