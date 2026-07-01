"""Tests for synthetic treatment-log generation."""

import pandas as pd
import pytest

from src.config import (
    CUSTOMER_ID_COLUMN,
    SNAPSHOT_MONTH_COLUMN,
    TARGET_COLUMN,
    get_profile,
)
from src.feature_engineering import build_feature_table
from src.intervention_policy import (
    ASSIGNED_GROUP_COLUMN,
    RECOMMENDED_ACTION_COLUMN,
    create_intervention_list,
)
from src.model_training import MODEL_V1_SCORE_COLUMN, run_model_v1
from src.synthetic_data import generate_customer_month_panel
from src.treatment_log import (
    CONTACT_ATTEMPTED_COLUMN,
    CONTACT_SUCCESS_COLUMN,
    INTERVENTION_COST_COLUMN,
    NET_VALUE_COLUMN,
    POST_INTERVENTION_TARGET_COLUMN,
    PREVENTED_ATTRITION_COLUMN,
    RETAINED_VALUE_COLUMN,
    TREATMENT_COMPLETED_COLUMN,
    generate_treatment_log,
    summarise_treatment_log,
)


@pytest.fixture(scope="module")
def compact_treatment_log() -> pd.DataFrame:
    """Generate a compact treatment log once for this test module."""
    panel = generate_customer_month_panel(profile=get_profile("compact"))
    feature_table = build_feature_table(panel).feature_table
    model_result = run_model_v1(feature_table)

    policy_result = create_intervention_list(
        scored_table=model_result.scored_table,
        feature_table=feature_table,
        profile_name="compact",
    )

    treatment_result = generate_treatment_log(policy_result.intervention_list)

    return treatment_result.treatment_log


def test_treatment_log_has_expected_row_count(compact_treatment_log: pd.DataFrame) -> None:
    """Treatment log should preserve compact intervention-list capacity."""
    assert len(compact_treatment_log) == 250


def test_treatment_log_required_columns_exist(compact_treatment_log: pd.DataFrame) -> None:
    """Treatment log should contain required audit and outcome columns."""
    required_columns = {
        "treatment_log_id",
        "policy_rank",
        CUSTOMER_ID_COLUMN,
        SNAPSHOT_MONTH_COLUMN,
        ASSIGNED_GROUP_COLUMN,
        RECOMMENDED_ACTION_COLUMN,
        "contact_channel",
        "reason_code_1",
        "reason_code_2",
        MODEL_V1_SCORE_COLUMN,
        "relationship_value_index",
        "value_at_risk_index",
        "relationship_health_score",
        TARGET_COLUMN,
        CONTACT_ATTEMPTED_COLUMN,
        CONTACT_SUCCESS_COLUMN,
        TREATMENT_COMPLETED_COLUMN,
        PREVENTED_ATTRITION_COLUMN,
        POST_INTERVENTION_TARGET_COLUMN,
        INTERVENTION_COST_COLUMN,
        RETAINED_VALUE_COLUMN,
        NET_VALUE_COLUMN,
    }

    assert required_columns.issubset(set(compact_treatment_log.columns))


def test_treatment_log_id_is_unique(compact_treatment_log: pd.DataFrame) -> None:
    """Each treatment-log row should have a unique treatment log ID."""
    assert compact_treatment_log["treatment_log_id"].is_unique


def test_customer_month_key_remains_unique(compact_treatment_log: pd.DataFrame) -> None:
    """Treatment log should not duplicate customer-month rows."""
    duplicated_keys = compact_treatment_log.duplicated(
        subset=[CUSTOMER_ID_COLUMN, SNAPSHOT_MONTH_COLUMN]
    ).sum()

    assert duplicated_keys == 0


def test_treatment_control_counts_are_expected(compact_treatment_log: pd.DataFrame) -> None:
    """Compact policy should produce 200 treatment rows and 50 control rows."""
    group_counts = compact_treatment_log[ASSIGNED_GROUP_COLUMN].value_counts().to_dict()

    assert group_counts["treatment"] == 200
    assert group_counts["control"] == 50


def test_control_group_has_no_contact_or_treatment(compact_treatment_log: pd.DataFrame) -> None:
    """Control rows should remain a clean holdout group."""
    control_rows = compact_treatment_log[ASSIGNED_GROUP_COLUMN].eq("control")

    assert compact_treatment_log.loc[control_rows, CONTACT_ATTEMPTED_COLUMN].sum() == 0
    assert compact_treatment_log.loc[control_rows, CONTACT_SUCCESS_COLUMN].sum() == 0
    assert compact_treatment_log.loc[control_rows, TREATMENT_COMPLETED_COLUMN].sum() == 0
    assert compact_treatment_log.loc[control_rows, INTERVENTION_COST_COLUMN].sum() == 0


def test_treatment_group_is_attempted(compact_treatment_log: pd.DataFrame) -> None:
    """Every treatment row should have a contact attempt."""
    treatment_rows = compact_treatment_log[ASSIGNED_GROUP_COLUMN].eq("treatment")

    assert compact_treatment_log.loc[treatment_rows, CONTACT_ATTEMPTED_COLUMN].eq(1).all()


def test_binary_treatment_columns_are_binary(compact_treatment_log: pd.DataFrame) -> None:
    """Treatment execution and outcome flags should be binary."""
    binary_columns = [
        CONTACT_ATTEMPTED_COLUMN,
        CONTACT_SUCCESS_COLUMN,
        TREATMENT_COMPLETED_COLUMN,
        PREVENTED_ATTRITION_COLUMN,
        POST_INTERVENTION_TARGET_COLUMN,
    ]

    for column in binary_columns:
        assert set(compact_treatment_log[column].unique()).issubset({0, 1}), column


def test_treatment_completion_requires_contact_success(
    compact_treatment_log: pd.DataFrame,
) -> None:
    """A treatment can only be completed when contact succeeded."""
    completed_rows = compact_treatment_log[TREATMENT_COMPLETED_COLUMN].eq(1)

    assert compact_treatment_log.loc[completed_rows, CONTACT_SUCCESS_COLUMN].eq(1).all()


def test_prevented_attrition_requires_original_attrition_and_completion(
    compact_treatment_log: pd.DataFrame,
) -> None:
    """Prevented attrition should only happen for completed treatment and original attrition."""
    prevented_rows = compact_treatment_log[PREVENTED_ATTRITION_COLUMN].eq(1)

    assert compact_treatment_log.loc[prevented_rows, TARGET_COLUMN].eq(1).all()
    assert compact_treatment_log.loc[prevented_rows, TREATMENT_COMPLETED_COLUMN].eq(1).all()
    assert compact_treatment_log.loc[prevented_rows, ASSIGNED_GROUP_COLUMN].eq("treatment").all()


def test_post_intervention_target_cannot_exceed_original_target(
    compact_treatment_log: pd.DataFrame,
) -> None:
    """Post-intervention attrition should not exceed original synthetic attrition."""
    assert (
        compact_treatment_log[POST_INTERVENTION_TARGET_COLUMN]
        <= compact_treatment_log[TARGET_COLUMN]
    ).all()


def test_cost_and_value_columns_are_valid(compact_treatment_log: pd.DataFrame) -> None:
    """Cost and retained value columns should be numerically coherent."""
    assert compact_treatment_log[INTERVENTION_COST_COLUMN].ge(0).all()
    assert compact_treatment_log[RETAINED_VALUE_COLUMN].ge(0).all()

    expected_net_value = (
        compact_treatment_log[RETAINED_VALUE_COLUMN]
        - compact_treatment_log[INTERVENTION_COST_COLUMN]
    ).round(2)

    assert compact_treatment_log[NET_VALUE_COLUMN].round(2).equals(expected_net_value)


def test_treatment_log_summary_matches_table(compact_treatment_log: pd.DataFrame) -> None:
    """Treatment-log summary should match table diagnostics."""
    summary = summarise_treatment_log(compact_treatment_log)

    assert summary["rows"] == len(compact_treatment_log)
    assert summary["treatment_rows"] == compact_treatment_log[ASSIGNED_GROUP_COLUMN].eq("treatment").sum()
    assert summary["control_rows"] == compact_treatment_log[ASSIGNED_GROUP_COLUMN].eq("control").sum()
    assert summary["prevented_attrition_count"] == compact_treatment_log[PREVENTED_ATTRITION_COLUMN].sum()
    assert summary["total_intervention_cost_index"] == pytest.approx(
        compact_treatment_log[INTERVENTION_COST_COLUMN].sum()
    )
    assert summary["total_retained_value_index"] == pytest.approx(
        compact_treatment_log[RETAINED_VALUE_COLUMN].sum()
    )
    assert summary["total_net_value_index"] == pytest.approx(
        compact_treatment_log[NET_VALUE_COLUMN].sum()
    )


def test_missing_intervention_columns_raise_error() -> None:
    """Treatment-log generation should fail clearly if required columns are missing."""
    panel = generate_customer_month_panel(profile=get_profile("compact"))
    feature_table = build_feature_table(panel).feature_table
    model_result = run_model_v1(feature_table)

    policy_result = create_intervention_list(
        scored_table=model_result.scored_table,
        feature_table=feature_table,
        profile_name="compact",
    )

    broken_intervention_list = policy_result.intervention_list.drop(
        columns=[RECOMMENDED_ACTION_COLUMN]
    )

    with pytest.raises(ValueError, match="Missing required intervention-list columns"):
        generate_treatment_log(broken_intervention_list)