"""Tests for intervention policy creation and treatment/control assignment."""

import pandas as pd
import pytest

from src.config import (
    CUSTOMER_ID_COLUMN,
    INTERVENTION_CAPACITY_BY_PROFILE,
    MODEL_V1_VERSION,
    SNAPSHOT_MONTH_COLUMN,
    TARGET_COLUMN,
    TREATMENT_SHARE,
    get_profile,
)
from src.feature_engineering import SPLIT_COLUMN, build_feature_table
from src.intervention_policy import (
    ASSIGNED_GROUP_COLUMN,
    REASON_CODE_1_COLUMN,
    REASON_CODE_2_COLUMN,
    RECOMMENDED_ACTION_COLUMN,
    create_intervention_list,
    summarise_intervention_list,
)
from src.model_training import (
    MODEL_V1_DECILE_COLUMN,
    MODEL_V1_SCORE_COLUMN,
    run_model_v1,
)
from src.synthetic_data import generate_customer_month_panel


@pytest.fixture(scope="module")
def compact_policy_result():
    """Create a compact intervention-policy result once for this module."""
    panel = generate_customer_month_panel(profile=get_profile("compact"))
    feature_table = build_feature_table(panel).feature_table
    model_result = run_model_v1(feature_table)

    return create_intervention_list(
        scored_table=model_result.scored_table,
        feature_table=feature_table,
        profile_name="compact",
        model_version=MODEL_V1_VERSION,
    )


def test_intervention_list_matches_profile_capacity(compact_policy_result) -> None:
    """The intervention list should match the configured profile capacity."""
    intervention_list = compact_policy_result.intervention_list
    expected_capacity = INTERVENTION_CAPACITY_BY_PROFILE["compact"]

    assert len(intervention_list) == expected_capacity


def test_intervention_list_has_unique_customers(compact_policy_result) -> None:
    """The intervention list should include each customer at most once."""
    intervention_list = compact_policy_result.intervention_list

    assert intervention_list[CUSTOMER_ID_COLUMN].nunique() == len(intervention_list)


def test_intervention_list_required_columns_exist(compact_policy_result) -> None:
    """The intervention list should contain all required audit columns."""
    intervention_list = compact_policy_result.intervention_list

    required_columns = {
        "policy_rank",
        CUSTOMER_ID_COLUMN,
        SNAPSHOT_MONTH_COLUMN,
        SPLIT_COLUMN,
        "model_version",
        TARGET_COLUMN,
        MODEL_V1_SCORE_COLUMN,
        MODEL_V1_DECILE_COLUMN,
        "relationship_value_index",
        "value_at_risk_index",
        "relationship_health_score",
        "digital_inactivity_flag",
        "engagement_decline_flag",
        "high_service_friction_flag",
        "low_satisfaction_flag",
        "credit_stress_score",
        REASON_CODE_1_COLUMN,
        REASON_CODE_2_COLUMN,
        RECOMMENDED_ACTION_COLUMN,
        ASSIGNED_GROUP_COLUMN,
        "intervention_capacity",
        "treatment_share",
        "control_share",
    }

    assert required_columns.issubset(set(intervention_list.columns))


def test_policy_rank_is_sequential(compact_policy_result) -> None:
    """Policy ranks should run sequentially from 1 to the list length."""
    intervention_list = compact_policy_result.intervention_list

    expected_ranks = list(range(1, len(intervention_list) + 1))

    assert intervention_list["policy_rank"].tolist() == expected_ranks


def test_intervention_list_uses_latest_snapshot_only(compact_policy_result) -> None:
    """The intervention list should be created from one latest snapshot month."""
    intervention_list = compact_policy_result.intervention_list

    assert intervention_list[SNAPSHOT_MONTH_COLUMN].nunique() == 1
    assert intervention_list[SNAPSHOT_MONTH_COLUMN].iloc[0] == compact_policy_result.snapshot_month


def test_intervention_list_uses_model_v1_version(compact_policy_result) -> None:
    """The intervention list should preserve the model version used for scoring."""
    intervention_list = compact_policy_result.intervention_list

    assert set(intervention_list["model_version"]) == {MODEL_V1_VERSION}


def test_treatment_control_assignment_matches_expected_share(compact_policy_result) -> None:
    """Treatment/control assignment should match the configured 80/20 policy."""
    intervention_list = compact_policy_result.intervention_list

    group_counts = intervention_list[ASSIGNED_GROUP_COLUMN].value_counts().to_dict()

    assert group_counts["treatment"] == int(len(intervention_list) * TREATMENT_SHARE)
    assert group_counts["control"] == len(intervention_list) - group_counts["treatment"]


def test_assigned_group_values_are_valid(compact_policy_result) -> None:
    """Assigned group should contain only treatment and control."""
    intervention_list = compact_policy_result.intervention_list

    assert set(intervention_list[ASSIGNED_GROUP_COLUMN].unique()) == {
        "treatment",
        "control",
    }


def test_recommended_actions_are_non_empty(compact_policy_result) -> None:
    """Every intervention row should have a recommended action."""
    intervention_list = compact_policy_result.intervention_list

    assert intervention_list[RECOMMENDED_ACTION_COLUMN].notna().all()
    assert (intervention_list[RECOMMENDED_ACTION_COLUMN].str.len() > 0).all()


def test_reason_codes_are_non_empty(compact_policy_result) -> None:
    """Every intervention row should have two reason codes."""
    intervention_list = compact_policy_result.intervention_list

    assert intervention_list[REASON_CODE_1_COLUMN].notna().all()
    assert intervention_list[REASON_CODE_2_COLUMN].notna().all()
    assert (intervention_list[REASON_CODE_1_COLUMN].str.len() > 0).all()
    assert (intervention_list[REASON_CODE_2_COLUMN].str.len() > 0).all()


def test_intervention_list_is_high_risk_subset(compact_policy_result) -> None:
    """The intervention list should concentrate higher-risk observations."""
    intervention_list = compact_policy_result.intervention_list

    assert intervention_list[MODEL_V1_SCORE_COLUMN].mean() > 0.10
    assert intervention_list[TARGET_COLUMN].mean() > 0.10


def test_intervention_summary_matches_list(compact_policy_result) -> None:
    """The intervention summary helper should match the list diagnostics."""
    intervention_list = compact_policy_result.intervention_list
    summary = summarise_intervention_list(intervention_list)

    assert summary["rows"] == len(intervention_list)
    assert summary["unique_customers"] == intervention_list[CUSTOMER_ID_COLUMN].nunique()
    assert summary["treatment_rows"] == intervention_list[ASSIGNED_GROUP_COLUMN].eq("treatment").sum()
    assert summary["control_rows"] == intervention_list[ASSIGNED_GROUP_COLUMN].eq("control").sum()
    assert summary["recommended_actions"] == intervention_list[RECOMMENDED_ACTION_COLUMN].nunique()


def test_missing_scored_columns_raise_error() -> None:
    """The policy should fail clearly if scored-table columns are missing."""
    panel = generate_customer_month_panel(profile=get_profile("compact"))
    feature_table = build_feature_table(panel).feature_table
    model_result = run_model_v1(feature_table)

    broken_scored_table = model_result.scored_table.drop(columns=[MODEL_V1_SCORE_COLUMN])

    with pytest.raises(ValueError, match="Missing required scored-table columns"):
        create_intervention_list(
            scored_table=broken_scored_table,
            feature_table=feature_table,
            profile_name="compact",
        )


def test_missing_feature_columns_raise_error() -> None:
    """The policy should fail clearly if feature-table columns are missing."""
    panel = generate_customer_month_panel(profile=get_profile("compact"))
    feature_table = build_feature_table(panel).feature_table
    model_result = run_model_v1(feature_table)

    broken_feature_table = feature_table.drop(columns=["digital_inactivity_flag"])

    with pytest.raises(ValueError, match="Missing required feature-table columns"):
        create_intervention_list(
            scored_table=model_result.scored_table,
            feature_table=broken_feature_table,
            profile_name="compact",
        )