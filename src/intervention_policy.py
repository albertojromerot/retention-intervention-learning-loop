"""Intervention policy for the synthetic retention learning loop.

This module converts model scores into an auditable intervention list. It
selects high-risk customer-month rows from the latest available snapshot,
assigns reason codes, recommends supportive actions, and creates a deterministic
treatment/control split.

The policy is synthetic and educational. It is not intended for autonomous
real-world customer decisioning.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    CONTROL_SHARE,
    CUSTOMER_ID_COLUMN,
    INTERVENTION_CAPACITY_BY_PROFILE,
    INTERVENTION_LIST_V1_FILE,
    MODEL_V1_VERSION,
    OUTPUTS_DIR,
    RANDOM_SEED,
    SNAPSHOT_MONTH_COLUMN,
    TARGET_COLUMN,
    TREATMENT_SHARE,
    get_profile,
)
from src.feature_engineering import SPLIT_COLUMN
from src.model_training import MODEL_V1_DECILE_COLUMN, MODEL_V1_SCORE_COLUMN


ASSIGNED_GROUP_COLUMN = "assigned_group"
RECOMMENDED_ACTION_COLUMN = "recommended_action"
REASON_CODE_1_COLUMN = "reason_code_1"
REASON_CODE_2_COLUMN = "reason_code_2"


@dataclass(frozen=True)
class InterventionPolicyResult:
    """Container for intervention-policy output and metadata."""

    intervention_list: pd.DataFrame
    profile_name: str
    model_version: str
    snapshot_month: str
    intervention_capacity: int
    treatment_share: float
    control_share: float


def _validate_inputs(scored_table: pd.DataFrame, feature_table: pd.DataFrame) -> None:
    """Validate scored and feature tables needed by the intervention policy."""
    required_scored_columns = {
        CUSTOMER_ID_COLUMN,
        SNAPSHOT_MONTH_COLUMN,
        SPLIT_COLUMN,
        TARGET_COLUMN,
        MODEL_V1_SCORE_COLUMN,
        MODEL_V1_DECILE_COLUMN,
        "relationship_value_index",
        "value_at_risk_index",
        "relationship_health_score",
    }

    required_feature_columns = {
        CUSTOMER_ID_COLUMN,
        SNAPSHOT_MONTH_COLUMN,
        "digital_inactivity_flag",
        "engagement_decline_flag",
        "high_service_friction_flag",
        "low_satisfaction_flag",
        "credit_stress_score",
        "process_friction_score",
        "relationship_depth_score",
        "relationship_value_index",
        "value_at_risk_index",
        "relationship_health_score",
    }

    missing_scored_columns = sorted(required_scored_columns.difference(scored_table.columns))
    missing_feature_columns = sorted(required_feature_columns.difference(feature_table.columns))

    if missing_scored_columns:
        raise ValueError(f"Missing required scored-table columns: {missing_scored_columns}")

    if missing_feature_columns:
        raise ValueError(f"Missing required feature-table columns: {missing_feature_columns}")


def _latest_snapshot_month(scored_table: pd.DataFrame) -> str:
    """Return the latest snapshot month in the scored table."""
    return str(sorted(scored_table[SNAPSHOT_MONTH_COLUMN].unique())[-1])


def _assign_reason_codes(row: pd.Series) -> tuple[str, str]:
    """Assign two human-readable synthetic reason codes for intervention."""
    reasons: list[str] = []

    if row["digital_inactivity_flag"] == 1:
        reasons.append("digital_inactivity")

    if row["engagement_decline_flag"] == 1:
        reasons.append("declining_engagement")

    if row["high_service_friction_flag"] == 1:
        reasons.append("service_friction")

    if row["low_satisfaction_flag"] == 1:
        reasons.append("low_satisfaction")

    if row["credit_stress_score"] >= 55:
        reasons.append("credit_stress")

    if row["relationship_health_score"] <= 45:
        reasons.append("weak_relationship_health")

    if row["value_at_risk_index"] >= 40:
        reasons.append("high_value_at_risk")

    if not reasons:
        reasons.append("high_model_risk")
        reasons.append("portfolio_priority")

    if len(reasons) == 1:
        reasons.append("portfolio_priority")

    return reasons[0], reasons[1]


def _assign_recommended_action(row: pd.Series) -> str:
    """Assign a supportive recommended action from synthetic policy rules."""
    if row["high_service_friction_flag"] == 1 or row["low_satisfaction_flag"] == 1:
        return "service_recovery_case"

    if row["credit_stress_score"] >= 55:
        return "financial_wellbeing_check"

    if row["digital_inactivity_flag"] == 1 or row["engagement_decline_flag"] == 1:
        return "digital_reactivation_nudge"

    if row["value_at_risk_index"] >= 40:
        return "relationship_call"

    return "relationship_check_in"


def _assign_treatment_control(
    intervention_list: pd.DataFrame,
    treatment_share: float = TREATMENT_SHARE,
    random_seed: int = RANDOM_SEED,
) -> pd.Series:
    """Assign deterministic treatment/control groups within the intervention list."""
    if not 0 < treatment_share < 1:
        raise ValueError("treatment_share must be between 0 and 1.")

    rng = np.random.default_rng(random_seed)
    n_rows = len(intervention_list)
    n_treatment = int(round(n_rows * treatment_share))

    assignments = np.array(["control"] * n_rows, dtype=object)
    treatment_indices = rng.choice(n_rows, size=n_treatment, replace=False)
    assignments[treatment_indices] = "treatment"

    return pd.Series(assignments, index=intervention_list.index)


def create_intervention_list(
    scored_table: pd.DataFrame,
    feature_table: pd.DataFrame,
    profile_name: str = "deep",
    model_version: str = MODEL_V1_VERSION,
) -> InterventionPolicyResult:
    """Create a ranked intervention list from model scores.

    Args:
        scored_table: Scored output from ML model v1.
        feature_table: Engineered feature table used to enrich policy reasons.
        profile_name: Synthetic run profile controlling intervention capacity.
        model_version: Model version used for scoring.

    Returns:
        InterventionPolicyResult with the intervention list and metadata.
    """
    _validate_inputs(scored_table=scored_table, feature_table=feature_table)

    profile = get_profile(profile_name)
    intervention_capacity = INTERVENTION_CAPACITY_BY_PROFILE[profile.name]
    snapshot_month = _latest_snapshot_month(scored_table)

    latest_scores = scored_table.loc[
        scored_table[SNAPSHOT_MONTH_COLUMN].eq(snapshot_month)
    ].copy()

    feature_columns_for_policy = [
        CUSTOMER_ID_COLUMN,
        SNAPSHOT_MONTH_COLUMN,
        "digital_inactivity_flag",
        "engagement_decline_flag",
        "high_service_friction_flag",
        "low_satisfaction_flag",
        "credit_stress_score",
        "process_friction_score",
        "relationship_depth_score",
        "relationship_value_index",
        "value_at_risk_index",
        "relationship_health_score",
    ]

    latest_features = feature_table.loc[
        feature_table[SNAPSHOT_MONTH_COLUMN].eq(snapshot_month),
        feature_columns_for_policy,
    ].copy()

    policy_frame = latest_scores.merge(
        latest_features,
        on=[CUSTOMER_ID_COLUMN, SNAPSHOT_MONTH_COLUMN],
        how="left",
        suffixes=("", "_feature"),
    )

    selected = (
        policy_frame.sort_values(
            by=[MODEL_V1_SCORE_COLUMN, "value_at_risk_index"],
            ascending=[False, False],
        )
        .head(intervention_capacity)
        .copy()
    )

    reason_codes = selected.apply(_assign_reason_codes, axis=1)
    selected[REASON_CODE_1_COLUMN] = [reason[0] for reason in reason_codes]
    selected[REASON_CODE_2_COLUMN] = [reason[1] for reason in reason_codes]

    selected[RECOMMENDED_ACTION_COLUMN] = selected.apply(
        _assign_recommended_action,
        axis=1,
    )

    selected[ASSIGNED_GROUP_COLUMN] = _assign_treatment_control(
        selected,
        treatment_share=TREATMENT_SHARE,
        random_seed=RANDOM_SEED,
    )

    selected["policy_rank"] = np.arange(1, len(selected) + 1)
    selected["model_version"] = model_version
    selected["intervention_capacity"] = intervention_capacity
    selected["treatment_share"] = TREATMENT_SHARE
    selected["control_share"] = CONTROL_SHARE

    output_columns = [
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
    ]

    intervention_list = selected[output_columns].reset_index(drop=True)

    return InterventionPolicyResult(
        intervention_list=intervention_list,
        profile_name=profile.name,
        model_version=model_version,
        snapshot_month=snapshot_month,
        intervention_capacity=intervention_capacity,
        treatment_share=TREATMENT_SHARE,
        control_share=CONTROL_SHARE,
    )


def summarise_intervention_list(intervention_list: pd.DataFrame) -> dict[str, int | float]:
    """Return concise diagnostics for an intervention list."""
    return {
        "rows": int(len(intervention_list)),
        "unique_customers": int(intervention_list[CUSTOMER_ID_COLUMN].nunique()),
        "target_rate": float(intervention_list[TARGET_COLUMN].mean()),
        "average_risk_score": float(intervention_list[MODEL_V1_SCORE_COLUMN].mean()),
        "treatment_rows": int(intervention_list[ASSIGNED_GROUP_COLUMN].eq("treatment").sum()),
        "control_rows": int(intervention_list[ASSIGNED_GROUP_COLUMN].eq("control").sum()),
        "recommended_actions": int(intervention_list[RECOMMENDED_ACTION_COLUMN].nunique()),
    }


def intervention_summary_as_text(summary: dict[str, int | float]) -> str:
    """Format intervention-list diagnostics for terminal output."""
    return (
        "Intervention policy summary\n"
        f"Rows: {summary['rows']:,}\n"
        f"Unique customers: {summary['unique_customers']:,}\n"
        f"Observed synthetic target rate: {summary['target_rate']:.2%}\n"
        f"Average model risk score: {summary['average_risk_score']:.2%}\n"
        f"Treatment rows: {summary['treatment_rows']:,}\n"
        f"Control rows: {summary['control_rows']:,}\n"
        f"Recommended action types: {summary['recommended_actions']:,}"
    )


def save_intervention_list(result: InterventionPolicyResult) -> Path:
    """Save intervention list to the outputs directory."""
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    output_path = OUTPUTS_DIR / INTERVENTION_LIST_V1_FILE
    result.intervention_list.to_csv(output_path, index=False)

    return output_path

def save_intervention_outputs(result) -> dict:
    """Save the intervention policy list to the outputs directory."""
    from src.config import INTERVENTION_LIST_V1_FILE, OUTPUTS_DIR

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    intervention_list_path = OUTPUTS_DIR / INTERVENTION_LIST_V1_FILE
    result.intervention_list.to_csv(intervention_list_path, index=False)

    return {
        "intervention_list": intervention_list_path,
    }