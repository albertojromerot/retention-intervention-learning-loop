"""Synthetic treatment log for the retention intervention learning loop.

This module simulates the execution of a retention intervention policy. It
creates an auditable treatment log with treatment/control assignment, contact
outcomes, completion outcomes, synthetic intervention costs, retained value,
and post-intervention attrition outcomes.

The simulation is educational and synthetic. It does not represent real
customer contact data, operational performance, or business outcomes.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import (
    CUSTOMER_ID_COLUMN,
    OUTPUTS_DIR,
    RANDOM_SEED,
    SNAPSHOT_MONTH_COLUMN,
    TARGET_COLUMN,
    TREATMENT_LOG_V1_FILE,
)
from src.intervention_policy import (
    ASSIGNED_GROUP_COLUMN,
    REASON_CODE_1_COLUMN,
    REASON_CODE_2_COLUMN,
    RECOMMENDED_ACTION_COLUMN,
)
from src.model_training import MODEL_V1_SCORE_COLUMN


POST_INTERVENTION_TARGET_COLUMN = "post_intervention_attrition_next_90d"
CONTACT_ATTEMPTED_COLUMN = "contact_attempted"
CONTACT_SUCCESS_COLUMN = "contact_success"
TREATMENT_COMPLETED_COLUMN = "treatment_completed"
PREVENTED_ATTRITION_COLUMN = "prevented_attrition_flag"
INTERVENTION_COST_COLUMN = "intervention_cost_index"
RETAINED_VALUE_COLUMN = "retained_value_index"
NET_VALUE_COLUMN = "net_value_index"


@dataclass(frozen=True)
class TreatmentLogResult:
    """Container for the synthetic treatment log and summary metadata."""

    treatment_log: pd.DataFrame
    snapshot_month: str
    rows: int
    treatment_rows: int
    control_rows: int


def _validate_intervention_list(intervention_list: pd.DataFrame) -> None:
    """Validate required intervention-list columns."""
    required_columns = {
        "policy_rank",
        CUSTOMER_ID_COLUMN,
        SNAPSHOT_MONTH_COLUMN,
        TARGET_COLUMN,
        MODEL_V1_SCORE_COLUMN,
        "relationship_value_index",
        "value_at_risk_index",
        "relationship_health_score",
        RECOMMENDED_ACTION_COLUMN,
        ASSIGNED_GROUP_COLUMN,
        REASON_CODE_1_COLUMN,
        REASON_CODE_2_COLUMN,
    }

    missing_columns = sorted(required_columns.difference(intervention_list.columns))

    if missing_columns:
        raise ValueError(f"Missing required intervention-list columns: {missing_columns}")


def _base_contact_success_probability(action: str) -> float:
    """Return synthetic contact-success probability by recommended action."""
    probabilities = {
        "digital_reactivation_nudge": 0.72,
        "relationship_check_in": 0.62,
        "service_recovery_case": 0.58,
        "financial_wellbeing_check": 0.55,
        "relationship_call": 0.50,
    }

    return probabilities.get(action, 0.55)


def _base_completion_probability(action: str) -> float:
    """Return synthetic treatment-completion probability by recommended action."""
    probabilities = {
        "digital_reactivation_nudge": 0.64,
        "relationship_check_in": 0.56,
        "service_recovery_case": 0.52,
        "financial_wellbeing_check": 0.48,
        "relationship_call": 0.46,
    }

    return probabilities.get(action, 0.50)


def _base_treatment_effect(action: str) -> float:
    """Return synthetic probability of preventing attrition after completion.

    This value applies only when the row is in treatment, the customer was
    contacted successfully, treatment was completed, and the original synthetic
    attrition label equals 1.
    """
    effects = {
        "digital_reactivation_nudge": 0.22,
        "relationship_check_in": 0.18,
        "service_recovery_case": 0.28,
        "financial_wellbeing_check": 0.24,
        "relationship_call": 0.20,
    }

    return effects.get(action, 0.18)


def _base_cost_index(action: str) -> float:
    """Return synthetic intervention cost index by recommended action."""
    costs = {
        "digital_reactivation_nudge": 1.5,
        "relationship_check_in": 3.0,
        "service_recovery_case": 5.0,
        "financial_wellbeing_check": 6.0,
        "relationship_call": 4.0,
    }

    return costs.get(action, 3.0)


def _contact_channel(action: str) -> str:
    """Map a recommended action to a synthetic contact channel."""
    channels = {
        "digital_reactivation_nudge": "in_app_message",
        "relationship_check_in": "email_or_phone",
        "service_recovery_case": "specialist_call",
        "financial_wellbeing_check": "advisory_call",
        "relationship_call": "relationship_manager_call",
    }

    return channels.get(action, "email_or_phone")


def generate_treatment_log(
    intervention_list: pd.DataFrame,
    random_seed: int = RANDOM_SEED,
) -> TreatmentLogResult:
    """Generate a synthetic treatment log from an intervention list.

    Args:
        intervention_list: Output from the intervention policy module.
        random_seed: Deterministic random seed for reproducibility.

    Returns:
        TreatmentLogResult containing the treatment log and metadata.
    """
    _validate_intervention_list(intervention_list)

    rng = np.random.default_rng(random_seed)

    treatment_log = intervention_list.copy().reset_index(drop=True)
    n_rows = len(treatment_log)

    treatment_log["treatment_log_id"] = [
        f"TLOG-{index:06d}" for index in range(1, n_rows + 1)
    ]

    treatment_log["contact_channel"] = treatment_log[RECOMMENDED_ACTION_COLUMN].map(
        _contact_channel
    )

    is_treatment = treatment_log[ASSIGNED_GROUP_COLUMN].eq("treatment")

    treatment_log[CONTACT_ATTEMPTED_COLUMN] = is_treatment.astype("int8")

    contact_probability = treatment_log[RECOMMENDED_ACTION_COLUMN].map(
        _base_contact_success_probability
    )

    contact_probability = (
        contact_probability
        + 0.10 * (1 - treatment_log[MODEL_V1_SCORE_COLUMN])
        - 0.05 * treatment_log["value_at_risk_index"].clip(0, 100) / 100
    ).clip(0.05, 0.95)

    contact_success = rng.binomial(
        n=1,
        p=contact_probability.to_numpy(),
        size=n_rows,
    )

    treatment_log[CONTACT_SUCCESS_COLUMN] = np.where(
        is_treatment,
        contact_success,
        0,
    ).astype("int8")

    completion_probability = treatment_log[RECOMMENDED_ACTION_COLUMN].map(
        _base_completion_probability
    )

    completion_probability = (
        completion_probability
        + 0.08 * treatment_log[CONTACT_SUCCESS_COLUMN]
        - 0.04 * treatment_log[MODEL_V1_SCORE_COLUMN]
    ).clip(0.05, 0.90)

    treatment_completed = rng.binomial(
        n=1,
        p=completion_probability.to_numpy(),
        size=n_rows,
    )

    treatment_log[TREATMENT_COMPLETED_COLUMN] = np.where(
        is_treatment & treatment_log[CONTACT_SUCCESS_COLUMN].eq(1),
        treatment_completed,
        0,
    ).astype("int8")

    base_effect = treatment_log[RECOMMENDED_ACTION_COLUMN].map(_base_treatment_effect)

    reason_alignment_boost = np.where(
        treatment_log[REASON_CODE_1_COLUMN].isin(
            ["service_friction", "low_satisfaction", "digital_inactivity"]
        ),
        0.05,
        0.00,
    )

    treatment_effect_probability = (
        base_effect
        + reason_alignment_boost
        + 0.05 * (treatment_log["relationship_value_index"].clip(0, 100) / 100)
    ).clip(0.00, 0.45)

    preventable_rows = (
        is_treatment
        & treatment_log[TREATMENT_COMPLETED_COLUMN].eq(1)
        & treatment_log[TARGET_COLUMN].eq(1)
    )

    prevented_attrition = rng.binomial(
        n=1,
        p=treatment_effect_probability.to_numpy(),
        size=n_rows,
    )

    treatment_log[PREVENTED_ATTRITION_COLUMN] = np.where(
        preventable_rows,
        prevented_attrition,
        0,
    ).astype("int8")

    treatment_log[POST_INTERVENTION_TARGET_COLUMN] = (
        treatment_log[TARGET_COLUMN] - treatment_log[PREVENTED_ATTRITION_COLUMN]
    ).clip(0, 1).astype("int8")

    attempted_cost = treatment_log[RECOMMENDED_ACTION_COLUMN].map(_base_cost_index)

    completion_cost_multiplier = np.where(
        treatment_log[TREATMENT_COMPLETED_COLUMN].eq(1),
        1.35,
        1.00,
    )

    treatment_log[INTERVENTION_COST_COLUMN] = np.where(
        is_treatment,
        attempted_cost * completion_cost_multiplier,
        0.0,
    ).round(2)

    treatment_log[RETAINED_VALUE_COLUMN] = (
        treatment_log[PREVENTED_ATTRITION_COLUMN]
        * treatment_log["relationship_value_index"]
    ).round(2)

    treatment_log[NET_VALUE_COLUMN] = (
        treatment_log[RETAINED_VALUE_COLUMN] - treatment_log[INTERVENTION_COST_COLUMN]
    ).round(2)

    output_columns = [
        "treatment_log_id",
        "policy_rank",
        CUSTOMER_ID_COLUMN,
        SNAPSHOT_MONTH_COLUMN,
        ASSIGNED_GROUP_COLUMN,
        RECOMMENDED_ACTION_COLUMN,
        "contact_channel",
        REASON_CODE_1_COLUMN,
        REASON_CODE_2_COLUMN,
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
    ]

    treatment_log = treatment_log[output_columns].copy()

    snapshot_month = str(treatment_log[SNAPSHOT_MONTH_COLUMN].iloc[0])

    return TreatmentLogResult(
        treatment_log=treatment_log,
        snapshot_month=snapshot_month,
        rows=n_rows,
        treatment_rows=int(is_treatment.sum()),
        control_rows=int((~is_treatment).sum()),
    )


def summarise_treatment_log(treatment_log: pd.DataFrame) -> dict[str, Any]:
    """Return concise diagnostics for the synthetic treatment log."""
    treatment_rows = treatment_log[ASSIGNED_GROUP_COLUMN].eq("treatment")
    control_rows = treatment_log[ASSIGNED_GROUP_COLUMN].eq("control")

    attempted_treatment_rows = treatment_rows & treatment_log[CONTACT_ATTEMPTED_COLUMN].eq(1)

    if attempted_treatment_rows.sum() > 0:
        contact_success_rate = float(
            treatment_log.loc[attempted_treatment_rows, CONTACT_SUCCESS_COLUMN].mean()
        )
    else:
        contact_success_rate = 0.0

    if treatment_rows.sum() > 0:
        treatment_completion_rate = float(
            treatment_log.loc[treatment_rows, TREATMENT_COMPLETED_COLUMN].mean()
        )
    else:
        treatment_completion_rate = 0.0

    return {
        "rows": int(len(treatment_log)),
        "treatment_rows": int(treatment_rows.sum()),
        "control_rows": int(control_rows.sum()),
        "pre_intervention_target_rate": float(treatment_log[TARGET_COLUMN].mean()),
        "post_intervention_target_rate": float(
            treatment_log[POST_INTERVENTION_TARGET_COLUMN].mean()
        ),
        "contact_success_rate": contact_success_rate,
        "treatment_completion_rate": treatment_completion_rate,
        "prevented_attrition_count": int(treatment_log[PREVENTED_ATTRITION_COLUMN].sum()),
        "total_intervention_cost_index": float(treatment_log[INTERVENTION_COST_COLUMN].sum()),
        "total_retained_value_index": float(treatment_log[RETAINED_VALUE_COLUMN].sum()),
        "total_net_value_index": float(treatment_log[NET_VALUE_COLUMN].sum()),
    }


def treatment_log_summary_as_text(summary: dict[str, Any]) -> str:
    """Format treatment-log diagnostics for terminal output."""
    return (
        "Treatment log summary\n"
        f"Rows: {summary['rows']:,}\n"
        f"Treatment rows: {summary['treatment_rows']:,}\n"
        f"Control rows: {summary['control_rows']:,}\n"
        f"Pre-intervention target rate: {summary['pre_intervention_target_rate']:.2%}\n"
        f"Post-intervention target rate: {summary['post_intervention_target_rate']:.2%}\n"
        f"Contact success rate: {summary['contact_success_rate']:.2%}\n"
        f"Treatment completion rate: {summary['treatment_completion_rate']:.2%}\n"
        f"Prevented attrition count: {summary['prevented_attrition_count']:,}\n"
        f"Total intervention cost index: {summary['total_intervention_cost_index']:,.2f}\n"
        f"Total retained value index: {summary['total_retained_value_index']:,.2f}\n"
        f"Total net value index: {summary['total_net_value_index']:,.2f}"
    )


def save_treatment_log(result: TreatmentLogResult) -> Path:
    """Save the synthetic treatment log to the outputs directory."""
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    output_path = OUTPUTS_DIR / TREATMENT_LOG_V1_FILE
    result.treatment_log.to_csv(output_path, index=False)

    return output_path

def save_treatment_log_outputs(result) -> dict:
    """Save treatment-log outputs to the outputs directory."""
    from src.config import OUTPUTS_DIR, TREATMENT_LOG_V1_FILE

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    treatment_log_path = OUTPUTS_DIR / TREATMENT_LOG_V1_FILE
    result.treatment_log.to_csv(treatment_log_path, index=False)

    return {
        "treatment_log": treatment_log_path,
    }