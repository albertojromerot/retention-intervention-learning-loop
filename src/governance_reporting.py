"""Governance and model-comparison reporting for the learning-loop pipeline.

This module creates two portfolio-ready outputs:

1. A model comparison table across the rules baseline, ML model v1 and
   treatment-aware ML model v2.
2. A governance checks table validating data integrity, treatment/control
   design, score ranges, clean-room assumptions and treatment-log consistency.

The comparison table is intentionally explicit that model v1 and model v2 have
different prediction objectives. Model v1 predicts pre-intervention attrition
risk across the full customer-month population. Model v2 predicts
post-intervention attrition among intervention candidates after treatment-log
feedback is available.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import (
    CONTROL_SHARE,
    CUSTOMER_ID_COLUMN,
    GOVERNANCE_CHECKS_FILE,
    MODEL_COMPARISON_FILE,
    OUTPUTS_DIR,
    SNAPSHOT_MONTH_COLUMN,
    TARGET_COLUMN,
    TREATMENT_SHARE,
)
from src.feature_engineering import SPLIT_COLUMN
from src.intervention_policy import ASSIGNED_GROUP_COLUMN
from src.model_training import MODEL_V1_SCORE_COLUMN
from src.model_v2_learning import (
    MODEL_V2_SCORE_COLUMN,
    TREATMENT_LEARNING_SPLIT_COLUMN,
)
from src.treatment_log import (
    CONTACT_ATTEMPTED_COLUMN,
    INTERVENTION_COST_COLUMN,
    NET_VALUE_COLUMN,
    POST_INTERVENTION_TARGET_COLUMN,
    RETAINED_VALUE_COLUMN,
)


COMMON_METRIC_COLUMNS = [
    "model_version",
    "model_family",
    "prediction_objective",
    "population",
    "direct_comparison_group",
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
    "precision_at_500",
    "lift_at_500",
    "precision_at_1000",
    "lift_at_1000",
]


@dataclass(frozen=True)
class GovernanceReportingResult:
    """Container for governance and comparison reporting outputs."""

    model_comparison: pd.DataFrame
    governance_checks: pd.DataFrame


def _normalise_metrics(
    metrics: pd.DataFrame,
    model_family: str,
    prediction_objective: str,
    population: str,
    direct_comparison_group: str,
) -> list[dict[str, Any]]:
    """Normalise metric rows into one comparison schema.

    Returning records instead of partially all-NA dataframes avoids pandas
    concatenation warnings and keeps the comparison schema explicit.
    """
    normalised_rows: list[dict[str, Any]] = []

    for _, metric_row in metrics.iterrows():
        row: dict[str, Any] = {column: None for column in COMMON_METRIC_COLUMNS}

        for column in metrics.columns:
            if column in row:
                value = metric_row[column]
                row[column] = None if pd.isna(value) else value

        row["model_family"] = model_family
        row["prediction_objective"] = prediction_objective
        row["population"] = population
        row["direct_comparison_group"] = direct_comparison_group

        normalised_rows.append(row)

    return normalised_rows


def build_model_comparison(
    rules_metrics: pd.DataFrame,
    model_v1_metrics: pd.DataFrame,
    model_v2_metrics: pd.DataFrame,
) -> pd.DataFrame:
    """Build a portfolio-ready model comparison table.

    The rules baseline and ML model v1 are directly comparable because both
    rank the same full customer-month population before intervention. Model v2
    is separated because it learns from the treatment log and predicts a
    post-intervention outcome on the intervention-candidate population.
    """
    comparison_rows: list[dict[str, Any]] = []

    comparison_rows.extend(
        _normalise_metrics(
            metrics=rules_metrics,
            model_family="rules_baseline",
            prediction_objective="pre_intervention_attrition_risk",
            population="full_customer_month_population",
            direct_comparison_group="pre_intervention_ranking",
        )
    )

    comparison_rows.extend(
        _normalise_metrics(
            metrics=model_v1_metrics,
            model_family="ml_model_v1",
            prediction_objective="pre_intervention_attrition_risk",
            population="full_customer_month_population",
            direct_comparison_group="pre_intervention_ranking",
        )
    )

    comparison_rows.extend(
        _normalise_metrics(
            metrics=model_v2_metrics,
            model_family="treatment_aware_ml_model_v2",
            prediction_objective="post_intervention_attrition_risk",
            population="intervention_candidate_population",
            direct_comparison_group="post_intervention_learning_loop",
        )
    )

    comparison = pd.DataFrame.from_records(
        comparison_rows,
        columns=COMMON_METRIC_COLUMNS,
    )

    sort_columns = [
        "direct_comparison_group",
        "split",
        "model_family",
    ]

    return comparison.sort_values(sort_columns).reset_index(drop=True)


def model_comparison_as_text(model_comparison: pd.DataFrame) -> str:
    """Format model-comparison output for terminal display."""
    if model_comparison.empty:
        return "Model comparison: no rows to report."

    display_columns = [
        "model_family",
        "model_version",
        "prediction_objective",
        "split",
        "rows",
        "target_rate",
        "roc_auc",
        "pr_auc",
        "brier_score",
        "precision_at_100",
        "lift_at_100",
        "direct_comparison_group",
    ]

    available_columns = [
        column for column in display_columns if column in model_comparison.columns
    ]

    display = model_comparison[available_columns].copy()
    numeric_columns = display.select_dtypes(include=["float", "float64"]).columns
    display[numeric_columns] = display[numeric_columns].round(4)

    return "Model comparison\n" + display.to_string(index=False)


def _make_check(
    check_name: str,
    passed: bool,
    severity: str,
    details: str,
) -> dict[str, str]:
    """Create a governance-check row."""
    return {
        "check_name": check_name,
        "status": "pass" if passed else "fail",
        "severity": severity,
        "details": details,
    }


def _missing_values_count(frame: pd.DataFrame) -> int:
    """Return total missing values in a dataframe."""
    return int(frame.isna().sum().sum())


def _duplicate_customer_month_keys(frame: pd.DataFrame) -> int:
    """Return duplicated customer-month key count."""
    return int(
        frame.duplicated(
            subset=[CUSTOMER_ID_COLUMN, SNAPSHOT_MONTH_COLUMN]
        ).sum()
    )


def build_governance_checks(
    synthetic_panel: pd.DataFrame,
    feature_table: pd.DataFrame,
    rules_scored_table: pd.DataFrame,
    model_v1_scored_table: pd.DataFrame,
    intervention_list: pd.DataFrame,
    treatment_log: pd.DataFrame,
    model_v2_learning_table: pd.DataFrame,
    model_v2_scored_table: pd.DataFrame,
) -> pd.DataFrame:
    """Build governance checks for the full learning-loop pipeline."""
    checks: list[dict[str, str]] = []

    synthetic_missing = _missing_values_count(synthetic_panel)
    feature_missing = _missing_values_count(feature_table)

    checks.append(
        _make_check(
            "synthetic_panel_has_rows",
            len(synthetic_panel) > 0,
            "critical",
            f"Synthetic panel rows: {len(synthetic_panel):,}",
        )
    )

    checks.append(
        _make_check(
            "synthetic_customer_month_key_unique",
            _duplicate_customer_month_keys(synthetic_panel) == 0,
            "critical",
            f"Duplicated synthetic customer-month keys: "
            f"{_duplicate_customer_month_keys(synthetic_panel):,}",
        )
    )

    checks.append(
        _make_check(
            "synthetic_panel_has_no_missing_values",
            synthetic_missing == 0,
            "critical",
            f"Missing values in synthetic panel: {synthetic_missing:,}",
        )
    )

    checks.append(
        _make_check(
            "feature_table_has_no_missing_values",
            feature_missing == 0,
            "critical",
            f"Missing values in feature table: {feature_missing:,}",
        )
    )

    expected_feature_splits = {"train", "validation", "test"}
    observed_feature_splits = set(feature_table[SPLIT_COLUMN].unique())

    checks.append(
        _make_check(
            "feature_table_has_train_validation_test_splits",
            expected_feature_splits.issubset(observed_feature_splits),
            "critical",
            f"Observed feature-table splits: {sorted(observed_feature_splits)}",
        )
    )

    checks.append(
        _make_check(
            "feature_customer_month_key_unique",
            _duplicate_customer_month_keys(feature_table) == 0,
            "critical",
            f"Duplicated feature-table customer-month keys: "
            f"{_duplicate_customer_month_keys(feature_table):,}",
        )
    )

    checks.append(
        _make_check(
            "rules_scored_rows_match_feature_table",
            len(rules_scored_table) == len(feature_table),
            "critical",
            f"Rules scored rows: {len(rules_scored_table):,}; "
            f"feature rows: {len(feature_table):,}",
        )
    )

    checks.append(
        _make_check(
            "model_v1_scores_are_probabilities",
            model_v1_scored_table[MODEL_V1_SCORE_COLUMN].between(0, 1).all(),
            "critical",
            "All model v1 scores must be between 0 and 1.",
        )
    )

    checks.append(
        _make_check(
            "intervention_list_uses_single_snapshot",
            intervention_list[SNAPSHOT_MONTH_COLUMN].nunique() == 1,
            "critical",
            f"Intervention snapshot count: "
            f"{intervention_list[SNAPSHOT_MONTH_COLUMN].nunique()}",
        )
    )

    if "intervention_capacity" in intervention_list.columns:
        expected_capacity = int(intervention_list["intervention_capacity"].iloc[0])
        capacity_passed = len(intervention_list) == expected_capacity
        capacity_details = (
            f"Rows: {len(intervention_list):,}; "
            f"configured capacity: {expected_capacity:,}"
        )
    else:
        capacity_passed = False
        capacity_details = "Missing intervention_capacity column."

    checks.append(
        _make_check(
            "intervention_capacity_is_respected",
            capacity_passed,
            "critical",
            capacity_details,
        )
    )

    group_values = set(intervention_list[ASSIGNED_GROUP_COLUMN].unique())

    checks.append(
        _make_check(
            "intervention_has_treatment_and_control_groups",
            group_values == {"treatment", "control"},
            "critical",
            f"Observed assigned groups: {sorted(group_values)}",
        )
    )

    treatment_share_observed = float(
        intervention_list[ASSIGNED_GROUP_COLUMN].eq("treatment").mean()
    )
    control_share_observed = float(
        intervention_list[ASSIGNED_GROUP_COLUMN].eq("control").mean()
    )

    checks.append(
        _make_check(
            "treatment_control_share_matches_policy",
            abs(treatment_share_observed - TREATMENT_SHARE) <= 0.01
            and abs(control_share_observed - CONTROL_SHARE) <= 0.01,
            "critical",
            f"Observed treatment share: {treatment_share_observed:.2%}; "
            f"observed control share: {control_share_observed:.2%}",
        )
    )

    control_rows = treatment_log[ASSIGNED_GROUP_COLUMN].eq("control")

    checks.append(
        _make_check(
            "control_group_has_no_contact_attempts",
            treatment_log.loc[control_rows, CONTACT_ATTEMPTED_COLUMN].sum() == 0,
            "critical",
            "Control rows should remain a clean holdout.",
        )
    )

    checks.append(
        _make_check(
            "post_intervention_target_not_above_original_target",
            (
                treatment_log[POST_INTERVENTION_TARGET_COLUMN]
                <= treatment_log[TARGET_COLUMN]
            ).all(),
            "critical",
            "Post-intervention target should not exceed original target.",
        )
    )

    checks.append(
        _make_check(
            "intervention_cost_is_non_negative",
            treatment_log[INTERVENTION_COST_COLUMN].ge(0).all(),
            "critical",
            "All intervention cost values should be non-negative.",
        )
    )

    expected_net_value = (
        treatment_log[RETAINED_VALUE_COLUMN] - treatment_log[INTERVENTION_COST_COLUMN]
    ).round(2)

    checks.append(
        _make_check(
            "net_value_arithmetic_is_correct",
            treatment_log[NET_VALUE_COLUMN].round(2).equals(expected_net_value),
            "critical",
            "Net value should equal retained value minus intervention cost.",
        )
    )

    expected_v2_splits = {"train", "validation", "test"}
    observed_v2_splits = set(
        model_v2_learning_table[TREATMENT_LEARNING_SPLIT_COLUMN].unique()
    )

    checks.append(
        _make_check(
            "model_v2_learning_table_has_required_splits",
            expected_v2_splits == observed_v2_splits,
            "critical",
            f"Observed model v2 learning splits: {sorted(observed_v2_splits)}",
        )
    )

    checks.append(
        _make_check(
            "model_v2_scores_are_probabilities",
            model_v2_scored_table[MODEL_V2_SCORE_COLUMN].between(0, 1).all(),
            "critical",
            "All model v2 scores must be between 0 and 1.",
        )
    )

    disallowed_column_tokens = [
        "real_customer",
        "real_member",
        "employer_name",
        "national_id",
        "phone",
        "email",
        "address",
    ]

    all_column_names = " ".join(
        [
            " ".join(map(str, synthetic_panel.columns)),
            " ".join(map(str, feature_table.columns)),
            " ".join(map(str, intervention_list.columns)),
            " ".join(map(str, treatment_log.columns)),
        ]
    ).lower()

    found_disallowed_tokens = [
        token for token in disallowed_column_tokens if token in all_column_names
    ]

    checks.append(
        _make_check(
            "clean_room_column_names_check",
            len(found_disallowed_tokens) == 0,
            "warning",
            f"Disallowed column-name tokens found: {found_disallowed_tokens}",
        )
    )

    governance_checks = pd.DataFrame(checks)
    governance_checks.insert(
        0,
        "check_id",
        [f"GOV-{index:03d}" for index in range(1, len(governance_checks) + 1)],
    )

    return governance_checks


def governance_checks_as_text(governance_checks: pd.DataFrame) -> str:
    """Format governance checks for terminal output."""
    if governance_checks.empty:
        return "Governance checks: no rows to report."

    summary = (
        governance_checks.groupby(["severity", "status"])
        .size()
        .reset_index(name="checks")
        .sort_values(["severity", "status"])
    )

    failed = governance_checks.loc[governance_checks["status"].eq("fail")]

    text = "Governance checks summary\n"
    text += summary.to_string(index=False)

    if not failed.empty:
        text += "\n\nFailed checks\n"
        text += failed[["check_id", "check_name", "severity", "details"]].to_string(
            index=False
        )
    else:
        text += "\n\nAll governance checks passed."

    return text


def run_governance_reporting(
    rules_metrics: pd.DataFrame,
    model_v1_metrics: pd.DataFrame,
    model_v2_metrics: pd.DataFrame,
    synthetic_panel: pd.DataFrame,
    feature_table: pd.DataFrame,
    rules_scored_table: pd.DataFrame,
    model_v1_scored_table: pd.DataFrame,
    intervention_list: pd.DataFrame,
    treatment_log: pd.DataFrame,
    model_v2_learning_table: pd.DataFrame,
    model_v2_scored_table: pd.DataFrame,
) -> GovernanceReportingResult:
    """Build model comparison and governance checks."""
    model_comparison = build_model_comparison(
        rules_metrics=rules_metrics,
        model_v1_metrics=model_v1_metrics,
        model_v2_metrics=model_v2_metrics,
    )

    governance_checks = build_governance_checks(
        synthetic_panel=synthetic_panel,
        feature_table=feature_table,
        rules_scored_table=rules_scored_table,
        model_v1_scored_table=model_v1_scored_table,
        intervention_list=intervention_list,
        treatment_log=treatment_log,
        model_v2_learning_table=model_v2_learning_table,
        model_v2_scored_table=model_v2_scored_table,
    )

    return GovernanceReportingResult(
        model_comparison=model_comparison,
        governance_checks=governance_checks,
    )


def save_governance_reporting_outputs(
    result: GovernanceReportingResult,
) -> dict[str, Path]:
    """Save governance reporting outputs to the outputs directory."""
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    model_comparison_path = OUTPUTS_DIR / MODEL_COMPARISON_FILE
    governance_checks_path = OUTPUTS_DIR / GOVERNANCE_CHECKS_FILE

    result.model_comparison.to_csv(model_comparison_path, index=False)
    result.governance_checks.to_csv(governance_checks_path, index=False)

    return {
        "model_comparison": model_comparison_path,
        "governance_checks": governance_checks_path,
    }