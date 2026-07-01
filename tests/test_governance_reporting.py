"""Tests for governance reporting and model-comparison outputs."""

import pandas as pd
import pytest

from src.config import (
    CUSTOMER_ID_COLUMN,
    GOVERNANCE_CHECKS_FILE,
    MODEL_COMPARISON_FILE,
    MODEL_V1_VERSION,
    MODEL_V2_VERSION,
    OUTPUTS_DIR,
    RULES_BASELINE_VERSION,
    SNAPSHOT_MONTH_COLUMN,
    TARGET_COLUMN,
    get_profile,
)
from src.feature_engineering import build_feature_table
from src.governance_reporting import (
    COMMON_METRIC_COLUMNS,
    GovernanceReportingResult,
    build_governance_checks,
    build_model_comparison,
    governance_checks_as_text,
    model_comparison_as_text,
    run_governance_reporting,
    save_governance_reporting_outputs,
)
from src.intervention_policy import create_intervention_list
from src.model_training import MODEL_V1_SCORE_COLUMN, run_model_v1
from src.model_v2_learning import MODEL_V2_SCORE_COLUMN, run_model_v2
from src.rules_baseline import run_rules_baseline
from src.synthetic_data import generate_customer_month_panel
from src.treatment_log import (
    CONTACT_ATTEMPTED_COLUMN,
    INTERVENTION_COST_COLUMN,
    NET_VALUE_COLUMN,
    POST_INTERVENTION_TARGET_COLUMN,
    RETAINED_VALUE_COLUMN,
    generate_treatment_log,
)


@pytest.fixture(scope="module")
def compact_governance_inputs():
    """Generate compact end-to-end inputs for governance tests."""
    panel = generate_customer_month_panel(profile=get_profile("compact"))
    feature_result = build_feature_table(panel)

    rules_result = run_rules_baseline(feature_result.feature_table)
    model_v1_result = run_model_v1(feature_result.feature_table)

    intervention_result = create_intervention_list(
        scored_table=model_v1_result.scored_table,
        feature_table=feature_result.feature_table,
        profile_name="compact",
    )

    treatment_result = generate_treatment_log(
        intervention_list=intervention_result.intervention_list
    )

    model_v2_result = run_model_v2(
        feature_table=feature_result.feature_table,
        treatment_log=treatment_result.treatment_log,
    )

    return {
        "panel": panel,
        "feature_table": feature_result.feature_table,
        "rules_result": rules_result,
        "model_v1_result": model_v1_result,
        "intervention_result": intervention_result,
        "treatment_result": treatment_result,
        "model_v2_result": model_v2_result,
    }


@pytest.fixture(scope="module")
def compact_governance_result(compact_governance_inputs):
    """Build compact governance result once for this test module."""
    inputs = compact_governance_inputs

    return run_governance_reporting(
        rules_metrics=inputs["rules_result"].metrics,
        model_v1_metrics=inputs["model_v1_result"].metrics,
        model_v2_metrics=inputs["model_v2_result"].metrics,
        synthetic_panel=inputs["panel"],
        feature_table=inputs["feature_table"],
        rules_scored_table=inputs["rules_result"].scored_table,
        model_v1_scored_table=inputs["model_v1_result"].scored_table,
        intervention_list=inputs["intervention_result"].intervention_list,
        treatment_log=inputs["treatment_result"].treatment_log,
        model_v2_learning_table=inputs["model_v2_result"].learning_table,
        model_v2_scored_table=inputs["model_v2_result"].scored_table,
    )


def test_model_comparison_has_expected_rows(compact_governance_result) -> None:
    """Model comparison should include rules, model v1 and model v2 rows."""
    comparison = compact_governance_result.model_comparison

    assert len(comparison) == 6


def test_model_comparison_has_common_schema(compact_governance_result) -> None:
    """Model comparison should expose the common comparison schema."""
    comparison = compact_governance_result.model_comparison

    assert list(comparison.columns) == COMMON_METRIC_COLUMNS


def test_model_comparison_contains_expected_model_versions(
    compact_governance_result,
) -> None:
    """Comparison should include all expected model and baseline versions."""
    comparison = compact_governance_result.model_comparison

    assert {
        RULES_BASELINE_VERSION,
        MODEL_V1_VERSION,
        MODEL_V2_VERSION,
    }.issubset(set(comparison["model_version"]))


def test_model_comparison_separates_prediction_objectives(
    compact_governance_result,
) -> None:
    """Model v2 should be clearly separated from pre-intervention models."""
    comparison = compact_governance_result.model_comparison

    objectives = comparison.groupby("model_family")["prediction_objective"].first()

    assert objectives["rules_baseline"] == "pre_intervention_attrition_risk"
    assert objectives["ml_model_v1"] == "pre_intervention_attrition_risk"
    assert (
        objectives["treatment_aware_ml_model_v2"]
        == "post_intervention_attrition_risk"
    )


def test_model_comparison_has_direct_comparison_groups(
    compact_governance_result,
) -> None:
    """Model comparison should identify fair comparison groups."""
    comparison = compact_governance_result.model_comparison

    assert set(comparison["direct_comparison_group"]) == {
        "pre_intervention_ranking",
        "post_intervention_learning_loop",
    }


def test_model_comparison_metrics_are_in_valid_ranges(
    compact_governance_result,
) -> None:
    """Core model-comparison metrics should be valid probabilities/scores."""
    comparison = compact_governance_result.model_comparison

    for column in ["target_rate", "roc_auc", "pr_auc", "brier_score"]:
        assert comparison[column].notna().all(), column
        assert comparison[column].between(0.0, 1.0).all(), column


def test_model_comparison_text_is_generated(compact_governance_result) -> None:
    """Model comparison text should be printable."""
    text = model_comparison_as_text(compact_governance_result.model_comparison)

    assert "Model comparison" in text
    assert "ml_model_v1" in text
    assert "treatment_aware_ml_model_v2" in text


def test_governance_checks_are_generated(compact_governance_result) -> None:
    """Governance checks should be generated and non-empty."""
    checks = compact_governance_result.governance_checks

    assert not checks.empty
    assert {"check_id", "check_name", "status", "severity", "details"}.issubset(
        set(checks.columns)
    )


def test_governance_checks_all_pass_for_clean_pipeline(
    compact_governance_result,
) -> None:
    """The clean synthetic pipeline should pass all governance checks."""
    checks = compact_governance_result.governance_checks

    assert set(checks["status"]) == {"pass"}


def test_governance_checks_have_expected_severities(
    compact_governance_result,
) -> None:
    """Governance checks should include critical checks and one clean-room warning."""
    checks = compact_governance_result.governance_checks

    assert "critical" in set(checks["severity"])
    assert "warning" in set(checks["severity"])


def test_governance_checks_text_is_generated(compact_governance_result) -> None:
    """Governance checks text should be printable."""
    text = governance_checks_as_text(compact_governance_result.governance_checks)

    assert "Governance checks summary" in text
    assert "All governance checks passed" in text


def test_governance_reporting_result_type(compact_governance_result) -> None:
    """Governance reporting should return the expected result container."""
    assert isinstance(compact_governance_result, GovernanceReportingResult)


def test_save_governance_reporting_outputs_creates_files(
    compact_governance_result,
) -> None:
    """Governance reporting outputs should be saved to configured CSV files."""
    output_paths = save_governance_reporting_outputs(compact_governance_result)

    assert output_paths["model_comparison"] == OUTPUTS_DIR / MODEL_COMPARISON_FILE
    assert output_paths["governance_checks"] == OUTPUTS_DIR / GOVERNANCE_CHECKS_FILE

    assert output_paths["model_comparison"].exists()
    assert output_paths["governance_checks"].exists()


def test_governance_detects_duplicate_synthetic_customer_month_keys(
    compact_governance_inputs,
) -> None:
    """Governance checks should fail if synthetic panel keys are duplicated."""
    inputs = compact_governance_inputs

    broken_panel = pd.concat(
        [inputs["panel"], inputs["panel"].head(1)],
        ignore_index=True,
    )

    checks = build_governance_checks(
        synthetic_panel=broken_panel,
        feature_table=inputs["feature_table"],
        rules_scored_table=inputs["rules_result"].scored_table,
        model_v1_scored_table=inputs["model_v1_result"].scored_table,
        intervention_list=inputs["intervention_result"].intervention_list,
        treatment_log=inputs["treatment_result"].treatment_log,
        model_v2_learning_table=inputs["model_v2_result"].learning_table,
        model_v2_scored_table=inputs["model_v2_result"].scored_table,
    )

    failed_check = checks.loc[
        checks["check_name"].eq("synthetic_customer_month_key_unique")
    ].iloc[0]

    assert failed_check["status"] == "fail"


def test_governance_detects_invalid_model_v1_scores(
    compact_governance_inputs,
) -> None:
    """Governance checks should fail if model v1 scores are outside [0, 1]."""
    inputs = compact_governance_inputs

    broken_scores = inputs["model_v1_result"].scored_table.copy()
    broken_scores.loc[broken_scores.index[0], MODEL_V1_SCORE_COLUMN] = 1.5

    checks = build_governance_checks(
        synthetic_panel=inputs["panel"],
        feature_table=inputs["feature_table"],
        rules_scored_table=inputs["rules_result"].scored_table,
        model_v1_scored_table=broken_scores,
        intervention_list=inputs["intervention_result"].intervention_list,
        treatment_log=inputs["treatment_result"].treatment_log,
        model_v2_learning_table=inputs["model_v2_result"].learning_table,
        model_v2_scored_table=inputs["model_v2_result"].scored_table,
    )

    failed_check = checks.loc[
        checks["check_name"].eq("model_v1_scores_are_probabilities")
    ].iloc[0]

    assert failed_check["status"] == "fail"


def test_governance_detects_control_group_contact_attempts(
    compact_governance_inputs,
) -> None:
    """Governance checks should fail if control rows receive contact attempts."""
    inputs = compact_governance_inputs

    broken_treatment_log = inputs["treatment_result"].treatment_log.copy()
    control_index = broken_treatment_log.loc[
        broken_treatment_log["assigned_group"].eq("control")
    ].index[0]
    broken_treatment_log.loc[control_index, CONTACT_ATTEMPTED_COLUMN] = 1

    checks = build_governance_checks(
        synthetic_panel=inputs["panel"],
        feature_table=inputs["feature_table"],
        rules_scored_table=inputs["rules_result"].scored_table,
        model_v1_scored_table=inputs["model_v1_result"].scored_table,
        intervention_list=inputs["intervention_result"].intervention_list,
        treatment_log=broken_treatment_log,
        model_v2_learning_table=inputs["model_v2_result"].learning_table,
        model_v2_scored_table=inputs["model_v2_result"].scored_table,
    )

    failed_check = checks.loc[
        checks["check_name"].eq("control_group_has_no_contact_attempts")
    ].iloc[0]

    assert failed_check["status"] == "fail"


def test_governance_detects_invalid_post_intervention_target(
    compact_governance_inputs,
) -> None:
    """Governance checks should fail if post-intervention target exceeds original target."""
    inputs = compact_governance_inputs

    broken_treatment_log = inputs["treatment_result"].treatment_log.copy()
    zero_target_index = broken_treatment_log.loc[
        broken_treatment_log[TARGET_COLUMN].eq(0)
    ].index[0]
    broken_treatment_log.loc[
        zero_target_index,
        POST_INTERVENTION_TARGET_COLUMN,
    ] = 1

    checks = build_governance_checks(
        synthetic_panel=inputs["panel"],
        feature_table=inputs["feature_table"],
        rules_scored_table=inputs["rules_result"].scored_table,
        model_v1_scored_table=inputs["model_v1_result"].scored_table,
        intervention_list=inputs["intervention_result"].intervention_list,
        treatment_log=broken_treatment_log,
        model_v2_learning_table=inputs["model_v2_result"].learning_table,
        model_v2_scored_table=inputs["model_v2_result"].scored_table,
    )

    failed_check = checks.loc[
        checks["check_name"].eq("post_intervention_target_not_above_original_target")
    ].iloc[0]

    assert failed_check["status"] == "fail"


def test_governance_detects_invalid_net_value_arithmetic(
    compact_governance_inputs,
) -> None:
    """Governance checks should fail if net value arithmetic is broken."""
    inputs = compact_governance_inputs

    broken_treatment_log = inputs["treatment_result"].treatment_log.copy()
    broken_treatment_log.loc[broken_treatment_log.index[0], NET_VALUE_COLUMN] = 999999

    checks = build_governance_checks(
        synthetic_panel=inputs["panel"],
        feature_table=inputs["feature_table"],
        rules_scored_table=inputs["rules_result"].scored_table,
        model_v1_scored_table=inputs["model_v1_result"].scored_table,
        intervention_list=inputs["intervention_result"].intervention_list,
        treatment_log=broken_treatment_log,
        model_v2_learning_table=inputs["model_v2_result"].learning_table,
        model_v2_scored_table=inputs["model_v2_result"].scored_table,
    )

    failed_check = checks.loc[
        checks["check_name"].eq("net_value_arithmetic_is_correct")
    ].iloc[0]

    assert failed_check["status"] == "fail"


def test_governance_detects_invalid_model_v2_scores(
    compact_governance_inputs,
) -> None:
    """Governance checks should fail if model v2 scores are outside [0, 1]."""
    inputs = compact_governance_inputs

    broken_scores = inputs["model_v2_result"].scored_table.copy()
    broken_scores.loc[broken_scores.index[0], MODEL_V2_SCORE_COLUMN] = -0.1

    checks = build_governance_checks(
        synthetic_panel=inputs["panel"],
        feature_table=inputs["feature_table"],
        rules_scored_table=inputs["rules_result"].scored_table,
        model_v1_scored_table=inputs["model_v1_result"].scored_table,
        intervention_list=inputs["intervention_result"].intervention_list,
        treatment_log=inputs["treatment_result"].treatment_log,
        model_v2_learning_table=inputs["model_v2_result"].learning_table,
        model_v2_scored_table=broken_scores,
    )

    failed_check = checks.loc[
        checks["check_name"].eq("model_v2_scores_are_probabilities")
    ].iloc[0]

    assert failed_check["status"] == "fail"


def test_build_model_comparison_accepts_metrics_directly(
    compact_governance_inputs,
) -> None:
    """Model comparison builder should work from direct metric inputs."""
    inputs = compact_governance_inputs

    comparison = build_model_comparison(
        rules_metrics=inputs["rules_result"].metrics,
        model_v1_metrics=inputs["model_v1_result"].metrics,
        model_v2_metrics=inputs["model_v2_result"].metrics,
    )

    assert len(comparison) == 6
    assert list(comparison.columns) == COMMON_METRIC_COLUMNS