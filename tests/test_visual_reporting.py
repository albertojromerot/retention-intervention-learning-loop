"""Tests for portfolio-ready visual reporting outputs."""

from pathlib import Path

import pandas as pd
import pytest

from src.visual_reporting import (
    INTERVENTION_ACTION_MIX_PLOT,
    MODEL_COMPARISON_AUC_PLOT,
    MODEL_COMPARISON_PR_AUC_PLOT,
    MODEL_V1_SCORE_COLUMN,
    MODEL_V2_SCORE_COLUMN,
    RISK_SCORE_DISTRIBUTION_V1_PLOT,
    RISK_SCORE_DISTRIBUTION_V2_PLOT,
    TREATMENT_OUTCOMES_PLOT,
    VisualReportingResult,
    plot_intervention_action_mix,
    plot_model_comparison_metric,
    plot_risk_score_distribution,
    plot_treatment_outcomes,
    run_visual_reporting,
)


@pytest.fixture()
def sample_model_comparison() -> pd.DataFrame:
    """Create a small model-comparison sample for visual tests."""
    return pd.DataFrame(
        {
            "model_family": [
                "rules_baseline",
                "ml_model_v1",
                "treatment_aware_ml_model_v2",
            ],
            "model_version": [
                "rules_baseline_v1",
                "ml_model_v1",
                "treatment_aware_ml_model_v2",
            ],
            "split": ["test", "test", "test"],
            "prediction_objective": [
                "pre_intervention_attrition_risk",
                "pre_intervention_attrition_risk",
                "post_intervention_attrition_risk",
            ],
            "roc_auc": [0.64, 0.68, 0.47],
            "pr_auc": [0.12, 0.15, 0.22],
        }
    )


@pytest.fixture()
def sample_intervention_list() -> pd.DataFrame:
    """Create a small intervention-list sample for visual tests."""
    return pd.DataFrame(
        {
            "customer_id": [1, 2, 3, 4, 5],
            "recommended_action": [
                "service_recovery_case",
                "service_recovery_case",
                "digital_reactivation_nudge",
                "relationship_check_in",
                "financial_wellbeing_check",
            ],
        }
    )


@pytest.fixture()
def sample_treatment_log() -> pd.DataFrame:
    """Create a small treatment-log sample for visual tests."""
    return pd.DataFrame(
        {
            "assigned_group": [
                "treatment",
                "treatment",
                "treatment",
                "control",
                "control",
            ],
            "contact_attempted": [1, 1, 1, 0, 0],
            "contact_success": [1, 1, 0, 0, 0],
            "treatment_completed": [1, 0, 0, 0, 0],
            "prevented_attrition_flag": [1, 0, 0, 0, 0],
        }
    )


@pytest.fixture()
def sample_scored_customers_v1() -> pd.DataFrame:
    """Create a small model v1 scoring sample for visual tests."""
    return pd.DataFrame(
        {
            "customer_id": [1, 2, 3, 4, 5],
            MODEL_V1_SCORE_COLUMN: [0.10, 0.25, 0.40, 0.55, 0.80],
        }
    )


@pytest.fixture()
def sample_scored_customers_v2() -> pd.DataFrame:
    """Create a small model v2 scoring sample for visual tests."""
    return pd.DataFrame(
        {
            "customer_id": [1, 2, 3, 4, 5],
            MODEL_V2_SCORE_COLUMN: [0.08, 0.20, 0.35, 0.50, 0.70],
        }
    )


def _assert_png_created(path: Path) -> None:
    """Assert that a PNG plot was created and is non-empty."""
    assert path.exists()
    assert path.suffix == ".png"
    assert path.stat().st_size > 0


def test_plot_model_comparison_auc_creates_png(
    sample_model_comparison,
    tmp_path,
) -> None:
    """AUC comparison plot should be generated as a PNG file."""
    output_path = tmp_path / MODEL_COMPARISON_AUC_PLOT

    result_path = plot_model_comparison_metric(
        model_comparison=sample_model_comparison,
        metric_column="roc_auc",
        output_path=output_path,
    )

    assert result_path == output_path
    _assert_png_created(result_path)


def test_plot_model_comparison_pr_auc_creates_png(
    sample_model_comparison,
    tmp_path,
) -> None:
    """PR-AUC comparison plot should be generated as a PNG file."""
    output_path = tmp_path / MODEL_COMPARISON_PR_AUC_PLOT

    result_path = plot_model_comparison_metric(
        model_comparison=sample_model_comparison,
        metric_column="pr_auc",
        output_path=output_path,
    )

    assert result_path == output_path
    _assert_png_created(result_path)


def test_plot_intervention_action_mix_creates_png(
    sample_intervention_list,
    tmp_path,
) -> None:
    """Intervention action mix plot should be generated as a PNG file."""
    output_path = tmp_path / INTERVENTION_ACTION_MIX_PLOT

    result_path = plot_intervention_action_mix(
        intervention_list=sample_intervention_list,
        output_path=output_path,
    )

    assert result_path == output_path
    _assert_png_created(result_path)


def test_plot_treatment_outcomes_creates_png(
    sample_treatment_log,
    tmp_path,
) -> None:
    """Treatment outcomes plot should be generated as a PNG file."""
    output_path = tmp_path / TREATMENT_OUTCOMES_PLOT

    result_path = plot_treatment_outcomes(
        treatment_log=sample_treatment_log,
        output_path=output_path,
    )

    assert result_path == output_path
    _assert_png_created(result_path)


def test_plot_risk_score_distribution_v1_creates_png(
    sample_scored_customers_v1,
    tmp_path,
) -> None:
    """Model v1 score distribution plot should be generated as a PNG file."""
    output_path = tmp_path / RISK_SCORE_DISTRIBUTION_V1_PLOT

    result_path = plot_risk_score_distribution(
        scored_table=sample_scored_customers_v1,
        score_column=MODEL_V1_SCORE_COLUMN,
        model_label="ML model v1",
        output_path=output_path,
    )

    assert result_path == output_path
    _assert_png_created(result_path)


def test_plot_risk_score_distribution_v2_creates_png(
    sample_scored_customers_v2,
    tmp_path,
) -> None:
    """Model v2 score distribution plot should be generated as a PNG file."""
    output_path = tmp_path / RISK_SCORE_DISTRIBUTION_V2_PLOT

    result_path = plot_risk_score_distribution(
        scored_table=sample_scored_customers_v2,
        score_column=MODEL_V2_SCORE_COLUMN,
        model_label="Treatment-aware ML model v2",
        output_path=output_path,
    )

    assert result_path == output_path
    _assert_png_created(result_path)


def test_run_visual_reporting_returns_expected_result_type(
    sample_model_comparison,
    sample_intervention_list,
    sample_treatment_log,
    sample_scored_customers_v1,
    sample_scored_customers_v2,
) -> None:
    """Full visual reporting should return the expected result container."""
    result = run_visual_reporting(
        model_comparison=sample_model_comparison,
        intervention_list=sample_intervention_list,
        treatment_log=sample_treatment_log,
        scored_customers_v1=sample_scored_customers_v1,
        scored_customers_v2=sample_scored_customers_v2,
    )

    assert isinstance(result, VisualReportingResult)


def test_run_visual_reporting_creates_all_expected_plots(
    sample_model_comparison,
    sample_intervention_list,
    sample_treatment_log,
    sample_scored_customers_v1,
    sample_scored_customers_v2,
) -> None:
    """Full visual reporting should create all expected plot files."""
    result = run_visual_reporting(
        model_comparison=sample_model_comparison,
        intervention_list=sample_intervention_list,
        treatment_log=sample_treatment_log,
        scored_customers_v1=sample_scored_customers_v1,
        scored_customers_v2=sample_scored_customers_v2,
    )

    assert set(result.plot_paths.keys()) == {
        "model_comparison_auc",
        "model_comparison_pr_auc",
        "intervention_action_mix",
        "treatment_outcomes",
        "risk_score_distribution_v1",
        "risk_score_distribution_v2",
    }

    for plot_path in result.plot_paths.values():
        _assert_png_created(plot_path)


def test_visual_reporting_result_text_lists_outputs(
    sample_model_comparison,
    sample_intervention_list,
    sample_treatment_log,
    sample_scored_customers_v1,
    sample_scored_customers_v2,
) -> None:
    """Visual reporting result should produce readable terminal text."""
    result = run_visual_reporting(
        model_comparison=sample_model_comparison,
        intervention_list=sample_intervention_list,
        treatment_log=sample_treatment_log,
        scored_customers_v1=sample_scored_customers_v1,
        scored_customers_v2=sample_scored_customers_v2,
    )

    text = result.as_text()

    assert "Visual reporting outputs" in text
    assert "model_comparison_auc" in text
    assert "treatment_outcomes" in text


def test_plot_model_comparison_metric_fails_with_missing_column(
    sample_model_comparison,
    tmp_path,
) -> None:
    """Visual reporting should fail clearly when required columns are missing."""
    broken_comparison = sample_model_comparison.drop(columns=["roc_auc"])

    with pytest.raises(ValueError, match="Missing required columns"):
        plot_model_comparison_metric(
            model_comparison=broken_comparison,
            metric_column="roc_auc",
            output_path=tmp_path / MODEL_COMPARISON_AUC_PLOT,
        )


def test_plot_risk_score_distribution_fails_with_missing_score_column(
    sample_scored_customers_v1,
    tmp_path,
) -> None:
    """Risk distribution plot should fail clearly if the score column is missing."""
    broken_scores = sample_scored_customers_v1.drop(columns=[MODEL_V1_SCORE_COLUMN])

    with pytest.raises(ValueError, match="Missing required columns"):
        plot_risk_score_distribution(
            scored_table=broken_scores,
            score_column=MODEL_V1_SCORE_COLUMN,
            model_label="ML model v1",
            output_path=tmp_path / RISK_SCORE_DISTRIBUTION_V1_PLOT,
        )