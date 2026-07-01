"""Portfolio-ready visual reporting for the learning-loop pipeline.

This module creates simple matplotlib plots from the saved pipeline outputs.
The plots are designed for GitHub, academic review and business explanation.

The visuals intentionally avoid dashboard dependencies. They can be generated
locally on a standard Python environment using pandas and matplotlib only.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from src.config import OUTPUTS_DIR


PLOTS_DIR = OUTPUTS_DIR / "plots"

MODEL_COMPARISON_INPUT_FILE = OUTPUTS_DIR / "model_comparison.csv"
INTERVENTION_LIST_INPUT_FILE = OUTPUTS_DIR / "intervention_list_v1.csv"
TREATMENT_LOG_INPUT_FILE = OUTPUTS_DIR / "treatment_log_v1.csv"
SCORED_CUSTOMERS_V1_INPUT_FILE = OUTPUTS_DIR / "scored_customers_v1.csv"
SCORED_CUSTOMERS_V2_INPUT_FILE = OUTPUTS_DIR / "scored_customers_v2.csv"

MODEL_COMPARISON_AUC_PLOT = "model_comparison_auc.png"
MODEL_COMPARISON_PR_AUC_PLOT = "model_comparison_pr_auc.png"
INTERVENTION_ACTION_MIX_PLOT = "intervention_action_mix.png"
TREATMENT_OUTCOMES_PLOT = "treatment_outcomes.png"
RISK_SCORE_DISTRIBUTION_V1_PLOT = "risk_score_distribution_v1.png"
RISK_SCORE_DISTRIBUTION_V2_PLOT = "risk_score_distribution_v2.png"

MODEL_V1_SCORE_COLUMN = "model_v1_risk_score"
MODEL_V2_SCORE_COLUMN = "model_v2_post_intervention_risk_score"


@dataclass(frozen=True)
class VisualReportingResult:
    """Container for generated plot output paths."""

    plot_paths: dict[str, Path]

    def as_text(self) -> str:
        """Format generated plot paths for terminal output."""
        lines = ["Visual reporting outputs"]
        for plot_name, plot_path in self.plot_paths.items():
            lines.append(f"- {plot_name}: {plot_path}")
        return "\n".join(lines)


def _ensure_required_columns(
    frame: pd.DataFrame,
    required_columns: list[str],
    frame_name: str,
) -> None:
    """Fail clearly if a required visual-reporting column is missing."""
    missing_columns = [
        column for column in required_columns if column not in frame.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns in {frame_name}: {missing_columns}"
        )


def _save_figure(fig: plt.Figure, output_path: Path) -> Path:
    """Save a matplotlib figure and close it cleanly."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _metric_display_name(metric_column: str) -> str:
    """Convert metric column names into readable chart labels."""
    return metric_column.replace("_", " ").upper()


def plot_model_comparison_metric(
    model_comparison: pd.DataFrame,
    metric_column: str,
    output_path: Path,
) -> Path:
    """Create a horizontal bar chart for one model-comparison metric."""
    required_columns = [
        "model_family",
        "model_version",
        "split",
        "prediction_objective",
        metric_column,
    ]
    _ensure_required_columns(
        model_comparison,
        required_columns,
        "model_comparison",
    )

    plot_data = model_comparison.dropna(subset=[metric_column]).copy()

    if plot_data.empty:
        raise ValueError(f"No non-missing values found for {metric_column}")

    plot_data["display_label"] = (
        plot_data["model_family"].astype(str)
        + " | "
        + plot_data["model_version"].astype(str)
        + " | "
        + plot_data["split"].astype(str)
    )

    plot_data = plot_data.sort_values(metric_column, ascending=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(plot_data["display_label"], plot_data[metric_column])
    ax.set_title(f"Model comparison — {_metric_display_name(metric_column)}")
    ax.set_xlabel(_metric_display_name(metric_column))
    ax.set_ylabel("Model, version and split")
    ax.grid(axis="x", alpha=0.3)

    return _save_figure(fig, output_path)


def plot_intervention_action_mix(
    intervention_list: pd.DataFrame,
    output_path: Path,
) -> Path:
    """Create a chart showing the recommended action mix."""
    _ensure_required_columns(
        intervention_list,
        ["recommended_action"],
        "intervention_list",
    )

    action_counts = (
        intervention_list["recommended_action"]
        .value_counts()
        .sort_values(ascending=True)
    )

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(action_counts.index.astype(str), action_counts.values)
    ax.set_title("Intervention action mix")
    ax.set_xlabel("Customers")
    ax.set_ylabel("Recommended action")
    ax.grid(axis="x", alpha=0.3)

    return _save_figure(fig, output_path)


def plot_treatment_outcomes(
    treatment_log: pd.DataFrame,
    output_path: Path,
) -> Path:
    """Create a treatment/control outcome-rate chart."""
    outcome_columns = [
        "contact_attempted",
        "contact_success",
        "treatment_completed",
        "prevented_attrition_flag",
    ]

    required_columns = ["assigned_group", *outcome_columns]
    _ensure_required_columns(treatment_log, required_columns, "treatment_log")

    outcome_rates = (
        treatment_log.groupby("assigned_group")[outcome_columns]
        .mean()
        .mul(100)
        .T
    )

    outcome_rates.index = [
        label.replace("_", " ").title() for label in outcome_rates.index
    ]

    fig, ax = plt.subplots(figsize=(9, 5))
    outcome_rates.plot(kind="bar", ax=ax)
    ax.set_title("Treatment/control process and outcome rates")
    ax.set_xlabel("Outcome")
    ax.set_ylabel("Rate (%)")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(title="Assigned group")

    return _save_figure(fig, output_path)


def plot_risk_score_distribution(
    scored_table: pd.DataFrame,
    score_column: str,
    model_label: str,
    output_path: Path,
) -> Path:
    """Create a risk-score distribution histogram."""
    _ensure_required_columns(scored_table, [score_column], model_label)

    scores = scored_table[score_column].dropna()

    if scores.empty:
        raise ValueError(f"No scores found in {score_column}")

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(scores, bins=30)
    ax.set_title(f"Risk score distribution — {model_label}")
    ax.set_xlabel("Predicted risk score")
    ax.set_ylabel("Rows")
    ax.set_xlim(0, 1)
    ax.grid(axis="y", alpha=0.3)

    return _save_figure(fig, output_path)


def run_visual_reporting(
    model_comparison: pd.DataFrame,
    intervention_list: pd.DataFrame,
    treatment_log: pd.DataFrame,
    scored_customers_v1: pd.DataFrame,
    scored_customers_v2: pd.DataFrame,
) -> VisualReportingResult:
    """Generate all portfolio-ready visual reports."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    plot_paths = {
        "model_comparison_auc": plot_model_comparison_metric(
            model_comparison=model_comparison,
            metric_column="roc_auc",
            output_path=PLOTS_DIR / MODEL_COMPARISON_AUC_PLOT,
        ),
        "model_comparison_pr_auc": plot_model_comparison_metric(
            model_comparison=model_comparison,
            metric_column="pr_auc",
            output_path=PLOTS_DIR / MODEL_COMPARISON_PR_AUC_PLOT,
        ),
        "intervention_action_mix": plot_intervention_action_mix(
            intervention_list=intervention_list,
            output_path=PLOTS_DIR / INTERVENTION_ACTION_MIX_PLOT,
        ),
        "treatment_outcomes": plot_treatment_outcomes(
            treatment_log=treatment_log,
            output_path=PLOTS_DIR / TREATMENT_OUTCOMES_PLOT,
        ),
        "risk_score_distribution_v1": plot_risk_score_distribution(
            scored_table=scored_customers_v1,
            score_column=MODEL_V1_SCORE_COLUMN,
            model_label="ML model v1 pre-intervention ranking",
            output_path=PLOTS_DIR / RISK_SCORE_DISTRIBUTION_V1_PLOT,
        ),
        "risk_score_distribution_v2": plot_risk_score_distribution(
            scored_table=scored_customers_v2,
            score_column=MODEL_V2_SCORE_COLUMN,
            model_label="Treatment-aware ML model v2",
            output_path=PLOTS_DIR / RISK_SCORE_DISTRIBUTION_V2_PLOT,
        ),
    }

    return VisualReportingResult(plot_paths=plot_paths)


def run_visual_reporting_from_outputs() -> VisualReportingResult:
    """Generate visual reports from existing CSV outputs."""
    input_files = [
        MODEL_COMPARISON_INPUT_FILE,
        INTERVENTION_LIST_INPUT_FILE,
        TREATMENT_LOG_INPUT_FILE,
        SCORED_CUSTOMERS_V1_INPUT_FILE,
        SCORED_CUSTOMERS_V2_INPUT_FILE,
    ]

    missing_files = [path for path in input_files if not path.exists()]

    if missing_files:
        raise FileNotFoundError(
            "Missing required output files. Run `python -m src.run_pipeline` "
            f"before visual reporting. Missing files: {missing_files}"
        )

    model_comparison = pd.read_csv(MODEL_COMPARISON_INPUT_FILE)
    intervention_list = pd.read_csv(INTERVENTION_LIST_INPUT_FILE)
    treatment_log = pd.read_csv(TREATMENT_LOG_INPUT_FILE)
    scored_customers_v1 = pd.read_csv(SCORED_CUSTOMERS_V1_INPUT_FILE)
    scored_customers_v2 = pd.read_csv(SCORED_CUSTOMERS_V2_INPUT_FILE)

    return run_visual_reporting(
        model_comparison=model_comparison,
        intervention_list=intervention_list,
        treatment_log=treatment_log,
        scored_customers_v1=scored_customers_v1,
        scored_customers_v2=scored_customers_v2,
    )