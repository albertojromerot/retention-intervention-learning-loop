"""Rules-based baseline for voluntary attrition prioritisation.

This module creates a transparent, non-ML benchmark. The baseline uses
interpretable business rules based on inactivity, declining engagement,
service friction, satisfaction, credit stress, and relationship health.

The purpose is not to be perfect. The purpose is to provide a credible
comparison point for later ML models.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)

from src.config import (
    CUSTOMER_ID_COLUMN,
    OUTPUTS_DIR,
    RULES_BASELINE_METRICS_FILE,
    RULES_BASELINE_SCORED_FILE,
    RULES_BASELINE_VERSION,
    SNAPSHOT_MONTH_COLUMN,
    TARGET_COLUMN,
)
from src.feature_engineering import SPLIT_COLUMN


RULES_SCORE_COLUMN = "rules_risk_score"
RULES_DECILE_COLUMN = "rules_risk_decile"


@dataclass(frozen=True)
class RulesBaselineResult:
    """Container for rules-based scoring and evaluation outputs."""

    scored_table: pd.DataFrame
    metrics: pd.DataFrame
    score_column: str
    model_version: str


def _min_max_scale(values: pd.Series) -> pd.Series:
    """Scale values to the [0, 1] interval.

    If all values are identical, return a neutral 0.5 score.
    """
    min_value = values.min()
    max_value = values.max()

    if max_value == min_value:
        return pd.Series(0.5, index=values.index)

    return (values - min_value) / (max_value - min_value)


def _precision_at_k(y_true: pd.Series, scores: pd.Series, k: int) -> float:
    """Compute precision among the top-K highest-scored rows."""
    if len(y_true) == 0:
        return float("nan")

    effective_k = min(k, len(y_true))

    ranked = pd.DataFrame(
        {
            "target": y_true.to_numpy(),
            "score": scores.to_numpy(),
        }
    ).sort_values("score", ascending=False)

    return float(ranked.head(effective_k)["target"].mean())


def _lift_at_k(y_true: pd.Series, scores: pd.Series, k: int) -> float:
    """Compute lift at K versus the base target rate."""
    base_rate = float(y_true.mean())

    if base_rate == 0:
        return float("nan")

    return _precision_at_k(y_true=y_true, scores=scores, k=k) / base_rate


def score_with_rules_baseline(feature_table: pd.DataFrame) -> pd.DataFrame:
    """Score customer-month rows using transparent business rules.

    Args:
        feature_table: Engineered feature table.

    Returns:
        DataFrame with identifiers, target, split, rules score and risk decile.
    """
    required_columns = {
        CUSTOMER_ID_COLUMN,
        SNAPSHOT_MONTH_COLUMN,
        SPLIT_COLUMN,
        TARGET_COLUMN,
        "digital_inactivity_flag",
        "engagement_decline_flag",
        "high_service_friction_flag",
        "low_satisfaction_flag",
        "credit_stress_score",
        "process_friction_score",
        "last_digital_activity_days",
        "value_at_risk_index",
        "complaint_count_180d",
        "relationship_health_score",
        "relationship_depth_score",
        "relationship_value_index",
    }

    missing_columns = sorted(required_columns.difference(feature_table.columns))
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    scored_table = feature_table[
        [
            CUSTOMER_ID_COLUMN,
            SNAPSHOT_MONTH_COLUMN,
            SPLIT_COLUMN,
            TARGET_COLUMN,
            "relationship_value_index",
            "value_at_risk_index",
            "relationship_health_score",
        ]
    ].copy()

    raw_score = (
        0.22 * feature_table["digital_inactivity_flag"]
        + 0.17 * feature_table["engagement_decline_flag"]
        + 0.16 * feature_table["high_service_friction_flag"]
        + 0.14 * feature_table["low_satisfaction_flag"]
        + 0.12 * (feature_table["credit_stress_score"] / 100)
        + 0.11 * (feature_table["process_friction_score"] / 100)
        + 0.09 * (feature_table["last_digital_activity_days"].clip(0, 180) / 180)
        + 0.09 * (feature_table["value_at_risk_index"] / 100)
        + 0.07 * (feature_table["complaint_count_180d"].clip(0, 5) / 5)
        - 0.12 * (feature_table["relationship_health_score"] / 100)
        - 0.08 * (feature_table["relationship_depth_score"] / 100)
    )

    scored_table[RULES_SCORE_COLUMN] = _min_max_scale(raw_score).round(6)

    scored_table[RULES_DECILE_COLUMN] = pd.qcut(
        scored_table[RULES_SCORE_COLUMN].rank(method="first"),
        q=10,
        labels=False,
        duplicates="drop",
    )

    scored_table[RULES_DECILE_COLUMN] = (
        scored_table[RULES_DECILE_COLUMN].astype(int) + 1
    )

    return scored_table


def evaluate_rules_baseline(
    scored_table: pd.DataFrame,
    evaluation_splits: Iterable[str] = ("validation", "test"),
    k_values: Iterable[int] = (100, 500, 1000),
) -> pd.DataFrame:
    """Evaluate the rules-based baseline by temporal split.

    Args:
        scored_table: Output from score_with_rules_baseline.
        evaluation_splits: Splits to evaluate.
        k_values: Ranking cut-offs for precision and lift.

    Returns:
        Metrics table with one row per evaluated split.
    """
    metrics_rows: list[dict[str, float | int | str]] = []

    for split_name in evaluation_splits:
        split_frame = scored_table.loc[scored_table[SPLIT_COLUMN].eq(split_name)].copy()

        if split_frame.empty:
            continue

        y_true = split_frame[TARGET_COLUMN]
        scores = split_frame[RULES_SCORE_COLUMN]

        row: dict[str, float | int | str] = {
            "model_version": RULES_BASELINE_VERSION,
            "split": split_name,
            "rows": int(len(split_frame)),
            "target_rate": float(y_true.mean()),
            "roc_auc": float(roc_auc_score(y_true, scores)),
            "pr_auc": float(average_precision_score(y_true, scores)),
            "brier_score": float(brier_score_loss(y_true, scores)),
        }

        for k in k_values:
            row[f"precision_at_{k}"] = _precision_at_k(y_true, scores, k)
            row[f"lift_at_{k}"] = _lift_at_k(y_true, scores, k)

        metrics_rows.append(row)

    return pd.DataFrame(metrics_rows)


def run_rules_baseline(feature_table: pd.DataFrame) -> RulesBaselineResult:
    """Run rules-based scoring and evaluation."""
    scored_table = score_with_rules_baseline(feature_table)
    metrics = evaluate_rules_baseline(scored_table)

    return RulesBaselineResult(
        scored_table=scored_table,
        metrics=metrics,
        score_column=RULES_SCORE_COLUMN,
        model_version=RULES_BASELINE_VERSION,
    )


def rules_metrics_as_text(metrics: pd.DataFrame) -> str:
    """Format rules-baseline metrics for terminal output."""
    if metrics.empty:
        return "Rules baseline metrics: no rows to report."

    display_columns = [
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
    ]

    available_columns = [column for column in display_columns if column in metrics.columns]

    rounded = metrics[available_columns].copy()

    numeric_columns = rounded.select_dtypes(include=["float", "float64"]).columns
    rounded[numeric_columns] = rounded[numeric_columns].round(4)

    return "Rules baseline metrics\n" + rounded.to_string(index=False)


def save_rules_baseline_outputs(result: RulesBaselineResult) -> dict[str, Path]:
    """Save rules-baseline scored rows and metrics to the outputs directory.

    Args:
        result: RulesBaselineResult produced by run_rules_baseline.

    Returns:
        Dictionary with output names and saved file paths.
    """
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    metrics_path = OUTPUTS_DIR / RULES_BASELINE_METRICS_FILE
    scored_path = OUTPUTS_DIR / RULES_BASELINE_SCORED_FILE

    result.metrics.to_csv(metrics_path, index=False)
    result.scored_table.to_csv(scored_path, index=False)

    return {
        "metrics": metrics_path,
        "scored_table": scored_path,
    }