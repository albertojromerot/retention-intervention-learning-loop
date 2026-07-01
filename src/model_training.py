"""Machine learning model training for voluntary attrition prediction.

This module trains the first supervised ML model for the synthetic retention
pipeline. It uses a temporal split, trains only on historical training months,
and evaluates on validation and test months.

The first model is intentionally a standard, interpretable production-style
pipeline using preprocessing plus HistGradientBoostingClassifier.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.config import (
    CUSTOMER_ID_COLUMN,
    MODEL_V1_METRICS_FILE,
    MODEL_V1_VERSION,
    OUTPUTS_DIR,
    RANDOM_SEED,
    SCORED_CUSTOMERS_V1_FILE,
    SNAPSHOT_MONTH_COLUMN,
    TARGET_COLUMN,
)
from src.feature_engineering import (
    CATEGORICAL_FEATURE_COLUMNS,
    FEATURE_COLUMNS,
    NUMERIC_FEATURE_COLUMNS,
    SPLIT_COLUMN,
)


MODEL_V1_SCORE_COLUMN = "model_v1_risk_score"
MODEL_V1_DECILE_COLUMN = "model_v1_risk_decile"


@dataclass(frozen=True)
class ModelTrainingResult:
    """Container for trained model, scored rows, metrics, and metadata."""

    model: Pipeline
    scored_table: pd.DataFrame
    metrics: pd.DataFrame
    feature_columns: list[str]
    score_column: str
    model_version: str


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


def _validate_feature_table(feature_table: pd.DataFrame) -> None:
    """Validate that the feature table contains required modelling columns."""
    required_columns = {
        CUSTOMER_ID_COLUMN,
        SNAPSHOT_MONTH_COLUMN,
        SPLIT_COLUMN,
        TARGET_COLUMN,
        *FEATURE_COLUMNS,
    }

    missing_columns = sorted(required_columns.difference(feature_table.columns))

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    required_splits = {"train", "validation", "test"}
    observed_splits = set(feature_table[SPLIT_COLUMN].unique())

    missing_splits = sorted(required_splits.difference(observed_splits))

    if missing_splits:
        raise ValueError(f"Missing required data splits: {missing_splits}")


def build_model_v1_pipeline() -> Pipeline:
    """Build the preprocessing and classifier pipeline for ML model v1."""
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                "passthrough",
                NUMERIC_FEATURE_COLUMNS,
            ),
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
                CATEGORICAL_FEATURE_COLUMNS,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    classifier = HistGradientBoostingClassifier(
        learning_rate=0.06,
        max_iter=160,
        max_leaf_nodes=31,
        l2_regularization=0.05,
        early_stopping=True,
        validation_fraction=0.12,
        n_iter_no_change=15,
        random_state=RANDOM_SEED,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )


def train_model_v1(feature_table: pd.DataFrame) -> Pipeline:
    """Train ML model v1 on the temporal training split only."""
    _validate_feature_table(feature_table)

    train_frame = feature_table.loc[feature_table[SPLIT_COLUMN].eq("train")].copy()

    x_train = train_frame[FEATURE_COLUMNS]
    y_train = train_frame[TARGET_COLUMN]

    model = build_model_v1_pipeline()
    model.fit(x_train, y_train)

    return model


def score_model_v1(model: Pipeline, feature_table: pd.DataFrame) -> pd.DataFrame:
    """Score all rows in the feature table with ML model v1."""
    _validate_feature_table(feature_table)

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

    scores = model.predict_proba(feature_table[FEATURE_COLUMNS])[:, 1]

    scored_table[MODEL_V1_SCORE_COLUMN] = pd.Series(scores).round(6).to_numpy()

    scored_table[MODEL_V1_DECILE_COLUMN] = pd.qcut(
        scored_table[MODEL_V1_SCORE_COLUMN].rank(method="first"),
        q=10,
        labels=False,
        duplicates="drop",
    )

    scored_table[MODEL_V1_DECILE_COLUMN] = (
        scored_table[MODEL_V1_DECILE_COLUMN].astype(int) + 1
    )

    return scored_table


def evaluate_model_scores(
    scored_table: pd.DataFrame,
    score_column: str,
    model_version: str,
    evaluation_splits: Iterable[str] = ("validation", "test"),
    k_values: Iterable[int] = (100, 500, 1000),
) -> pd.DataFrame:
    """Evaluate model scores by temporal split."""
    required_columns = {
        SPLIT_COLUMN,
        TARGET_COLUMN,
        score_column,
    }

    missing_columns = sorted(required_columns.difference(scored_table.columns))
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    metrics_rows: list[dict[str, float | int | str]] = []

    for split_name in evaluation_splits:
        split_frame = scored_table.loc[scored_table[SPLIT_COLUMN].eq(split_name)].copy()

        if split_frame.empty:
            continue

        y_true = split_frame[TARGET_COLUMN]
        scores = split_frame[score_column]

        row: dict[str, float | int | str] = {
            "model_version": model_version,
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


def run_model_v1(feature_table: pd.DataFrame) -> ModelTrainingResult:
    """Train, score, and evaluate ML model v1."""
    model = train_model_v1(feature_table)
    scored_table = score_model_v1(model, feature_table)
    metrics = evaluate_model_scores(
        scored_table=scored_table,
        score_column=MODEL_V1_SCORE_COLUMN,
        model_version=MODEL_V1_VERSION,
    )

    return ModelTrainingResult(
        model=model,
        scored_table=scored_table,
        metrics=metrics,
        feature_columns=FEATURE_COLUMNS,
        score_column=MODEL_V1_SCORE_COLUMN,
        model_version=MODEL_V1_VERSION,
    )


def model_metrics_as_text(metrics: pd.DataFrame) -> str:
    """Format model metrics for terminal output."""
    if metrics.empty:
        return "Model metrics: no rows to report."

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

    return "Model metrics\n" + rounded.to_string(index=False)


def save_model_v1_outputs(result: ModelTrainingResult) -> dict[str, Path]:
    """Save ML model v1 scored rows and metrics to the outputs directory.

    Args:
        result: ModelTrainingResult produced by run_model_v1.

    Returns:
        Dictionary with output names and saved file paths.
    """
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    metrics_path = OUTPUTS_DIR / MODEL_V1_METRICS_FILE
    scored_path = OUTPUTS_DIR / SCORED_CUSTOMERS_V1_FILE

    result.metrics.to_csv(metrics_path, index=False)
    result.scored_table.to_csv(scored_path, index=False)

    return {
        "metrics": metrics_path,
        "scored_table": scored_path,
    }