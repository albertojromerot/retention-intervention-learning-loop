"""Treatment-aware model v2 for the retention intervention learning loop.

This module learns from the synthetic treatment log. Unlike ML model v1, which
predicts pre-intervention attrition risk, model v2 predicts post-intervention
attrition among customers selected for intervention.

This is a learning-loop simulation:
- model v1 ranks customers for intervention;
- the intervention policy assigns treatment/control groups;
- the synthetic treatment log records execution and outcomes;
- model v2 learns from that feedback.

This model is educational and synthetic. It should not be interpreted as a
real-world causal model.
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
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.config import (
    CUSTOMER_ID_COLUMN,
    MODEL_V2_METRICS_FILE,
    MODEL_V2_VERSION,
    OUTPUTS_DIR,
    RANDOM_SEED,
    SCORED_CUSTOMERS_V2_FILE,
    SNAPSHOT_MONTH_COLUMN,
    TREATMENT_LEARNING_TABLE_V2_FILE,
)
from src.feature_engineering import (
    CATEGORICAL_FEATURE_COLUMNS,
    FEATURE_COLUMNS,
    NUMERIC_FEATURE_COLUMNS,
)
from src.intervention_policy import (
    ASSIGNED_GROUP_COLUMN,
    REASON_CODE_1_COLUMN,
    REASON_CODE_2_COLUMN,
    RECOMMENDED_ACTION_COLUMN,
)
from src.model_training import MODEL_V1_SCORE_COLUMN
from src.treatment_log import (
    CONTACT_ATTEMPTED_COLUMN,
    CONTACT_SUCCESS_COLUMN,
    INTERVENTION_COST_COLUMN,
    POST_INTERVENTION_TARGET_COLUMN,
    TREATMENT_COMPLETED_COLUMN,
)


TREATMENT_LEARNING_SPLIT_COLUMN = "treatment_learning_split"
MODEL_V2_SCORE_COLUMN = "model_v2_post_intervention_risk_score"
MODEL_V2_DECILE_COLUMN = "model_v2_risk_decile"


TREATMENT_NUMERIC_FEATURE_COLUMNS = [
    MODEL_V1_SCORE_COLUMN,
    CONTACT_ATTEMPTED_COLUMN,
    CONTACT_SUCCESS_COLUMN,
    TREATMENT_COMPLETED_COLUMN,
    INTERVENTION_COST_COLUMN,
    "assigned_treatment_flag",
]


TREATMENT_CATEGORICAL_FEATURE_COLUMNS = [
    ASSIGNED_GROUP_COLUMN,
    RECOMMENDED_ACTION_COLUMN,
    "contact_channel",
    REASON_CODE_1_COLUMN,
    REASON_CODE_2_COLUMN,
]


MODEL_V2_FEATURE_COLUMNS = (
    FEATURE_COLUMNS
    + TREATMENT_NUMERIC_FEATURE_COLUMNS
    + TREATMENT_CATEGORICAL_FEATURE_COLUMNS
)


@dataclass(frozen=True)
class ModelV2Result:
    """Container for model v2 outputs."""

    model: Pipeline
    learning_table: pd.DataFrame
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


def _validate_inputs(feature_table: pd.DataFrame, treatment_log: pd.DataFrame) -> None:
    """Validate feature table and treatment log before model v2 learning."""
    required_feature_columns = {
        CUSTOMER_ID_COLUMN,
        SNAPSHOT_MONTH_COLUMN,
        *FEATURE_COLUMNS,
    }

    required_treatment_columns = {
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
        CONTACT_ATTEMPTED_COLUMN,
        CONTACT_SUCCESS_COLUMN,
        TREATMENT_COMPLETED_COLUMN,
        INTERVENTION_COST_COLUMN,
        POST_INTERVENTION_TARGET_COLUMN,
    }

    missing_feature_columns = sorted(required_feature_columns.difference(feature_table.columns))
    missing_treatment_columns = sorted(required_treatment_columns.difference(treatment_log.columns))

    if missing_feature_columns:
        raise ValueError(f"Missing required feature-table columns: {missing_feature_columns}")

    if missing_treatment_columns:
        raise ValueError(f"Missing required treatment-log columns: {missing_treatment_columns}")


def build_treatment_learning_table(
    feature_table: pd.DataFrame,
    treatment_log: pd.DataFrame,
    random_seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Build model-v2 learning table from features and treatment feedback.

    The learning table includes only rows that entered the intervention policy.
    The target is post-intervention attrition.
    """
    _validate_inputs(feature_table=feature_table, treatment_log=treatment_log)

    treatment_columns = [
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
        CONTACT_ATTEMPTED_COLUMN,
        CONTACT_SUCCESS_COLUMN,
        TREATMENT_COMPLETED_COLUMN,
        INTERVENTION_COST_COLUMN,
        POST_INTERVENTION_TARGET_COLUMN,
    ]

    base_columns = [
        CUSTOMER_ID_COLUMN,
        SNAPSHOT_MONTH_COLUMN,
        *FEATURE_COLUMNS,
    ]

    learning_table = treatment_log[treatment_columns].merge(
        feature_table[base_columns],
        on=[CUSTOMER_ID_COLUMN, SNAPSHOT_MONTH_COLUMN],
        how="left",
    )

    learning_table["assigned_treatment_flag"] = (
        learning_table[ASSIGNED_GROUP_COLUMN].eq("treatment").astype("int8")
    )

    if learning_table[FEATURE_COLUMNS].isna().sum().sum() > 0:
        raise ValueError("Feature merge produced missing values for model v2.")

    train_validation, test = train_test_split(
        learning_table,
        test_size=0.20,
        random_state=random_seed,
        stratify=learning_table[POST_INTERVENTION_TARGET_COLUMN],
    )

    train, validation = train_test_split(
        train_validation,
        test_size=0.25,
        random_state=random_seed,
        stratify=train_validation[POST_INTERVENTION_TARGET_COLUMN],
    )

    learning_table[TREATMENT_LEARNING_SPLIT_COLUMN] = "unassigned"
    learning_table.loc[train.index, TREATMENT_LEARNING_SPLIT_COLUMN] = "train"
    learning_table.loc[validation.index, TREATMENT_LEARNING_SPLIT_COLUMN] = "validation"
    learning_table.loc[test.index, TREATMENT_LEARNING_SPLIT_COLUMN] = "test"

    return learning_table.sort_values("policy_rank").reset_index(drop=True)


def build_model_v2_pipeline() -> Pipeline:
    """Build the preprocessing and classifier pipeline for treatment-aware model v2."""
    numeric_columns = NUMERIC_FEATURE_COLUMNS + TREATMENT_NUMERIC_FEATURE_COLUMNS
    categorical_columns = (
        CATEGORICAL_FEATURE_COLUMNS + TREATMENT_CATEGORICAL_FEATURE_COLUMNS
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", "passthrough", numeric_columns),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical_columns,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    classifier = HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=120,
        max_leaf_nodes=15,
        l2_regularization=0.10,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=12,
        random_state=RANDOM_SEED,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )


def train_model_v2(learning_table: pd.DataFrame) -> Pipeline:
    """Train treatment-aware model v2 on treatment-log learning rows."""
    train_frame = learning_table.loc[
        learning_table[TREATMENT_LEARNING_SPLIT_COLUMN].eq("train")
    ].copy()

    x_train = train_frame[MODEL_V2_FEATURE_COLUMNS]
    y_train = train_frame[POST_INTERVENTION_TARGET_COLUMN]

    model = build_model_v2_pipeline()
    model.fit(x_train, y_train)

    return model


def score_model_v2(model: Pipeline, learning_table: pd.DataFrame) -> pd.DataFrame:
    """Score treatment-log learning rows with model v2."""
    scored_table = learning_table[
        [
            "treatment_log_id",
            "policy_rank",
            CUSTOMER_ID_COLUMN,
            SNAPSHOT_MONTH_COLUMN,
            TREATMENT_LEARNING_SPLIT_COLUMN,
            ASSIGNED_GROUP_COLUMN,
            RECOMMENDED_ACTION_COLUMN,
            "contact_channel",
            POST_INTERVENTION_TARGET_COLUMN,
            MODEL_V1_SCORE_COLUMN,
            CONTACT_ATTEMPTED_COLUMN,
            CONTACT_SUCCESS_COLUMN,
            TREATMENT_COMPLETED_COLUMN,
            INTERVENTION_COST_COLUMN,
        ]
    ].copy()

    scores = model.predict_proba(learning_table[MODEL_V2_FEATURE_COLUMNS])[:, 1]

    scored_table[MODEL_V2_SCORE_COLUMN] = pd.Series(scores).round(6).to_numpy()

    scored_table[MODEL_V2_DECILE_COLUMN] = pd.qcut(
        scored_table[MODEL_V2_SCORE_COLUMN].rank(method="first"),
        q=10,
        labels=False,
        duplicates="drop",
    )

    scored_table[MODEL_V2_DECILE_COLUMN] = (
        scored_table[MODEL_V2_DECILE_COLUMN].astype(int) + 1
    )

    return scored_table


def evaluate_model_v2(
    scored_table: pd.DataFrame,
    evaluation_splits: Iterable[str] = ("validation", "test"),
    k_values: Iterable[int] = (50, 100, 200),
) -> pd.DataFrame:
    """Evaluate treatment-aware model v2 by learning split."""
    metrics_rows: list[dict[str, float | int | str]] = []

    for split_name in evaluation_splits:
        split_frame = scored_table.loc[
            scored_table[TREATMENT_LEARNING_SPLIT_COLUMN].eq(split_name)
        ].copy()

        if split_frame.empty:
            continue

        y_true = split_frame[POST_INTERVENTION_TARGET_COLUMN]
        scores = split_frame[MODEL_V2_SCORE_COLUMN]

        row: dict[str, float | int | str] = {
            "model_version": MODEL_V2_VERSION,
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


def run_model_v2(
    feature_table: pd.DataFrame,
    treatment_log: pd.DataFrame,
) -> ModelV2Result:
    """Build learning table, train, score, and evaluate model v2."""
    learning_table = build_treatment_learning_table(
        feature_table=feature_table,
        treatment_log=treatment_log,
    )

    model = train_model_v2(learning_table)
    scored_table = score_model_v2(model, learning_table)
    metrics = evaluate_model_v2(scored_table)

    return ModelV2Result(
        model=model,
        learning_table=learning_table,
        scored_table=scored_table,
        metrics=metrics,
        feature_columns=MODEL_V2_FEATURE_COLUMNS,
        score_column=MODEL_V2_SCORE_COLUMN,
        model_version=MODEL_V2_VERSION,
    )


def model_v2_metrics_as_text(metrics: pd.DataFrame) -> str:
    """Format model v2 metrics for terminal output."""
    if metrics.empty:
        return "Model v2 metrics: no rows to report."

    display_columns = [
        "model_version",
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
    ]

    available_columns = [column for column in display_columns if column in metrics.columns]

    rounded = metrics[available_columns].copy()

    numeric_columns = rounded.select_dtypes(include=["float", "float64"]).columns
    rounded[numeric_columns] = rounded[numeric_columns].round(4)

    return "Model v2 metrics\n" + rounded.to_string(index=False)


def save_model_v2_outputs(result: ModelV2Result) -> dict[str, Path]:
    """Save model v2 learning table, scored rows and metrics.

    Args:
        result: ModelV2Result produced by run_model_v2.

    Returns:
        Dictionary with saved output file paths.
    """
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    learning_table_path = OUTPUTS_DIR / TREATMENT_LEARNING_TABLE_V2_FILE
    metrics_path = OUTPUTS_DIR / MODEL_V2_METRICS_FILE
    scored_path = OUTPUTS_DIR / SCORED_CUSTOMERS_V2_FILE

    result.learning_table.to_csv(learning_table_path, index=False)
    result.metrics.to_csv(metrics_path, index=False)
    result.scored_table.to_csv(scored_path, index=False)

    return {
        "learning_table": learning_table_path,
        "metrics": metrics_path,
        "scored_table": scored_path,
    }