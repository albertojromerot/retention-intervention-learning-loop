"""Feature engineering for the synthetic retention intervention pipeline.

This module converts the synthetic customer-month panel into a model-ready
feature table. It adds interpretable behavioural, relationship, service, and
credit-health features, then assigns a temporal train/validation/test split.

The split is time-based rather than random to better reflect how retention
models are evaluated in real operational settings.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import (
    CUSTOMER_ID_COLUMN,
    FEATURE_TABLE_FILE,
    OUTPUTS_DIR,
    SNAPSHOT_MONTH_COLUMN,
    TARGET_COLUMN,
)


SPLIT_COLUMN = "data_split"


NUMERIC_FEATURE_COLUMNS = [
    "customer_month_index",
    "tenure_months",
    "active_products",
    "has_savings_product",
    "has_credit_product",
    "has_digital_wallet",
    "savings_balance_index",
    "credit_balance_index",
    "relationship_value_index",
    "app_logins_30d",
    "web_logins_30d",
    "digital_transactions_30d",
    "last_digital_activity_days",
    "digital_engagement_score",
    "digital_engagement_trend_3m",
    "contact_attempts_90d",
    "successful_contacts_90d",
    "complaint_count_180d",
    "average_resolution_days",
    "process_friction_score",
    "satisfaction_proxy_score",
    "repayment_behaviour_score",
    "recent_credit_application_flag",
    "relationship_depth_score",
    "digital_inactivity_flag",
    "engagement_decline_flag",
    "high_service_friction_flag",
    "low_satisfaction_flag",
    "credit_stress_score",
    "relationship_health_score",
    "value_at_risk_index",
]


CATEGORICAL_FEATURE_COLUMNS = [
    "age_band",
    "region_type",
    "relationship_segment",
    "days_past_due_band",
]


FEATURE_COLUMNS = NUMERIC_FEATURE_COLUMNS + CATEGORICAL_FEATURE_COLUMNS


@dataclass(frozen=True)
class FeatureBuildResult:
    """Container with the feature table and modelling metadata."""

    feature_table: pd.DataFrame
    feature_columns: list[str]
    numeric_feature_columns: list[str]
    categorical_feature_columns: list[str]
    target_column: str
    split_column: str


def _validate_required_columns(panel: pd.DataFrame) -> None:
    """Validate that the synthetic panel contains columns needed for features."""
    required_columns = {
        CUSTOMER_ID_COLUMN,
        SNAPSHOT_MONTH_COLUMN,
        TARGET_COLUMN,
        "customer_month_index",
        "tenure_months",
        "active_products",
        "has_savings_product",
        "has_credit_product",
        "has_digital_wallet",
        "savings_balance_index",
        "credit_balance_index",
        "relationship_value_index",
        "app_logins_30d",
        "web_logins_30d",
        "digital_transactions_30d",
        "last_digital_activity_days",
        "digital_engagement_score",
        "digital_engagement_trend_3m",
        "contact_attempts_90d",
        "successful_contacts_90d",
        "complaint_count_180d",
        "average_resolution_days",
        "process_friction_score",
        "satisfaction_proxy_score",
        "repayment_behaviour_score",
        "days_past_due_band",
        "recent_credit_application_flag",
        "age_band",
        "region_type",
        "relationship_segment",
    }

    missing_columns = sorted(required_columns.difference(panel.columns))
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")


def _add_temporal_split(feature_table: pd.DataFrame) -> pd.DataFrame:
    """Add train/validation/test split labels using snapshot order.

    The split is based on unique snapshot months:

    - First 70% of months: train.
    - Next 15% of months: validation.
    - Final 15% of months: test.

    For small profiles, the logic guarantees at least one validation month and
    one test month.
    """
    output = feature_table.copy()

    ordered_months = sorted(output[SNAPSHOT_MONTH_COLUMN].unique())
    n_months = len(ordered_months)

    if n_months < 4:
        raise ValueError("At least four monthly snapshots are required.")

    train_end = max(1, int(np.floor(n_months * 0.70)))
    validation_end = max(train_end + 1, int(np.floor(n_months * 0.85)))

    # Guarantee that the test set has at least one month.
    validation_end = min(validation_end, n_months - 1)

    train_months = set(ordered_months[:train_end])
    validation_months = set(ordered_months[train_end:validation_end])

    conditions = [
        output[SNAPSHOT_MONTH_COLUMN].isin(train_months),
        output[SNAPSHOT_MONTH_COLUMN].isin(validation_months),
    ]

    choices = ["train", "validation"]

    output[SPLIT_COLUMN] = np.select(
        conditions,
        choices,
        default="test",
    )

    return output


def build_feature_table(panel: pd.DataFrame) -> FeatureBuildResult:
    """Build a model-ready feature table from the synthetic panel.

    Args:
        panel: Synthetic customer-month panel.

    Returns:
        FeatureBuildResult with the feature table and metadata.
    """
    _validate_required_columns(panel)

    feature_table = panel.copy()

    feature_table["relationship_depth_score"] = (
        12.0 * feature_table["active_products"]
        + 0.25 * feature_table["tenure_months"]
        + 0.20 * feature_table["relationship_value_index"]
    ).clip(0, 100).round(2)

    feature_table["digital_inactivity_flag"] = (
        feature_table["last_digital_activity_days"] >= 45
    ).astype("int8")

    feature_table["engagement_decline_flag"] = (
        feature_table["digital_engagement_trend_3m"] <= -10
    ).astype("int8")

    feature_table["high_service_friction_flag"] = (
        feature_table["process_friction_score"] >= 60
    ).astype("int8")

    feature_table["low_satisfaction_flag"] = (
        feature_table["satisfaction_proxy_score"] <= 45
    ).astype("int8")

    feature_table["credit_stress_score"] = np.select(
        [
            feature_table["days_past_due_band"].eq("no_credit_product"),
            feature_table["days_past_due_band"].eq("current"),
            feature_table["days_past_due_band"].eq("1-30"),
            feature_table["days_past_due_band"].eq("31-60"),
            feature_table["days_past_due_band"].eq("61+"),
        ],
        [0, 5, 25, 55, 85],
        default=0,
    ).astype("int16")

    feature_table["relationship_health_score"] = (
        0.30 * feature_table["relationship_depth_score"]
        + 0.25 * feature_table["digital_engagement_score"]
        + 0.25 * feature_table["satisfaction_proxy_score"]
        + 0.20 * feature_table["repayment_behaviour_score"]
        - 0.30 * feature_table["process_friction_score"]
        - 0.15 * feature_table["credit_stress_score"]
    ).clip(0, 100).round(2)

    feature_table["value_at_risk_index"] = (
        feature_table["relationship_value_index"]
        * (100 - feature_table["relationship_health_score"])
        / 100
    ).clip(0, 100).round(2)

    feature_table = _add_temporal_split(feature_table)

    ordered_columns = [
        CUSTOMER_ID_COLUMN,
        SNAPSHOT_MONTH_COLUMN,
        SPLIT_COLUMN,
        *FEATURE_COLUMNS,
        TARGET_COLUMN,
    ]

    feature_table = feature_table[ordered_columns].copy()

    return FeatureBuildResult(
        feature_table=feature_table,
        feature_columns=FEATURE_COLUMNS,
        numeric_feature_columns=NUMERIC_FEATURE_COLUMNS,
        categorical_feature_columns=CATEGORICAL_FEATURE_COLUMNS,
        target_column=TARGET_COLUMN,
        split_column=SPLIT_COLUMN,
    )


def summarise_feature_table(feature_table: pd.DataFrame) -> dict[str, Any]:
    """Return concise diagnostics for the engineered feature table."""
    split_counts = feature_table[SPLIT_COLUMN].value_counts().to_dict()

    return {
        "rows": int(len(feature_table)),
        "customers": int(feature_table[CUSTOMER_ID_COLUMN].nunique()),
        "months": int(feature_table[SNAPSHOT_MONTH_COLUMN].nunique()),
        "columns": int(feature_table.shape[1]),
        "features": int(len(FEATURE_COLUMNS)),
        "numeric_features": int(len(NUMERIC_FEATURE_COLUMNS)),
        "categorical_features": int(len(CATEGORICAL_FEATURE_COLUMNS)),
        "target_rate": float(feature_table[TARGET_COLUMN].mean()),
        "missing_values": int(feature_table.isna().sum().sum()),
        "train_rows": int(split_counts.get("train", 0)),
        "validation_rows": int(split_counts.get("validation", 0)),
        "test_rows": int(split_counts.get("test", 0)),
    }


def feature_table_summary_as_text(summary: dict[str, Any]) -> str:
    """Format feature-table diagnostics for terminal output."""
    return (
        "Feature table summary\n"
        f"Rows: {summary['rows']:,}\n"
        f"Customers: {summary['customers']:,}\n"
        f"Months: {summary['months']:,}\n"
        f"Columns: {summary['columns']:,}\n"
        f"Features: {summary['features']:,}\n"
        f"Numeric features: {summary['numeric_features']:,}\n"
        f"Categorical features: {summary['categorical_features']:,}\n"
        f"Target rate: {summary['target_rate']:.2%}\n"
        f"Missing values: {summary['missing_values']:,}\n"
        f"Train rows: {summary['train_rows']:,}\n"
        f"Validation rows: {summary['validation_rows']:,}\n"
        f"Test rows: {summary['test_rows']:,}"
    )


def save_feature_table(feature_table: pd.DataFrame) -> Path:
    """Save the engineered feature table to the outputs directory."""
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    output_path = OUTPUTS_DIR / FEATURE_TABLE_FILE
    feature_table.to_csv(output_path, index=False)

    return output_path

def save_feature_table_outputs(result) -> dict:
    """Save the engineered feature table to the outputs directory."""
    from src.config import FEATURE_TABLE_FILE, OUTPUTS_DIR

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    feature_table_path = OUTPUTS_DIR / FEATURE_TABLE_FILE
    result.feature_table.to_csv(feature_table_path, index=False)

    return {
        "feature_table": feature_table_path,
    }