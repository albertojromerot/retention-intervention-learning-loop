"""Synthetic data generation for the retention intervention learning loop.

This module creates a clean-room synthetic customer-month panel for a generic
financial-services context. The data is generated from invented assumptions
and does not represent any real organisation, customer base, schema, metric,
or operational process.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

import numpy as np
import pandas as pd

from src.config import (
    CUSTOMER_ID_COLUMN,
    RANDOM_SEED,
    SNAPSHOT_MONTH_COLUMN,
    SYNTHETIC_DATA_DIR,
    SYNTHETIC_PANEL_FILE,
    TARGET_COLUMN,
    SyntheticProfile,
)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Convert a numeric risk score into a probability between 0 and 1."""
    return 1.0 / (1.0 + np.exp(-x))


def _clip_array(values: np.ndarray, lower: float, upper: float) -> np.ndarray:
    """Clip numeric values to a defined interval."""
    return np.clip(values, lower, upper)


def generate_customer_month_panel(
    profile: SyntheticProfile,
    random_seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Generate a synthetic customer-month panel.

    The dataset is intentionally designed as a generic financial-services
    portfolio dataset. It includes relationship depth, product activity,
    digital engagement, service friction, credit-health-style signals, and a
    synthetic voluntary attrition label over a future 90-day horizon.

    Args:
        profile: Synthetic dataset size profile.
        random_seed: Deterministic random seed for reproducibility.

    Returns:
        A pandas DataFrame with one row per synthetic customer-month.
    """
    rng = np.random.default_rng(random_seed)

    n_customers = profile.n_customers
    n_months = profile.n_months
    n_rows = n_customers * n_months

    customer_ids = np.arange(1, n_customers + 1, dtype=np.int32)
    month_index = np.arange(1, n_months + 1, dtype=np.int16)

    repeated_customer_ids = np.repeat(customer_ids, n_months)
    repeated_month_index = np.tile(month_index, n_customers)

    snapshot_months = pd.period_range(
        start="2024-01",
        periods=n_months,
        freq="M",
    ).astype(str)

    repeated_snapshot_months = np.tile(snapshot_months, n_customers)

    # ---------------------------------------------------------------------
    # Customer-level attributes
    # ---------------------------------------------------------------------

    base_tenure = rng.integers(1, 121, size=n_customers, dtype=np.int16)

    age_bands = rng.choice(
        ["18-29", "30-44", "45-59", "60+"],
        size=n_customers,
        p=[0.22, 0.38, 0.27, 0.13],
    )

    region_types = rng.choice(
        ["urban", "suburban", "regional", "rural"],
        size=n_customers,
        p=[0.48, 0.24, 0.18, 0.10],
    )

    relationship_segments = rng.choice(
        ["emerging", "core", "established", "high_value"],
        size=n_customers,
        p=[0.30, 0.42, 0.20, 0.08],
    )

    customer_engagement_base = rng.normal(loc=0.0, scale=1.0, size=n_customers)
    customer_value_base = rng.normal(loc=0.0, scale=1.0, size=n_customers)
    customer_service_sensitivity = rng.normal(loc=0.0, scale=1.0, size=n_customers)

    # Repeat customer-level attributes to customer-month level.
    tenure_months = np.repeat(base_tenure, n_months) + repeated_month_index - 1
    age_band = np.repeat(age_bands, n_months)
    region_type = np.repeat(region_types, n_months)
    relationship_segment = np.repeat(relationship_segments, n_months)
    engagement_base = np.repeat(customer_engagement_base, n_months)
    value_base = np.repeat(customer_value_base, n_months)
    service_sensitivity = np.repeat(customer_service_sensitivity, n_months)

    # ---------------------------------------------------------------------
    # Time-varying synthetic behavioural signals
    # ---------------------------------------------------------------------

    seasonal_effect = 0.15 * np.sin((repeated_month_index / 12.0) * 2.0 * np.pi)
    time_noise = rng.normal(loc=0.0, scale=0.35, size=n_rows)

    relationship_strength = (
        0.015 * tenure_months
        + 0.55 * value_base
        + 0.35 * engagement_base
        + seasonal_effect
        + time_noise
    )

    active_products = np.round(
        _clip_array(
            2.2 + 0.018 * tenure_months + 0.55 * value_base + rng.normal(0, 0.9, n_rows),
            1,
            7,
        )
    ).astype(np.int8)

    has_savings_product = rng.binomial(
        n=1,
        p=_clip_array(0.72 + 0.03 * active_products, 0.50, 0.98),
        size=n_rows,
    ).astype(np.int8)

    has_credit_product = rng.binomial(
        n=1,
        p=_clip_array(0.30 + 0.05 * active_products + 0.04 * value_base, 0.10, 0.88),
        size=n_rows,
    ).astype(np.int8)

    has_digital_wallet = rng.binomial(
        n=1,
        p=_clip_array(0.42 + 0.10 * engagement_base + 0.02 * active_products, 0.08, 0.92),
        size=n_rows,
    ).astype(np.int8)

    savings_balance_index = _clip_array(
        50
        + 9.0 * value_base
        + 2.0 * active_products
        + 0.08 * tenure_months
        + rng.normal(0, 12, n_rows),
        0,
        100,
    ).round(2)

    credit_balance_index = _clip_array(
        35
        + 8.5 * has_credit_product
        + 5.0 * value_base
        + rng.normal(0, 15, n_rows),
        0,
        100,
    ).round(2)

    relationship_value_index = _clip_array(
        0.55 * savings_balance_index
        + 0.30 * credit_balance_index
        + 4.0 * active_products
        + rng.normal(0, 5, n_rows),
        0,
        100,
    ).round(2)

    # ---------------------------------------------------------------------
    # Digital engagement
    # ---------------------------------------------------------------------

    digital_engagement_latent = (
        1.2
        + 0.65 * engagement_base
        + 0.35 * has_digital_wallet
        + 0.03 * active_products
        + seasonal_effect
        + rng.normal(0, 0.45, n_rows)
    )

    app_logins_30d = rng.poisson(
        lam=_clip_array(np.exp(digital_engagement_latent), 0.2, 45.0)
    ).astype(np.int16)

    web_logins_30d = rng.poisson(
        lam=_clip_array(np.exp(0.75 + 0.45 * engagement_base + rng.normal(0, 0.35, n_rows)), 0.1, 25.0)
    ).astype(np.int16)

    digital_transactions_30d = rng.poisson(
        lam=_clip_array(
            2.0 + 0.35 * app_logins_30d + 0.12 * web_logins_30d + 0.25 * active_products,
            0.2,
            60.0,
        )
    ).astype(np.int16)

    last_digital_activity_days = np.round(
        _clip_array(
            35
            - 1.6 * app_logins_30d
            - 0.65 * digital_transactions_30d
            + rng.normal(0, 10, n_rows),
            0,
            180,
        )
    ).astype(np.int16)

    digital_engagement_score = _clip_array(
        20
        + 2.0 * app_logins_30d
        + 1.2 * web_logins_30d
        + 1.7 * digital_transactions_30d
        - 0.25 * last_digital_activity_days,
        0,
        100,
    ).round(2)

    digital_engagement_trend_3m = _clip_array(
        rng.normal(loc=0.0, scale=12.0, size=n_rows)
        + 0.10 * (digital_engagement_score - 50)
        - 0.06 * last_digital_activity_days,
        -50,
        50,
    ).round(2)

    # ---------------------------------------------------------------------
    # Service, satisfaction, and process-friction signals
    # ---------------------------------------------------------------------

    complaint_lambda = _clip_array(
        0.20
        + 0.18 * np.maximum(service_sensitivity, 0)
        + 0.012 * last_digital_activity_days
        - 0.0025 * digital_engagement_score,
        0.02,
        3.50,
    )

    complaint_count_180d = rng.poisson(lam=complaint_lambda).astype(np.int8)

    contact_attempts_90d = rng.poisson(
        lam=_clip_array(0.50 + 0.45 * complaint_count_180d + 0.006 * last_digital_activity_days, 0.05, 8.0)
    ).astype(np.int8)

    successful_contacts_90d = rng.binomial(
        n=np.maximum(contact_attempts_90d, 0),
        p=_clip_array(0.62 + 0.004 * digital_engagement_score - 0.03 * complaint_count_180d, 0.10, 0.92),
    ).astype(np.int8)

    average_resolution_days = _clip_array(
        3.0
        + 2.2 * complaint_count_180d
        + 1.5 * np.maximum(service_sensitivity, 0)
        + rng.normal(0, 2.0, n_rows),
        0,
        45,
    ).round(2)

    process_friction_score = _clip_array(
        15
        + 8.5 * complaint_count_180d
        + 1.2 * average_resolution_days
        + 0.15 * last_digital_activity_days
        - 0.12 * digital_engagement_score
        + rng.normal(0, 7, n_rows),
        0,
        100,
    ).round(2)

    satisfaction_proxy_score = _clip_array(
        82
        - 0.55 * process_friction_score
        - 3.2 * complaint_count_180d
        + 0.12 * digital_engagement_score
        + 0.06 * relationship_value_index
        + rng.normal(0, 6, n_rows),
        0,
        100,
    ).round(2)

    # ---------------------------------------------------------------------
    # Credit-health-style signals
    # ---------------------------------------------------------------------

    repayment_behaviour_score = _clip_array(
        78
        + 0.12 * relationship_value_index
        + 0.08 * satisfaction_proxy_score
        - 0.25 * process_friction_score
        + rng.normal(0, 10, n_rows),
        0,
        100,
    ).round(2)

    delinquency_probability = _clip_array(
        0.22
        - 0.0017 * repayment_behaviour_score
        + 0.0012 * process_friction_score
        + 0.03 * has_credit_product,
        0.01,
        0.35,
    )

    delinquency_flag = rng.binomial(n=1, p=delinquency_probability, size=n_rows)

    days_past_due_band = np.where(
        has_credit_product == 0,
        "no_credit_product",
        np.where(
            delinquency_flag == 0,
            "current",
            rng.choice(
                ["1-30", "31-60", "61+"],
                size=n_rows,
                p=[0.70, 0.22, 0.08],
            ),
        ),
    )

    recent_credit_application_flag = rng.binomial(
        n=1,
        p=_clip_array(0.06 + 0.006 * active_products + 0.001 * relationship_value_index, 0.02, 0.28),
        size=n_rows,
    ).astype(np.int8)

    # ---------------------------------------------------------------------
    # Synthetic target: voluntary attrition within next 90 days
    # ---------------------------------------------------------------------

    risk_score = (
        -0.80
        - 0.020 * active_products
        - 0.010 * tenure_months
        - 0.011 * relationship_value_index
        - 0.010 * satisfaction_proxy_score
        - 0.006 * repayment_behaviour_score
        + 0.020 * last_digital_activity_days
        + 0.055 * complaint_count_180d
        + 0.012 * process_friction_score
        - 0.014 * digital_engagement_trend_3m
        + 0.20 * (days_past_due_band == "31-60").astype(float)
        + 0.42 * (days_past_due_band == "61+").astype(float)
        + rng.normal(0, 0.40, n_rows)
    )

    attrition_probability = _clip_array(_sigmoid(risk_score), 0.005, 0.65)

    voluntary_attrition_next_90d = rng.binomial(
        n=1,
        p=attrition_probability,
        size=n_rows,
    ).astype(np.int8)

    panel = pd.DataFrame(
        {
            CUSTOMER_ID_COLUMN: repeated_customer_ids,
            SNAPSHOT_MONTH_COLUMN: repeated_snapshot_months,
            "customer_month_index": repeated_month_index,
            "tenure_months": tenure_months.astype(np.int16),
            "age_band": age_band,
            "region_type": region_type,
            "relationship_segment": relationship_segment,
            "active_products": active_products,
            "has_savings_product": has_savings_product,
            "has_credit_product": has_credit_product,
            "has_digital_wallet": has_digital_wallet,
            "savings_balance_index": savings_balance_index,
            "credit_balance_index": credit_balance_index,
            "relationship_value_index": relationship_value_index,
            "app_logins_30d": app_logins_30d,
            "web_logins_30d": web_logins_30d,
            "digital_transactions_30d": digital_transactions_30d,
            "last_digital_activity_days": last_digital_activity_days,
            "digital_engagement_score": digital_engagement_score,
            "digital_engagement_trend_3m": digital_engagement_trend_3m,
            "contact_attempts_90d": contact_attempts_90d,
            "successful_contacts_90d": successful_contacts_90d,
            "complaint_count_180d": complaint_count_180d,
            "average_resolution_days": average_resolution_days,
            "process_friction_score": process_friction_score,
            "satisfaction_proxy_score": satisfaction_proxy_score,
            "repayment_behaviour_score": repayment_behaviour_score,
            "days_past_due_band": days_past_due_band,
            "recent_credit_application_flag": recent_credit_application_flag,
            TARGET_COLUMN: voluntary_attrition_next_90d,
        }
    )

    return panel


def summarise_synthetic_panel(panel: pd.DataFrame) -> dict[str, Any]:
    """Return a concise summary of the synthetic customer-month panel.

    Args:
        panel: Synthetic customer-month panel.

    Returns:
        Dictionary with basic dataset diagnostics.
    """
    return {
        "rows": int(len(panel)),
        "customers": int(panel[CUSTOMER_ID_COLUMN].nunique()),
        "months": int(panel[SNAPSHOT_MONTH_COLUMN].nunique()),
        "columns": int(panel.shape[1]),
        "target_rate": float(panel[TARGET_COLUMN].mean()),
        "missing_values": int(panel.isna().sum().sum()),
        "memory_mb": float(panel.memory_usage(deep=True).sum() / 1_000_000),
    }


def save_synthetic_panel(panel: pd.DataFrame) -> Path:
    """Save the synthetic customer-month panel to the synthetic data directory.

    Args:
        panel: Synthetic customer-month panel.

    Returns:
        Path to the saved CSV file.
    """
    SYNTHETIC_DATA_DIR.mkdir(parents=True, exist_ok=True)

    output_path = SYNTHETIC_DATA_DIR / SYNTHETIC_PANEL_FILE
    panel.to_csv(output_path, index=False)

    return output_path


def synthetic_panel_summary_as_text(summary: dict[str, Any]) -> str:
    """Format a synthetic panel summary for terminal output."""
    return (
        "Synthetic panel summary\n"
        f"Rows: {summary['rows']:,}\n"
        f"Customers: {summary['customers']:,}\n"
        f"Months: {summary['months']:,}\n"
        f"Columns: {summary['columns']:,}\n"
        f"Target rate: {summary['target_rate']:.2%}\n"
        f"Missing values: {summary['missing_values']:,}\n"
        f"Memory usage: {summary['memory_mb']:.2f} MB"
    )


def profile_to_dict(profile: SyntheticProfile) -> dict[str, Any]:
    """Convert a synthetic profile dataclass to a plain dictionary."""
    return asdict(profile)

def save_synthetic_panel_outputs(panel) -> dict:
    """Save the synthetic customer-month panel to the data/synthetic directory."""
    from pathlib import Path

    from src.config import SYNTHETIC_PANEL_FILE, get_profile

    profile = get_profile()
    synthetic_dir = Path(Path.cwd()) / "data" / "synthetic"
    synthetic_dir.mkdir(parents=True, exist_ok=True)

    synthetic_panel_path = synthetic_dir / SYNTHETIC_PANEL_FILE
    panel.to_csv(synthetic_panel_path, index=False)

    return {
        "synthetic_panel": synthetic_panel_path,
    }