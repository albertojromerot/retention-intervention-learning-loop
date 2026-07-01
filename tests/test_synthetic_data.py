"""Tests for synthetic customer-month data generation."""

import pandas as pd

from src.config import (
    CUSTOMER_ID_COLUMN,
    SNAPSHOT_MONTH_COLUMN,
    TARGET_COLUMN,
    get_profile,
)
from src.synthetic_data import (
    generate_customer_month_panel,
    summarise_synthetic_panel,
)


def _generate_compact_panel() -> pd.DataFrame:
    """Generate a compact synthetic panel for fast tests."""
    return generate_customer_month_panel(profile=get_profile("compact"))


def test_synthetic_panel_has_expected_shape() -> None:
    """The compact profile should generate the expected number of rows."""
    profile = get_profile("compact")
    panel = _generate_compact_panel()

    assert panel.shape[0] == profile.n_customers * profile.n_months
    assert panel[CUSTOMER_ID_COLUMN].nunique() == profile.n_customers
    assert panel[SNAPSHOT_MONTH_COLUMN].nunique() == profile.n_months


def test_synthetic_panel_required_columns_exist() -> None:
    """The synthetic panel should contain all planned baseline columns."""
    panel = _generate_compact_panel()

    required_columns = {
        CUSTOMER_ID_COLUMN,
        SNAPSHOT_MONTH_COLUMN,
        "customer_month_index",
        "tenure_months",
        "age_band",
        "region_type",
        "relationship_segment",
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
        TARGET_COLUMN,
    }

    assert required_columns.issubset(set(panel.columns))


def test_customer_month_key_is_unique() -> None:
    """Each customer should have only one row per snapshot month."""
    panel = _generate_compact_panel()

    duplicated_keys = panel.duplicated(
        subset=[CUSTOMER_ID_COLUMN, SNAPSHOT_MONTH_COLUMN]
    ).sum()

    assert duplicated_keys == 0


def test_target_is_binary_and_reasonably_imbalanced() -> None:
    """The synthetic target should be binary with a realistic event rate."""
    panel = _generate_compact_panel()

    target_values = set(panel[TARGET_COLUMN].unique())
    target_rate = panel[TARGET_COLUMN].mean()

    assert target_values.issubset({0, 1})
    assert 0.05 <= target_rate <= 0.15


def test_no_missing_values_are_generated() -> None:
    """The first synthetic panel version should not contain missing values."""
    panel = _generate_compact_panel()

    assert panel.isna().sum().sum() == 0


def test_numeric_score_columns_are_within_expected_bounds() -> None:
    """Synthetic score/index columns should remain between 0 and 100."""
    panel = _generate_compact_panel()

    bounded_score_columns = [
        "savings_balance_index",
        "credit_balance_index",
        "relationship_value_index",
        "digital_engagement_score",
        "process_friction_score",
        "satisfaction_proxy_score",
        "repayment_behaviour_score",
    ]

    for column in bounded_score_columns:
        assert panel[column].between(0, 100).all(), column


def test_count_columns_are_non_negative() -> None:
    """Synthetic count columns should not contain negative values."""
    panel = _generate_compact_panel()

    count_columns = [
        "active_products",
        "app_logins_30d",
        "web_logins_30d",
        "digital_transactions_30d",
        "last_digital_activity_days",
        "contact_attempts_90d",
        "successful_contacts_90d",
        "complaint_count_180d",
    ]

    for column in count_columns:
        assert (panel[column] >= 0).all(), column


def test_categorical_values_are_expected() -> None:
    """Categorical values should stay within the synthetic design space."""
    panel = _generate_compact_panel()

    assert set(panel["age_band"].unique()).issubset(
        {"18-29", "30-44", "45-59", "60+"}
    )

    assert set(panel["region_type"].unique()).issubset(
        {"urban", "suburban", "regional", "rural"}
    )

    assert set(panel["relationship_segment"].unique()).issubset(
        {"emerging", "core", "established", "high_value"}
    )

    assert set(panel["days_past_due_band"].unique()).issubset(
        {"no_credit_product", "current", "1-30", "31-60", "61+"}
    )


def test_synthetic_summary_matches_panel() -> None:
    """The summary helper should report core panel diagnostics correctly."""
    panel = _generate_compact_panel()
    summary = summarise_synthetic_panel(panel)

    assert summary["rows"] == len(panel)
    assert summary["customers"] == panel[CUSTOMER_ID_COLUMN].nunique()
    assert summary["months"] == panel[SNAPSHOT_MONTH_COLUMN].nunique()
    assert summary["columns"] == panel.shape[1]
    assert summary["missing_values"] == 0
    assert 0.05 <= summary["target_rate"] <= 0.15