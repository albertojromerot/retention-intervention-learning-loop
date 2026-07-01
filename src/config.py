"""Project configuration for the synthetic retention intervention learning loop.

This module centralises project paths, synthetic-data profiles, model version
labels, file names, and reusable constants.

The project is a clean-room portfolio simulation. It must not contain or depend
on proprietary, confidential, employer, customer, member, financial, or
operational data from any real organisation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
SYNTHETIC_DATA_DIR = DATA_DIR / "synthetic"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
DOCS_DIR = PROJECT_ROOT / "docs"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
DASHBOARD_DIR = PROJECT_ROOT / "dashboard"


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

RANDOM_SEED = 42


# ---------------------------------------------------------------------------
# Core column names
# ---------------------------------------------------------------------------

CUSTOMER_ID_COLUMN = "customer_id"
SNAPSHOT_MONTH_COLUMN = "snapshot_month"
TARGET_COLUMN = "voluntary_attrition_next_90d"


# ---------------------------------------------------------------------------
# Synthetic run profiles
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SyntheticProfile:
    """Configuration for a synthetic data generation profile."""

    name: str
    n_customers: int
    n_months: int
    start_month: str


SYNTHETIC_PROFILES: dict[str, SyntheticProfile] = {
    "compact": SyntheticProfile(
        name="compact",
        n_customers=5_000,
        n_months=12,
        start_month="2024-01",
    ),
    "deep": SyntheticProfile(
        name="deep",
        n_customers=25_000,
        n_months=24,
        start_month="2024-01",
    ),
}

DEFAULT_PROFILE = "deep"


# ---------------------------------------------------------------------------
# Intervention policy configuration
# ---------------------------------------------------------------------------

INTERVENTION_CAPACITY_BY_PROFILE = {
    "compact": 250,
    "deep": 1_000,
}

TREATMENT_SHARE = 0.80
CONTROL_SHARE = 0.20

RISK_DECILES = 10


# ---------------------------------------------------------------------------
# Model and policy version labels
# ---------------------------------------------------------------------------

RULES_BASELINE_VERSION = "rules_baseline_v1"
MODEL_V1_VERSION = "ml_model_v1"
MODEL_V2_VERSION = "treatment_aware_ml_model_v2"


# ---------------------------------------------------------------------------
# Output file names
# ---------------------------------------------------------------------------

SYNTHETIC_PANEL_FILE = "synthetic_customer_month_panel.csv"
FEATURE_TABLE_FILE = "feature_table.csv"

RULES_BASELINE_METRICS_FILE = "metrics_rules_baseline.csv"
RULES_BASELINE_SCORED_FILE = "scored_customers_rules_baseline.csv"

MODEL_V1_METRICS_FILE = "metrics_model_v1.csv"
SCORED_CUSTOMERS_V1_FILE = "scored_customers_v1.csv"

INTERVENTION_LIST_V1_FILE = "intervention_list_v1.csv"
TREATMENT_LOG_V1_FILE = "treatment_log_v1.csv"

MODEL_V2_METRICS_FILE = "metrics_model_v2.csv"
TREATMENT_LEARNING_TABLE_V2_FILE = "treatment_learning_table_v2.csv"
SCORED_CUSTOMERS_V2_FILE = "scored_customers_v2.csv"

MODEL_COMPARISON_FILE = "model_comparison.csv"
GOVERNANCE_CHECKS_FILE = "governance_checks.csv"


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def get_profile(profile_name: str = DEFAULT_PROFILE) -> SyntheticProfile:
    """Return a synthetic data profile by name.

    Args:
        profile_name: Name of the synthetic profile.

    Returns:
        SyntheticProfile configuration.

    Raises:
        ValueError: If the profile name is unknown.
    """
    if profile_name not in SYNTHETIC_PROFILES:
        available_profiles = ", ".join(sorted(SYNTHETIC_PROFILES))
        raise ValueError(
            f"Unknown profile '{profile_name}'. "
            f"Available profiles: {available_profiles}"
        )

    return SYNTHETIC_PROFILES[profile_name]


def ensure_project_directories() -> None:
    """Create project directories required by the pipeline."""
    for directory in [
        DATA_DIR,
        SYNTHETIC_DATA_DIR,
        OUTPUTS_DIR,
        DOCS_DIR,
        NOTEBOOKS_DIR,
        DASHBOARD_DIR,
    ]:
        directory.mkdir(parents=True, exist_ok=True)