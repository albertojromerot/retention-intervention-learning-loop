"""Tests for the project configuration module."""

from pathlib import Path

import pytest

from src.config import (
    CONTROL_SHARE,
    DEFAULT_PROFILE,
    INTERVENTION_CAPACITY_BY_PROFILE,
    PROJECT_ROOT,
    SYNTHETIC_PROFILES,
    TARGET_COLUMN,
    TREATMENT_SHARE,
    get_profile,
)


def test_project_root_exists() -> None:
    """The inferred project root should exist."""
    assert PROJECT_ROOT.exists()
    assert PROJECT_ROOT.is_dir()


def test_default_profile_is_available() -> None:
    """The default synthetic profile should be registered."""
    assert DEFAULT_PROFILE in SYNTHETIC_PROFILES


def test_profile_sizes_are_positive() -> None:
    """All synthetic profiles should define positive customer and month counts."""
    for profile in SYNTHETIC_PROFILES.values():
        assert profile.n_customers > 0
        assert profile.n_months > 0


def test_deep_profile_matches_expected_size() -> None:
    """The deep profile should match the planned portfolio-scale run."""
    profile = get_profile("deep")
    assert profile.n_customers == 25_000
    assert profile.n_months == 24


def test_invalid_profile_raises_error() -> None:
    """Unknown profiles should raise a clear error."""
    with pytest.raises(ValueError, match="Unknown profile"):
        get_profile("invalid_profile")


def test_treatment_and_control_shares_sum_to_one() -> None:
    """Treatment and control shares should form a complete assignment split."""
    assert TREATMENT_SHARE + CONTROL_SHARE == pytest.approx(1.0)


def test_intervention_capacity_defined_for_each_profile() -> None:
    """Every synthetic profile should have an intervention capacity setting."""
    for profile_name in SYNTHETIC_PROFILES:
        assert profile_name in INTERVENTION_CAPACITY_BY_PROFILE
        assert INTERVENTION_CAPACITY_BY_PROFILE[profile_name] > 0


def test_target_column_name_is_specific() -> None:
    """The target column should clearly describe the prediction window."""
    assert TARGET_COLUMN == "voluntary_attrition_next_90d"


def test_project_root_is_path_object() -> None:
    """Project root should be represented as a pathlib Path object."""
    assert isinstance(PROJECT_ROOT, Path)