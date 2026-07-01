"""Basic project-structure tests for the initial scaffold."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_required_directories_exist() -> None:
    """The initial project directories should exist."""
    required_directories = [
        "src",
        "data",
        "data/synthetic",
        "outputs",
        "notebooks",
        "tests",
        "dashboard",
        "docs",
    ]

    for directory in required_directories:
        path = PROJECT_ROOT / directory
        assert path.exists(), f"Missing required directory: {directory}"
        assert path.is_dir(), f"Expected a directory: {directory}"


def test_required_root_files_exist() -> None:
    """The initial governance and setup files should exist."""
    required_files = [
        "README.md",
        "DISCLAIMER.md",
        "CLEAN_ROOM_STATEMENT.md",
        "DATA_PROVENANCE.md",
        "MODEL_CARD.md",
        "ETHICS_AND_GOVERNANCE.md",
        "LICENSE",
        "requirements.txt",
        ".gitignore",
    ]

    for file_name in required_files:
        path = PROJECT_ROOT / file_name
        assert path.exists(), f"Missing required file: {file_name}"
        assert path.is_file(), f"Expected a file: {file_name}"