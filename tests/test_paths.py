"""Tests verifying test isolation and path resolution."""

import os
from pathlib import Path

from deskflow.paths import get_config_dir, get_data_dir


def test_paths_resolved_inside_temp_dir() -> None:
    """Verify that get_data_dir and get_config_dir resolve to the test temporary directory."""
    data_dir = get_data_dir()
    config_dir = get_config_dir()

    env_data = os.environ.get("DESKFLOW_DATA_DIR")
    env_config = os.environ.get("DESKFLOW_CONFIG_DIR")

    assert env_data is not None, "DESKFLOW_DATA_DIR must be set by conftest fixture"
    assert env_config is not None, "DESKFLOW_CONFIG_DIR must be set by conftest fixture"

    assert data_dir.resolve() == Path(env_data).resolve()
    assert config_dir.resolve() == Path(env_config).resolve()

    # Ensure the resolved directories exist and are writable
    test_file = data_dir / "write_test.tmp"
    test_file.write_text("ok", encoding="utf-8")
    assert test_file.exists()
    test_file.unlink()
