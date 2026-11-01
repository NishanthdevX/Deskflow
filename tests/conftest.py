"""Pytest configuration and shared fixtures."""

import os
from collections.abc import Generator
from pathlib import Path

import pytest


@pytest.fixture(scope="session", autouse=True)
def isolate_deskflow_dirs(
    tmp_path_factory: pytest.TempPathFactory,
) -> Generator[tuple[Path, Path], None, None]:
    """Isolate DESKFLOW user data and configuration directories during testing."""
    test_data_dir = tmp_path_factory.mktemp("deskflow_test_data")
    test_config_dir = tmp_path_factory.mktemp("deskflow_test_config")

    old_data = os.environ.get("DESKFLOW_DATA_DIR")
    old_config = os.environ.get("DESKFLOW_CONFIG_DIR")

    os.environ["DESKFLOW_DATA_DIR"] = str(test_data_dir)
    os.environ["DESKFLOW_CONFIG_DIR"] = str(test_config_dir)

    yield test_data_dir, test_config_dir

    if old_data is not None:
        os.environ["DESKFLOW_DATA_DIR"] = old_data
    else:
        os.environ.pop("DESKFLOW_DATA_DIR", None)

    if old_config is not None:
        os.environ["DESKFLOW_CONFIG_DIR"] = old_config
    else:
        os.environ.pop("DESKFLOW_CONFIG_DIR", None)
