"""Filesystem path helpers respecting environment overrides for test isolation."""

import os
from pathlib import Path

import platformdirs


def get_data_dir() -> Path:
    """Return root user data directory for deskflow.

    Honours DESKFLOW_DATA_DIR environment variable if set;
    otherwise falls back to platformdirs.user_data_dir.
    """
    env_dir = os.environ.get("DESKFLOW_DATA_DIR")
    if env_dir:
        p = Path(env_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p
    return Path(platformdirs.user_data_dir("deskflow", ensure_exists=True))


def get_config_dir() -> Path:
    """Return root user configuration directory for deskflow.

    Honours DESKFLOW_CONFIG_DIR environment variable if set;
    otherwise falls back to platformdirs.user_config_dir.
    """
    env_dir = os.environ.get("DESKFLOW_CONFIG_DIR")
    if env_dir:
        p = Path(env_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p
    return Path(platformdirs.user_config_dir("deskflow", ensure_exists=True))
