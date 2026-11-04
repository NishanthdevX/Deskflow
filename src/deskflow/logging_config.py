"""Logging infrastructure for Deskflow with rotating file handler.

Logs rotating output to <data_dir>/logs/deskflow.log (1 MB x 3 backups).
"""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from deskflow.paths import get_data_dir


class _LoggingState:
    initialized: bool = False


def setup_logging(verbose: bool = False, debug: bool = False) -> Path:
    """Initialize rotating file logger and optional console logging."""
    log_dir = get_data_dir() / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "deskflow.log"

    if _LoggingState.initialized:
        return log_file

    level = logging.DEBUG if (debug or verbose) else logging.INFO

    # Root deskflow logger
    root_logger = logging.getLogger("deskflow")
    root_logger.setLevel(level)

    # 1 MB x 3 rotating file handler
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=1_048_576,  # 1 MB
        backupCount=3,
        encoding="utf-8",
    )
    file_handler.setLevel(level)
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    _LoggingState.initialized = True
    return log_file


def get_log_file_path() -> Path:
    """Return path to the active deskflow.log file."""
    return get_data_dir() / "logs" / "deskflow.log"
