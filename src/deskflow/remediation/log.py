"""Audit logging for remediation actions."""

import json
from pathlib import Path
from typing import Any

from deskflow.paths import get_data_dir


def append_fix_log(record: dict[str, Any], data_dir: Path | None = None) -> Path:
    """Append a fix execution record to <data_dir>/fixlog.jsonl.

    Args:
        record: Serialized dictionary describing executed fix actions,
            before/after state metrics, and success status.
        data_dir: Directory containing fixlog.jsonl (defaults to user data dir).

    Returns:
        Path to the appended fixlog.jsonl file.
    """
    target_dir = data_dir or get_data_dir()
    target_dir.mkdir(parents=True, exist_ok=True)
    log_file = target_dir / "fixlog.jsonl"
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
    return log_file
