"""Tests for deskflow fix command, dry-run mode, confirmation, and log persistence."""

import json
from pathlib import Path

from typer.testing import CliRunner

from deskflow.cli import app

runner = CliRunner()


def test_fix_dry_run(mock_fix_environment: Path) -> None:
    """Test deskflow fix --dry-run prints plan and does not modify system."""
    res = runner.invoke(app, ["fix", "--action", "clean_temp", "--dry-run", "--yes"])
    assert res.exit_code == 0, f"Command failed: {res.output}"
    assert "Diagnosing current system state" in res.output
    assert "[Dry-run]" in res.output or "clean_temp" in res.output

    # Check that fixlog.jsonl was created with dry_run: true
    log_file = mock_fix_environment / "fixlog.jsonl"
    assert log_file.exists()
    lines = log_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["dry_run"] is True
    assert "clean_temp" in record["actions"]


def test_fix_execution_with_rescan(mock_fix_environment: Path) -> None:
    """Test deskflow fix with an action executes and performs re-scan."""
    res = runner.invoke(app, ["fix", "--action", "clean_temp", "--yes"])
    assert res.exit_code == 0, f"Command failed: {res.output}"
    assert "Executing actions" in res.output
    assert "Remediation Verification" in res.output
    assert "Health Score" in res.output

    # Check audit log
    log_file = mock_fix_environment / "fixlog.jsonl"
    assert log_file.exists()
    lines = log_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) >= 1
    record = json.loads(lines[-1])
    assert record["dry_run"] is False
    assert record["before"] is not None
    assert record["after"] is not None
