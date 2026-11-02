"""Tests for fix audit log honesty: dry-run records must contain no measured fields."""

from pathlib import Path

from typer.testing import CliRunner

from deskflow.cli import app

runner = CliRunner()


def test_dry_run_log_honesty(mock_fix_environment: Path) -> None:
    """Dry-run records store estimated values in estimated_*; bytes_freed and after stay null."""
    import json

    res = runner.invoke(app, ["fix", "--action", "clean_temp", "--dry-run", "--yes"])
    assert res.exit_code == 0

    log_file = mock_fix_environment / "fixlog.jsonl"
    assert log_file.exists()

    record = json.loads(log_file.read_text(encoding="utf-8").strip().splitlines()[-1])
    assert record["dry_run"] is True
    assert record["after"] is None
    assert record["before"] is not None

    for item in record["results"]:
        # Must be null/None in dry run!
        assert item["bytes_freed"] is None
        assert "estimated_bytes_freed" in item
        assert isinstance(item["estimated_bytes_freed"], int)


def test_non_dry_run_log_records_measured_values(mock_fix_environment: Path) -> None:
    """Non-dry-run records store measured values in bytes_freed and populated after block."""
    import json

    res = runner.invoke(app, ["fix", "--action", "clean_temp", "--yes"])
    assert res.exit_code == 0

    log_file = mock_fix_environment / "fixlog.jsonl"
    assert log_file.exists()

    record = json.loads(log_file.read_text(encoding="utf-8").strip().splitlines()[-1])
    assert record["dry_run"] is False
    assert record["after"] is not None
    assert isinstance(record["after"]["score"], int)

    for item in record["results"]:
        assert isinstance(item["bytes_freed"], int)
        assert item["bytes_freed"] is not None
