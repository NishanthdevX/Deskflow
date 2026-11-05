"""Tests for the CLI entry point and subcommands."""

import json

from typer.testing import CliRunner

from deskflow import __version__
from deskflow.cli import app

runner = CliRunner()


def test_version_exits_zero_and_prints_version() -> None:
    """--version should exit 0 and print the package version string."""
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert __version__ in result.output


def test_help_exits_zero() -> None:
    """--help should exit 0 and display usage information."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "Usage" in result.output
    assert "battery" in result.output
    assert "plugged" in result.output
    assert "tools" in result.output


def test_snapshot_json_output() -> None:
    """deskflow snapshot --json should exit 0 and output valid Record JSON."""
    result = runner.invoke(app, ["snapshot", "--json"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["schema_version"] == 1
    assert "sample" in data
    assert "probe" in data
    assert "top_processes" in data


def test_doctor_exits_zero() -> None:
    """deskflow doctor should exit 0 and report capabilities."""
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0
    assert "Platform:" in result.output
    assert "Verdict:" in result.output
