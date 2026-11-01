"""Tests for the CLI entry point."""

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
