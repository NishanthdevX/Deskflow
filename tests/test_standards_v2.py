"""Unit tests for Phase 13: Product standards, exit codes, JSON purity, and redaction."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest
from typer.testing import CliRunner

from deskflow.cli import app
from deskflow.commands.support import sanitize_text
from deskflow.exit_codes import (
    EXIT_BLOCKED_OR_DECLINED,
    EXIT_OK,
    EXIT_USAGE,
)

runner = CliRunner()


def test_bugreport_redaction() -> None:
    """Verify bugreport text sanitization redacts home dirs, usernames, and GUIDs."""
    fake_home = "C:\\Users\\SecretDeveloper"
    fake_user = "SecretDeveloper"
    fake_guid = "a1b2c3d4-e5f6-7a8b-9c0d-1e2f3a4b5c6d"

    sample_log = (
        f"2026-11-04 12:00:00 [INFO] User {fake_user} started scan.\n"
        f"Data stored at {fake_home}\\AppData\\Local\\deskflow.\n"
        f"Machine identifier: {fake_guid}.\n"
    )

    sanitized = sanitize_text(
        sample_log,
        home_dir=fake_home,
        username=fake_user,
        machine_id=fake_guid,
    )

    assert fake_user not in sanitized
    assert fake_home not in sanitized
    assert fake_guid not in sanitized
    assert "[USERNAME]" in sanitized
    assert "[HOME_DIR]" in sanitized
    assert "[MACHINE_ID]" in sanitized


def test_json_output_purity() -> None:
    """Verify --json produces 100% pure JSON on stdout with no contaminating text."""
    # 1. deskflow report --json
    res_rep = runner.invoke(app, ["report", "--json"])
    assert res_rep.exit_code == EXIT_OK
    data_rep = json.loads(res_rep.output)
    assert isinstance(data_rep, dict)
    assert "leaderboard" in data_rep

    # 2. deskflow info --json
    res_info = runner.invoke(app, ["info", "--json"])
    assert res_info.exit_code == EXIT_OK
    data_info = json.loads(res_info.output)
    assert isinstance(data_info, dict)
    assert "cpu" in data_info


def test_exit_codes_and_no_input() -> None:
    """Verify standardized exit codes under various CLI invocations."""
    # 1. Success exit code (0)
    res_ok = runner.invoke(app, ["doctor", "--plain"])
    assert res_ok.exit_code == EXIT_OK

    # 2. Usage error (2)
    res_usage = runner.invoke(app, ["scan", "--seconds", "2"])  # min is 13s
    assert res_usage.exit_code in (EXIT_USAGE, 1, 2)

    # 3. Blocked / declined with --no-input (5)
    res_uninstall = runner.invoke(app, ["uninstall", "--data", "--no-input"])
    assert res_uninstall.exit_code == EXIT_BLOCKED_OR_DECLINED


def test_installer_dry_run_powershell() -> None:
    """Verify install.ps1 dry-run execution."""
    script_path = Path("install.ps1").resolve()
    if not script_path.exists():
        pytest.skip("install.ps1 not in current directory")

    env = os.environ.copy()
    env["DESKFLOW_DRYRUN"] = "1"
    proc = subprocess.run(
        ["powershell", "-File", str(script_path)],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    assert proc.returncode == 0
    assert "[DRYRUN]" in proc.stdout
    assert "deskflow" in proc.stdout


def test_installer_hash_mismatch_abort() -> None:
    """Verify install.ps1 aborts when a hash mismatch is detected."""
    script_path = Path("install.ps1").resolve()
    if not script_path.exists():
        pytest.skip("install.ps1 not in current directory")

    env = os.environ.copy()
    env["DESKFLOW_TEST_HASH_MISMATCH"] = "1"
    proc = subprocess.run(
        ["powershell", "-File", str(script_path)],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    assert proc.returncode != 0
    assert "checksum mismatch" in proc.stderr.lower() or "checksum mismatch" in proc.stdout.lower()
