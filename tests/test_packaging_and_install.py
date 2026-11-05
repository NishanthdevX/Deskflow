"""Tests for packaging, model export, model resolution, and installer behavior."""

import os
import subprocess
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

from deskflow.commands.models import models_app
from deskflow.ml.artifacts import (
    get_bundled_model_dir,
    resolve_model_dir,
)

runner = CliRunner()


def test_bundled_model_dir_exists() -> None:
    """Verify that default pre-bundled models exist and contain a valid model card."""
    bundled = get_bundled_model_dir()
    assert bundled.exists(), f"Bundled models directory missing: {bundled}"
    assert (bundled / "model_card.json").exists(), "Bundled model_card.json missing"
    assert (bundled / "regressor_xgb.ubj").exists(), "Bundled regressor_xgb.ubj missing"


def test_resolve_model_dir_fallback(tmp_path: Path) -> None:
    """Verify model resolution falls back to bundled default when user models are empty."""
    empty_root = tmp_path / "models"
    empty_root.mkdir()

    resolved_path, is_bundled = resolve_model_dir(models_root=empty_root)
    assert is_bundled is True
    assert (resolved_path / "model_card.json").exists()


def test_models_export_cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test 'deskflow models export' CLI command with synthetic check."""
    out_dir = tmp_path / "exported"

    # Mock load_model_card to simulate synthetic model set refusal
    import deskflow.commands.models as models_cmd

    orig_load_card = models_cmd.load_model_card
    monkeypatch.setattr(
        models_cmd,
        "load_model_card",
        lambda p: {**orig_load_card(p), "synthetic": True},
    )

    # 1. Export synthetic without allow-synthetic should fail
    result_refuse = runner.invoke(
        models_app,
        ["export", "--to", str(out_dir)],
    )
    assert result_refuse.exit_code != 0
    assert "synthetic" in result_refuse.output.lower()

    # 2. Export with allow-synthetic should succeed
    result_ok = runner.invoke(
        models_app,
        ["export", "--to", str(out_dir), "--allow-synthetic"],
    )
    assert result_ok.exit_code == 0
    assert (out_dir / "model_card.json").exists()
    assert (out_dir / "regressor_xgb.ubj").exists()


@pytest.mark.skipif(sys.platform != "win32", reason="PowerShell test runs on Windows")
def test_install_ps1_dryrun() -> None:
    """Test that install.ps1 executes cleanly in dryrun mode on Windows."""
    repo_root = Path(__file__).resolve().parent.parent
    ps1_path = repo_root / "install.ps1"
    assert ps1_path.exists()

    env = os.environ.copy()
    env["DESKFLOW_DRYRUN"] = "1"
    env["DESKFLOW_ARGS"] = "--version"

    proc = subprocess.run(
        ["powershell", "-ExecutionPolicy", "Bypass", "-File", str(ps1_path)],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    assert proc.returncode == 0
    assert "[DRYRUN]" in proc.stdout
    assert "NishanthdevX/Deskflow" in proc.stdout
