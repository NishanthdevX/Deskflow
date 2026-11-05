"""Unit tests for Phase 14: Truth contract, five-command interface, and power checks."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from deskflow.cli import app
from deskflow.commands.check_power import _handle_fail_on, run_power_check
from deskflow.exit_codes import (
    EXIT_BLOCKED_OR_DECLINED,
    EXIT_ERROR,
    EXIT_OK,
)
from deskflow.truth import (
    ModelProvenance,
    compute_file_sha256,
    load_provenance,
    verify_model_hashes,
)

runner = CliRunner()


def test_model_provenance_loading() -> None:
    """Verify ModelProvenance loads successfully and inspects synthetic card properties."""
    prov = load_provenance()
    assert isinstance(prov, ModelProvenance)
    assert prov.synthetic is True
    assert prov.model_dir.exists()
    assert isinstance(prov.model_hashes, dict)
    assert len(prov.model_hashes) > 0
    assert "synthetic demo" in prov.footer_string().lower()


def test_verify_model_hashes_valid_and_tampered(tmp_path: Path) -> None:
    """Verify that verify_model_hashes passes on intact models and catches modifications."""
    prov = load_provenance()
    passed, errors = verify_model_hashes(prov.model_dir)
    assert passed is True
    assert errors == []

    # Tampered test in a temp dir
    fake_model = tmp_path / "model.onnx"
    fake_model.write_bytes(b"intact bytes")
    card_path = tmp_path / "model_card.json"
    card_data = {
        "synthetic": True,
        "models_sha256": {"model.onnx": compute_file_sha256(fake_model)},
    }
    card_path.write_text(json.dumps(card_data), encoding="utf-8")

    # Should pass
    p1, errs1 = verify_model_hashes(tmp_path)
    assert p1 is True

    # Tamper with the model file
    fake_model.write_bytes(b"corrupted bytes")
    p2, errs2 = verify_model_hashes(tmp_path)
    assert p2 is False
    assert len(errs2) == 1
    assert "checksum mismatch" in errs2[0].lower()


def test_truth_contract_synthetic_suppression(tmp_path: Path) -> None:
    """Verify user-facing reports suppress synthetic accuracy metrics and show truth banner."""
    # 1. Plain report
    res_plain = runner.invoke(app, ["report", "--plain"])
    assert res_plain.exit_code == EXIT_OK
    assert "NO REAL DATA YET" in res_plain.output
    normalized = " ".join(res_plain.output.split())
    assert "Run deskflow full to measure real accuracy on this machine." in normalized
    # No synthetic MAE numbers or feature weights
    assert "XGBoost: MAE" not in res_plain.output

    # 2. HTML report
    report_html_path = tmp_path / "report.html"
    res_html = runner.invoke(app, ["report", "--html", str(report_html_path)])
    assert res_html.exit_code == EXIT_OK
    assert report_html_path.exists()
    html_text = report_html_path.read_text(encoding="utf-8")
    assert "NO REAL DATA YET" in html_text

    # 3. Tools verify command
    res_verify = runner.invoke(app, ["tools", "verify"])
    assert res_verify.exit_code == EXIT_OK
    assert "PASS: Model files verified against card hashes" in res_verify.output


def test_default_help_primary_commands() -> None:
    """Verify deskflow --help lists only the 7 primary commands and tools."""
    res = runner.invoke(app, ["--help"])
    assert res.exit_code == EXIT_OK
    out = res.output

    # Visible primary commands
    assert "battery" in out
    assert "plugged" in out
    assert "full" in out
    assert "pipeline" in out
    assert "fix" in out
    assert "info" in out
    assert "tools" in out

    # Subcommands hidden from default help
    lines = [line.strip().split()[0] for line in out.splitlines() if line.strip().startswith("│ ")]
    # Hidden commands should not be listed as top-level commands
    for hidden_cmd in ("scan", "watch", "check", "prep", "dataset", "lab", "snapshot"):
        assert hidden_cmd not in lines


def test_legacy_commands_still_callable() -> None:
    """Verify hidden legacy commands still execute by name."""
    # 1. doctor
    res_doc = runner.invoke(app, ["doctor", "--plain"])
    assert res_doc.exit_code == EXIT_OK

    # 2. tools catalog
    res_tools = runner.invoke(app, ["tools"])
    assert res_tools.exit_code == EXIT_OK
    assert "[DESKFLOW ADVANCED TOOLS]" in res_tools.output
    assert "verify" in res_tools.output

    # 3. stubs full & pipeline
    res_full = runner.invoke(app, ["full"])
    assert res_full.exit_code == EXIT_OK
    assert "deskflow full: available in the next release (Phase 15)." in res_full.output

    res_pipe = runner.invoke(app, ["pipeline"])
    assert res_pipe.exit_code == EXIT_OK
    assert "deskflow pipeline: available in the next release (Phase 15)." in res_pipe.output


def test_power_check_preflight_mismatch() -> None:
    """Verify battery check aborts on AC, and plugged check aborts on battery with code 5."""
    mock_adapter_ac = MagicMock()
    mock_bat_ac = MagicMock()
    mock_bat_ac.plugged_in = True
    mock_bat_ac.percent = 99.0
    mock_adapter_ac.battery.return_value = mock_bat_ac

    with (
        patch("deskflow.commands.check_power.get_adapter", return_value=mock_adapter_ac),
        patch("deskflow.commands.check_power.get_current_power_state", return_value="ac"),
    ):
        with pytest.raises(Exception) as exc_info:
            run_power_check(target_state="battery")
        assert getattr(exc_info.value, "exit_code", None) == EXIT_BLOCKED_OR_DECLINED

    mock_adapter_bat = MagicMock()
    mock_bat_bat = MagicMock()
    mock_bat_bat.plugged_in = False
    mock_bat_bat.percent = 60.0
    mock_adapter_bat.battery.return_value = mock_bat_bat

    with (
        patch("deskflow.commands.check_power.get_adapter", return_value=mock_adapter_bat),
        patch("deskflow.commands.check_power.get_current_power_state", return_value="battery"),
    ):
        with pytest.raises(Exception) as exc_info:
            run_power_check(target_state="ac")
        assert getattr(exc_info.value, "exit_code", None) == EXIT_BLOCKED_OR_DECLINED


def test_fail_on_thresholds() -> None:
    """Verify --fail-on fair|poor threshold logic."""
    # Good health never triggers fail-on
    _handle_fail_on("Good", "fair")
    _handle_fail_on("Good", "poor")

    # Fair health triggers --fail-on fair but not --fail-on poor
    _handle_fail_on("Fair", "poor")
    with pytest.raises(Exception) as exc_fair:
        _handle_fail_on("Fair", "fair")
    assert getattr(exc_fair.value, "exit_code", None) == EXIT_ERROR

    # Poor health triggers both
    with pytest.raises(Exception) as exc_p1:
        _handle_fail_on("Poor", "fair")
    assert getattr(exc_p1.value, "exit_code", None) == EXIT_ERROR

    with pytest.raises(Exception) as exc_p2:
        _handle_fail_on("Poor", "poor")
    assert getattr(exc_p2.value, "exit_code", None) == EXIT_ERROR


def test_fix_effectiveness_table_columns(tmp_path: Path) -> None:
    """Verify fix effectiveness reports distinct columns for file bytes and closed app memory."""
    from deskflow.commands.report import _collect_fix_effectiveness

    sample_logs = [
        {
            "ts_utc": "2026-11-05T00:00:00Z",
            "dry_run": False,
            "actions": ["clean_temp", "close_idle_apps"],
            "results": [
                {
                    "tool_id": "clean_temp",
                    "success": True,
                    "bytes_freed": 104857600,  # 100 MB
                },
                {
                    "tool_id": "close_idle_apps",
                    "success": True,
                    "bytes_freed": 524288000,  # 500 MB app RSS
                },
            ],
            "before": {"available_memory_bytes": 4000000000},
            "after": {"available_memory_bytes": 4200000000},  # +200 MB
        }
    ]

    fixlog_file = tmp_path / "fixlog.jsonl"
    with fixlog_file.open("w", encoding="utf-8") as f:
        for entry in sample_logs:
            f.write(json.dumps(entry) + "\n")

    _, eff, _ = _collect_fix_effectiveness(fixlog_file)
    clean_stat = eff.get("clean_temp")
    assert clean_stat is not None
    assert clean_stat["bytes_deleted"] == 104857600
    assert clean_stat["app_memory_closed"] == 0

    close_stat = eff.get("close_idle_apps")
    assert close_stat is not None
    assert close_stat["bytes_deleted"] == 0
    assert close_stat["app_memory_closed"] == 524288000
