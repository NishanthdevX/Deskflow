"""Unit and integration tests for Phase 15.

Tests:
1. Shared prediction pair tracker engine with fake clock.
2. Block bootstrap reproducibility and confidence interval properties.
3. Pipeline static frame rendering (80x24, 100x30, no emoji, tables).
4. Full scan timeline execution in quick_test mode and artifact persistence.
5. Automated selftest on healthy installation and intentionally broken fixtures.
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from rich.console import Console

from deskflow.commands.full import compute_block_bootstrap, full_command
from deskflow.commands.pipeline import pipeline_command
from deskflow.commands.selftest import (
    check_command_surface,
    check_exit_codes,
    check_html_report,
    check_installer_dry_runs,
    check_json_purity,
    check_model_hashes,
    check_no_network_sockets,
    check_probe_stability,
    check_truth_contract,
)
from deskflow.commands.watch import PredictionPairTracker
from deskflow.truth import ModelProvenance


def test_prediction_engine_fake_clock() -> None:
    """Verify PredictionPairTracker matches future probe measurements deterministically."""
    tracker = PredictionPairTracker(horizon_s=20.0, tolerance_s=3.0)

    # Prediction made at t=100.0s for target horizon t=120.0s
    tracker.add_prediction(
        t_mono=100.0,
        t_utc="2026-11-05T00:00:00Z",
        pred_xgb=1.20,
        pred_rf=1.15,
        pred_ridge=1.10,
        persistence_ratio=1.00,
        pred_tag="cpu_pressure",
        top_factor="cpu_percent",
    )

    # Before horizon: probe at t=110.0s does not resolve
    tracker.process_probe(110.0, "2026-11-05T00:00:10Z", 1.05)
    assert len(tracker.resolved) == 0

    # At horizon: probe at t=120.5s resolves the prediction from t=100.0s
    tracker.process_probe(120.5, "2026-11-05T00:00:20Z", 1.25)
    assert len(tracker.resolved) == 1

    pair = tracker.resolved[0]
    assert pair["pred_xgb"] == 1.20
    assert pair["actual_ratio"] == 1.25
    assert pytest.approx(pair["abs_error"], 1e-4) == 0.05
    assert pytest.approx(pair["abs_error_rf"], 1e-4) == 0.10
    assert pytest.approx(pair["abs_error_ridge"], 1e-4) == 0.15
    assert pytest.approx(pair["persistence_error"], 1e-4) == 0.25

    board = tracker.compute_leaderboard()
    assert len(board) == 4
    # XGBoost should lead since its error is lowest (0.05)
    assert board[0]["model"] == "XGBoost"


def test_block_bootstrap_reproducibility() -> None:
    """Verify moving block bootstrap produces bitwise reproducible confidence intervals."""
    # Create synthetic series of resolved prediction pairs
    resolved_pairs: list[dict[str, Any]] = []
    for i in range(30):
        resolved_pairs.append(
            {
                "abs_error": 0.08 + 0.01 * (i % 5),
                "abs_error_rf": 0.10 + 0.01 * (i % 4),
                "abs_error_ridge": 0.12 + 0.01 * (i % 3),
                "persistence_error": 0.20 + 0.02 * (i % 6),
            }
        )

    res1 = compute_block_bootstrap(resolved_pairs, block_size=5, n_resamples=1000, seed=42)
    res2 = compute_block_bootstrap(resolved_pairs, block_size=5, n_resamples=1000, seed=42)

    assert res1 == res2
    assert res1["n"] == 30

    for model in ("XGBoost", "RandomForest", "Ridge", "Persistence"):
        ci = res1[model]["ci_95"]
        mae = res1[model]["mae"]
        assert ci[0] - 1e-6 <= mae <= ci[1] + 1e-6

    # Delta vs Persistence
    delta_info = res1["improvement_over_persistence"]
    assert delta_info["ci_95"][0] - 1e-6 <= delta_info["mean"] <= delta_info["ci_95"][1] + 1e-6
    assert delta_info["excludes_zero"] is True


def test_pipeline_static_frame_rendering() -> None:
    """Verify static rendering of pipeline diagram and tables with zero emoji."""
    buf = io.StringIO()
    con = Console(file=buf, width=100, no_color=True)

    with patch("deskflow.commands.pipeline.Console", return_value=con):
        pipeline_command(static=True, plain=True)

    output = buf.getvalue()
    assert "LIVE TELEMETRY & INFERENCE PIPELINE" in output
    assert "SENSORS" in output
    assert "PROBE" in output
    assert "FEATURES" in output
    assert "MODELS" in output
    assert "SCORE+" in output
    assert "FIX" in output
    assert "COLLECT (lab + normal)" in output
    assert "[ MODELS USED ]" in output
    assert "[ HOW IT IS TRAINED ]" in output

    # Check that no emoji exist
    from deskflow.commands.selftest import EMOJI_PATTERN

    assert len(EMOJI_PATTERN.findall(output)) == 0


def test_full_scan_quick_test(tmp_path: Path) -> None:
    """Verify full scan execution in quick_test mode and session artifact saving."""
    from deskflow.baseline import Baseline

    dummy_b = Baseline(
        machine_id="test",
        created_utc="2026-11-05T00:00:00Z",
        platform="TestOS",
        cpu_ns=10_000_000.0,
        mem_ns=2_000_000.0,
        disk_ns=3_000_000.0,
        total_ns=15_000_000.0,
        p10_total_ns=12_000_000.0,
        p90_total_ns=18_000_000.0,
        sample_count=20,
        deskflow_version="0.2.0",
        power_state="ac",
    )

    with (
        patch("deskflow.commands.full.load_baseline", return_value=dummy_b),
        patch("deskflow.commands.full.get_data_dir", return_value=tmp_path),
    ):
        full_command(yes=True, plain=True, quick_test=True)

        latest_json = tmp_path / "sessions" / "full_scan_latest.json"
        assert latest_json.exists()
        with open(latest_json, encoding="utf-8") as f:
            data = json.load(f)

        assert data["completed"] is True
        assert len(data["stage_results"]) == 5
        stages = [s["name"] for s in data["stage_results"]]
        assert stages == ["idle", "cpu ramp", "memory ramp", "disk", "background"]


def test_selftest_checks_healthy() -> None:
    """Verify that individual selftest checks pass on current healthy workspace."""
    # 1. Truth contract
    prov = ModelProvenance(
        synthetic=True,
        dataset_hash="",
        train_rows=1000,
        test_rows=300,
        sessions_count=8,
        machines_count=1,
        power_states=["ac"],
        created_utc="2026-11-05T00:00:00Z",
        git_commit="abc12345",
        model_hashes={},
        model_dir=Path("."),
    )
    res_truth = check_truth_contract(prov)
    assert res_truth.status == "PASS"

    # 2. Command surface
    res_cmd = check_command_surface()
    assert res_cmd.status == "PASS"

    # 3. JSON purity
    res_json = check_json_purity()
    assert res_json.status == "PASS"

    # 4. Exit codes
    res_exit = check_exit_codes()
    assert res_exit.status == "PASS"

    # 5. Installer dry runs
    res_inst = check_installer_dry_runs()
    assert res_inst.status == "PASS"

    # 6. HTML report
    res_html = check_html_report()
    assert res_html.status == "PASS"

    # 7. No network sockets
    res_net = check_no_network_sockets()
    assert res_net.status == "PASS"


def test_selftest_broken_fixtures(tmp_path: Path) -> None:
    """Verify that selftest correctly catches deliberately broken configurations."""
    # Broken Fixture 1: Modified model file / SHA-256 mismatch
    fake_model_dir = tmp_path / "models"
    fake_model_dir.mkdir()
    fake_file = fake_model_dir / "regressor_xgboost.joblib"
    fake_file.write_bytes(b"tampered content")

    broken_prov = ModelProvenance(
        synthetic=False,
        dataset_hash="11223344",
        train_rows=500,
        test_rows=100,
        sessions_count=8,
        machines_count=1,
        power_states=["ac"],
        created_utc="2026-11-05T00:00:00Z",
        model_hashes={"regressor_xgboost.joblib": "expected_valid_sha256_hash_here"},
        git_commit="deadbeef",
        model_dir=fake_model_dir,
    )

    res_hashes = check_model_hashes(broken_prov)
    assert res_hashes.status == "FAIL"
    assert "file hash mismatch" in res_hashes.measured

    # Broken Fixture 2: Probe stability check with high jitter
    with patch("deskflow.commands.selftest.run_probe") as mock_probe:
        # Create erratic latencies with p95 / median > 1.4 (1 warmup + 12 samples)
        mock_res1 = MagicMock(total_ns=10_000_000.0)
        mock_res_spike = MagicMock(total_ns=30_000_000.0)  # 3x spike
        mock_probe.side_effect = [mock_res1] * 11 + [mock_res_spike] * 4

        res_stability = check_probe_stability()
        assert res_stability.status == "FAIL"
        assert "exceeds threshold 1.40" in res_stability.reason
