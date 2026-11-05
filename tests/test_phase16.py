"""Automated tests for Phase 16: Hard quality gates, build-model orchestration, and docs."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from typer.testing import CliRunner

# Ensure root scripts directory is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.update_readme_accuracy import generate_accuracy_markdown

from deskflow.cli import app as cli_app
from deskflow.ml.gates import (
    ALLOW_NAN_OR_OUT_OF_RANGE,
    MAX_VALID_PRED_RATIO,
    MIN_CLASS_ROWS,
    MIN_TEST_ROWS,
    MIN_TEST_SESSIONS,
    MIN_TRAIN_ROWS,
    MIN_VALID_PRED_RATIO,
    REQUIRE_BOTH_POWER_STATES,
    VALIDATE_PASSES,
    evaluate_dataset_gates,
    evaluate_prediction_gates,
)

runner = CliRunner()


def test_hard_gates_named_constants() -> None:
    """Verify that all quality gate constants are named, typed, and documented."""
    assert MIN_TRAIN_ROWS == 1000
    assert MIN_TEST_ROWS == 300
    assert MIN_TEST_SESSIONS == 2
    assert MIN_CLASS_ROWS == 100
    assert REQUIRE_BOTH_POWER_STATES is True
    assert ALLOW_NAN_OR_OUT_OF_RANGE is False
    assert VALIDATE_PASSES is True
    assert MIN_VALID_PRED_RATIO == 0.0
    assert MAX_VALID_PRED_RATIO == 20.0


def test_hard_gates_evaluation_passing() -> None:
    """Verify gate evaluation passes when all thresholds are met."""
    class_counts = {
        "healthy": 300,
        "cpu_pressure": 200,
        "memory_pressure": 200,
        "disk_pressure": 200,
    }
    gates = evaluate_dataset_gates(
        train_rows=1200,
        test_rows=400,
        test_sessions_count=2,
        class_counts=class_counts,
        power_states_train=["ac", "battery"],
        has_both_baselines=True,
        validation_errors=[],
    )

    assert all(g.passed for g in gates)
    assert len(gates) == 6


def test_hard_gates_evaluation_failing() -> None:
    """Verify gate evaluation catches insufficient rows, missing classes, and errors."""
    class_counts = {
        "healthy": 50,  # below 100
        "cpu_pressure": 200,
        "memory_pressure": 200,
        "disk_pressure": 200,
    }
    gates = evaluate_dataset_gates(
        train_rows=800,  # below 1000
        test_rows=150,  # below 300
        test_sessions_count=1,  # below 2
        class_counts=class_counts,
        power_states_train=["ac"],  # missing battery when both baselines exist
        has_both_baselines=True,
        validation_errors=["Schema violation: missing feature"],
    )

    failing = [g.name for g in gates if not g.passed]
    assert "Train Rows" in failing
    assert "Test Rows" in failing
    assert "Test Sessions" in failing
    assert "Class Balance" in failing
    assert "Power States" in failing
    assert "Dataset Validation" in failing


def test_prediction_gates_integrity() -> None:
    """Verify prediction gate checks for finite bounds and absence of NaN/Inf."""
    valid_preds = {
        "XGBoost": np.array([0.9, 1.2, 1.5, 2.0]),
        "RandomForest": np.array([1.0, 1.1, 1.4, 1.8]),
    }
    gates_valid = evaluate_prediction_gates(valid_preds)
    assert all(g.passed for g in gates_valid)

    nan_preds = {
        "XGBoost": np.array([1.0, np.nan, 1.5]),
    }
    gates_nan = evaluate_prediction_gates(nan_preds)
    assert not gates_nan[0].passed
    assert "contains NaN" in gates_nan[0].reason

    oor_preds = {
        "Ridge": np.array([1.0, 25.4, 1.5]),
    }
    gates_oor = evaluate_prediction_gates(oor_preds)
    assert not gates_oor[0].passed
    assert "25.40 > 20.0" in gates_oor[0].reason


def test_build_model_export_refusal_on_gate_failure(tmp_path: Path) -> None:
    """Verify build-model command refuses export and exits with code 1 if gates fail."""
    from deskflow.collector.models import SCHEMA_VERSION, Record, SystemSample

    fake_run_dir = tmp_path / "tiny_runs"
    fake_run_dir.mkdir()
    fake_run = fake_run_dir / "tiny_run.jsonl"
    fake_meta = fake_run_dir / "tiny_run.meta.json"

    # Write 5 records only (far below 1000 rows)
    sample = SystemSample(
        ts_utc="2026-11-05T00:00:00Z",
        monotonic_s=1.0,
        cpu_percent_mean=5.0,
        cpu_percent_max_core=10.0,
        cpu_freq_current_mhz=3000.0,
        cpu_freq_max_mhz=4000.0,
        cpu_freq_reliable=True,
        ctx_switches_per_s=1000.0,
        interrupts_per_s=500.0,
        mem_total_bytes=16 * (1024**3),
        mem_available_bytes=8 * (1024**3),
        mem_percent=50.0,
        swap_used_bytes=0,
        swap_percent=0.0,
        disk_read_bytes_per_s=0.0,
        disk_write_bytes_per_s=0.0,
        disk_busy_ms_per_s=0.0,
        net_sent_bytes_per_s=0.0,
        net_recv_bytes_per_s=0.0,
        net_errors_per_s=0.0,
        net_drops_per_s=0.0,
        process_count=100,
        thread_count=1000,
        battery_percent=100.0,
        battery_plugged=True,
        max_temperature_c=50.0,
        uptime_s=1000.0,
    )
    rec = Record(
        schema_version=SCHEMA_VERSION,
        run_id="tiny_run",
        label="healthy",
        sample=sample,
        probe=None,
        top_processes=[],
        storage=None,
    )
    fake_run.write_text(json.dumps(rec.to_dict()) + "\n")
    fake_meta.write_text(
        json.dumps(
            {
                "scenario": "idle",
                "intensity": "none",
                "pattern": "none",
                "machine_id": "test-mach",
                "session_id": "test-session",
                "deskflow_version": "0.2.0",
                "aborted": False,
                "worker_count": 0,
                "bytes_allocated": 0,
            }
        )
    )

    res = runner.invoke(
        cli_app,
        ["tools", "build-model", "--runs", str(fake_run_dir), "--export", "--plain"],
    )
    # Refusal should exit with error (code 1)
    assert res.exit_code != 0


def test_update_readme_accuracy_markdown() -> None:
    """Verify that update_readme_accuracy produces structured real benchmark markdown."""
    md = generate_accuracy_markdown()
    assert "Hardware Verification Dataset" in md
    assert "Held-Out Test Set Leaderboard" in md
    assert "XGBoost" in md
    assert "Superior" in md
    assert "Yes (Superior)" in md
