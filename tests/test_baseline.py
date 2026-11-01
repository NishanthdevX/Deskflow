"""Unit tests for machine identity and baseline calibration."""

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from typer.testing import CliRunner

from deskflow.baseline import (
    Baseline,
    CalibrationError,
    calibrate_baseline,
    get_or_create_machine_id,
    load_baseline,
    save_baseline,
)
from deskflow.cli import app
from deskflow.collector.probe import ProbeResult


def test_machine_id_generation_and_persistence(tmp_path: Path) -> None:
    """Verify machine ID is created once as a UUID4 and persisted."""
    id1 = get_or_create_machine_id(config_dir=tmp_path)
    assert len(id1) == 36
    assert (tmp_path / "machine.json").exists()

    # Re-reading should return identical UUID
    id2 = get_or_create_machine_id(config_dir=tmp_path)
    assert id1 == id2

    with open(tmp_path / "machine.json", encoding="utf-8") as f:
        data = json.load(f)
    assert data["machine_id"] == id1


def test_baseline_save_and_load(tmp_path: Path) -> None:
    """Verify Baseline dataclass serialization, saving, and loading."""
    b = Baseline(
        machine_id="test-uuid",
        created_utc="2026-11-01T12:00:00Z",
        platform="windows",
        cpu_ns=10_000_000.0,
        mem_ns=2_000_000.0,
        disk_ns=3_000_000.0,
        total_ns=15_000_000.0,
        p10_total_ns=14_000_000.0,
        p90_total_ns=16_000_000.0,
        sample_count=30,
        deskflow_version="0.1.0",
    )

    out_file = tmp_path / "baseline.json"
    saved_path = save_baseline(b, path=out_file)
    assert saved_path == out_file
    assert out_file.exists()

    loaded = load_baseline(path=out_file)
    assert loaded == b
    assert loaded.to_dict() == b.to_dict()


def test_calibrate_baseline_success(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify calibration with low-noise probes succeeds and computes medians."""
    monkeypatch.setattr("time.sleep", lambda _: None)
    # Produce 10 probes: first 3 warmup, 7 steady
    probe_seq = [
        ProbeResult(
            ts_utc="2026-11-01T12:00:00Z", cpu_ns=50e6, mem_ns=10e6, disk_ns=20e6, total_ns=80e6
        ),  # warmup 1
        ProbeResult(
            ts_utc="2026-11-01T12:00:02Z", cpu_ns=40e6, mem_ns=10e6, disk_ns=20e6, total_ns=70e6
        ),  # warmup 2
        ProbeResult(
            ts_utc="2026-11-01T12:00:04Z", cpu_ns=30e6, mem_ns=10e6, disk_ns=20e6, total_ns=60e6
        ),  # warmup 3
        # Retained 7 probes: stable total ~20ms
        ProbeResult(
            ts_utc="2026-11-01T12:00:06Z", cpu_ns=10e6, mem_ns=4e6, disk_ns=6e6, total_ns=20e6
        ),
        ProbeResult(
            ts_utc="2026-11-01T12:00:08Z", cpu_ns=11e6, mem_ns=4e6, disk_ns=6e6, total_ns=21e6
        ),
        ProbeResult(
            ts_utc="2026-11-01T12:00:10Z", cpu_ns=10e6, mem_ns=4e6, disk_ns=5e6, total_ns=19e6
        ),
        ProbeResult(
            ts_utc="2026-11-01T12:00:12Z", cpu_ns=10e6, mem_ns=4e6, disk_ns=6e6, total_ns=20e6
        ),
        ProbeResult(
            ts_utc="2026-11-01T12:00:14Z", cpu_ns=11e6, mem_ns=4e6, disk_ns=5e6, total_ns=20e6
        ),
        ProbeResult(
            ts_utc="2026-11-01T12:00:16Z", cpu_ns=10e6, mem_ns=4e6, disk_ns=6e6, total_ns=20e6
        ),
        ProbeResult(
            ts_utc="2026-11-01T12:00:18Z", cpu_ns=10e6, mem_ns=4e6, disk_ns=6e6, total_ns=20e6
        ),
    ]

    mock_probe = MagicMock(side_effect=probe_seq)
    out_file = tmp_path / "baseline.json"

    baseline = calibrate_baseline(
        duration_s=20.0,
        interval_s=2.0,
        warmup_samples=3,
        probe_fn=mock_probe,
        config_dir=tmp_path,
        out_path=out_file,
    )

    assert baseline.sample_count == 7
    assert baseline.total_ns == 20e6
    assert baseline.cpu_ns == 10e6
    assert baseline.mem_ns == 4e6
    assert baseline.disk_ns == 6e6
    assert out_file.exists()


def test_calibrate_baseline_refuses_when_noisy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify calibration refuses to save when coefficient of variation exceeds 0.5."""
    monkeypatch.setattr("time.sleep", lambda _: None)
    # Discard 3 warmup, then wildly varying totals: CV will exceed 0.5
    noisy_seq = [
        ProbeResult(
            ts_utc="2026-11-01T12:00:00Z", cpu_ns=10e6, mem_ns=5e6, disk_ns=5e6, total_ns=20e6
        ),
        ProbeResult(
            ts_utc="2026-11-01T12:00:02Z", cpu_ns=10e6, mem_ns=5e6, disk_ns=5e6, total_ns=20e6
        ),
        ProbeResult(
            ts_utc="2026-11-01T12:00:04Z", cpu_ns=10e6, mem_ns=5e6, disk_ns=5e6, total_ns=20e6
        ),
        # Usable:
        ProbeResult(
            ts_utc="2026-11-01T12:00:06Z", cpu_ns=10e6, mem_ns=5e6, disk_ns=5e6, total_ns=10e6
        ),
        ProbeResult(
            ts_utc="2026-11-01T12:00:08Z", cpu_ns=100e6, mem_ns=50e6, disk_ns=50e6, total_ns=200e6
        ),
        ProbeResult(
            ts_utc="2026-11-01T12:00:10Z", cpu_ns=10e6, mem_ns=5e6, disk_ns=5e6, total_ns=15e6
        ),
        ProbeResult(
            ts_utc="2026-11-01T12:00:12Z", cpu_ns=200e6, mem_ns=50e6, disk_ns=50e6, total_ns=300e6
        ),
    ]

    mock_probe = MagicMock(side_effect=noisy_seq)
    out_file = tmp_path / "baseline.json"

    with pytest.raises(CalibrationError, match="exceeded 0.5 threshold"):
        calibrate_baseline(
            duration_s=14.0,
            interval_s=2.0,
            warmup_samples=3,
            probe_fn=mock_probe,
            config_dir=tmp_path,
            out_path=out_file,
        )

    # File must not be saved
    assert not out_file.exists()


def test_baseline_cli_commands(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify CLI baseline commands show and calibrate."""
    runner = CliRunner()
    baseline_path = tmp_path / "baseline.json"

    # Show on missing file
    res = runner.invoke(app, ["baseline", "show", "--path", str(baseline_path)])
    assert res.exit_code == 1

    # Calibrate with mocked low-noise probe
    monkeypatch.setattr(
        "deskflow.commands.baseline.calibrate_baseline",
        lambda **kwargs: Baseline(
            machine_id="mock-id",
            created_utc="2026-11-01T12:00:00Z",
            platform="windows",
            cpu_ns=10e6,
            mem_ns=5e6,
            disk_ns=5e6,
            total_ns=20e6,
            p10_total_ns=19e6,
            p90_total_ns=21e6,
            sample_count=10,
            deskflow_version="0.1.0",
        ),
    )

    cal_res = runner.invoke(
        app, ["baseline", "calibrate", "--duration", "6", "--out", str(baseline_path)]
    )
    assert cal_res.exit_code == 0
    assert "Baseline calibration successful" in cal_res.stdout

    # Show on existing file
    save_baseline(
        Baseline(
            machine_id="mock-id",
            created_utc="2026-11-01T12:00:00Z",
            platform="windows",
            cpu_ns=10e6,
            mem_ns=5e6,
            disk_ns=5e6,
            total_ns=20e6,
            p10_total_ns=19e6,
            p90_total_ns=21e6,
            sample_count=10,
            deskflow_version="0.1.0",
        ),
        path=baseline_path,
    )
    show_res = runner.invoke(app, ["baseline", "show", "--path", str(baseline_path)])
    assert show_res.exit_code == 0
    assert "mock-id" in show_res.stdout


def test_baseline_calibrate_cli_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify CLI baseline calibrate exits with code 1 when CalibrationError is raised."""
    runner = CliRunner()

    def _fail_cal(**kwargs):
        raise CalibrationError("CV exceeded")

    monkeypatch.setattr("deskflow.commands.baseline.calibrate_baseline", _fail_cal)
    res = runner.invoke(app, ["baseline", "calibrate", "--duration", "6"])
    assert res.exit_code == 1
    assert "Calibration rejected" in res.stdout


def test_machine_id_corrupt_file_recovery(tmp_path: Path) -> None:
    """Verify corrupted machine.json is overwritten with a valid UUID."""
    bad_file = tmp_path / "machine.json"
    bad_file.write_text("{corrupt json", encoding="utf-8")

    mid = get_or_create_machine_id(config_dir=tmp_path)
    assert len(mid) == 36


def test_load_baseline_missing_raises(tmp_path: Path) -> None:
    """Verify load_baseline raises FileNotFoundError when path is missing."""
    with pytest.raises(FileNotFoundError, match="Baseline file not found"):
        load_baseline(path=tmp_path / "nonexistent.json")


def test_calibrate_progress_callback(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify calibrate_baseline invokes progress callback on each tick."""
    monkeypatch.setattr("time.sleep", lambda _: None)
    calls = []

    def _cb(idx: int, total: int, lat_ms: float) -> None:
        calls.append((idx, total, lat_ms))

    mock_probe = MagicMock(
        return_value=ProbeResult(
            ts_utc="2026-11-01T12:00:00Z", cpu_ns=10e6, mem_ns=2e6, disk_ns=3e6, total_ns=15e6
        )
    )
    calibrate_baseline(
        duration_s=12.0,
        interval_s=2.0,
        warmup_samples=1,
        probe_fn=mock_probe,
        config_dir=tmp_path,
        progress_callback=_cb,
    )
    assert len(calls) == 6
