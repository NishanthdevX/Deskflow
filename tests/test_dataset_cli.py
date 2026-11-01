"""Unit tests for dataset CLI commands: build, describe, validate, and merge."""

import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

from deskflow.baseline import Baseline, save_baseline
from deskflow.cli import app
from deskflow.collector.models import ProbeResult, Record, StorageReport, SystemSample
from deskflow.features.builder import build_dataset, merge_datasets
from deskflow.features.splits import assign_splits


def _create_synthetic_run(
    run_dir: Path, run_id: str, duration_s: int = 120, label: str = "idle"
) -> None:
    """Helper to write a valid synthetic run with 120s of records."""
    run_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = run_dir / f"{run_id}.jsonl"
    meta_path = run_dir / f"{run_id}.meta.json"

    records: list[Record] = []
    base_epoch = 1700000000.0
    for sec in range(0, duration_s, 2):
        mono = float(sec)
        dt_str = datetime.fromtimestamp(base_epoch + sec, tz=UTC).isoformat()
        sample = SystemSample(
            ts_utc=dt_str,
            monotonic_s=mono,
            cpu_percent_mean=20.0,
            cpu_percent_max_core=30.0,
            cpu_freq_current_mhz=2500.0,
            cpu_freq_max_mhz=3000.0,
            cpu_freq_reliable=True,
            ctx_switches_per_s=1000.0,
            interrupts_per_s=500.0,
            mem_total_bytes=16 * (1024**3),
            mem_available_bytes=8 * (1024**3),
            mem_percent=50.0,
            swap_used_bytes=1024**2,
            swap_percent=1.0,
            disk_read_bytes_per_s=1024.0,
            disk_write_bytes_per_s=2048.0,
            disk_busy_ms_per_s=50.0,
            net_recv_bytes_per_s=500.0,
            net_sent_bytes_per_s=500.0,
            net_errors_per_s=0.0,
            net_drops_per_s=0.0,
            process_count=150,
            thread_count=1200,
            battery_percent=85.0,
            battery_plugged=True,
            max_temperature_c=55.0,
            uptime_s=1000.0,
        )
        probe = None
        if sec % 4 == 0:
            probe = ProbeResult(
                ts_utc=dt_str,
                cpu_ns=20_000_000,
                mem_ns=2_000_000,
                disk_ns=5_000_000,
                total_ns=27_000_000,
            )
        rec = Record(
            schema_version=1,
            run_id=run_id,
            label=label,
            sample=sample,
            probe=probe,
            top_processes=[],
            storage=StorageReport(ts_utc=dt_str, entries=[], truncated=False, duration_s=0.1),
        )
        records.append(rec)

    with open(jsonl_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r.to_dict()) + "\n")

    meta_dict = {
        "scenario": label,
        "intensity": "none",
        "pattern": "none",
        "machine_id": "test-machine",
        "session_id": f"session-{run_id}",
        "deskflow_version": "0.1.0",
        "aborted": False,
        "actual": {
            "pre_start": "2026-11-01T12:00:00Z",
            "inject_start": "2026-11-01T12:00:30Z",
            "inject_end": "2026-11-01T12:01:30Z",
            "post_end": "2026-11-01T12:02:00Z",
        },
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta_dict, f)


@pytest.fixture
def dummy_baseline(tmp_path: Path) -> Path:
    b = Baseline(
        machine_id="test-machine",
        created_utc="2026-11-01T12:00:00Z",
        platform="windows",
        cpu_ns=20e6,
        mem_ns=2e6,
        disk_ns=5e6,
        total_ns=27e6,
        p10_total_ns=25e6,
        p90_total_ns=30e6,
        sample_count=20,
        deskflow_version="0.1.0",
    )
    b_path = tmp_path / "baseline.json"
    save_baseline(b, path=b_path)
    return b_path


def test_dataset_cli_build_describe_validate(tmp_path: Path, dummy_baseline: Path) -> None:
    """End-to-end test of dataset build, describe, and validate CLI commands."""
    runner = CliRunner()
    runs_dir = tmp_path / "runs"
    _create_synthetic_run(runs_dir, "run_01", duration_s=120)

    out_parquet = tmp_path / "dataset.parquet"

    # 1. deskflow dataset build
    build_res = runner.invoke(
        app,
        [
            "dataset",
            "build",
            "--runs",
            str(runs_dir),
            "--baseline",
            str(dummy_baseline),
            "--horizon",
            "30",
            "--out",
            str(out_parquet),
            "--keep-transition",
        ],
    )
    assert build_res.exit_code == 0, build_res.stdout
    assert "Per-Run Build Summary" in build_res.stdout
    assert "run_01" in build_res.stdout
    assert "Dataset built successfully." in build_res.stdout
    assert out_parquet.exists()

    # 2. deskflow dataset describe
    desc_res = runner.invoke(app, ["dataset", "describe", str(out_parquet)])
    assert desc_res.exit_code == 0, desc_res.stdout
    assert "Dataset Summary:" in desc_res.stdout
    assert "Total Rows" in desc_res.stdout

    # 3. deskflow dataset validate
    val_res = runner.invoke(app, ["dataset", "validate", str(out_parquet)])
    assert val_res.exit_code == 0, val_res.stdout
    assert "Dataset validation passed" in val_res.stdout


def test_dataset_cli_validate_failure(tmp_path: Path) -> None:
    """Verify deskflow dataset validate detects missing columns and issues errors."""
    runner = CliRunner()
    bad_df = pd.DataFrame({"cpu_percent_mean_mean_10s": [1.0], "run_id": ["r1"]})
    bad_parquet = tmp_path / "invalid.parquet"
    bad_df.to_parquet(bad_parquet)

    val_res = runner.invoke(app, ["dataset", "validate", str(bad_parquet)])
    assert val_res.exit_code == 1
    assert "Dataset validation failed" in val_res.stdout


def test_dataset_cli_describe_missing_file(tmp_path: Path) -> None:
    """Verify deskflow dataset describe handles missing file."""
    runner = CliRunner()
    res = runner.invoke(app, ["dataset", "describe", str(tmp_path / "missing.parquet")])
    assert res.exit_code == 1
    assert "File not found" in res.stdout


def test_dataset_cli_merge(tmp_path: Path, dummy_baseline: Path) -> None:
    """Verify deskflow dataset merge combines multiple datasets and applies splits."""
    runner = CliRunner()
    runs_dir_a = tmp_path / "runs_a"
    runs_dir_b = tmp_path / "runs_b"
    _create_synthetic_run(runs_dir_a, "run_a", duration_s=120)
    _create_synthetic_run(runs_dir_b, "run_b", duration_s=120)

    ds_a = tmp_path / "ds_a.parquet"
    ds_b = tmp_path / "ds_b.parquet"
    build_dataset([runs_dir_a], dummy_baseline, horizon_s=30.0, out_path=ds_a)
    build_dataset([runs_dir_b], dummy_baseline, horizon_s=30.0, out_path=ds_b)

    merged_out = tmp_path / "merged.parquet"
    res = runner.invoke(app, ["dataset", "merge", str(ds_a), str(ds_b), "--out", str(merged_out)])
    assert res.exit_code == 0, res.stdout
    assert "Merging 2 dataset files" in res.stdout
    assert "Total rows:" in res.stdout
    assert merged_out.exists()


def test_dataset_merge_differing_horizons_fails(tmp_path: Path, dummy_baseline: Path) -> None:
    """Verify merge_datasets raises ValueError when horizons do not match."""
    runs_dir = tmp_path / "runs"
    _create_synthetic_run(runs_dir, "run_x", duration_s=120)

    ds_1 = tmp_path / "ds_1.parquet"
    ds_2 = tmp_path / "ds_2.parquet"
    build_dataset([runs_dir], dummy_baseline, horizon_s=10.0, out_path=ds_1)
    build_dataset([runs_dir], dummy_baseline, horizon_s=30.0, out_path=ds_2)

    with pytest.raises(ValueError, match="Cannot merge datasets with differing horizons"):
        merge_datasets([ds_1, ds_2], tmp_path / "fail.parquet")


def test_splits_edge_cases() -> None:
    """Verify assign_splits behavior for empty dataframes and all-test selection."""
    empty_df = pd.DataFrame()
    res, strategy = assign_splits(empty_df)
    assert strategy == "empty"

    # Multi-session where all sessions get selected by test_ratio -> keep 1 train
    df = pd.DataFrame(
        {
            "session_id": ["s1", "s2"],
            "run_id": ["r1", "r2"],
            "ts_utc": ["2026-11-01T12:00:00Z", "2026-11-01T12:01:00Z"],
        }
    )
    res_df, strat = assign_splits(df, test_ratio=1.0)
    assert strat == "session"
    assert (res_df["split"] == "train").sum() >= 1
