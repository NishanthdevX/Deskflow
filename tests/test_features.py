"""Unit tests for feature engineering, window math, no-lookahead assertions, and splits."""

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from deskflow.baseline import Baseline
from deskflow.collector.models import (
    ProbeResult,
    ProcessSample,
    Record,
    StorageReport,
    SystemSample,
)
from deskflow.features.builder import (
    build_dataset,
    describe_dataset,
    validate_dataset,
)
from deskflow.features.extract import extract_run_features
from deskflow.features.registry import (
    FEATURE_NAMES,
    FEATURE_SET_VERSION,
    FEATURE_SPECS,
)
from deskflow.features.splits import assign_splits
from deskflow.features.window import (
    compute_drop,
    compute_max,
    compute_mean,
    compute_min,
    compute_slope,
    compute_std,
)


def _make_sample(
    mono: float, ts_utc: str, mem_pct: float = 50.0, cpu_pct: float = 20.0
) -> SystemSample:
    return SystemSample(
        ts_utc=ts_utc,
        monotonic_s=mono,
        cpu_percent_mean=cpu_pct,
        cpu_percent_max_core=cpu_pct * 1.5,
        cpu_freq_current_mhz=2500.0,
        cpu_freq_max_mhz=3000.0,
        cpu_freq_reliable=True,
        ctx_switches_per_s=1000.0,
        interrupts_per_s=500.0,
        mem_total_bytes=16 * (1024**3),
        mem_available_bytes=int(16 * (1024**3) * (1.0 - mem_pct / 100.0)),
        mem_percent=mem_pct,
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


def _make_dummy_baseline() -> Baseline:
    return Baseline(
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


def test_window_statistics_math() -> None:
    """Verify window statistics: slope, mean, max, drop with linear and constant data."""
    # Constant series -> zero slope
    ts = [0.0, 1.0, 2.0, 3.0, 4.0]
    vals = [10.0, 10.0, 10.0, 10.0, 10.0]
    assert compute_slope(ts, vals) == 0.0
    assert compute_mean(vals) == 10.0
    assert compute_std(vals) == 0.0
    assert compute_drop(vals) == 0.0

    # Linear series: y = 2.5 * t + 1.0 -> slope = 2.5
    y_linear = [2.5 * t + 1.0 for t in ts]
    assert pytest.approx(compute_slope(ts, y_linear), rel=1e-5) == 2.5
    assert compute_max(y_linear) == 11.0
    assert compute_min(y_linear) == 1.0
    assert compute_drop(y_linear) == 0.0

    # Dropping series: max 10.0, last 4.0 -> drop 6.0
    y_drop = [10.0, 8.0, 6.0, 4.0]
    assert compute_drop(y_drop) == 6.0

    # Irregular timestamps
    ts_irreg = [0.0, 0.5, 3.0, 4.0]
    ys_irreg = [0.0, 1.0, 6.0, 8.0]  # y = 2 * t
    assert pytest.approx(compute_slope(ts_irreg, ys_irreg), rel=1e-5) == 2.0


def test_feature_registry_consistency() -> None:
    """Verify feature registry versioning and uniqueness."""
    assert FEATURE_SET_VERSION == 1
    names = [spec.name for spec in FEATURE_SPECS]
    assert len(names) == len(set(names)), "Feature names must be strictly unique"
    assert len(FEATURE_NAMES) == len(FEATURE_SPECS)


def test_warmup_drop_and_no_lookahead() -> None:
    """Verify no-lookahead guarantee and that points before 60s are dropped."""
    baseline = _make_dummy_baseline()
    records: list[Record] = []

    # Generate 120s of continuous records every 2s, probe every 4s
    for sec in range(0, 122, 2):
        mono = float(sec)
        dt_str = datetime.fromtimestamp(1700000000 + sec, tz=UTC).isoformat()
        sample = _make_sample(mono=mono, ts_utc=dt_str, mem_pct=40.0 + (sec * 0.1))
        probe = None
        if sec % 4 == 0:
            probe = ProbeResult(
                ts_utc=dt_str,
                cpu_ns=20_000_000,
                mem_ns=2_000_000,
                disk_ns=5_000_000,
                total_ns=27_000_000,
            )

        top_procs = [
            ProcessSample(
                pid=100,
                name="app.exe",
                rss_bytes=1024**3,
                cpu_percent=15.0,
                num_threads=10,
                age_s=700.0,
                is_protected=False,
            )
        ]
        storage = StorageReport(ts_utc=dt_str, entries=[], truncated=False, duration_s=0.1)

        rec = Record(
            schema_version=1,
            run_id="run-test",
            label="idle",
            sample=sample,
            probe=probe,
            top_processes=top_procs,
            storage=storage,
        )
        records.append(rec)

    # Extract features from full series (horizon 30s)
    full_rows = extract_run_features(records, baseline=baseline, horizon_s=30.0)
    assert len(full_rows) > 0

    # Decision points before 60s must be dropped
    first_row_ts = full_rows[0]["ts_utc"]
    first_epoch = datetime.fromisoformat(first_row_ts).timestamp()
    assert first_epoch - 1700000000 >= 60.0

    # No-lookahead test:
    # Pick a decision record around t=80s.
    t_target_idx = [i for i, r in enumerate(records) if r.sample.monotonic_s == 80.0][0]
    records[: t_target_idx + 1]

    # Target alignment needs probe up to t + H (80 + 30 = 110s)
    # The feature computation at t=80 depends ONLY on data <= 80s!
    # Let's extract row for t=80s in full_rows:
    row_full_80 = [
        r for r in full_rows if datetime.fromisoformat(r["ts_utc"]).timestamp() == 1700000080
    ][0]

    # Now verify that every feature in FEATURE_NAMES matches when computed with only data <= 80s
    records_up_to_80 = [r for r in records if r.sample.monotonic_s <= 80.0]
    # To satisfy target alignment at 110s for the test assertion:
    records_with_target = records_up_to_80 + [
        r for r in records if r.sample.monotonic_s in {110.0, 112.0}
    ]
    row_trunc_80 = extract_run_features(records_with_target, baseline=baseline, horizon_s=30.0)[-1]

    for fname in FEATURE_NAMES:
        v_full = row_full_80[fname]
        v_trunc = row_trunc_80[fname]
        if np.isnan(v_full):
            assert np.isnan(v_trunc), f"Mismatch for {fname}: {v_full} != {v_trunc}"
        else:
            assert pytest.approx(v_full, rel=1e-5) == v_trunc, (
                f"Mismatch for {fname}: {v_full} != {v_trunc}"
            )


def test_gap_rule_drops_row() -> None:
    """Verify that a time gap > 3x interval inside the 60s window drops the decision row."""
    baseline = _make_dummy_baseline()
    records: list[Record] = []

    # Nominal interval 2s -> gap limit 6.0s
    # Insert a 10s gap between 70s and 80s
    timestamps = list(range(0, 72, 2)) + list(range(82, 130, 2))

    for sec in timestamps:
        mono = float(sec)
        dt_str = datetime.fromtimestamp(1700000000 + sec, tz=UTC).isoformat()
        sample = _make_sample(mono=mono, ts_utc=dt_str)
        probe = None
        if sec % 4 == 0 or sec == 82:
            probe = ProbeResult(
                ts_utc=dt_str,
                cpu_ns=20_000_000,
                mem_ns=2_000_000,
                disk_ns=5_000_000,
                total_ns=27_000_000,
            )

        rec = Record(
            schema_version=1,
            run_id="run-gap",
            label="idle",
            sample=sample,
            probe=probe,
            top_processes=[],
            storage=StorageReport(ts_utc=dt_str, entries=[], truncated=False, duration_s=0.1),
        )
        records.append(rec)

    rows = extract_run_features(records, baseline=baseline, horizon_s=30.0)

    # Any decision point t where window [t-60, t] contains the gap [70, 82] must be dropped.
    # The gap is active between mono=70 and mono=82.
    # Decision points from mono=82 up to mono=120 will see the gap if t-60 <= 70.
    for r in rows:
        row_mono = datetime.fromisoformat(r["ts_utc"]).timestamp() - 1700000000
        # Decision point cannot be between 82 and 130 if window contains the gap
        assert not (82 <= row_mono < 132), f"Row at {row_mono} should have been dropped due to gap"


def test_labeling_and_transition_rules() -> None:
    """Verify labeling transitions from sidecar metadata timeline."""
    baseline = _make_dummy_baseline()
    meta = {
        "scenario": "cpu_pressure",
        "intensity": "high",
        "pattern": "constant",
        "aborted": False,
        "actual": {
            "pre_start": "2026-11-01T12:00:00Z",
            "inject_start": "2026-11-01T12:00:30Z",
            "inject_end": "2026-11-01T12:03:30Z",
            "post_end": "2026-11-01T12:04:30Z",
        },
    }

    # Generate synthetic records across timeline
    records: list[Record] = []
    base_epoch = datetime.fromisoformat("2026-11-01T12:00:00+00:00").timestamp()

    for sec in range(0, 300, 2):
        mono = float(sec)
        dt_str = datetime.fromtimestamp(base_epoch + sec, tz=UTC).isoformat()
        sample = _make_sample(mono=mono, ts_utc=dt_str)
        probe = ProbeResult(
            ts_utc=dt_str,
            cpu_ns=20_000_000,
            mem_ns=2_000_000,
            disk_ns=5_000_000,
            total_ns=27_000_000,
        )
        rec = Record(
            schema_version=1,
            run_id="run-lab",
            label="lab:cpu_pressure:high:constant",
            sample=sample,
            probe=probe,
            top_processes=[],
            storage=StorageReport(ts_utc=dt_str, entries=[], truncated=False, duration_s=0.1),
        )
        records.append(rec)

    # Without keep_transition (default)
    rows_default = extract_run_features(
        records, baseline=baseline, meta=meta, keep_transition=False
    )
    issues_default = {r["issue_now"] for r in rows_default}
    assert "transition" not in issues_default
    assert "cpu_pressure" in issues_default

    # With keep_transition
    rows_kept = extract_run_features(records, baseline=baseline, meta=meta, keep_transition=True)
    issues_kept = {r["issue_now"] for r in rows_kept}
    assert "transition" in issues_kept
    assert len(rows_kept) > len(rows_default)


def test_splitting_sessions_and_runs() -> None:
    """Verify session-level and run-level splitting without leakage."""
    # Two sessions: session A (older), session B (newer)
    data = []
    for s_idx, session in enumerate(["session-1", "session-2"]):
        for r_idx in range(5):
            run_id = f"run-{session}-{r_idx}"
            for i in range(10):
                data.append(
                    {
                        "session_id": session,
                        "run_id": run_id,
                        "ts_utc": f"2026-11-0{s_idx + 1}T12:{r_idx:02d}:{i:02d}Z",
                        "target_total_ratio": 1.0,
                    }
                )

    df = pd.DataFrame(data)
    split_df, strategy = assign_splits(df, test_ratio=0.20)
    assert strategy == "session"

    # Entire session-2 must be test
    test_runs = set(split_df[split_df["split"] == "test"]["run_id"])
    train_runs = set(split_df[split_df["split"] == "train"]["run_id"])
    assert test_runs.isdisjoint(train_runs), "No run can exist in both train and test"
    assert (split_df["split"] == "test").mean() >= 0.20

    # Single session fallback to run-level
    df_single = df[df["session_id"] == "session-1"].copy()
    split_single, strategy_single = assign_splits(df_single, test_ratio=0.20)
    assert strategy_single == "run"
    test_runs_s = set(split_single[split_single["split"] == "test"]["run_id"])
    train_runs_s = set(split_single[split_single["split"] == "train"]["run_id"])
    assert test_runs_s.isdisjoint(train_runs_s)


def test_dataset_build_validate_describe_e2e(tmp_path: Path) -> None:
    """End-to-end dataset generation from synthetic JSONL, validation, and description."""
    baseline = _make_dummy_baseline()
    base_file = tmp_path / "baseline.json"
    from deskflow.baseline import save_baseline

    save_baseline(baseline, path=base_file)

    run_dir = tmp_path / "runs"
    run_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = run_dir / "test_run.jsonl"
    meta_path = run_dir / "test_run.meta.json"

    # Write 150s of synthetic records
    records: list[Record] = []
    base_epoch = 1700000000.0
    for sec in range(0, 160, 2):
        mono = float(sec)
        dt_str = datetime.fromtimestamp(base_epoch + sec, tz=UTC).isoformat()
        sample = _make_sample(mono=mono, ts_utc=dt_str)
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
            run_id="test_run",
            label="idle",
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
        "scenario": "idle",
        "intensity": "none",
        "pattern": "none",
        "machine_id": "test-machine",
        "session_id": "test-session",
        "deskflow_version": "0.1.0",
        "aborted": False,
        "actual": {
            "pre_start": "2026-11-01T12:00:00Z",
            "inject_start": "2026-11-01T12:00:30Z",
            "inject_end": "2026-11-01T12:02:00Z",
            "post_end": "2026-11-01T12:02:30Z",
        },
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta_dict, f)

    out_parquet = tmp_path / "test.parquet"
    parquet_path, manifest_path = build_dataset(
        run_paths=[run_dir],
        baseline_path=base_file,
        horizon_s=30.0,
        out_path=out_parquet,
    )

    assert parquet_path.exists()
    assert manifest_path.exists()

    # Validate dataset
    errors = validate_dataset(parquet_path)
    assert not errors, f"Validation errors: {errors}"

    # Describe dataset
    summary = describe_dataset(parquet_path)
    assert summary["total_rows"] > 0
    assert summary["train_rows"] + summary["test_rows"] == summary["total_rows"]
    assert "test-session" in summary["sessions"]
