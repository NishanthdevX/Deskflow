"""Parity test: verifies live path and dataset builder produce identical feature vectors."""

from datetime import UTC, datetime

import numpy as np

from deskflow.baseline import Baseline
from deskflow.collector.models import (
    ProbeResult,
    ProcessSample,
    Record,
    SystemSample,
)
from deskflow.features.extract import build_row, extract_run_features
from deskflow.features.registry import FEATURE_NAMES


def _make_dummy_sample(mono: float, cpu: float = 25.0) -> SystemSample:
    secs = int(mono) % 60
    mins = int(mono) // 60
    return SystemSample(
        ts_utc=f"2026-11-01T12:{mins:02d}:{secs:02d}Z",
        monotonic_s=mono,
        cpu_percent_mean=cpu,
        cpu_percent_max_core=cpu * 1.2,
        cpu_freq_current_mhz=2800.0,
        cpu_freq_max_mhz=3200.0,
        cpu_freq_reliable=True,
        ctx_switches_per_s=1200.0,
        interrupts_per_s=600.0,
        mem_total_bytes=16 * (1024**3),
        mem_available_bytes=int(16 * (1024**3) * 0.6),
        mem_percent=40.0,
        swap_used_bytes=1024**2,
        swap_percent=2.5,
        disk_read_bytes_per_s=1024.0 * 500,
        disk_write_bytes_per_s=1024.0 * 800,
        disk_busy_ms_per_s=50.0,
        net_sent_bytes_per_s=100.0,
        net_recv_bytes_per_s=200.0,
        net_errors_per_s=0.0,
        net_drops_per_s=0.0,
        process_count=180,
        thread_count=1400,
        battery_percent=100.0,
        battery_plugged=True,
        max_temperature_c=45.0,
        uptime_s=10000.0,
    )


def test_feature_parity_live_vs_dataset_builder() -> None:
    """Test that build_row on records_up_to_t yields exact parity with extract_run_features."""
    baseline = Baseline(
        machine_id="test-machine-1",
        created_utc="2026-11-01T00:00:00Z",
        platform="windows",
        cpu_ns=10_000_000,
        mem_ns=20_000_000,
        disk_ns=30_000_000,
        total_ns=60_000_000,
        p10_total_ns=55_000_000,
        p90_total_ns=65_000_000,
        sample_count=30,
        deskflow_version="0.1.0",
        cv=0.04,
        power_state="ac",
    )

    # Generate 60 seconds of records (sample every 2s, probe every 4s)
    records: list[Record] = []
    top_proc = [
        ProcessSample(
            pid=100,
            name="worker.exe",
            rss_bytes=1024**3,
            cpu_percent=15.0,
            num_threads=12,
            age_s=1200.0,
            is_protected=False,
        )
    ]

    for i in range(30):
        t = float(i * 2)
        has_probe = i % 2 == 0
        probe = (
            ProbeResult(
                ts_utc=datetime.now(UTC).isoformat(),
                cpu_ns=11_000_000,
                mem_ns=21_000_000,
                disk_ns=31_000_000,
                total_ns=63_000_000,
            )
            if has_probe
            else None
        )
        records.append(
            Record(
                schema_version=1,
                run_id="run-parity-test",
                label="idle",
                sample=_make_dummy_sample(t, cpu=20.0 + (i % 5)),
                probe=probe,
                top_processes=top_proc,
                storage=None,
            )
        )

    # Extract dataset rows
    dataset_rows = extract_run_features(
        records=records,
        baseline=baseline,
        horizon_s=20.0,
    )
    assert len(dataset_rows) > 0, "Expected at least one valid feature row extracted"

    # For each decision point in dataset_rows, test parity with live path
    for ds_row in dataset_rows:
        target_ts_utc = ds_row["ts_utc"]
        # Find record corresponding to this ts_utc
        rec_idx = next(i for i, r in enumerate(records) if r.sample.ts_utc == target_ts_utc)
        cur_rec = records[rec_idx]

        # Live path: pass only records strictly up to cur_rec
        records_up_to_t = records[: rec_idx + 1]
        live_features = build_row(
            records_up_to_t=records_up_to_t,
            baseline=baseline,
            current_record=cur_rec,
        )

        # Compare every single canonical feature in FEATURE_NAMES
        for feat in FEATURE_NAMES:
            ds_val = ds_row[feat]
            live_val = live_features[feat]
            if np.isnan(ds_val):
                assert np.isnan(live_val), f"Expected NaN for {feat}"
            else:
                assert np.isclose(ds_val, live_val, rtol=1e-5, atol=1e-5), (
                    f"Mismatch in feature '{feat}': dataset={ds_val}, live={live_val}"
                )

        # Also verify numpy allclose over feature vectors
        vec_ds = np.array([ds_row[f] for f in FEATURE_NAMES], dtype=np.float64)
        vec_live = np.array([live_features[f] for f in FEATURE_NAMES], dtype=np.float64)

        # Handle NaNs safely in allclose
        mask = ~np.isnan(vec_ds)
        assert np.allclose(vec_ds[mask], vec_live[mask], rtol=1e-5, atol=1e-5)
