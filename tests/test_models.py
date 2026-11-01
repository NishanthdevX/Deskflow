"""Unit tests for telemetry data models and serialization."""

import pytest

from deskflow.collector.models import (
    SCHEMA_VERSION,
    ProbeResult,
    ProcessSample,
    Record,
    StorageEntry,
    StorageReport,
    SystemSample,
)


def _make_dummy_sample() -> SystemSample:
    return SystemSample(
        ts_utc="2026-11-01T12:00:00Z",
        monotonic_s=100.0,
        cpu_percent_mean=15.5,
        cpu_percent_max_core=30.0,
        cpu_freq_current_mhz=2400.0,
        cpu_freq_max_mhz=3200.0,
        cpu_freq_reliable=True,
        ctx_switches_per_s=1200.0,
        interrupts_per_s=500.0,
        mem_total_bytes=16000000000,
        mem_available_bytes=8000000000,
        mem_percent=50.0,
        swap_used_bytes=1000000000,
        swap_percent=10.0,
        disk_read_bytes_per_s=1024000.0,
        disk_write_bytes_per_s=512000.0,
        disk_busy_ms_per_s=15.0,
        net_recv_bytes_per_s=2048.0,
        net_sent_bytes_per_s=1024.0,
        net_errors_per_s=0.0,
        net_drops_per_s=0.0,
        process_count=150,
        thread_count=1200,
        battery_percent=85.0,
        battery_plugged=True,
        max_temperature_c=48.5,
        uptime_s=3600.0,
    )


def test_system_sample_roundtrip() -> None:
    """SystemSample must serialize to dict and deserialize losslessly."""
    sample = _make_dummy_sample()
    data = sample.to_dict()
    restored = SystemSample.from_dict(data)
    assert restored == sample


def test_probe_result_roundtrip() -> None:
    """ProbeResult must serialize to dict and deserialize losslessly."""
    probe = ProbeResult(
        ts_utc="2026-11-01T12:00:00Z",
        cpu_ns=15_000_000,
        mem_ns=2_000_000,
        disk_ns=10_000_000,
        total_ns=27_000_000,
    )
    data = probe.to_dict()
    restored = ProbeResult.from_dict(data)
    assert restored == probe


def test_record_roundtrip() -> None:
    """Record must serialize to dict and deserialize losslessly."""
    sample = _make_dummy_sample()
    probe = ProbeResult(
        ts_utc="2026-11-01T12:00:00Z",
        cpu_ns=15_000_000,
        mem_ns=2_000_000,
        disk_ns=10_000_000,
        total_ns=27_000_000,
    )
    proc = ProcessSample(
        pid=123,
        name="test.exe",
        rss_bytes=50_000_000,
        cpu_percent=5.0,
        num_threads=4,
        age_s=120.0,
        is_protected=False,
    )
    storage = StorageReport(
        ts_utc="2026-11-01T12:00:00Z",
        entries=[StorageEntry(path="/tmp", kind="temp", bytes=1000)],
        truncated=False,
        duration_s=0.05,
    )

    record = Record(
        schema_version=SCHEMA_VERSION,
        run_id="20261101T120000Z_aabbcc",
        label="test_run",
        sample=sample,
        probe=probe,
        top_processes=[proc],
        storage=storage,
    )

    data = record.to_dict()
    restored = Record.from_dict(data)
    assert restored == record


def test_record_schema_version_mismatch() -> None:
    """Record deserialization must raise ValueError on schema version mismatch."""
    sample = _make_dummy_sample()
    record = Record(
        schema_version=SCHEMA_VERSION,
        run_id="run_1",
        label="unlabeled",
        sample=sample,
        probe=None,
        top_processes=[],
        storage=None,
    )
    data = record.to_dict()
    data["schema_version"] = 999  # Incompatible future version

    with pytest.raises(ValueError, match="Schema version mismatch"):
        Record.from_dict(data)
