"""Unit tests for RunRecorder and load_run persistence."""

from pathlib import Path

import pytest

from deskflow.collector.models import (
    SCHEMA_VERSION,
    Record,
    SystemSample,
)
from deskflow.collector.recorder import RunRecorder, load_run


def _make_dummy_record(run_id: str = "run_test", label: str = "test") -> Record:
    sample = SystemSample(
        ts_utc="2026-11-01T12:00:00Z",
        monotonic_s=10.0,
        cpu_percent_mean=10.0,
        cpu_percent_max_core=20.0,
        cpu_freq_current_mhz=2000.0,
        cpu_freq_max_mhz=3000.0,
        cpu_freq_reliable=True,
        ctx_switches_per_s=100.0,
        interrupts_per_s=50.0,
        mem_total_bytes=8000000000,
        mem_available_bytes=4000000000,
        mem_percent=50.0,
        swap_used_bytes=0,
        swap_percent=0.0,
        disk_read_bytes_per_s=0.0,
        disk_write_bytes_per_s=0.0,
        disk_busy_ms_per_s=0.0,
        net_recv_bytes_per_s=0.0,
        net_sent_bytes_per_s=0.0,
        net_errors_per_s=0.0,
        net_drops_per_s=0.0,
        process_count=50,
        thread_count=300,
        battery_percent=100.0,
        battery_plugged=True,
        max_temperature_c=40.0,
        uptime_s=1000.0,
    )
    return Record(
        schema_version=SCHEMA_VERSION,
        run_id=run_id,
        label=label,
        sample=sample,
        probe=None,
        top_processes=[],
        storage=None,
    )


def test_recorder_writes_and_flushes(tmp_path: Path) -> None:
    """RunRecorder writes lines and increments records_written."""
    out_file = tmp_path / "custom_run.jsonl"
    with RunRecorder(out_path=out_file, run_id="r1") as recorder:
        rec1 = _make_dummy_record("r1", "label1")
        rec2 = _make_dummy_record("r1", "label2")
        recorder.write(rec1)
        recorder.write(rec2)
        assert recorder.records_written == 2

    # Verify file content
    lines = out_file.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 2

    loaded = load_run(out_file)
    assert len(loaded) == 2
    assert loaded[0].label == "label1"
    assert loaded[1].label == "label2"


def test_load_run_invalid_json(tmp_path: Path) -> None:
    """load_run must raise ValueError on corrupted JSON line."""
    p = tmp_path / "corrupt.jsonl"
    p.write_text("not a valid json\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid JSON"):
        load_run(p)


def test_load_run_schema_version_mismatch(tmp_path: Path) -> None:
    """load_run must raise ValueError on schema version mismatch."""
    p = tmp_path / "bad_schema.jsonl"
    p.write_text('{"schema_version": 999}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="Schema version mismatch"):
        load_run(p)
