"""Unit tests for the performance micro-benchmark probe."""

from pathlib import Path

from deskflow.collector.probe import (
    probe_cpu,
    probe_disk,
    probe_mem,
    run_probe,
)


def test_probe_cpu_returns_positive_ns() -> None:
    """CPU probe should execute and return positive elapsed nanoseconds."""
    duration_ns = probe_cpu()
    assert duration_ns > 0
    # Sanity check: should be under 500 ms on any modern machine
    assert duration_ns < 500_000_000


def test_probe_mem_returns_positive_ns() -> None:
    """Memory probe should execute and return positive elapsed nanoseconds."""
    duration_ns = probe_mem()
    assert duration_ns > 0
    assert duration_ns < 500_000_000


def test_probe_disk_cleans_up_file(tmp_path: Path) -> None:
    """Disk probe should write, fsync, delete the file, and leave no temp files behind."""
    duration_ns = probe_disk(data_dir=tmp_path)
    assert duration_ns > 0

    # Ensure no leftover files in directory
    leftovers = list(tmp_path.glob("probe_*.tmp"))
    assert len(leftovers) == 0


def test_run_probe_total_equals_components(tmp_path: Path) -> None:
    """run_probe must return a ProbeResult where total_ns matches component sums."""
    result = run_probe(data_dir=tmp_path)
    assert result.cpu_ns > 0
    assert result.mem_ns > 0
    assert result.disk_ns > 0
    assert result.total_ns == result.cpu_ns + result.mem_ns + result.disk_ns
    assert result.ts_utc.endswith("Z") or "+00:00" in result.ts_utc
