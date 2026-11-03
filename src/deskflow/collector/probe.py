"""Micro-benchmark probe measuring ground truth system latency."""

import os
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path

from deskflow.collector.models import ProbeResult
from deskflow.paths import get_data_dir

# Fixed workload sizes
# Default 3 repeats for CPU and memory micro-workloads
PROBE_REPEATS: int = 3
PROBE_WARMUP_MS: int = 10

# 60,000 iterations of linear congruential generator arithmetic (~15-18ms on modern CPU)
CPU_PROBE_ITERATIONS: int = 60_000

# 8 MiB memory allocation with 4096-byte stride writes (simulates paging/cache touch)
MEM_PROBE_BYTES: int = 8 * 1024 * 1024
MEM_PROBE_STRIDE: int = 4096

# 1 MiB disk write + fsync
DISK_PROBE_BYTES: int = 1 * 1024 * 1024


def probe_cpu() -> int:
    """Run pure-Python integer LCG workload and return duration in nanoseconds."""
    state = 12345
    t0 = time.perf_counter_ns()
    for _ in range(CPU_PROBE_ITERATIONS):
        state = (state * 1103515245 + 12345) & 0x7FFFFFFF
    t1 = time.perf_counter_ns()
    # Ensure state is computed
    if state == -1:
        return 0
    return t1 - t0


def _warm_cpu() -> None:
    """Keep one core active briefly before the timed workload."""
    end = time.perf_counter_ns() + (PROBE_WARMUP_MS * 1_000_000)
    state = 1
    while time.perf_counter_ns() < end:
        state = (state * 1103515245 + 12345) & 0x7FFFFFFF
    if state == -1:
        raise RuntimeError("unreachable")


def probe_mem() -> int:
    """Allocate bytearray, touch memory with strided writes, and return duration in ns."""
    t0 = time.perf_counter_ns()
    buf = bytearray(MEM_PROBE_BYTES)
    for i in range(0, MEM_PROBE_BYTES, MEM_PROBE_STRIDE):
        buf[i] = 1
    t1 = time.perf_counter_ns()
    del buf
    return t1 - t0


def probe_disk(data_dir: Path | None = None) -> int:
    """Write 1 MiB to a temporary file, fsync, remove it, and return duration in ns.

    Cleans up the temporary file reliably even if an exception occurs.
    """
    target_dir: Path
    if data_dir is not None:
        target_dir = data_dir
    else:
        target_dir = get_data_dir()

    target_dir.mkdir(parents=True, exist_ok=True)

    payload = b"X" * 65536
    chunks = DISK_PROBE_BYTES // len(payload)

    fd, temp_path = tempfile.mkstemp(prefix="probe_", suffix=".tmp", dir=str(target_dir))
    try:
        t0 = time.perf_counter_ns()
        with os.fdopen(fd, "wb") as f:
            for _ in range(chunks):
                f.write(payload)
            f.flush()
            os.fsync(f.fileno())
        t1 = time.perf_counter_ns()
        return t1 - t0
    finally:
        try:
            if os.path.exists(temp_path):
                os.remove(temp_path)
        except OSError:
            pass


def run_probe(
    data_dir: Path | None = None,
    repeats: int = PROBE_REPEATS,
) -> ProbeResult:
    """Execute CPU, memory, and disk micro-workloads sequentially.

    CPU and memory workloads run `repeats` times and take the minimum duration
    to filter out transient scheduler noise. Disk stays single as it is syscall/I-O bound.

    Returns:
        ProbeResult with duration in nanoseconds for each component and total.
    """
    ts_utc = datetime.now(UTC).isoformat()
    _warm_cpu()
    cpu_repeats = tuple(probe_cpu() for _ in range(repeats))
    mem_repeats = tuple(probe_mem() for _ in range(repeats))
    cpu_ns = min(cpu_repeats)
    mem_ns = min(mem_repeats)
    disk_ns = probe_disk(data_dir=data_dir)
    total_ns = cpu_ns + mem_ns + disk_ns

    return ProbeResult(
        ts_utc=ts_utc,
        cpu_ns=cpu_ns,
        mem_ns=mem_ns,
        disk_ns=disk_ns,
        total_ns=total_ns,
        cpu_repeats_ns=cpu_repeats,
        mem_repeats_ns=mem_repeats,
    )
