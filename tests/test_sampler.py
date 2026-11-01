"""Unit tests for the telemetry sampler with monkeypatched psutil."""

from collections import namedtuple
from unittest.mock import MagicMock

import psutil
import pytest

from deskflow.collector.sampler import Sampler
from deskflow.platform.base import BatteryInfo, CacheLocation, CpuFreq, PlatformAdapter

# Mock counter tuples matching psutil structures
CpuStats = namedtuple("CpuStats", ["ctx_switches", "interrupts", "soft_interrupts", "syscalls"])
DiskIO = namedtuple(
    "DiskIO",
    [
        "read_count",
        "write_count",
        "read_bytes",
        "write_bytes",
        "read_time",
        "write_time",
        "busy_time",
    ],
)
NetIO = namedtuple(
    "NetIO",
    [
        "bytes_sent",
        "bytes_recv",
        "packets_sent",
        "packets_recv",
        "errin",
        "errout",
        "dropin",
        "dropout",
    ],
)


class DummyAdapter(PlatformAdapter):
    """Deterministic dummy platform adapter for testing."""

    @property
    def name(self) -> str:
        return "dummy"

    def cpu_freq(self) -> CpuFreq | None:
        return CpuFreq(current_mhz=2500.0, max_mhz=3000.0, reliable=True)

    def temperatures(self) -> dict[str, float]:
        return {"core0": 45.0}

    def battery(self) -> BatteryInfo | None:
        return BatteryInfo(percent=90.0, plugged_in=True)

    def temp_and_cache_dirs(self) -> list[CacheLocation]:
        return []

    def protected_process_names(self) -> frozenset[str]:
        return frozenset({"init", "system"})


def test_first_sample_rates_are_none(monkeypatch: pytest.MonkeyPatch) -> None:
    """The first sample must set delta rates to None."""
    adapter = DummyAdapter()
    sampler = Sampler(adapter=adapter)

    sample, _ = sampler.sample()

    assert sample.ctx_switches_per_s is None
    assert sample.interrupts_per_s is None
    assert sample.disk_read_bytes_per_s is None
    assert sample.disk_write_bytes_per_s is None
    assert sample.net_recv_bytes_per_s is None
    assert sample.net_sent_bytes_per_s is None


def test_rate_computation_from_deltas(monkeypatch: pytest.MonkeyPatch) -> None:
    """Subsequent samples must compute rates as deltas divided by elapsed monotonic time."""
    adapter = DummyAdapter()
    sampler = Sampler(adapter=adapter)

    time_values = [100.0, 102.0]  # dt = 2.0s
    monkeypatch.setattr("time.monotonic", lambda: time_values.pop(0) if time_values else 104.0)

    stats_seq = [
        CpuStats(ctx_switches=1000, interrupts=500, soft_interrupts=0, syscalls=0),
        CpuStats(ctx_switches=3000, interrupts=900, soft_interrupts=0, syscalls=0),
    ]
    monkeypatch.setattr(psutil, "cpu_stats", lambda: stats_seq.pop(0))

    disk_seq = [
        DiskIO(
            0, 0, read_bytes=10_000, write_bytes=20_000, read_time=0, write_time=0, busy_time=100
        ),
        DiskIO(
            0, 0, read_bytes=30_000, write_bytes=40_000, read_time=0, write_time=0, busy_time=300
        ),
    ]
    monkeypatch.setattr(psutil, "disk_io_counters", lambda: disk_seq.pop(0))

    net_seq = [
        NetIO(
            bytes_sent=1000,
            bytes_recv=2000,
            packets_sent=10,
            packets_recv=20,
            errin=0,
            errout=0,
            dropin=0,
            dropout=0,
        ),
        NetIO(
            bytes_sent=5000,
            bytes_recv=8000,
            packets_sent=50,
            packets_recv=80,
            errin=1,
            errout=1,
            dropin=0,
            dropout=0,
        ),
    ]
    monkeypatch.setattr(psutil, "net_io_counters", lambda: net_seq.pop(0))

    # First sample primes counters
    sampler.sample()

    # Second sample computes rates over dt = 2.0s
    sample2, _ = sampler.sample()

    # (3000 - 1000) / 2 = 1000.0
    assert sample2.ctx_switches_per_s == pytest.approx(1000.0)
    # (900 - 500) / 2 = 200.0
    assert sample2.interrupts_per_s == pytest.approx(200.0)
    # (30000 - 10000) / 2 = 10000.0
    assert sample2.disk_read_bytes_per_s == pytest.approx(10000.0)
    # (40000 - 20000) / 2 = 10000.0
    assert sample2.disk_write_bytes_per_s == pytest.approx(10000.0)
    # (300 - 100) / 2 = 100.0
    assert sample2.disk_busy_ms_per_s == pytest.approx(100.0)
    # (8000 - 2000) / 2 = 3000.0
    assert sample2.net_recv_bytes_per_s == pytest.approx(3000.0)
    # (5000 - 1000) / 2 = 2000.0
    assert sample2.net_sent_bytes_per_s == pytest.approx(2000.0)
    # (2 - 0) / 2 = 1.0
    assert sample2.net_errors_per_s == pytest.approx(1.0)


def test_per_process_exceptions_are_skipped(monkeypatch: pytest.MonkeyPatch) -> None:
    """Processes throwing NoSuchProcess, AccessDenied, or ZombieProcess are skipped."""
    adapter = DummyAdapter()
    sampler = Sampler(adapter=adapter)

    # Mock process_iter to return normal proc and failing procs
    normal_proc = MagicMock()
    normal_proc.info = {
        "pid": 100,
        "name": "good.exe",
        "memory_info": namedtuple("Mem", ["rss"])(rss=102400),
        "cpu_percent": 1.5,
        "num_threads": 2,
        "create_time": 100.0,
        "username": "user",
    }

    failing_proc1 = MagicMock()
    type(failing_proc1).info = property(
        lambda self: (_ for _ in ()).throw(psutil.NoSuchProcess(pid=101))
    )

    failing_proc2 = MagicMock()
    type(failing_proc2).info = property(
        lambda self: (_ for _ in ()).throw(psutil.AccessDenied(pid=102))
    )

    failing_proc3 = MagicMock()
    type(failing_proc3).info = property(
        lambda self: (_ for _ in ()).throw(psutil.ZombieProcess(pid=103))
    )

    monkeypatch.setattr(
        psutil,
        "process_iter",
        lambda attrs: [normal_proc, failing_proc1, failing_proc2, failing_proc3],
    )

    _, processes = sampler.sample()

    # Only normal_proc should be present
    assert len(processes) == 1
    assert processes[0].pid == 100
    assert processes[0].name == "good.exe"
