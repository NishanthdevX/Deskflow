"""System telemetry sampler using psutil and platform adapter."""

import contextlib
import os
import time
from collections.abc import Set
from datetime import UTC, datetime
from typing import Any

import psutil

from deskflow.collector.models import ProcessSample, SystemSample
from deskflow.platform.base import PlatformAdapter
from deskflow.safety import _get_current_username, _get_parent_pids, is_protected


class Sampler:
    """Samples hardware telemetry, system metrics, and running processes."""

    def __init__(self, adapter: PlatformAdapter, interval_s: float = 1.0) -> None:
        """Initialize the sampler, priming psutil system CPU counters.

        Note:
            Per-process cpu_percent counters are primed during the first sample pass;
            consequently, process cpu_percent will report 0.0 on the initial reading.
        """
        self._adapter = adapter
        self.interval_s = interval_s

        # Prime global CPU counters
        psutil.cpu_percent(interval=None, percpu=True)

        # Baseline state for delta rate calculation
        self._prev_monotonic_s: float | None = None
        self._prev_cpu_stats: Any = None
        self._prev_disk_io: Any = None
        self._prev_net_io: Any = None

        # Cache protected process evaluation environment
        self._current_pid = os.getpid()
        self._current_user = _get_current_username()
        self._parent_pids: Set[int] = _get_parent_pids(self._current_pid)
        self._protected_names: Set[str] = frozenset(
            p.lower() for p in self._adapter.protected_process_names()
        )

    def sample(self) -> tuple[SystemSample, list[ProcessSample]]:
        """Collect a system telemetry sample and snapshot of running processes.

        Returns:
            A tuple of (SystemSample, list of ProcessSample sorted descending by RSS).
        """
        now_utc = datetime.now(UTC).isoformat()
        monotonic_s = time.monotonic()
        now_epoch = time.time()

        # 1. CPU utilization
        cpu_percentages = psutil.cpu_percent(interval=None, percpu=True)
        if cpu_percentages:
            cpu_mean = float(sum(cpu_percentages) / len(cpu_percentages))
            cpu_max_core = float(max(cpu_percentages))
        else:
            cpu_mean = 0.0
            cpu_max_core = 0.0

        # CPU frequency
        freq = self._adapter.cpu_freq()
        freq_current = freq.current_mhz if freq else None
        freq_max = freq.max_mhz if freq else None
        freq_reliable = freq.reliable if freq else None

        # 2. Virtual and swap memory
        mem = psutil.virtual_memory()
        swap = psutil.swap_memory()

        # 3. Delta counters: CPU switches/interrupts, Disk I/O, Network I/O
        curr_cpu_stats = psutil.cpu_stats() if hasattr(psutil, "cpu_stats") else None
        curr_disk_io = psutil.disk_io_counters() if hasattr(psutil, "disk_io_counters") else None
        curr_net_io = psutil.net_io_counters() if hasattr(psutil, "net_io_counters") else None

        ctx_switches_per_s: float | None = None
        interrupts_per_s: float | None = None
        disk_read_bytes_per_s: float | None = None
        disk_write_bytes_per_s: float | None = None
        disk_busy_ms_per_s: float | None = None
        net_recv_bytes_per_s: float | None = None
        net_sent_bytes_per_s: float | None = None
        net_errors_per_s: float | None = None
        net_drops_per_s: float | None = None

        if self._prev_monotonic_s is not None:
            dt = monotonic_s - self._prev_monotonic_s
            if dt > 0:
                if curr_cpu_stats and self._prev_cpu_stats:
                    ctx_switches_per_s = max(
                        0.0,
                        (curr_cpu_stats.ctx_switches - self._prev_cpu_stats.ctx_switches) / dt,
                    )
                    interrupts_per_s = max(
                        0.0,
                        (curr_cpu_stats.interrupts - self._prev_cpu_stats.interrupts) / dt,
                    )

                if curr_disk_io and self._prev_disk_io:
                    disk_read_bytes_per_s = max(
                        0.0,
                        (curr_disk_io.read_bytes - self._prev_disk_io.read_bytes) / dt,
                    )
                    disk_write_bytes_per_s = max(
                        0.0,
                        (curr_disk_io.write_bytes - self._prev_disk_io.write_bytes) / dt,
                    )
                    curr_busy = getattr(curr_disk_io, "busy_time", None)
                    prev_busy = getattr(self._prev_disk_io, "busy_time", None)
                    if curr_busy is not None and prev_busy is not None:
                        disk_busy_ms_per_s = max(
                            0.0,
                            (curr_busy - prev_busy) / dt,
                        )

                if curr_net_io and self._prev_net_io:
                    net_recv_bytes_per_s = max(
                        0.0,
                        (curr_net_io.bytes_recv - self._prev_net_io.bytes_recv) / dt,
                    )
                    net_sent_bytes_per_s = max(
                        0.0,
                        (curr_net_io.bytes_sent - self._prev_net_io.bytes_sent) / dt,
                    )
                    prev_errors = self._prev_net_io.errin + self._prev_net_io.errout
                    curr_errors = curr_net_io.errin + curr_net_io.errout
                    net_errors_per_s = max(0.0, (curr_errors - prev_errors) / dt)

                    prev_drops = self._prev_net_io.dropin + self._prev_net_io.dropout
                    curr_drops = curr_net_io.dropin + curr_net_io.dropout
                    net_drops_per_s = max(0.0, (curr_drops - prev_drops) / dt)

        # Update previous state
        self._prev_monotonic_s = monotonic_s
        self._prev_cpu_stats = curr_cpu_stats
        self._prev_disk_io = curr_disk_io
        self._prev_net_io = curr_net_io

        # 4. Battery and Temperatures
        bat = self._adapter.battery()
        battery_percent = bat.percent if bat else None
        battery_plugged = bat.plugged_in if bat else None

        temps = self._adapter.temperatures()
        max_temp = max(temps.values()) if temps else None

        # 5. Uptime
        try:
            uptime = max(0.0, now_epoch - psutil.boot_time())
        except Exception:
            uptime = 0.0

        # 6. Fast process enumeration with attrs list (< 10ms without LSASS RPC)
        processes: list[ProcessSample] = []
        process_iter_attrs = ["pid", "name", "memory_info", "cpu_percent"]

        for proc in psutil.process_iter(attrs=process_iter_attrs):
            try:
                info = proc.info
                pid = int(info["pid"])
                name = str(info["name"] or "")
                mem_info = info.get("memory_info")
                rss = int(mem_info.rss) if mem_info else 0
                cpu_p = float(info["cpu_percent"]) if info.get("cpu_percent") is not None else 0.0

                # Fast check based on PID and known protected names without IPC/RPC calls
                fast_protected = (
                    pid in {0, 1, 4}
                    or pid == self._current_pid
                    or pid in self._parent_pids
                    or name.lower() in self._protected_names
                )

                processes.append(
                    ProcessSample(
                        pid=pid,
                        name=name,
                        rss_bytes=rss,
                        cpu_percent=cpu_p,
                        num_threads=1,
                        age_s=0.0,
                        is_protected=fast_protected,
                    )
                )
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                continue
            except Exception:
                continue

        # Sort descending by RSS memory
        processes.sort(key=lambda p: p.rss_bytes, reverse=True)

        # Enrich top 10 processes with exact threads, age, and full ownership check
        for i in range(min(10, len(processes))):
            p_sample = processes[i]
            try:
                p_obj = psutil.Process(p_sample.pid)
                th = p_obj.num_threads()
                ctime = p_obj.create_time()
                age = max(0.0, now_epoch - ctime)
                user: str | None = None
                with contextlib.suppress(Exception):
                    user = p_obj.username()

                full_protected = is_protected(
                    name=p_sample.name,
                    pid=p_sample.pid,
                    username=user,
                    current_pid=self._current_pid,
                    current_user=self._current_user,
                    parent_pids=self._parent_pids,
                    protected_names=self._protected_names,
                )

                user: str | None = None
                try:
                    user = p_obj.username()
                except Exception:
                    pass

                full_protected = is_protected(
                    name=p_sample.name,
                    pid=p_sample.pid,
                    username=user,
                    current_pid=self._current_pid,
                    current_user=self._current_user,
                    parent_pids=self._parent_pids,
                    protected_names=self._protected_names,
                )

                processes[i] = ProcessSample(
                    pid=p_sample.pid,
                    name=p_sample.name,
                    rss_bytes=p_sample.rss_bytes,
                    cpu_percent=p_sample.cpu_percent,
                    num_threads=th,
                    age_s=age,
                    is_protected=full_protected,
                )
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                pass

        # Total thread count from adapter (0.05s via Win32 snapshot) or process count fallback
        adapter_threads = self._adapter.thread_count()
        total_threads = adapter_threads if adapter_threads is not None else len(processes)

        system_sample = SystemSample(
            ts_utc=now_utc,
            monotonic_s=monotonic_s,
            cpu_percent_mean=cpu_mean,
            cpu_percent_max_core=cpu_max_core,
            cpu_freq_current_mhz=freq_current,
            cpu_freq_max_mhz=freq_max,
            cpu_freq_reliable=freq_reliable,
            ctx_switches_per_s=ctx_switches_per_s,
            interrupts_per_s=interrupts_per_s,
            mem_total_bytes=int(mem.total),
            mem_available_bytes=int(mem.available),
            mem_percent=float(mem.percent),
            swap_used_bytes=int(swap.used),
            swap_percent=float(swap.percent),
            disk_read_bytes_per_s=disk_read_bytes_per_s,
            disk_write_bytes_per_s=disk_write_bytes_per_s,
            disk_busy_ms_per_s=disk_busy_ms_per_s,
            net_recv_bytes_per_s=net_recv_bytes_per_s,
            net_sent_bytes_per_s=net_sent_bytes_per_s,
            net_errors_per_s=net_errors_per_s,
            net_drops_per_s=net_drops_per_s,
            process_count=len(processes),
            thread_count=total_threads,
            battery_percent=battery_percent,
            battery_plugged=battery_plugged,
            max_temperature_c=max_temp,
            uptime_s=uptime,
        )

        return system_sample, processes
