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

    def __init__(
        self,
        adapter: PlatformAdapter,
        interval_s: float = 1.0,
        process_interval_s: float = 10.0,
        net_interval_s: float = 10.0,
    ) -> None:
        """Initialize the sampler, priming psutil system CPU counters.

        Args:
            adapter: PlatformAdapter instance for OS metric calls.
            interval_s: Telemetry sampling interval in seconds.
            process_interval_s: Process table enumeration interval in seconds (default 10s).
            net_interval_s: Network I/O counter sampling interval in seconds (default 10s).
        """
        self._adapter = adapter
        self.interval_s = interval_s
        self.process_interval_s = process_interval_s
        self.net_interval_s = net_interval_s

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

        # Process cadence cache and ownership cache keyed by (pid, create_time)
        self._last_process_sample_mono: float | None = None
        self._cached_processes: list[ProcessSample] = []
        self._ownership_cache: dict[tuple[int, float], bool] = {}
        self._last_net_sample_mono: float | None = None
        self._cached_net_io: Any | None = None
        self._cached_net_rates: tuple[float, float, float, float] | None = None

        # Cache boot time to avoid disk/eventlog queries per tick
        try:
            self._boot_time = psutil.boot_time()
        except Exception:
            self._boot_time = time.time()

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
        swap_info = self._adapter.swap_memory()
        if swap_info is not None:
            swap_used_bytes, swap_percent = swap_info
        else:
            swap = psutil.swap_memory()
            swap_used_bytes = int(swap.used)
            swap_percent = float(swap.percent)

        # 3. Delta counters: CPU switches/interrupts, Disk I/O, Network I/O
        curr_cpu_stats = psutil.cpu_stats() if hasattr(psutil, "cpu_stats") else None
        curr_disk_io = psutil.disk_io_counters() if hasattr(psutil, "disk_io_counters") else None

        # Sample network counters on net_interval_s cadence to avoid syscall overhead per tick
        is_net_sample_due = (
            self._cached_net_io is None
            or self._last_net_sample_mono is None
            or (monotonic_s - self._last_net_sample_mono) >= self.net_interval_s
        )
        if is_net_sample_due:
            new_net_io = psutil.net_io_counters() if hasattr(psutil, "net_io_counters") else None
            if self._last_net_sample_mono is not None and new_net_io and self._cached_net_io:
                net_dt = max(0.001, monotonic_s - self._last_net_sample_mono)
                prev_errors = self._cached_net_io.errin + self._cached_net_io.errout
                curr_errors = new_net_io.errin + new_net_io.errout
                prev_drops = self._cached_net_io.dropin + self._cached_net_io.dropout
                curr_drops = new_net_io.dropin + new_net_io.dropout
                self._cached_net_rates = (
                    max(0.0, (new_net_io.bytes_recv - self._cached_net_io.bytes_recv) / net_dt),
                    max(0.0, (new_net_io.bytes_sent - self._cached_net_io.bytes_sent) / net_dt),
                    max(0.0, (curr_errors - prev_errors) / net_dt),
                    max(0.0, (curr_drops - prev_drops) / net_dt),
                )
            self._cached_net_io = new_net_io
            self._last_net_sample_mono = monotonic_s

        ctx_switches_per_s: float | None = None
        interrupts_per_s: float | None = None
        disk_read_bytes_per_s: float | None = None
        disk_write_bytes_per_s: float | None = None
        disk_busy_ms_per_s: float | None = None
        net_recv_bytes_per_s: float | None = None
        net_sent_bytes_per_s: float | None = None
        net_errors_per_s: float | None = None
        net_drops_per_s: float | None = None

        if self._cached_net_rates is not None:
            net_recv_bytes_per_s, net_sent_bytes_per_s, net_errors_per_s, net_drops_per_s = (
                self._cached_net_rates
            )

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

        # Update previous state
        self._prev_monotonic_s = monotonic_s
        self._prev_cpu_stats = curr_cpu_stats
        self._prev_disk_io = curr_disk_io
        self._prev_net_io = self._cached_net_io

        # 4. Battery and Temperatures
        bat = self._adapter.battery()
        battery_percent = bat.percent if bat else None
        battery_plugged = bat.plugged_in if bat else None

        temps = self._adapter.temperatures()
        max_temp = max(temps.values()) if temps else None

        # 5. Uptime
        uptime = max(0.0, now_epoch - self._boot_time)

        # 6. Process enumeration on its own cadence (--process-interval, default 10s)
        if (
            not self._cached_processes
            or self._last_process_sample_mono is None
            or (monotonic_s - self._last_process_sample_mono) >= self.process_interval_s
        ):
            processes: list[ProcessSample] = []
            snap = self._adapter.process_snapshot()
            if snap is not None:
                # Ultra-fast path (< 5ms) via platform snapshot
                for p_meta in snap:
                    fast_protected = (
                        p_meta.pid in {0, 1, 4}
                        or p_meta.pid == self._current_pid
                        or p_meta.pid in self._parent_pids
                        or p_meta.name.lower() in self._protected_names
                    )
                    age = (
                        max(0.0, now_epoch - p_meta.create_time_s)
                        if p_meta.create_time_s > 0
                        else 0.0
                    )
                    processes.append(
                        ProcessSample(
                            pid=p_meta.pid,
                            name=p_meta.name,
                            rss_bytes=p_meta.rss_bytes,
                            cpu_percent=0.0,
                            num_threads=p_meta.num_threads,
                            age_s=age,
                            is_protected=fast_protected,
                        )
                    )
            else:
                # Portable fallback using psutil
                process_iter_attrs = ["pid", "name", "memory_info"]
                for proc in psutil.process_iter(attrs=process_iter_attrs):
                    try:
                        info = proc.info
                        pid = int(info["pid"])
                        name = str(info["name"] or "")
                        mem_info = info.get("memory_info")
                        rss = int(mem_info.rss) if mem_info else 0
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
                                cpu_percent=0.0,
                                num_threads=1,
                                age_s=0.0,
                                is_protected=fast_protected,
                            )
                        )
                    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                        continue
                    except Exception:
                        continue

            # Pass 1: Sort descending by RSS memory and retain top 25 processes only.
            # Processes outside the top 25 are invisible to downstream features.
            processes.sort(key=lambda p: p.rss_bytes, reverse=True)
            top_25 = processes[:25]

            # Pass 2: compute cpu_percent and thread count only for top 25 by RSS
            # and enrich with cached ownership per (pid, create_time).
            for i in range(len(top_25)):
                p_sample = top_25[i]
                cpu_pct = p_sample.cpu_percent
                threads = p_sample.num_threads
                if snap is None:
                    with contextlib.suppress(Exception):
                        proc_obj = psutil.Process(p_sample.pid)
                        cpu_pct = proc_obj.cpu_percent(interval=None)
                        threads = proc_obj.num_threads()

                full_protected = p_sample.is_protected
                if not full_protected:
                    # Calculate or retrieve create_time for ownership cache
                    ctime = now_epoch - p_sample.age_s if p_sample.age_s > 0 else now_epoch
                    cache_key = (p_sample.pid, round(ctime, 1))
                    if cache_key in self._ownership_cache:
                        is_current_user = self._ownership_cache[cache_key]
                    else:
                        user_str: str | None = None
                        with contextlib.suppress(Exception):
                            user_str = psutil.Process(p_sample.pid).username()
                        is_current_user = bool(
                            user_str is not None
                            and self._current_user
                            and user_str.lower() == self._current_user.lower()
                        )
                        self._ownership_cache[cache_key] = is_current_user

                    full_protected = is_protected(
                        name=p_sample.name,
                        pid=p_sample.pid,
                        username=self._current_user if is_current_user else None,
                        current_pid=self._current_pid,
                        current_user=self._current_user,
                        parent_pids=self._parent_pids,
                        protected_names=self._protected_names,
                    )

                top_25[i] = ProcessSample(
                    pid=p_sample.pid,
                    name=p_sample.name,
                    rss_bytes=p_sample.rss_bytes,
                    cpu_percent=cpu_pct,
                    num_threads=threads,
                    age_s=p_sample.age_s,
                    is_protected=full_protected,
                )

            self._cached_processes = top_25
            self._last_process_sample_mono = monotonic_s
        else:
            processes = self._cached_processes

        # Process and thread totals via adapter
        adapter_procs = self._adapter.process_count()
        total_procs = adapter_procs if adapter_procs is not None else len(processes)
        total_threads = self._adapter.thread_count()

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
            swap_used_bytes=swap_used_bytes,
            swap_percent=swap_percent,
            disk_read_bytes_per_s=disk_read_bytes_per_s,
            disk_write_bytes_per_s=disk_write_bytes_per_s,
            disk_busy_ms_per_s=disk_busy_ms_per_s,
            net_recv_bytes_per_s=net_recv_bytes_per_s,
            net_sent_bytes_per_s=net_sent_bytes_per_s,
            net_errors_per_s=net_errors_per_s,
            net_drops_per_s=net_drops_per_s,
            process_count=total_procs,
            thread_count=total_threads,
            battery_percent=battery_percent,
            battery_plugged=battery_plugged,
            max_temperature_c=max_temp,
            uptime_s=uptime,
        )

        return system_sample, processes
