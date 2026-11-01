"""CPU pressure injection scenario."""

import multiprocessing as mp
import os
import time
from typing import Any

import psutil

from deskflow.lab.base import Intensity, Pattern, Scenario, ScenarioStats


def _cpu_worker(
    parent_pid: int,
    hold_duration_s: float,
    pattern: str,
    stop_event: Any,
) -> None:
    """Worker process function executing CPU micro-loops."""
    start_time = time.monotonic()
    slice_s = 0.1

    while not stop_event.is_set():
        # Check parent liveness
        if not psutil.pid_exists(parent_pid):
            break

        now = time.monotonic()
        elapsed = now - start_time
        if elapsed >= hold_duration_s:
            break

        if pattern == "constant":
            # 100% busy for slice
            busy_end = time.monotonic() + slice_s
            x = 0
            while time.monotonic() < busy_end:
                x = (x + 1) * 3 & 0x7FFFFFFF

        elif pattern == "ramp":
            # Linear growth of duty cycle from 0.05 to 1.0
            fraction = min(1.0, max(0.05, elapsed / max(1.0, hold_duration_s)))
            busy_time = slice_s * fraction
            sleep_time = slice_s * (1.0 - fraction)

            busy_end = time.monotonic() + busy_time
            x = 0
            while time.monotonic() < busy_end:
                x = (x + 1) * 3 & 0x7FFFFFFF
            if sleep_time > 0:
                time.sleep(sleep_time)

        elif pattern == "burst":
            # 20s busy, 20s sleep
            cycle_pos = elapsed % 40.0
            if cycle_pos < 20.0:
                busy_end = time.monotonic() + slice_s
                x = 0
                while time.monotonic() < busy_end:
                    x = (x + 1) * 3 & 0x7FFFFFFF
            else:
                time.sleep(slice_s)
        else:
            time.sleep(slice_s)


class CpuPressureScenario(Scenario):
    """Generates controlled CPU saturation using multiprocessing workers."""

    def __init__(
        self,
        intensity: Intensity = "medium",
        pattern: Pattern = "constant",
        max_cores: int | None = None,
    ) -> None:
        super().__init__(intensity=intensity, pattern=pattern)
        self.max_cores = max_cores
        self._processes: list[Any] = []
        self._stop_event: Any = None
        self._worker_count = 0

    @property
    def name(self) -> str:
        return "cpu_pressure"

    def _calculate_workers(self) -> int:
        total_cores = psutil.cpu_count(logical=True) or 1
        if self.intensity == "low":
            count = max(1, int(total_cores * 0.25))
        elif self.intensity == "medium":
            count = max(1, int(total_cores * 0.50))
        else:  # high
            count = max(1, total_cores)

        if self.max_cores is not None:
            count = min(count, max(1, self.max_cores))
        return count

    def start(self, hold_duration_s: float = 180.0) -> None:
        self._worker_count = self._calculate_workers()
        ctx = mp.get_context("spawn")
        self._stop_event = ctx.Event()
        self._processes = []
        parent_pid = os.getpid()

        for _ in range(self._worker_count):
            p = ctx.Process(
                target=_cpu_worker,
                args=(parent_pid, hold_duration_s, self.pattern, self._stop_event),
                daemon=True,
            )
            p.start()
            self._processes.append(p)

        self._is_running = True

    def stop(self) -> None:
        if self._stop_event is not None:
            self._stop_event.set()

        for p in self._processes:
            if p.is_alive():
                p.terminate()
                p.join(timeout=1.0)
                if p.is_alive():
                    p.kill()

        self._processes.clear()
        self._is_running = False

    def get_child_pids(self) -> list[int]:
        """Return list of spawned child process IDs."""
        return [p.pid for p in self._processes if p.pid is not None]

    def get_stats(self) -> ScenarioStats:
        return ScenarioStats(
            worker_count=self._worker_count,
            bytes_allocated=0,
            extra={"cores_targeted": self._worker_count},
        )
