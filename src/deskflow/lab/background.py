"""Background load injection scenario creating multiple idle processes holding memory."""

import contextlib
import os
import subprocess
import sys
from typing import Any

import psutil

from deskflow.lab.base import (
    Intensity,
    Pattern,
    Scenario,
    ScenarioStats,
    get_memory_floor_bytes,
)


class BackgroundLoadScenario(Scenario):
    """Spawns multiple idle background processes each holding pinned memory."""

    def __init__(
        self,
        intensity: Intensity = "medium",
        pattern: Pattern = "constant",
        worker_count_override: int | None = None,
    ) -> None:
        super().__init__(intensity=intensity, pattern=pattern)
        self.worker_count_override = worker_count_override
        self._processes: list[subprocess.Popen[Any]] = []
        self._target_proc_count = 0
        self._mib_per_proc = 0

    @property
    def name(self) -> str:
        return "background_load"

    def _calculate_plan(self) -> tuple[int, int]:
        """Return (proc_count, mib_per_proc) scaled to respect safety floor."""
        if self.worker_count_override is not None:
            return (self.worker_count_override, 10)

        if self.intensity == "low":
            count, mib = 10, 50
        elif self.intensity == "medium":
            count, mib = 25, 100
        else:  # high
            count, mib = 40, 150

        # Check memory floor
        vm = psutil.virtual_memory()
        floor = get_memory_floor_bytes()
        avail = vm.available
        safe_allocatable = max(0, avail - floor)

        target_bytes = count * mib * 1024 * 1024
        if target_bytes > safe_allocatable:
            # Scale down proc count
            bytes_per_proc = mib * 1024 * 1024
            count = max(1, safe_allocatable // bytes_per_proc)

        return count, mib

    def start(self, hold_duration_s: float = 180.0) -> None:
        count, mib = self._calculate_plan()
        self._target_proc_count = count
        self._mib_per_proc = mib
        self._processes = []
        parent_pid = os.getpid()

        # Script that allocates bytearray, touches pages, and polls parent PID
        script = (
            "import sys, time, psutil\n"
            f"parent_pid = {parent_pid}\n"
            f"alloc_bytes = {mib * 1024 * 1024}\n"
            f"hold_s = {hold_duration_s}\n"
            "data = bytearray(alloc_bytes)\n"
            "for i in range(0, len(data), 4096): data[i] = 1\n"
            "start = time.monotonic()\n"
            "while time.monotonic() - start < hold_s:\n"
            "    if not psutil.pid_exists(parent_pid): break\n"
            "    time.sleep(1.0)\n"
        )

        for _ in range(count):
            p = subprocess.Popen(
                [sys.executable, "-c", script],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            self._processes.append(p)

        self._spawned_count = len(self._processes)
        self._is_running = True

    def stop(self) -> None:
        for p in self._processes:
            try:
                p.terminate()
                p.wait(timeout=1.0)
            except Exception:
                with contextlib.suppress(Exception):
                    p.kill()

        self._processes.clear()
        self._is_running = False

    def get_child_pids(self) -> list[int]:
        return [p.pid for p in self._processes if p.pid is not None]

    def get_stats(self) -> ScenarioStats:
        total_bytes = self._spawned_count * self._mib_per_proc * 1024 * 1024
        return ScenarioStats(
            worker_count=self._spawned_count,
            bytes_allocated=total_bytes,
        )
