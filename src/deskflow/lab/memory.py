"""Memory pressure injection scenario."""

import multiprocessing as mp
import os
import time
from typing import Any

import psutil

from deskflow.lab.base import (
    Intensity,
    Pattern,
    Scenario,
    ScenarioStats,
    get_memory_floor_bytes,
)

CHUNK_SIZE = 64 * 1024 * 1024  # 64 MiB


def _touch_chunk(chunk: bytearray) -> None:
    """Touch one byte per 4096-byte OS page to ensure physical page allocation."""
    stride = 4096
    length = len(chunk)
    for i in range(0, length, stride):
        chunk[i] = (chunk[i] + 1) & 0xFF


def _memory_worker(
    parent_pid: int,
    target_bytes: int,
    hold_duration_s: float,
    pattern: str,
    stop_event: Any,
) -> None:
    """Worker process allocating and write-touching memory chunks."""
    start_time = time.monotonic()
    total_chunks = max(1, target_bytes // CHUNK_SIZE)
    chunks: list[bytearray] = []

    def _allocate_chunks(count: int) -> None:
        nonlocal chunks
        for _ in range(count):
            if stop_event.is_set() or not psutil.pid_exists(parent_pid):
                return
            try:
                c = bytearray(CHUNK_SIZE)
                _touch_chunk(c)
                chunks.append(c)
            except MemoryError:
                break

    def _free_chunks() -> None:
        nonlocal chunks
        chunks.clear()

    if pattern == "constant":
        _allocate_chunks(total_chunks)
        while not stop_event.is_set():
            if not psutil.pid_exists(parent_pid):
                break
            if time.monotonic() - start_time >= hold_duration_s:
                break
            # Periodic touch to keep pages active
            for c in chunks:
                if stop_event.is_set():
                    break
                c[0] = (c[0] + 1) & 0xFF
            time.sleep(1.0)

    elif pattern == "ramp":
        while not stop_event.is_set():
            if not psutil.pid_exists(parent_pid):
                break
            elapsed = time.monotonic() - start_time
            if elapsed >= hold_duration_s:
                break
            fraction = min(1.0, elapsed / max(1.0, hold_duration_s))
            desired_chunks = int(total_chunks * fraction)
            if len(chunks) < desired_chunks:
                _allocate_chunks(desired_chunks - len(chunks))

            # Touch active chunks
            for c in chunks:
                c[0] = (c[0] + 1) & 0xFF
            time.sleep(0.5)

    elif pattern == "burst":
        while not stop_event.is_set():
            if not psutil.pid_exists(parent_pid):
                break
            elapsed = time.monotonic() - start_time
            if elapsed >= hold_duration_s:
                break
            cycle_pos = elapsed % 40.0
            if cycle_pos < 20.0:
                if len(chunks) < total_chunks:
                    _allocate_chunks(total_chunks)
                for c in chunks:
                    c[0] = (c[0] + 1) & 0xFF
                time.sleep(1.0)
            else:
                if chunks:
                    _free_chunks()
                time.sleep(1.0)

    _free_chunks()


class MemoryPressureScenario(Scenario):
    """Generates controlled memory pressure respecting the safety floor."""

    def __init__(
        self,
        intensity: Intensity = "medium",
        pattern: Pattern = "constant",
        target_bytes_override: int | None = None,
    ) -> None:
        super().__init__(intensity=intensity, pattern=pattern)
        self.target_bytes_override = target_bytes_override
        self._process: Any = None
        self._stop_event: Any = None
        self._allocated_bytes = 0

    @property
    def name(self) -> str:
        return "memory_pressure"

    def _calculate_target_bytes(self) -> int:
        if self.target_bytes_override is not None:
            return self.target_bytes_override

        vm = psutil.virtual_memory()
        floor = get_memory_floor_bytes()
        avail = vm.available

        fraction = (
            0.25 if self.intensity == "low" else (0.50 if self.intensity == "medium" else 0.70)
        )
        target = int(avail * fraction)

        # Ensure remaining available does not breach floor
        if avail - target < floor:
            target = max(0, avail - floor)

        # Round down to nearest 64 MiB
        target = (target // CHUNK_SIZE) * CHUNK_SIZE
        return max(CHUNK_SIZE, target)

    def start(self, hold_duration_s: float = 180.0) -> None:
        self._allocated_bytes = self._calculate_target_bytes()
        ctx = mp.get_context("spawn")
        self._stop_event = ctx.Event()
        parent_pid = os.getpid()

        self._process = ctx.Process(
            target=_memory_worker,
            args=(
                parent_pid,
                self._allocated_bytes,
                hold_duration_s,
                self.pattern,
                self._stop_event,
            ),
            daemon=True,
        )
        self._process.start()
        self._is_running = True

    def stop(self) -> None:
        if self._stop_event is not None:
            self._stop_event.set()

        if self._process and self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=2.0)
            if self._process.is_alive():
                self._process.kill()

        self._process = None
        self._is_running = False

    def get_child_pids(self) -> list[int]:
        if self._process and self._process.pid is not None:
            return [self._process.pid]
        return []

    def get_stats(self) -> ScenarioStats:
        return ScenarioStats(
            worker_count=1 if self._process else 0,
            bytes_allocated=self._allocated_bytes,
        )
