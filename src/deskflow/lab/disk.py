"""Disk pressure injection scenario."""

import multiprocessing as mp
import os
import shutil
import threading
import time
from pathlib import Path
from typing import Any

import psutil

from deskflow.lab.base import (
    DISK_MIN_FREE_BYTES,
    Intensity,
    Pattern,
    Scenario,
    ScenarioStats,
)
from deskflow.lab.lock import get_scratch_dir

BLOCK_SIZE = 4 * 1024 * 1024  # 4 MiB
MAX_FILE_SIZE = 512 * 1024 * 1024  # 512 MiB per writer


def _disk_writer_thread(
    file_path: Path,
    hold_duration_s: float,
    pattern: str,
    stop_event: threading.Event,
) -> None:
    """Thread writing 4 MiB chunks cyclically to a fixed 512 MiB file with fsync."""
    start_time = time.monotonic()
    data_block = os.urandom(64 * 1024) * 64  # 4 MiB block

    try:
        with open(file_path, "wb+", buffering=0) as f:
            while not stop_event.is_set():
                elapsed = time.monotonic() - start_time
                if elapsed >= hold_duration_s:
                    break

                if pattern == "burst":
                    cycle_pos = elapsed % 40.0
                    if cycle_pos >= 20.0:
                        time.sleep(0.5)
                        continue

                # Write block
                current_offset = f.tell()
                if current_offset + BLOCK_SIZE > MAX_FILE_SIZE:
                    f.seek(0)
                f.write(data_block)
                f.flush()
                os.fsync(f.fileno())

                if pattern == "ramp":
                    # Sleep shrinks as time progresses
                    fraction = min(1.0, elapsed / max(1.0, hold_duration_s))
                    throttle_s = max(0.01, 0.2 * (1.0 - fraction))
                    time.sleep(throttle_s)
                elif pattern == "constant":
                    time.sleep(0.005)
    except Exception:
        pass


def _disk_worker(
    parent_pid: int,
    scratch_dir_str: str,
    num_writers: int,
    hold_duration_s: float,
    pattern: str,
    stop_event: Any,
) -> None:
    """Worker process managing writer threads and scratch volume."""
    sdir = Path(scratch_dir_str)
    sdir.mkdir(parents=True, exist_ok=True)
    threads: list[threading.Thread] = []
    thread_stop = threading.Event()

    for i in range(num_writers):
        fpath = sdir / f"disk_writer_{i}.tmp"
        t = threading.Thread(
            target=_disk_writer_thread,
            args=(fpath, hold_duration_s, pattern, thread_stop),
            daemon=True,
        )
        t.start()
        threads.append(t)

    start_time = time.monotonic()
    while not stop_event.is_set():
        if not psutil.pid_exists(parent_pid):
            break
        if time.monotonic() - start_time >= hold_duration_s:
            break
        time.sleep(0.5)

    thread_stop.set()
    for t in threads:
        t.join(timeout=1.0)


class DiskPressureScenario(Scenario):
    """Generates controlled disk I/O pressure via cyclic fsync writes."""

    def __init__(
        self,
        intensity: Intensity = "medium",
        pattern: Pattern = "constant",
        scratch_dir: Path | None = None,
        max_file_size_override: int | None = None,
    ) -> None:
        super().__init__(intensity=intensity, pattern=pattern)
        self.scratch_dir = scratch_dir or get_scratch_dir()
        self.max_file_size_override = max_file_size_override
        self._process: Any = None
        self._stop_event: Any = None
        self._writer_count = 0

    @property
    def name(self) -> str:
        return "disk_pressure"

    def _calculate_writers(self) -> int:
        if self.intensity == "low":
            return 1
        if self.intensity == "medium":
            return 2
        return 4  # high

    def start(self, hold_duration_s: float = 180.0) -> None:
        # Pre-check free disk space
        try:
            free_disk = shutil.disk_usage(self.scratch_dir).free
            if free_disk < DISK_MIN_FREE_BYTES:
                raise RuntimeError(
                    f"Insufficient free disk space ({free_disk / (1024**3):.2f} GB) "
                    f"to start disk pressure lab."
                )
        except Exception:
            pass

        self._writer_count = self._calculate_writers()
        ctx = mp.get_context("spawn")
        self._stop_event = ctx.Event()
        parent_pid = os.getpid()

        self._process = ctx.Process(
            target=_disk_worker,
            args=(
                parent_pid,
                str(self.scratch_dir),
                self._writer_count,
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

        # Clean scratch writer files
        for i in range(self._writer_count):
            p = self.scratch_dir / f"disk_writer_{i}.tmp"
            p.unlink(missing_ok=True)

    def get_child_pids(self) -> list[int]:
        if self._process and self._process.pid is not None:
            return [self._process.pid]
        return []

    def get_stats(self) -> ScenarioStats:
        file_size = self.max_file_size_override or MAX_FILE_SIZE
        return ScenarioStats(
            worker_count=self._writer_count,
            bytes_allocated=self._writer_count * file_size,
        )
