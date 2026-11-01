"""Safety watchdog thread monitoring resource limits during lab execution."""

import shutil
import threading
from collections.abc import Callable
from pathlib import Path

import psutil

from deskflow.lab.base import (
    DISK_MIN_FREE_BYTES,
    get_memory_floor_bytes,
)
from deskflow.lab.lock import get_scratch_dir


class LabWatchdog:
    """Monitors free memory and disk space every second and aborts on breach."""

    def __init__(
        self,
        on_abort: Callable[[str], None],
        scratch_dir: Path | None = None,
        check_interval_s: float = 1.0,
        mem_floor_override: int | None = None,
        disk_min_override: int | None = None,
    ) -> None:
        self.on_abort = on_abort
        self.scratch_dir = scratch_dir or get_scratch_dir()
        self.check_interval_s = check_interval_s
        self.mem_floor = mem_floor_override or get_memory_floor_bytes()
        self.disk_min = disk_min_override or DISK_MIN_FREE_BYTES

        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self.aborted = False
        self.abort_reason = ""

    def start(self) -> None:
        """Start the background watchdog monitor thread."""
        self._stop_event.clear()
        self.aborted = False
        self.abort_reason = ""
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True, name="lab-watchdog")
        self._thread.start()

    def stop(self) -> None:
        """Stop the background watchdog monitor thread."""
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)

    def _monitor_loop(self) -> None:
        while not self._stop_event.is_set():
            # 1. Check memory available
            try:
                avail_mem = psutil.virtual_memory().available
                if avail_mem < self.mem_floor:
                    self.aborted = True
                    self.abort_reason = (
                        f"Memory available ({avail_mem / (1024**3):.2f} GB) fell below "
                        f"safety floor ({self.mem_floor / (1024**3):.2f} GB)"
                    )
                    self.on_abort(self.abort_reason)
                    break
            except Exception:
                pass

            # 2. Check disk free
            try:
                disk_free = shutil.disk_usage(self.scratch_dir).free
                if disk_free < self.disk_min:
                    self.aborted = True
                    self.abort_reason = (
                        f"Free disk space ({disk_free / (1024**3):.2f} GB) fell below "
                        f"safety threshold ({self.disk_min / (1024**3):.2f} GB)"
                    )
                    self.on_abort(self.abort_reason)
                    break
            except Exception:
                pass

            self._stop_event.wait(self.check_interval_s)
