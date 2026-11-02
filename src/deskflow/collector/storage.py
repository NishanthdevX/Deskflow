"""Storage scanner for temporary, cache, and trash locations."""

import contextlib
import os
import threading
import time
from collections.abc import Generator
from datetime import UTC, datetime

from deskflow.collector.models import StorageEntry, StorageReport
from deskflow.platform.base import PlatformAdapter


class StorageScanner:
    """Scans cache and temporary directories to measure storage consumption."""

    def __init__(
        self,
        adapter: PlatformAdapter,
        time_budget_s: float = 3.0,
        max_entries: int = 200_000,
        cache_ttl_s: float = 300.0,
    ) -> None:
        """Initialize the storage scanner.

        Args:
            adapter: PlatformAdapter supplying known temporary/cache paths.
            time_budget_s: Maximum total wall time allowed for a single scan.
            max_entries: Maximum total directory entries to inspect before truncating.
            cache_ttl_s: Duration in seconds to reuse scan results.
        """
        self._adapter = adapter
        self.time_budget_s = time_budget_s
        self.max_entries = max_entries
        self.cache_ttl_s = cache_ttl_s

        self._cached_report: StorageReport | None = None
        self._cached_monotonic: float | None = None
        self._lock = threading.Lock()
        self._is_scanning = False
        self._pause_event = threading.Event()
        self._stop_event = threading.Event()

    def pause(self) -> None:
        """Pause scanning (e.g. during micro-benchmark probe execution)."""
        self._pause_event.set()

    def resume(self) -> None:
        """Resume scanning after probe completion."""
        self._pause_event.clear()

    @contextlib.contextmanager
    def pause_during_probe(self) -> Generator[None, None, None]:
        """Context manager ensuring the scanner pauses while probe is executing."""
        self._pause_event.set()
        try:
            yield
        finally:
            self._pause_event.clear()

    def stop(self) -> None:
        """Signal background scanner to stop."""
        self._stop_event.set()

    def scan(self, force: bool = False) -> StorageReport:
        """Scan configured locations, respecting time budget, entry cap, and symlink guards.

        Args:
            force: If True, bypasses the TTL cache and triggers a fresh scan.

        Returns:
            StorageReport containing per-location byte totals and truncation status.
        """
        now_mono = time.monotonic()
        with self._lock:
            if (
                not force
                and self._cached_report is not None
                and self._cached_monotonic is not None
                and (now_mono - self._cached_monotonic < self.cache_ttl_s)
            ):
                return self._cached_report

        start_mono = time.monotonic()
        now_epoch = time.time()
        ts_utc = datetime.now(UTC).isoformat()
        locations = self._adapter.temp_and_cache_dirs()
        entries: list[StorageEntry] = []
        truncated = False
        total_entries_visited = 0

        for loc in locations:
            loc_bytes = 0
            loc_cleanable_bytes = 0
            if not os.path.exists(loc.path):
                entries.append(
                    StorageEntry(path=loc.path, kind=loc.kind, bytes=0, cleanable_bytes=0)
                )
                continue

            # Explicit stack avoiding recursion depth limits
            stack: list[str] = [loc.path]

            while stack:
                if self._stop_event.is_set():
                    break
                # Pause while probe is in progress
                while self._pause_event.is_set() and not self._stop_event.is_set():
                    time.sleep(0.01)

                # Check budgets
                if time.monotonic() - start_mono >= self.time_budget_s:
                    truncated = True
                    break
                if total_entries_visited >= self.max_entries:
                    truncated = True
                    break

                curr_dir = stack.pop()
                try:
                    with os.scandir(curr_dir) as scanner:
                        for item in scanner:
                            total_entries_visited += 1

                            # Symlink guard: never follow symlinks
                            try:
                                if item.is_symlink():
                                    continue
                            except OSError:
                                continue

                            # Account file sizes
                            try:
                                if item.is_file(follow_symlinks=False):
                                    stat_res = item.stat(follow_symlinks=False)
                                    loc_bytes += stat_res.st_size
                                    if (now_epoch - stat_res.st_mtime) >= 86400.0:
                                        loc_cleanable_bytes += stat_res.st_size
                                elif item.is_dir(follow_symlinks=False):
                                    stack.append(item.path)
                            except OSError:
                                continue

                            if total_entries_visited >= self.max_entries:
                                truncated = True
                                break
                            if time.monotonic() - start_mono >= self.time_budget_s:
                                truncated = True
                                break
                except (PermissionError, FileNotFoundError, OSError):
                    continue

            entries.append(
                StorageEntry(
                    path=loc.path,
                    kind=loc.kind,
                    bytes=loc_bytes,
                    cleanable_bytes=loc_cleanable_bytes,
                )
            )

        duration_s = max(0.0, time.monotonic() - start_mono)
        report = StorageReport(
            ts_utc=ts_utc,
            entries=entries,
            truncated=truncated,
            duration_s=duration_s,
        )

        with self._lock:
            self._cached_report = report
            self._cached_monotonic = time.monotonic()

        return report

    def start_background_scan(self) -> None:
        """Launch an asynchronous scan in a background thread if not already running."""
        with self._lock:
            if self._is_scanning:
                return
            self._is_scanning = True

        def _worker() -> None:
            try:
                self.scan(force=True)
            finally:
                with self._lock:
                    self._is_scanning = False

        thread = threading.Thread(target=_worker, name="StorageScannerThread", daemon=True)
        thread.start()

    def get_cached_report(self) -> StorageReport | None:
        """Return currently cached report if available, without blocking on a scan."""
        with self._lock:
            return self._cached_report
