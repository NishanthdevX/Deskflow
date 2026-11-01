"""Unit tests for remediation actions, safe temp cleanup, and audit logging."""

import os
import time
from pathlib import Path

from deskflow.collector.models import ProcessSample
from deskflow.platform.base import CacheLocation, PlatformAdapter
from deskflow.remediation.actions import (
    clean_temp,
    close_idle_apps,
    end_busy_process,
    trim_memory,
)
from deskflow.remediation.log import append_fix_log


class DummyAdapter(PlatformAdapter):
    """Mock platform adapter for testing remediation."""

    def __init__(self, temp_dir: Path, is_win: bool = False) -> None:
        self._temp_dir = temp_dir
        self._is_win = is_win

    @property
    def name(self) -> str:
        return "windows" if self._is_win else "macos"

    def cpu_freq(self):
        return None

    def temperatures(self):
        return {}

    def battery(self):
        return None

    def temp_and_cache_dirs(self) -> list[CacheLocation]:
        return [
            CacheLocation(path=str(self._temp_dir), kind="temp", safe_to_clean=True),
        ]

    def protected_process_names(self) -> frozenset[str]:
        return frozenset({"explorer.exe", "system", "launchd"})

    def is_trim_memory_supported(self) -> bool:
        return self._is_win

    def trim_process_memory(self, pid: int) -> bool:
        return True


def test_clean_temp_rules(tmp_path: Path) -> None:
    """Test old file deleted, new file kept, symlinks skipped, dry-run changes nothing."""
    adapter = DummyAdapter(temp_dir=tmp_path)

    # 1. Create an old file (> 24 hours)
    old_file = tmp_path / "old.log"
    old_file.write_text("old content", encoding="utf-8")
    old_mtime = time.time() - 90000.0  # 25 hours ago
    os.utime(old_file, (old_mtime, old_mtime))

    # 2. Create a new file (< 24 hours)
    new_file = tmp_path / "new.log"
    new_file.write_text("new content", encoding="utf-8")

    # 3. Create a subdirectory with an old file
    sub_dir = tmp_path / "subdir"
    sub_dir.mkdir()
    sub_old = sub_dir / "sub_old.tmp"
    sub_old.write_text("sub old content", encoding="utf-8")
    os.utime(sub_old, (old_mtime, old_mtime))

    # 4. Create a symlink to outside directory if supported
    target_outside = tmp_path.parent / "outside.txt"
    target_outside.write_text("do not touch", encoding="utf-8")
    symlink_file = tmp_path / "link_outside.txt"
    try:
        symlink_file.symlink_to(target_outside)
    except (OSError, NotImplementedError):
        # Symlink creation may require privilege on Windows, skip symlink creation if not allowed
        symlink_file = None

    # Test Dry-Run first
    res_dry = clean_temp(adapter=adapter, dry_run=True)
    assert res_dry.dry_run is True
    assert res_dry.bytes_freed > 0
    # Files must still exist after dry-run!
    assert old_file.exists()
    assert new_file.exists()
    assert sub_old.exists()
    if symlink_file and symlink_file.exists():
        assert target_outside.exists()

    # Test Actual Execution
    res_exec = clean_temp(adapter=adapter, dry_run=False)
    assert res_exec.dry_run is False
    assert res_exec.bytes_freed > 0

    # Old files should be deleted
    assert not old_file.exists()
    assert not sub_old.exists()
    # New file must remain untouched
    assert new_file.exists()
    # Outside file must remain untouched
    assert target_outside.exists()
    # Subdir should be removed if empty
    assert not sub_dir.exists()


def test_close_idle_apps_safety() -> None:
    """Test protected processes are never closed and only >= 300MB idle groups offered."""
    procs = [
        # Protected system process
        ProcessSample(
            pid=10,
            name="explorer.exe",
            rss_bytes=500 * 1024 * 1024,
            cpu_percent=0.1,
            num_threads=20,
            age_s=2000.0,
            is_protected=True,
        ),
        # Idle app but under 300MB
        ProcessSample(
            pid=200,
            name="notepad.exe",
            rss_bytes=20 * 1024 * 1024,
            cpu_percent=0.0,
            num_threads=2,
            age_s=1500.0,
            is_protected=False,
        ),
        # Idle app over 300MB
        ProcessSample(
            pid=301,
            name="idle_browser.exe",
            rss_bytes=200 * 1024 * 1024,
            cpu_percent=0.2,
            num_threads=10,
            age_s=1200.0,
            is_protected=False,
        ),
        ProcessSample(
            pid=302,
            name="idle_browser.exe",
            rss_bytes=150 * 1024 * 1024,
            cpu_percent=0.1,
            num_threads=8,
            age_s=1200.0,
            is_protected=False,
        ),
    ]

    res = close_idle_apps(procs, dry_run=True)
    assert res.dry_run is True
    assert "idle_browser.exe" in res.details
    assert "explorer.exe" not in res.details
    assert "notepad.exe" not in res.details
    assert res.bytes_freed == (350 * 1024 * 1024)


def test_end_busy_process() -> None:
    """Test that end_busy_process targets up to 3 non-protected >=50% CPU processes."""
    procs = [
        ProcessSample(
            pid=501,
            name="worker1.exe",
            rss_bytes=100 * 1024 * 1024,
            cpu_percent=85.0,
            num_threads=4,
            age_s=100.0,
            is_protected=False,
        ),
        ProcessSample(
            pid=502,
            name="worker2.exe",
            rss_bytes=100 * 1024 * 1024,
            cpu_percent=70.0,
            num_threads=4,
            age_s=100.0,
            is_protected=False,
        ),
        ProcessSample(
            pid=503,
            name="protected_cpu.exe",
            rss_bytes=100 * 1024 * 1024,
            cpu_percent=95.0,
            num_threads=4,
            age_s=100.0,
            is_protected=True,
        ),
    ]

    res = end_busy_process(procs, dry_run=True)
    assert res.dry_run is True
    assert "worker1.exe_501" in res.details
    assert "worker2.exe_502" in res.details
    assert "protected_cpu.exe_503" not in res.details


def test_trim_memory_adapter() -> None:
    """Test trim_memory respecting adapter support."""
    adapter_win = DummyAdapter(temp_dir=Path("."), is_win=True)
    adapter_mac = DummyAdapter(temp_dir=Path("."), is_win=False)

    procs = [
        ProcessSample(
            pid=601,
            name="app.exe",
            rss_bytes=100 * 1024 * 1024,
            cpu_percent=1.0,
            num_threads=4,
            age_s=1000.0,
            is_protected=False,
        )
    ]

    # Windows adapter supports trim
    res_win = trim_memory(procs, adapter=adapter_win, dry_run=True)
    assert res_win.success is True

    # Mac adapter does not support trim
    res_mac = trim_memory(procs, adapter=adapter_mac, dry_run=True)
    assert res_mac.success is False


def test_fix_audit_logging(tmp_path: Path) -> None:
    """Test append_fix_log writes JSON lines to fixlog.jsonl."""
    record = {
        "ts_utc": "2026-11-01T12:00:00Z",
        "action": "clean_temp",
        "dry_run": False,
        "bytes_freed": 1024 * 1024 * 15,
    }
    log_file = append_fix_log(record, data_dir=tmp_path)
    assert log_file.exists()
    content = log_file.read_text(encoding="utf-8")
    assert "clean_temp" in content
    assert "15728640" in content
