"""Tests for storage cleanable vs total byte accounting and tmp-tree traversal with symlinks."""

import contextlib
import os
import time
from pathlib import Path

from deskflow.collector.storage import StorageScanner
from deskflow.platform.base import CacheLocation, PlatformAdapter
from deskflow.remediation.actions import clean_temp


class StorageTestAdapter(PlatformAdapter):
    """Platform adapter pointing to a controlled temporary directory tree."""

    def __init__(self, temp_root: Path) -> None:
        self._temp_root = temp_root

    @property
    def name(self) -> str:
        return "test-storage"

    def cpu_freq(self):
        return None

    def temperatures(self):
        return {}

    def battery(self):
        return None

    def temp_and_cache_dirs(self) -> list[CacheLocation]:
        return [
            CacheLocation(path=str(self._temp_root), kind="temp", safe_to_clean=True),
        ]

    def protected_process_names(self) -> frozenset[str]:
        return frozenset()


def test_tmp_tree_traversal_and_cleanable_accounting(tmp_path: Path) -> None:
    """Test StorageScanner computes total and cleanable bytes in nested tree with symlinks."""
    root = tmp_path / "temp_root"
    root.mkdir()

    now = time.time()
    old_time = now - (48 * 3600)  # 48 hours ago
    new_time = now - (1 * 3600)  # 1 hour ago

    # 1. Top-level files
    old_top = root / "old_top.dat"
    old_top.write_bytes(b"A" * 2048)
    os.utime(old_top, (old_time, old_time))

    new_top = root / "new_top.dat"
    new_top.write_bytes(b"B" * 1024)
    os.utime(new_top, (new_time, new_time))

    # 2. Nested subdirectories
    sub_dir = root / "nested1" / "nested2"
    sub_dir.mkdir(parents=True)

    old_nested = sub_dir / "old_nested.log"
    old_nested.write_bytes(b"C" * 4096)
    os.utime(old_nested, (old_time, old_time))

    new_nested = sub_dir / "new_nested.log"
    new_nested.write_bytes(b"D" * 512)
    os.utime(new_nested, (new_time, new_time))

    # 3. Outside file & symlink
    outside_file = tmp_path / "outside_huge.bin"
    outside_file.write_bytes(b"Z" * 100_000)

    symlink_file = root / "link_to_outside.bin"
    with contextlib.suppress(OSError, NotImplementedError):
        symlink_file.symlink_to(outside_file)

    adapter = StorageTestAdapter(temp_root=root)
    scanner = StorageScanner(adapter=adapter, time_budget_s=5.0)

    # First Scan
    report = scanner.scan(force=True)
    assert len(report.entries) == 1
    entry = report.entries[0]

    expected_total = 2048 + 1024 + 4096 + 512  # 7680 bytes
    expected_cleanable = 2048 + 4096  # 6144 bytes

    assert entry.bytes == expected_total
    assert entry.cleanable_bytes == expected_cleanable
    assert not report.truncated

    # Test Dry-Run clean_temp
    res_dry = clean_temp(adapter=adapter, dry_run=True)
    assert res_dry.bytes_freed == expected_cleanable
    assert old_top.exists()
    assert old_nested.exists()

    # Test Non-Dry-Run clean_temp
    res_exec = clean_temp(adapter=adapter, dry_run=False)
    assert res_exec.bytes_freed == expected_cleanable

    # Old files must be deleted
    assert not old_top.exists()
    assert not old_nested.exists()

    # New files must still exist!
    assert new_top.exists()
    assert new_nested.exists()

    # Re-scan after cleanup: cleanable bytes must now be 0
    report_after = scanner.scan(force=True)
    entry_after = report_after.entries[0]
    assert entry_after.bytes == (1024 + 512)
    assert entry_after.cleanable_bytes == 0
