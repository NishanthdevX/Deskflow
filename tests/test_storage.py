"""Unit tests for the storage scanner, symlink guards, and limits."""

from pathlib import Path

from deskflow.collector.storage import StorageScanner
from deskflow.platform.base import CacheLocation, PlatformAdapter


class MockStorageAdapter(PlatformAdapter):
    """Adapter pointing to custom test paths."""

    def __init__(self, locations: list[CacheLocation]) -> None:
        self._locations = locations

    @property
    def name(self) -> str:
        return "mock"

    def cpu_freq(self) -> None:
        return None

    def temperatures(self) -> dict[str, float]:
        return {}

    def battery(self) -> None:
        return None

    def temp_and_cache_dirs(self) -> list[CacheLocation]:
        return self._locations

    def protected_process_names(self) -> frozenset[str]:
        return frozenset()


def test_storage_scanner_computes_size(tmp_path: Path) -> None:
    """StorageScanner accurately tallies directory file sizes."""
    # Create 3 files of 1024 bytes each
    for i in range(3):
        p = tmp_path / f"file_{i}.dat"
        p.write_bytes(b"A" * 1024)

    sub = tmp_path / "subdir"
    sub.mkdir()
    (sub / "nested.dat").write_bytes(b"B" * 2048)

    loc = CacheLocation(path=str(tmp_path), kind="temp", safe_to_clean=True)
    adapter = MockStorageAdapter([loc])
    scanner = StorageScanner(adapter=adapter)

    report = scanner.scan()
    assert len(report.entries) == 1
    assert report.entries[0].bytes == 3 * 1024 + 2048
    assert not report.truncated


def test_storage_scanner_symlink_guard(tmp_path: Path) -> None:
    """StorageScanner must never follow symlinks or enter recursive symlink loops."""
    file_real = tmp_path / "real.dat"
    file_real.write_bytes(b"X" * 1000)

    sub = tmp_path / "sub"
    sub.mkdir()

    # Attempt symlink creation (may require privileges on Windows, so catch OSError)
    try:
        symlink_target = sub / "loop_symlink"
        symlink_target.symlink_to(tmp_path, target_is_directory=True)
    except OSError:
        pass

    loc = CacheLocation(path=str(tmp_path), kind="temp", safe_to_clean=True)
    adapter = MockStorageAdapter([loc])
    scanner = StorageScanner(adapter=adapter, time_budget_s=2.0)

    report = scanner.scan()
    assert not report.truncated
    # If symlink was followed, file_real would be counted multiple times or loop
    assert report.entries[0].bytes == 1000


def test_storage_scanner_entry_cap_truncation(tmp_path: Path) -> None:
    """StorageScanner must truncate when max_entries is exceeded."""
    # Create 20 files
    for i in range(20):
        (tmp_path / f"f_{i}.dat").write_bytes(b"Z")

    loc = CacheLocation(path=str(tmp_path), kind="temp", safe_to_clean=True)
    adapter = MockStorageAdapter([loc])
    # Cap at 5 entries
    scanner = StorageScanner(adapter=adapter, max_entries=5)

    report = scanner.scan()
    assert report.truncated


def test_storage_scanner_caching(tmp_path: Path) -> None:
    """StorageScanner returns cached report if within cache_ttl_s."""
    f = tmp_path / "test.dat"
    f.write_bytes(b"12345")

    loc = CacheLocation(path=str(tmp_path), kind="cache", safe_to_clean=True)
    adapter = MockStorageAdapter([loc])
    scanner = StorageScanner(adapter=adapter, cache_ttl_s=100.0)

    report1 = scanner.scan()
    assert report1.entries[0].bytes == 5

    # Write additional bytes
    f.write_bytes(b"1234567890")

    # Second call returns cached result
    report2 = scanner.scan()
    assert report2.entries[0].bytes == 5

    # Force bypasses cache
    report3 = scanner.scan(force=True)
    assert report3.entries[0].bytes == 10
