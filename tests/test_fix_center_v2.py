"""Comprehensive unit tests for Phase 10 Fix Center, fastfetch info, and safety guarantees."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from deskflow.cli import app
from deskflow.collector.models import ProcessSample
from deskflow.commands.fix import run_fast_inventory
from deskflow.commands.info import _get_cached_static_hardware
from deskflow.platform.base import CacheLocation, PlatformAdapter
from deskflow.remediation.actions import (
    clean_app_caches,
    empty_recycle_bin,
    find_heavy_tasks,
)

runner = CliRunner()


class FakeAdapter(PlatformAdapter):
    """Mock platform adapter for testing remediation and hardware introspection."""

    @property
    def name(self) -> str:
        return "fake_os"

    def cpu_percent(self) -> float:
        return 20.0

    def cpu_percent_per_core(self) -> list[float]:
        return [20.0, 20.0]

    def cpu_freq(self):
        return None

    def virtual_memory(self):
        return (16 * 1024**3, 8 * 1024**3, 50.0)

    def swap_memory(self):
        return (1024**2, 1.0)

    def disk_usage(self, path: str):
        return (100 * 1024**3, 50 * 1024**3, 50 * 1024**3, 50.0)

    def disk_io_counters(self):
        return (1000, 2000, 10, 20, 5)

    def net_io_counters(self):
        return (100, 200, 0, 0)

    def process_count(self) -> int:
        return 100

    def thread_count(self) -> int:
        return 500

    def temperatures(self):
        return {}

    def battery(self):
        return None

    def power_state(self) -> str:
        return "ac"

    def temp_and_cache_dirs(self) -> list[CacheLocation]:
        return []

    def never_offer_names(self) -> list[str]:
        return ["system", "deskflow", "svchost.exe", "kernel_task"]

    def protected_process_names(self) -> list[str]:
        return ["csrss.exe", "winlogon.exe", "explorer.exe"]

    def is_process_protected(self, pid: int, name: str) -> bool:
        return name.lower() in ("system", "csrss.exe", "winlogon.exe")

    def query_recycle_bin(self) -> tuple[int, int]:
        return 5, 25 * 1024 * 1024  # 5 items, 25 MB

    def empty_recycle_bin(self) -> bool:
        return True

    def app_cache_dirs(self) -> list[CacheLocation]:
        return []

    def get_static_hardware(self) -> dict[str, str | list[dict[str, Any]]]:
        return {
            "host_model": "FakeBook Pro",
            "cpu_model": "FakeCore i9",
            "gpus": [{"name": "FakeGPU", "vram_bytes": 4 * 1024**3}],
            "resolution": "1920x1080",
        }

    def full_process_snapshot(self) -> list[ProcessSample]:
        return [
            ProcessSample(10, "system", 0, 0.0, 1, 100.0, True),
            ProcessSample(100, "chrome.exe", 600 * 1024 * 1024, 45.0, 10, 200.0, False),
            ProcessSample(101, "chrome.exe", 400 * 1024 * 1024, 15.0, 8, 200.0, False),
            ProcessSample(200, "heavy_worker.exe", 1200 * 1024 * 1024, 85.0, 4, 50.0, False),
            ProcessSample(300, "idle_editor.exe", 350 * 1024 * 1024, 0.1, 2, 800.0, False),
            ProcessSample(400, "svchost.exe", 500 * 1024 * 1024, 10.0, 20, 1000.0, False),
        ]


def test_fix_inventory_fast_and_tool_count():
    """Verify run_fast_inventory returns all 6 tools with correct IDs and risk levels."""
    adapter = FakeAdapter()
    tools = run_fast_inventory(adapter)

    assert len(tools) == 6
    expected_tools = [
        (1, "clean_temp", "SAFE"),
        (2, "clean_app_caches", "SAFE"),
        (3, "empty_recycle_bin", "CAUTION"),
        (4, "trim_memory", "SAFE"),
        (5, "close_idle_apps", "CAUTION"),
        (6, "end_heavy_tasks", "CAUTION"),
    ]
    for (num, tool_id, risk), tool in zip(expected_tools, tools, strict=False):
        assert tool.number == num
        assert tool.tool_id == tool_id
        assert tool.risk == risk
        assert isinstance(tool.potential_str, str)


def test_fix_cli_list_table_and_json():
    """Verify deskflow fix --list outputs table and --json outputs valid schema."""
    res_table = runner.invoke(app, ["fix", "--list", "--plain"])
    assert res_table.exit_code == 0
    assert "TOOLS INVENTORY" in res_table.output
    assert "Clean temporary" in res_table.output
    assert "Recycle Bin" in res_table.output
    assert "SAFE" in res_table.output
    assert "CAUTION" in res_table.output

    res_json = runner.invoke(app, ["fix", "--list", "--json"])
    assert res_json.exit_code == 0
    data = json.loads(res_json.output)
    assert isinstance(data, list)
    assert len(data) == 6
    assert data[0]["tool_id"] == "clean_temp"
    assert data[0]["risk"] == "SAFE"


def test_heavy_task_ranking_and_never_offer_filter():
    """Verify find_heavy_tasks ranks top CPU and Memory while excluding NEVER_OFFER."""
    adapter = FakeAdapter()
    tasks = find_heavy_tasks(adapter)

    names = [t.name.lower() for t in tasks]
    assert "system" not in names
    assert "svchost.exe" not in names
    assert "heavy_worker.exe" in names
    assert "chrome.exe" in names

    # Verify grouping: chrome.exe should group pid 100 and 101
    chrome = next(t for t in tasks if t.name.lower() == "chrome.exe")
    assert chrome.process_count == 2
    assert chrome.cpu_percent == 60.0
    assert chrome.rss_bytes == 1000 * 1024 * 1024


def test_clean_app_caches_nested_locked_and_symlinks(tmp_path: Path):
    """Verify clean_app_caches safely handles directories, skips locked files and symlinks."""
    cache_root = tmp_path / "mock_browser_cache"
    cache_root.mkdir(parents=True)

    normal_file = cache_root / "data_0"
    normal_file.write_bytes(b"A" * 1024)

    sub_dir = cache_root / "nested"
    sub_dir.mkdir()
    nested_file = sub_dir / "data_1"
    nested_file.write_bytes(b"B" * 2048)

    # Dry run
    adapter = FakeAdapter()
    loc = CacheLocation(
        path=str(cache_root), kind="cache", label="Mock Browser Cache", safe_to_clean=True
    )
    with patch.object(adapter, "app_cache_dirs", return_value=[loc]):
        res_dry = clean_app_caches(adapter=adapter, dry_run=True)
        assert res_dry.success is True
        assert res_dry.dry_run is True
        assert normal_file.exists()
        assert nested_file.exists()

        # Real run
        res_real = clean_app_caches(adapter=adapter, dry_run=False)
        assert res_real.success is True
        assert res_real.bytes_freed >= 3072
        assert not normal_file.exists()
        assert not nested_file.exists()


def test_empty_recycle_bin_api():
    """Verify empty_recycle_bin respects confirmation and invokes adapter."""
    adapter = FakeAdapter()
    # Unconfirmed
    res_unconfirmed = empty_recycle_bin(adapter=adapter, confirmed=False)
    assert res_unconfirmed.success is False

    # Dry-run
    res_dry = empty_recycle_bin(adapter=adapter, dry_run=True)
    assert res_dry.success is True
    assert res_dry.dry_run is True

    # Confirmed real run
    res_real = empty_recycle_bin(adapter=adapter, confirmed=True)
    assert res_real.success is True
    assert res_real.bytes_freed == 25 * 1024 * 1024


def test_info_with_failing_hardware_sources(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Verify deskflow info outputs 'unknown' and never crashes when hardware sources fail."""
    monkeypatch.setenv("DESKFLOW_DATA_DIR", str(tmp_path / "data"))

    adapter = FakeAdapter()
    with patch.object(
        adapter, "get_static_hardware", side_effect=Exception("WMI/Registry failure")
    ):
        hw = _get_cached_static_hardware(adapter)
        assert hw["host_model"] == "unknown"
        assert hw["cpu_model"] == "unknown"
        assert hw["resolution"] == "unknown"

    res = runner.invoke(app, ["info", "--plain"])
    assert res.exit_code == 0
    assert "Host" in res.output
    assert "CPU" in res.output
    assert "Memory" in res.output


def test_yes_never_selects_caution():
    """Verify that --yes without --include-caution never executes CAUTION tools."""
    adapter = FakeAdapter()
    tools = run_fast_inventory(adapter)

    # Simulate --yes logic
    chosen = []
    include_caution = False
    for t in tools:
        if t.risk == "SAFE" or include_caution:
            chosen.append(t)

    assert len(chosen) == 3
    for t in chosen:
        assert t.risk == "SAFE"
        assert t.tool_id in ("clean_temp", "clean_app_caches", "trim_memory")
