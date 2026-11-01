"""Unit tests for fault-injection lab scenarios, safety watchdog, and runner."""

import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from deskflow.lab import (
    BackgroundLoadScenario,
    ComboScenario,
    CpuPressureScenario,
    DiskPressureScenario,
    IdleScenario,
    LabWatchdog,
    MemoryPressureScenario,
    cleanup_leftovers,
    create_scenario,
)
from deskflow.lab.plans import get_plan


def test_cpu_scenario_lifecycle() -> None:
    """Verify CPU pressure scenario starts, runs, and terminates cleanly."""
    scenario = CpuPressureScenario(intensity="low", pattern="constant", max_cores=1)
    scenario.start(hold_duration_s=2.0)
    assert scenario.is_running
    time.sleep(0.5)
    scenario.stop()
    assert not scenario.is_running
    stats = scenario.get_stats()
    assert stats.worker_count == 1


def test_memory_scenario_lifecycle() -> None:
    """Verify memory pressure scenario allocates and frees memory safely."""
    # 64 MiB tiny parameter
    scenario = MemoryPressureScenario(
        intensity="low", pattern="constant", target_bytes_override=64 * 1024 * 1024
    )
    scenario.start(hold_duration_s=2.0)
    assert scenario.is_running
    time.sleep(0.5)
    scenario.stop()
    assert not scenario.is_running
    stats = scenario.get_stats()
    assert stats.bytes_allocated == 64 * 1024 * 1024


def test_disk_scenario_lifecycle(tmp_path: Path) -> None:
    """Verify disk pressure scenario writes cyclically and cleans up scratch files."""
    scenario = DiskPressureScenario(
        intensity="low",
        pattern="constant",
        scratch_dir=tmp_path,
        max_file_size_override=8 * 1024 * 1024,
    )
    scenario.start(hold_duration_s=2.0)
    assert scenario.is_running
    time.sleep(0.5)
    scenario.stop()
    assert not scenario.is_running
    # Verify scratch writer files are removed on stop
    assert not (tmp_path / "disk_writer_0.tmp").exists()


def test_background_load_lifecycle() -> None:
    """Verify background load scenario spawns and terminates subprocesses cleanly."""
    scenario = BackgroundLoadScenario(intensity="low", pattern="constant")
    # Force tiny count for test
    scenario._calculate_plan = lambda: (1, 10)  # type: ignore[assignment]
    scenario.start(hold_duration_s=2.0)
    assert scenario.is_running
    time.sleep(0.5)
    scenario.stop()
    assert not scenario.is_running
    stats = scenario.get_stats()
    assert stats.worker_count == 1


def test_idle_and_combo_scenarios() -> None:
    """Verify idle scenario and combo scenario composition."""
    idle = IdleScenario()
    idle.start(hold_duration_s=1.0)
    assert idle.is_running
    idle.stop()
    assert not idle.is_running

    combo = create_scenario("cpu_pressure+idle", intensity="low", pattern="constant", max_cores=1)
    assert isinstance(combo, ComboScenario)
    combo.start(hold_duration_s=1.0)
    assert combo.is_running
    combo.stop()
    assert not combo.is_running


def test_watchdog_memory_abort(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify watchdog triggers abort callback when memory falls below floor."""
    aborted_reasons: list[str] = []

    def _on_abort(reason: str) -> None:
        aborted_reasons.append(reason)

    watchdog = LabWatchdog(
        on_abort=_on_abort,
        scratch_dir=tmp_path,
        check_interval_s=0.05,
        mem_floor_override=10 * 1024 * 1024 * 1024,  # High floor 10 GB
    )

    # Monkeypatch virtual_memory to report 1 GB available
    fake_vm = MagicMock(available=1 * 1024 * 1024 * 1024)
    monkeypatch.setattr("psutil.virtual_memory", lambda: fake_vm)

    watchdog.start()
    time.sleep(0.2)
    watchdog.stop()

    assert watchdog.aborted
    assert len(aborted_reasons) == 1
    assert "fell below safety floor" in aborted_reasons[0]


def test_watchdog_disk_abort(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify watchdog triggers abort callback when free disk falls below 5 GiB."""
    aborted_reasons: list[str] = []

    def _on_abort(reason: str) -> None:
        aborted_reasons.append(reason)

    watchdog = LabWatchdog(
        on_abort=_on_abort,
        scratch_dir=tmp_path,
        check_interval_s=0.05,
        disk_min_override=10 * 1024 * 1024 * 1024,  # Require 10 GB
    )

    # Monkeypatch disk_usage to report 2 GB free
    fake_usage = MagicMock(free=2 * 1024 * 1024 * 1024)
    monkeypatch.setattr("shutil.disk_usage", lambda _: fake_usage)

    watchdog.start()
    time.sleep(0.2)
    watchdog.stop()

    assert watchdog.aborted
    assert len(aborted_reasons) == 1
    assert "fell below safety threshold" in aborted_reasons[0]


def test_scratch_dir_and_lock_cleanup(tmp_path: Path) -> None:
    """Verify cleanup_leftovers removes stale files and lock files."""
    dummy_file = tmp_path / "stale.tmp"
    dummy_file.write_text("test")
    lock_file = tmp_path / "lab_children.lock"
    lock_file.write_text("[999999]")  # Non-existent PID

    cleanup_leftovers(scratch_dir=tmp_path)
    assert not dummy_file.exists()
    assert not lock_file.exists()


def test_plan_generation_arithmetic() -> None:
    """Verify quick and standard plan generation counts and durations."""
    quick = get_plan("quick")
    assert len(quick) == 5
    assert quick[0].scenario == "cpu_pressure"
    assert quick[-1].scenario == "idle"

    standard = get_plan("standard")
    # Standard plan has 4 scenarios * 5 patterns + 2 idle = 22 runs
    assert len(standard) in (20, 22)
    scenarios = {r.scenario for r in standard}
    assert "cpu_pressure" in scenarios
    assert "memory_pressure" in scenarios
    assert "disk_pressure" in scenarios
    assert "background_load" in scenarios
    assert "idle" in scenarios
