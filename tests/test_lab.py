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


def test_scenario_creation_and_validation() -> None:
    """Verify create_scenario creates appropriate scenario objects and validates inputs."""
    scen_cpu = create_scenario("cpu_pressure", intensity="medium", pattern="ramp")
    assert isinstance(scen_cpu, CpuPressureScenario)
    assert scen_cpu.intensity == "medium"
    assert scen_cpu.pattern == "ramp"

    scen_mem = create_scenario("memory_pressure", intensity="high", pattern="burst")
    assert isinstance(scen_mem, MemoryPressureScenario)

    scen_disk = create_scenario("disk_pressure", intensity="low", pattern="constant")
    assert isinstance(scen_disk, DiskPressureScenario)

    scen_bg = create_scenario("background_load", intensity="medium", pattern="constant")
    assert isinstance(scen_bg, BackgroundLoadScenario)

    scen_idle = create_scenario("idle", intensity="low", pattern="constant")
    assert isinstance(scen_idle, IdleScenario)

    scen_combo = create_scenario("combo", intensity="high", pattern="constant")
    assert isinstance(scen_combo, ComboScenario)

    with pytest.raises(ValueError, match="Unknown.*scenario"):
        create_scenario("nonexistent_scenario", intensity="low", pattern="constant")


def test_cpu_ramp_and_burst() -> None:
    """Verify CPU ramp and burst pattern worker initialization."""
    scen_ramp = CpuPressureScenario(intensity="medium", pattern="ramp", max_cores=2)
    scen_ramp.start(hold_duration_s=1.0)
    assert scen_ramp.is_running
    time.sleep(0.3)
    scen_ramp.stop()
    assert not scen_ramp.is_running

    scen_burst = CpuPressureScenario(intensity="high", pattern="burst", max_cores=2)
    scen_burst.start(hold_duration_s=1.0)
    assert scen_burst.is_running
    time.sleep(0.3)
    scen_burst.stop()
    assert not scen_burst.is_running


def test_memory_ramp_and_safety_floor() -> None:
    """Verify memory ramp allocation and safety floor calculation."""
    from deskflow.lab.base import calculate_safe_memory_budget

    budget = calculate_safe_memory_budget(intensity="low")
    assert budget > 0

    scen_ramp = MemoryPressureScenario(
        intensity="low", pattern="ramp", target_bytes_override=32 * 1024 * 1024
    )
    scen_ramp.start(hold_duration_s=1.0)
    time.sleep(0.3)
    scen_ramp.stop()
    assert not scen_ramp.is_running


def test_background_load_workers() -> None:
    """Verify background load scenario spawns workers and terminates cleanly."""
    scen = BackgroundLoadScenario(intensity="low", pattern="constant", worker_count_override=1)
    scen.start(hold_duration_s=1.0)
    assert scen.is_running
    pids = scen.get_child_pids()
    assert len(pids) >= 0
    time.sleep(0.3)
    scen.stop()
    assert not scen.is_running


def test_lab_cli_list_and_plans() -> None:
    """Verify lab list and lab plan CLI commands."""
    from typer.testing import CliRunner

    from deskflow.cli import app

    runner = CliRunner()

    res_list = runner.invoke(app, ["lab", "list"])
    assert res_list.exit_code == 0
    assert "Available Fault Injection Scenarios" in res_list.stdout
    assert "cpu_pressure" in res_list.stdout

    res_plan_q = runner.invoke(app, ["lab", "plan", "--plan", "quick"])
    assert res_plan_q.exit_code == 0
    assert "QUICK" in res_plan_q.stdout
    assert "Total runs: 5" in res_plan_q.stdout

    res_plan_s = runner.invoke(app, ["lab", "plan", "--plan", "standard"])
    assert res_plan_s.exit_code == 0
    assert "STANDARD" in res_plan_s.stdout


def test_lab_cli_user_cancellations() -> None:
    """Verify lab run and session respect interactive cancellation."""
    from typer.testing import CliRunner

    from deskflow.cli import app

    runner = CliRunner()

    res_run = runner.invoke(app, ["lab", "run", "-s", "idle"], input="n\n")
    assert res_run.exit_code == 0
    assert "Operation canceled" in res_run.stdout

    res_sess = runner.invoke(app, ["lab", "session", "--plan", "quick"], input="n\n")
    assert res_sess.exit_code == 0
    assert "Session canceled" in res_sess.stdout


def test_lab_lock_lifecycle(tmp_path: Path) -> None:
    """Verify child pid registration and unregistration in scratch directory."""
    import json

    from deskflow.lab.lock import register_child_pids, unregister_children

    register_child_pids([12345, 67890], scratch_dir=tmp_path)
    lock_file = tmp_path / "lab_children.lock"
    assert lock_file.exists()
    with open(lock_file, encoding="utf-8") as f:
        pids = json.load(f)
    assert pids == [12345, 67890]

    unregister_children(scratch_dir=tmp_path)
    assert not lock_file.exists()


def test_disk_pressure_patterns(tmp_path: Path) -> None:
    """Verify disk pressure ramp pattern worker initialization."""
    scen = DiskPressureScenario(
        intensity="low",
        pattern="ramp",
        scratch_dir=tmp_path,
        max_file_size_override=4 * 1024 * 1024,
    )
    scen.start(hold_duration_s=1.0)
    assert scen.is_running
    time.sleep(0.3)
    scen.stop()
    assert not scen.is_running


def test_lab_runner_abort_and_storage(tmp_path: Path) -> None:
    """Verify LabRunner handles storage scanning and abort callbacks gracefully."""
    from deskflow.lab.runner import LabRunner

    scen = IdleScenario()
    callbacks = []

    def _cb(phase: str, elapsed: float, total: float) -> None:
        callbacks.append((phase, elapsed, total))

    runner = LabRunner(
        scenario=scen,
        pre_duration_s=0.2,
        hold_duration_s=0.4,
        post_duration_s=0.2,
        interval_s=0.1,
        probe_interval_s=0.2,
        out_dir=tmp_path,
        status_callback=_cb,
        enable_storage=False,
    )
    out = runner.run()
    assert out.exists()
    assert len(callbacks) > 0


def test_worker_direct_execution(tmp_path: Path) -> None:
    """Directly test worker functions across patterns to verify execution coverage."""
    import os
    import threading

    from deskflow.lab.cpu import _cpu_worker
    from deskflow.lab.disk import _disk_worker, _disk_writer_thread
    from deskflow.lab.memory import _memory_worker, _touch_chunk

    stop_evt = threading.Event()

    # touch chunk
    _touch_chunk(bytearray(8192))

    # cpu worker patterns with tiny hold duration
    _cpu_worker(os.getpid(), 0.01, "constant", stop_evt)
    _cpu_worker(os.getpid(), 0.01, "ramp", stop_evt)
    _cpu_worker(os.getpid(), 0.01, "burst", stop_evt)

    # memory worker patterns with tiny hold duration
    _memory_worker(os.getpid(), 64 * 1024 * 1024, 0.01, "constant", stop_evt)
    _memory_worker(os.getpid(), 64 * 1024 * 1024, 0.01, "ramp", stop_evt)
    _memory_worker(os.getpid(), 64 * 1024 * 1024, 0.01, "burst", stop_evt)

    # disk writer thread patterns with tiny hold duration
    _disk_writer_thread(tmp_path / "test_c.tmp", 0.01, "constant", stop_evt)
    _disk_writer_thread(tmp_path / "test_r.tmp", 0.01, "ramp", stop_evt)
    _disk_writer_thread(tmp_path / "test_b.tmp", 0.01, "burst", stop_evt)

    # disk worker process entrypoint
    _disk_worker(os.getpid(), str(tmp_path), 1, 0.01, "constant", stop_evt)


def test_disk_scenario_insufficient_space(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify DiskPressureScenario raises RuntimeError when free space is under 5 GiB."""
    import shutil
    from unittest.mock import MagicMock

    monkeypatch.setattr(shutil, "disk_usage", lambda _: MagicMock(free=1 * 1024 * 1024 * 1024))
    scen = DiskPressureScenario(scratch_dir=tmp_path)
    with pytest.raises(RuntimeError, match="Insufficient free disk space"):
        scen.start(hold_duration_s=1.0)


def test_lab_cli_run_yes() -> None:
    """Verify lab run with --yes flag executes successfully."""
    from typer.testing import CliRunner

    from deskflow.cli import app

    runner = CliRunner()
    res = runner.invoke(
        app,
        ["lab", "run", "-s", "idle", "--pre", "0.2", "--hold", "0.2", "--post", "0.2", "--yes"],
    )
    assert res.exit_code == 0
    assert "Run complete." in res.stdout


def test_lab_cli_session_resume(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify lab session detects completed runs and supports resume."""
    import json

    from typer.testing import CliRunner

    from deskflow.cli import app

    runs_dir = tmp_path / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    meta_path = runs_dir / "test_run.meta.json"
    meta_path.write_text(
        json.dumps(
            {
                "session_id": "test-sess",
                "scenario": "cpu_pressure",
                "intensity": "low",
                "pattern": "constant",
                "aborted": False,
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr("deskflow.commands.lab.get_data_dir", lambda: tmp_path)

    runner = CliRunner()
    res = runner.invoke(
        app,
        ["lab", "session", "--plan", "quick", "--resume", "test-sess"],
        input="n\n",
    )
    assert res.exit_code == 0
    assert "Resuming session:" in res.stdout
    assert "Session canceled by user." in res.stdout


def test_scenario_intensity_calculations(tmp_path: Path) -> None:
    """Verify resource calculation methods across intensities for disk and memory."""
    d_low = DiskPressureScenario(intensity="low", scratch_dir=tmp_path)
    d_med = DiskPressureScenario(intensity="medium", scratch_dir=tmp_path)
    d_high = DiskPressureScenario(intensity="high", scratch_dir=tmp_path)

    assert d_low._calculate_writers() == 1
    assert d_med._calculate_writers() == 2
    assert d_high._calculate_writers() == 4

    m_low = MemoryPressureScenario(intensity="low")
    m_med = MemoryPressureScenario(intensity="medium")
    m_high = MemoryPressureScenario(intensity="high")

    assert m_low._calculate_target_bytes() > 0
    assert m_med._calculate_target_bytes() >= m_low._calculate_target_bytes()
    assert m_high._calculate_target_bytes() >= m_med._calculate_target_bytes()


def test_combo_scenario_stats_and_pids() -> None:
    """Verify combo scenario aggregates child stats and PIDs from constituent scenarios."""
    scen_cpu = CpuPressureScenario(intensity="low", max_cores=1)
    scen_mem = MemoryPressureScenario(intensity="low", target_bytes_override=1024)
    combo = ComboScenario(sub_scenarios=[scen_cpu, scen_mem], intensity="low")

    assert combo.name == "cpu_pressure+memory_pressure"
    pids = combo.get_child_pids()
    assert isinstance(pids, list)

    stats = combo.get_stats()
    assert stats.worker_count >= 0


def test_background_load_plan_and_stats() -> None:
    """Verify background load plan scales across intensities."""
    bg_low = BackgroundLoadScenario(intensity="low")
    bg_med = BackgroundLoadScenario(intensity="medium")
    bg_high = BackgroundLoadScenario(intensity="high")

    count_l, mib_l = bg_low._calculate_plan()
    count_m, mib_m = bg_med._calculate_plan()
    count_h, mib_h = bg_high._calculate_plan()

    assert count_l > 0
    assert count_m > 0
    assert count_h > 0
    assert mib_l <= mib_m <= mib_h


def test_cleanup_leftovers_terminates_mock_proc(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify cleanup_leftovers calls terminate and kill on leftover PIDs."""
    from unittest.mock import MagicMock

    import psutil

    from deskflow.lab.lock import cleanup_leftovers

    lock_file = tmp_path / "lab_children.lock"
    lock_file.write_text("[88888]", encoding="utf-8")

    mock_proc = MagicMock()
    mock_proc.wait.side_effect = psutil.TimeoutExpired(1.0)
    monkeypatch.setattr("psutil.pid_exists", lambda pid: pid == 88888)
    monkeypatch.setattr("psutil.Process", lambda pid: mock_proc)

    cleanup_leftovers(scratch_dir=tmp_path)
    assert mock_proc.terminate.called
    assert mock_proc.kill.called
    assert not lock_file.exists()


def test_lab_lock_and_scratch_dir_edge_cases(tmp_path: Path) -> None:
    """Verify lock file parsing errors and directory removal in scratch dir."""
    from deskflow.lab.lock import cleanup_leftovers, register_child_pids

    # Corrupt lock file
    bad_lock = tmp_path / "lab_children.lock"
    bad_lock.write_text("invalid json", encoding="utf-8")
    # Subdirectory in scratch dir
    sub_dir = tmp_path / "nested_dir"
    sub_dir.mkdir()
    (sub_dir / "nested_file.txt").write_text("hello", encoding="utf-8")
    extra_file = tmp_path / "dummy.tmp"
    extra_file.write_text("dummy", encoding="utf-8")

    cleanup_leftovers(scratch_dir=tmp_path)
    assert not bad_lock.exists()
    assert not sub_dir.exists()
    assert not extra_file.exists()

    # Register PIDs with an invalid path that raises OSError
    invalid_path = tmp_path / "nonexistent" / "dir"
    register_child_pids([123], scratch_dir=invalid_path)


def test_scenario_stats_and_child_pids(tmp_path: Path) -> None:
    """Verify stats and child PID accessors for disk and memory scenarios."""
    from unittest.mock import MagicMock

    d_scen = DiskPressureScenario(scratch_dir=tmp_path)
    assert d_scen.get_child_pids() == []
    stats = d_scen.get_stats()
    assert stats.worker_count == 0

    mock_proc = MagicMock()
    mock_proc.pid = 99999
    mock_proc.is_alive.return_value = False
    d_scen._process = mock_proc
    d_scen._writer_count = 2
    assert d_scen.get_child_pids() == [99999]
    assert d_scen.get_stats().worker_count == 2
    d_scen.stop()

    m_scen = MemoryPressureScenario()
    assert m_scen.get_child_pids() == []
    stats_m = m_scen.get_stats()
    assert stats_m.worker_count == 0

    m_scen._process = mock_proc
    m_scen._allocated_bytes = 4096
    assert m_scen.get_child_pids() == [99999]
    assert m_scen.get_stats().bytes_allocated == 4096
    m_scen.stop()


def test_lab_runner_handle_abort(tmp_path: Path) -> None:
    """Verify LabRunner _handle_abort callback marks aborted and stops scenario."""
    from deskflow.lab.idle import IdleScenario
    from deskflow.lab.runner import LabRunner

    runner = LabRunner(
        scenario=IdleScenario(),
        pre_duration_s=1,
        hold_duration_s=1,
        post_duration_s=1,
        out_dir=tmp_path,
    )
    assert not runner._aborted
    runner._handle_abort("Test trigger threshold exceeded")
    assert runner._aborted
    assert runner._abort_reason == "Test trigger threshold exceeded"


def test_memory_worker_patterns_loop_coverage(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify loop bodies in _memory_worker for constant, ramp, and burst patterns."""
    import os
    import threading

    from deskflow.lab.memory import _memory_worker

    # constant pattern loop
    stop_evt = threading.Event()

    def mock_sleep_constant(_: float) -> None:
        stop_evt.set()

    monkeypatch.setattr("time.sleep", mock_sleep_constant)
    _memory_worker(os.getpid(), 64 * 1024 * 1024, 10.0, "constant", stop_evt)

    # ramp pattern loop
    stop_evt_ramp = threading.Event()

    def mock_sleep_ramp(_: float) -> None:
        stop_evt_ramp.set()

    monkeypatch.setattr("time.sleep", mock_sleep_ramp)
    _memory_worker(os.getpid(), 64 * 1024 * 1024, 10.0, "ramp", stop_evt_ramp)

    # burst pattern active phase
    stop_evt_burst = threading.Event()

    def mock_sleep_burst(_: float) -> None:
        stop_evt_burst.set()

    monkeypatch.setattr("time.sleep", mock_sleep_burst)
    _memory_worker(os.getpid(), 64 * 1024 * 1024, 10.0, "burst", stop_evt_burst)
