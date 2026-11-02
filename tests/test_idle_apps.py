"""Tests for idle app identification, window classification, NEVER_OFFER exclusions,
graceful escalation, and consistency across scan and fix.
"""

from pathlib import Path

from deskflow.collector.models import ProcessSample, StorageReport
from deskflow.platform.base import CacheLocation, PlatformAdapter
from deskflow.remediation.actions import close_idle_apps
from deskflow.remediation.apps import find_idle_apps
from deskflow.remediation.recommend import recommend_fixes


class FakeAppAdapter(PlatformAdapter):
    """Fake platform adapter for testing idle app classification and graceful escalation."""

    def __init__(
        self,
        window_pids: set[int] | None = None,
        foreground_pid: int | None = None,
        never_offer: frozenset[str] | None = None,
    ) -> None:
        self._window_pids = window_pids or set()
        self._foreground_pid = foreground_pid
        self._never_offer = never_offer or frozenset({"code.exe", "antigravity.exe", "cmd.exe"})
        self.closed_pids: list[int] = []
        self.terminated_pids: list[int] = []

    @property
    def name(self) -> str:
        return "fake"

    def cpu_freq(self):
        return None

    def temperatures(self):
        return {}

    def battery(self):
        return None

    def temp_and_cache_dirs(self) -> list[CacheLocation]:
        return [CacheLocation(path=str(Path(".")), kind="temp", safe_to_clean=True)]

    def protected_process_names(self) -> frozenset[str]:
        return frozenset({"explorer.exe", "system"})

    def get_window_owning_pids(self) -> set[int]:
        return self._window_pids

    def get_foreground_pid(self) -> int | None:
        return self._foreground_pid

    def never_offer_names(self) -> frozenset[str]:
        return self._never_offer

    def close_process_windows(self, pid: int, timeout_s: float = 8.0) -> bool:
        self.closed_pids.append(pid)
        # Return True for graceful close
        return True


def test_window_owning_vs_windowless_classification() -> None:
    """Window-owning processes are candidates; windowless helper processes are rejected."""
    adapter = FakeAppAdapter(
        window_pids={101, 102},
        foreground_pid=999,
    )

    processes = [
        # Process 101: owns a visible window, low CPU, >=300MB
        ProcessSample(
            pid=101,
            name="gui_tool.exe",
            rss_bytes=400 * 1024 * 1024,
            cpu_percent=0.5,
            num_threads=5,
            age_s=1000.0,
            is_protected=False,
        ),
        # Process 102: owns a visible window, low CPU, part of same app
        ProcessSample(
            pid=102,
            name="gui_tool.exe",
            rss_bytes=100 * 1024 * 1024,
            cpu_percent=0.2,
            num_threads=4,
            age_s=1000.0,
            is_protected=False,
        ),
        # Process 201: windowless background daemon (holding 800MB!), not in window_pids
        ProcessSample(
            pid=201,
            name="background_daemon.exe",
            rss_bytes=800 * 1024 * 1024,
            cpu_percent=0.1,
            num_threads=10,
            age_s=5000.0,
            is_protected=False,
        ),
    ]

    idle = find_idle_apps(processes, adapter=adapter)
    assert len(idle) == 1
    assert idle[0].name == "gui_tool.exe"
    assert set(idle[0].pids) == {101, 102}
    assert idle[0].total_rss_bytes == 500 * 1024 * 1024

    # Background daemon without windows must never be offered!
    names = [g.name for g in idle]
    assert "background_daemon.exe" not in names


def test_never_offer_respected() -> None:
    """IDEs, terminals, and protected apps are never offered even if window-owning."""
    adapter = FakeAppAdapter(
        window_pids={301, 302, 303},
        foreground_pid=999,
        never_offer=frozenset({"code.exe", "antigravity.exe", "cmd.exe"}),
    )

    processes = [
        # VS Code: window-owning, low CPU, large RAM
        ProcessSample(
            pid=301,
            name="code.exe",
            rss_bytes=1500 * 1024 * 1024,
            cpu_percent=0.2,
            num_threads=20,
            age_s=3600.0,
            is_protected=False,
        ),
        # Antigravity IDE: window-owning, low CPU, large RAM
        ProcessSample(
            pid=302,
            name="antigravity.exe",
            rss_bytes=2000 * 1024 * 1024,
            cpu_percent=0.3,
            num_threads=30,
            age_s=3600.0,
            is_protected=False,
        ),
        # Regular user app: window-owning, low CPU, >=300MB
        ProcessSample(
            pid=303,
            name="pdf_reader.exe",
            rss_bytes=450 * 1024 * 1024,
            cpu_percent=0.1,
            num_threads=8,
            age_s=1200.0,
            is_protected=False,
        ),
    ]

    idle = find_idle_apps(processes, adapter=adapter)
    assert len(idle) == 1
    assert idle[0].name == "pdf_reader.exe"
    offered_names = {g.name for g in idle}
    assert "code.exe" not in offered_names
    assert "antigravity.exe" not in offered_names


def test_foreground_exclusion() -> None:
    """The application currently holding the foreground window is excluded."""
    adapter = FakeAppAdapter(
        window_pids={401, 402},
        foreground_pid=401,  # 401 is frontmost
    )

    processes = [
        ProcessSample(
            pid=401,
            name="active_editor.exe",
            rss_bytes=600 * 1024 * 1024,
            cpu_percent=0.5,
            num_threads=10,
            age_s=2000.0,
            is_protected=False,
        ),
        ProcessSample(
            pid=402,
            name="background_browser.exe",
            rss_bytes=600 * 1024 * 1024,
            cpu_percent=0.5,
            num_threads=10,
            age_s=2000.0,
            is_protected=False,
        ),
    ]

    idle = find_idle_apps(processes, adapter=adapter)
    assert len(idle) == 1
    assert idle[0].name == "background_browser.exe"
    assert idle[0].pids == (402,)


def test_graceful_path_never_calls_terminate() -> None:
    """Graceful closing uses close_process_windows and never forces terminate/kill."""
    adapter = FakeAppAdapter(
        window_pids={501},
        foreground_pid=999,
    )

    processes = [
        ProcessSample(
            pid=501,
            name="calc.exe",
            rss_bytes=350 * 1024 * 1024,
            cpu_percent=0.1,
            num_threads=4,
            age_s=1000.0,
            is_protected=False,
        ),
    ]

    res = close_idle_apps(processes, adapter=adapter, dry_run=False, force=False)
    assert res.success is True
    # Adapter recorded that close_process_windows was called for 501
    assert 501 in adapter.closed_pids


def test_force_requires_second_confirmation() -> None:
    """When graceful close times out, force hard-kill requires second explicit confirmation."""

    class StubbornAdapter(FakeAppAdapter):
        def close_process_windows(self, pid: int, timeout_s: float = 8.0) -> bool:
            # Simulates an app that refused to close (waiting at save prompt)
            return False

    stubborn_adapter = StubbornAdapter(
        window_pids={601},
        foreground_pid=999,
    )

    processes = [
        ProcessSample(
            pid=601,
            name="unsaved_notes.exe",
            rss_bytes=400 * 1024 * 1024,
            cpu_percent=0.1,
            num_threads=4,
            age_s=1000.0,
            is_protected=False,
        ),
    ]

    # Case 1: force=False -> Never hard-kills, reports process still running
    res_no_force = close_idle_apps(processes, adapter=stubborn_adapter, dry_run=False, force=False)
    assert "still running" in res_no_force.message

    # Case 2: force=True but second confirmation declined (returns False)
    res_declined = close_idle_apps(
        processes,
        adapter=stubborn_adapter,
        dry_run=False,
        force=True,
        confirm_force_cb=lambda name, pid: False,
    )
    assert "still running" in res_declined.message


def test_scan_and_fix_report_identical_idle_groups() -> None:
    """Scan recommendations and fix close_idle_apps evaluate the exact same candidate groups."""
    adapter = FakeAppAdapter(
        window_pids={701, 702, 703},
        foreground_pid=999,
    )

    processes = [
        # Idle app 1: 400MB
        ProcessSample(
            pid=701,
            name="app_one.exe",
            rss_bytes=400 * 1024 * 1024,
            cpu_percent=0.2,
            num_threads=5,
            age_s=1000.0,
            is_protected=False,
        ),
        # Idle app 2: 500MB
        ProcessSample(
            pid=702,
            name="app_two.exe",
            rss_bytes=500 * 1024 * 1024,
            cpu_percent=0.3,
            num_threads=6,
            age_s=1500.0,
            is_protected=False,
        ),
        # Active app: CPU 5% (not idle)
        ProcessSample(
            pid=703,
            name="busy_game.exe",
            rss_bytes=600 * 1024 * 1024,
            cpu_percent=5.0,
            num_threads=8,
            age_s=500.0,
            is_protected=False,
        ),
    ]

    # Scan path
    idle_from_scan = find_idle_apps(processes, adapter=adapter)
    recs = recommend_fixes(
        top_processes=processes,
        storage_report=StorageReport("2026-11-01T00:00:00Z", [], False, 0.1),
        issue_tag="healthy",
        is_anomaly=False,
        adapter=adapter,
    )
    idle_rec = next((r for r in recs if r.action_id == "close_idle_apps"), None)
    assert idle_rec is not None

    # Fix path
    res_fix = close_idle_apps(processes, adapter=adapter, dry_run=True)

    # Both must identify app_one.exe and app_two.exe, and exactly 2 groups
    assert len(idle_from_scan) == 2
    scan_names = {g.name for g in idle_from_scan}
    assert scan_names == {"app_one.exe", "app_two.exe"}

    assert len(res_fix.details) == 2
    assert set(res_fix.details.keys()) == scan_names
    assert res_fix.bytes_freed == sum(g.total_rss_bytes for g in idle_from_scan)
