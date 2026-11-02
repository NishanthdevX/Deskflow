"""Identification and grouping of idle user-facing applications."""

import logging
import os
from dataclasses import dataclass

import psutil

from deskflow.collector.models import ProcessSample
from deskflow.platform.base import PlatformAdapter

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class IdleAppGroup:
    """An application group eligible to be closed."""

    name: str
    pids: tuple[int, ...]
    total_cpu_percent: float
    total_rss_bytes: int
    window_count: int = 1


def get_deskflow_process_tree_pids() -> set[int]:
    """Return all PIDs belonging to Deskflow process tree (self, ancestors, children)."""
    pids: set[int] = {os.getpid()}
    try:
        proc = psutil.Process(os.getpid())
        for p in proc.parents():
            pids.add(p.pid)
        for c in proc.children(recursive=True):
            pids.add(c.pid)
    except Exception:
        pass
    return pids


def find_idle_apps(
    top_processes: list[ProcessSample],
    adapter: PlatformAdapter,
    min_rss_bytes: int = 300 * 1024 * 1024,
    max_cpu_percent: float = 2.0,
) -> list[IdleAppGroup]:
    """Identify user-facing applications that are idle and holding substantial memory.

    Requirements:
    1. Candidates are user-facing apps only:
       Must own at least one visible top-level window with a non-empty title (Windows)
       or background-only false (macOS), as reported by adapter.get_window_owning_pids().
       Processes without windows are never candidates.
    2. Exclude NEVER_OFFER apps (terminals, shells, IDEs, presentation/conferencing,
       shell UI, and the DESKFLOW process tree and all its ancestors).
    3. Exclude foreground window owner (adapter.get_foreground_pid()).
    4. Group candidate processes by executable name (case-insensitive).
    5. A group is idle when:
       - Sum of CPU% < max_cpu_percent (default 2.0%)
       - Sum of RSS >= min_rss_bytes (default 300 MB)
       - None of the PIDs in the group is the foreground window owner.
    """
    window_pids = adapter.get_window_owning_pids()
    foreground_pid = adapter.get_foreground_pid()
    deskflow_tree_pids = get_deskflow_process_tree_pids()
    never_offer = {n.lower() for n in adapter.never_offer_names()}
    protected_names = {n.lower() for n in adapter.protected_process_names()}

    # Group candidate processes by lowercase executable name
    candidates_by_name: dict[str, list[ProcessSample]] = {}

    for p in top_processes:
        # Must be window-owning
        if p.pid not in window_pids:
            continue

        name_lower = p.name.lower()

        # Check NEVER_OFFER and protection
        if name_lower in never_offer or name_lower in protected_names:
            continue
        if p.pid in deskflow_tree_pids or p.pid in (0, 1, 4):
            continue
        if p.is_protected:
            continue

        candidates_by_name.setdefault(name_lower, []).append(p)

    idle_groups: list[IdleAppGroup] = []

    for _name_lower, procs in candidates_by_name.items():
        group_pids = tuple(p.pid for p in procs)

        # Must not own the foreground window
        if foreground_pid is not None and foreground_pid in group_pids:
            continue

        total_cpu = sum(p.cpu_percent for p in procs)
        total_rss = sum(p.rss_bytes for p in procs)

        # Check idle thresholds: total CPU < 2%, total RSS >= 300 MB
        if total_cpu < max_cpu_percent and total_rss >= min_rss_bytes:
            display_name = procs[0].name
            idle_groups.append(
                IdleAppGroup(
                    name=display_name,
                    pids=group_pids,
                    total_cpu_percent=total_cpu,
                    total_rss_bytes=total_rss,
                    window_count=len(group_pids),
                )
            )

    return idle_groups
