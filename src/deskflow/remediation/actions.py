"""Remediation actions: safe, confirmed, dry-run-capable system maintenance actions."""

import logging
import os
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import psutil

from deskflow.collector.models import ProcessSample
from deskflow.platform.base import PlatformAdapter
from deskflow.remediation.apps import find_idle_apps, get_deskflow_process_tree_pids
from deskflow.safety import is_protected

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ActionResult:
    """Outcome of a remediation action execution."""

    action: str
    success: bool
    message: str
    bytes_freed: int = 0
    dry_run: bool = False
    details: dict[str, Any] = field(default_factory=dict)


def clean_temp(
    adapter: PlatformAdapter,
    dry_run: bool = False,
    max_age_seconds: float = 86400.0,
) -> ActionResult:
    """Clean temporary and cache directories older than max_age_seconds (default 24h).

    Safety:
    - Only deletes inside roots returned by adapter where safe_to_clean is True.
    - Never follows symlinks out of or within directories.
    - Deletes only regular files older than 24 hours.
    - Silently skips locked or in-use files.
    - Removes empty directories after file cleanup.
    - Reports exact measured bytes actually deleted.
    """
    now = time.time()
    total_bytes_deleted = 0
    total_files_deleted = 0
    skipped_files = 0

    roots = [loc.path for loc in adapter.temp_and_cache_dirs() if loc.safe_to_clean]

    for root_str in roots:
        root_path = Path(root_str)
        if not root_path.exists() or not root_path.is_dir():
            continue

        # Top-down file walk with no symlink traversal
        for dirpath, dirnames, filenames in os.walk(str(root_path), followlinks=False):
            # Prune symlinked subdirectories from walking
            dirnames[:] = [d for d in dirnames if not os.path.islink(os.path.join(dirpath, d))]

            for fname in filenames:
                file_full = os.path.join(dirpath, fname)
                try:
                    # Never follow symlinks
                    if os.path.islink(file_full):
                        continue

                    # Regular files only
                    if not os.path.isfile(file_full):
                        continue

                    stat_res = os.stat(file_full)
                    # Must be older than 24 hours
                    if (now - stat_res.st_mtime) < max_age_seconds:
                        continue

                    file_size = stat_res.st_size

                    if dry_run:
                        total_bytes_deleted += file_size
                        total_files_deleted += 1
                    else:
                        try:
                            os.remove(file_full)
                            total_bytes_deleted += file_size
                            total_files_deleted += 1
                        except (PermissionError, OSError):
                            skipped_files += 1
                except (PermissionError, OSError):
                    skipped_files += 1

        # Remove empty directories bottom-up
        if not dry_run:
            for dirpath, _dirnames, _filenames in os.walk(
                str(root_path), topdown=False, followlinks=False
            ):
                if dirpath == str(root_path):
                    continue  # Never remove root temp directory
                try:
                    if not os.listdir(dirpath):
                        os.rmdir(dirpath)
                except (PermissionError, OSError):
                    pass

    mb_freed = total_bytes_deleted / (1024 * 1024)
    prefix = "[Dry-run] Would delete" if dry_run else "Deleted"
    msg = f"{prefix} {total_files_deleted} files ({mb_freed:.1f} MB) older than 24 hours."

    return ActionResult(
        action="clean_temp",
        success=True,
        message=msg,
        bytes_freed=total_bytes_deleted,
        dry_run=dry_run,
        details={
            "files_deleted": total_files_deleted,
            "skipped_files": skipped_files,
            "bytes_deleted": total_bytes_deleted,
        },
    )


def clean_app_caches(
    adapter: PlatformAdapter,
    dry_run: bool = False,
    selected_browsers: list[str] | None = None,
) -> ActionResult:
    """Clean browser and app cache directories without touching profiles or history.

    SAFE: Never deletes profiles, cookies, history, or passwords.
    Note: 'pages load slower once'.
    """
    total_bytes_deleted = 0
    total_files_deleted = 0
    skipped_files = 0
    per_browser_bytes: dict[str, int] = {}

    cache_locs = adapter.app_cache_dirs()
    for loc in cache_locs:
        if not loc.safe_to_clean:
            continue
        if (
            selected_browsers
            and loc.label
            and not any(b.lower() in loc.label.lower() for b in selected_browsers)
        ):
            continue

        loc_path = Path(loc.path)
        if not loc_path.exists() or not loc_path.is_dir():
            continue

        browser_key = loc.label or loc_path.name
        loc_bytes = 0

        for dirpath, dirnames, filenames in os.walk(str(loc_path), followlinks=False):
            dirnames[:] = [d for d in dirnames if not os.path.islink(os.path.join(dirpath, d))]
            for fname in filenames:
                file_full = os.path.join(dirpath, fname)
                try:
                    if os.path.islink(file_full) or not os.path.isfile(file_full):
                        continue
                    file_size = os.path.getsize(file_full)
                    if dry_run:
                        total_bytes_deleted += file_size
                        loc_bytes += file_size
                        total_files_deleted += 1
                    else:
                        try:
                            os.remove(file_full)
                            total_bytes_deleted += file_size
                            loc_bytes += file_size
                            total_files_deleted += 1
                        except (PermissionError, OSError):
                            skipped_files += 1
                except (PermissionError, OSError):
                    skipped_files += 1

        per_browser_bytes[browser_key] = per_browser_bytes.get(browser_key, 0) + loc_bytes

    mb_freed = total_bytes_deleted / (1024 * 1024)
    prefix = "[Dry-run] Would clean" if dry_run else "Cleaned"
    msg = (
        f"{prefix} {total_files_deleted} browser/app cache files ({mb_freed:.1f} MB). "
        f"Note: pages load slower once."
    )
    return ActionResult(
        action="clean_app_caches",
        success=True,
        message=msg,
        bytes_freed=total_bytes_deleted,
        dry_run=dry_run,
        details={
            "files_deleted": total_files_deleted,
            "skipped_files": skipped_files,
            "per_browser_bytes": per_browser_bytes,
        },
    )


def empty_recycle_bin(
    adapter: PlatformAdapter,
    dry_run: bool = False,
    confirmed: bool = False,
) -> ActionResult:
    """Empty system recycle bin or trash.

    CAUTION: Permanent deletion.
    """
    items_count, size_bytes = adapter.query_recycle_bin()
    mb = size_bytes / (1024 * 1024)

    if items_count == 0 or size_bytes == 0:
        return ActionResult(
            action="empty_recycle_bin",
            success=True,
            message="Recycle Bin is already empty.",
            bytes_freed=0,
            dry_run=dry_run,
            details={"items_count": 0, "size_bytes": 0},
        )

    if dry_run:
        return ActionResult(
            action="empty_recycle_bin",
            success=True,
            message=f"[Dry-run] Would empty Recycle Bin: {items_count} items ({mb:.1f} MB).",
            bytes_freed=size_bytes,
            dry_run=True,
            details={"items_count": items_count, "size_bytes": size_bytes},
        )

    if not confirmed:
        return ActionResult(
            action="empty_recycle_bin",
            success=False,
            message="Recycle Bin emptying requires explicit confirmation.",
            bytes_freed=0,
            dry_run=False,
            details={"items_count": items_count, "size_bytes": size_bytes},
        )

    success = adapter.empty_recycle_bin()
    msg = (
        f"Emptied Recycle Bin: {items_count} items ({mb:.1f} MB freed)."
        if success
        else "Failed to empty Recycle Bin."
    )
    return ActionResult(
        action="empty_recycle_bin",
        success=success,
        message=msg,
        bytes_freed=size_bytes if success else 0,
        dry_run=False,
        details={"items_count": items_count, "size_bytes": size_bytes},
    )


def close_idle_apps(
    processes: list[ProcessSample],
    adapter: PlatformAdapter | None = None,
    dry_run: bool = False,
    force: bool = False,
    min_group_rss_mb: float = 300.0,
    confirm_force_cb: Callable[[str, int], bool] | None = None,
) -> ActionResult:
    """Close non-protected idle applications owned by user holding significant memory.

    Escalation rules:
    - Candidates found via unified find_idle_apps(processes, adapter).
    - Window-owning visible apps only; NEVER_OFFER and foreground apps strictly excluded.
    - Graceful-first: posts WM_CLOSE (Windows) or AppleScript quit (macOS), waits up to 8s.
    - If app is still running (e.g. at save prompt):
      NEVER hard-kill unless force=True AND confirm_force_cb returns True.
    """
    from deskflow.platform.factory import get_adapter

    eff_adapter = adapter or get_adapter()
    min_rss_bytes = int(min_group_rss_mb * 1024 * 1024)
    idle_groups = find_idle_apps(
        top_processes=processes,
        adapter=eff_adapter,
        min_rss_bytes=min_rss_bytes,
        max_cpu_percent=2.0,
    )

    if not idle_groups:
        return ActionResult(
            action="close_idle_apps",
            success=True,
            message=f"No idle application groups exceeding {min_group_rss_mb:.0f} MB found.",
            dry_run=dry_run,
        )

    closed_count = 0
    bytes_freed = 0
    details: dict[str, Any] = {}
    still_running: list[str] = []

    for group in idle_groups:
        details[group.name] = {
            "count": len(group.pids),
            "rss_bytes": group.total_rss_bytes,
            "pids": list(group.pids),
        }

        if dry_run:
            closed_count += len(group.pids)
            bytes_freed += group.total_rss_bytes
            continue

        group_freed = 0
        for pid in group.pids:
            # Graceful close first
            closed = eff_adapter.close_process_windows(pid, timeout_s=8.0)
            if not closed and force:
                confirmed = False
                if confirm_force_cb is not None:
                    confirmed = confirm_force_cb(group.name, pid)
                if confirmed:
                    try:
                        proc = psutil.Process(pid)
                        proc.kill()
                        closed = True
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        closed = True
                else:
                    closed = False

            if closed:
                closed_count += 1
                try:
                    p_match = next((p for p in processes if p.pid == pid), None)
                    group_freed += p_match.rss_bytes if p_match else 0
                except Exception:
                    pass
            else:
                still_running.append(f"{group.name} (PID {pid})")

        bytes_freed += group_freed

    prefix = "[Dry-run] Would close" if dry_run else "Closed"
    mb = bytes_freed / (1024 * 1024)
    msg = (
        f"{prefix} {closed_count} idle processes across "
        f"{len(idle_groups)} app groups holding {mb:.1f} MB."
    )
    if still_running:
        msg += f" Note: {len(still_running)} process(es) still running: {', '.join(still_running)}."

    return ActionResult(
        action="close_idle_apps",
        success=True,
        message=msg,
        bytes_freed=bytes_freed,
        dry_run=dry_run,
        details=details,
    )


def end_busy_process(
    processes: list[ProcessSample],
    adapter: PlatformAdapter | None = None,
    dry_run: bool = False,
    force: bool = False,
    confirm_force_cb: Callable[[str, int], bool] | None = None,
) -> ActionResult:
    """End up to 3 non-protected user processes using >= 50% CPU using graceful-first escalation."""
    from deskflow.platform.factory import get_adapter

    eff_adapter = adapter or get_adapter()
    deskflow_tree_pids = get_deskflow_process_tree_pids()
    never_offer = {n.lower() for n in eff_adapter.never_offer_names()}
    protected_names = {n.lower() for n in eff_adapter.protected_process_names()}

    busy_procs = [
        p
        for p in processes
        if not p.is_protected
        and p.cpu_percent >= 50.0
        and p.pid > 4
        and p.pid not in deskflow_tree_pids
        and p.name.lower() not in never_offer
        and p.name.lower() not in protected_names
    ]
    # Sort descending by CPU
    busy_procs = sorted(busy_procs, key=lambda p: p.cpu_percent, reverse=True)[:3]

    if not busy_procs:
        return ActionResult(
            action="end_busy_process",
            success=True,
            message="No non-protected processes using >= 50% CPU found.",
            dry_run=dry_run,
        )

    ended_count = 0
    details: dict[str, Any] = {}
    still_running: list[str] = []

    for p in busy_procs:
        details[f"{p.name}_{p.pid}"] = {
            "cpu_percent": p.cpu_percent,
            "rss_bytes": p.rss_bytes,
        }
        if dry_run:
            ended_count += 1
            continue

        closed = eff_adapter.close_process_windows(p.pid, timeout_s=8.0)
        if not closed:
            if force:
                confirmed = False
                if confirm_force_cb is not None:
                    confirmed = confirm_force_cb(p.name, p.pid)
                if confirmed:
                    try:
                        proc = psutil.Process(p.pid)
                        proc.kill()
                        closed = True
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        closed = True
            else:
                closed = False

        if closed:
            ended_count += 1
        else:
            still_running.append(f"{p.name} (PID {p.pid})")

    prefix = "[Dry-run] Would end" if dry_run else "Ended"
    msg = f"{prefix} {ended_count} high-CPU process(es)."
    if still_running:
        msg += f" Note: {len(still_running)} process(es) still running: {', '.join(still_running)}."

    return ActionResult(
        action="end_busy_process",
        success=True,
        message=msg,
        dry_run=dry_run,
        details=details,
    )


@dataclass(frozen=True)
class HeavyTaskGroup:
    """Group of processes sharing an executable name ranked as heavy."""

    name: str
    pids: tuple[int, ...]
    cpu_percent: float
    rss_bytes: int
    process_count: int


def find_heavy_tasks(
    adapter: PlatformAdapter,
    all_processes: list[ProcessSample] | None = None,
) -> list[HeavyTaskGroup]:
    """Find top 5 by CPU and top 5 by memory from a full process pass.

    Never touches deskflow tree, protected names, or NEVER_OFFER names.
    """
    procs = all_processes if all_processes is not None else adapter.full_process_snapshot()
    deskflow_tree_pids = get_deskflow_process_tree_pids()
    never_offer = {n.lower() for n in adapter.never_offer_names()}
    protected = {n.lower() for n in adapter.protected_process_names()}

    # Group by lowercase executable name
    by_name: dict[str, list[ProcessSample]] = {}
    for p in procs:
        if p.is_protected or p.pid <= 4 or p.pid in deskflow_tree_pids:
            continue
        p_name_lower = p.name.lower()
        p_base = p_name_lower[:-4] if p_name_lower.endswith(".exe") else p_name_lower
        if (
            p_name_lower in never_offer
            or p_base in never_offer
            or p_name_lower in protected
            or p_base in protected
        ):
            continue
        by_name.setdefault(p_name_lower, []).append(p)

    groups: list[HeavyTaskGroup] = []
    for _name_lower, plist in by_name.items():
        disp_name = plist[0].name
        tot_cpu = sum(p.cpu_percent for p in plist)
        tot_rss = sum(p.rss_bytes for p in plist)
        groups.append(
            HeavyTaskGroup(
                name=disp_name,
                pids=tuple(p.pid for p in plist),
                cpu_percent=tot_cpu,
                rss_bytes=tot_rss,
                process_count=len(plist),
            )
        )

    # Top 5 by CPU and Top 5 by memory
    top_cpu = sorted(groups, key=lambda g: g.cpu_percent, reverse=True)[:5]
    top_mem = sorted(groups, key=lambda g: g.rss_bytes, reverse=True)[:5]

    # Union preserving order (CPU top first, then MEM top)
    combined: list[HeavyTaskGroup] = []
    seen = set()
    for g in top_cpu + top_mem:
        if g.name.lower() not in seen:
            seen.add(g.name.lower())
            combined.append(g)

    return combined


def end_heavy_tasks(
    adapter: PlatformAdapter,
    dry_run: bool = False,
    force: bool = False,
    task_names: list[str] | None = None,
    all_processes: list[ProcessSample] | None = None,
    confirm_force_cb: Callable[[str, int], bool] | None = None,
) -> ActionResult:
    """End top heavy tasks identified from a full process pass.

    CAUTION: Graceful-first escalation, NEVER_OFFER protected.
    """
    candidates = find_heavy_tasks(adapter, all_processes=all_processes)
    if task_names:
        selected_set = {t.lower() for t in task_names}
        candidates = [g for g in candidates if g.name.lower() in selected_set]

    if not candidates:
        return ActionResult(
            action="end_heavy_tasks",
            success=True,
            message="No eligible heavy background tasks found.",
            dry_run=dry_run,
        )

    ended_count = 0
    total_freed_bytes = 0
    details: dict[str, Any] = {}
    still_running: list[str] = []

    for group in candidates:
        details[group.name] = {
            "pids": list(group.pids),
            "cpu_percent": group.cpu_percent,
            "rss_bytes": group.rss_bytes,
            "process_count": group.process_count,
        }
        if dry_run:
            ended_count += len(group.pids)
            total_freed_bytes += group.rss_bytes
            continue

        for pid in group.pids:
            closed = adapter.close_process_windows(pid, timeout_s=8.0)
            if not closed and force:
                confirmed = False
                if confirm_force_cb is not None:
                    confirmed = confirm_force_cb(group.name, pid)
                if confirmed:
                    try:
                        proc = psutil.Process(pid)
                        proc.kill()
                        closed = True
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        closed = True
            if closed:
                ended_count += 1
                total_freed_bytes += group.rss_bytes // max(1, group.process_count)
            else:
                still_running.append(f"{group.name} (PID {pid})")

    prefix = "[Dry-run] Would end" if dry_run else "Ended"
    mb = total_freed_bytes / (1024 * 1024)
    msg = (
        f"{prefix} {ended_count} process(es) across "
        f"{len(candidates)} heavy task(s) holding {mb:.1f} MB."
    )
    if still_running:
        msg += f" Note: {len(still_running)} process(es) still running: {', '.join(still_running)}."

    return ActionResult(
        action="end_heavy_tasks",
        success=True,
        message=msg,
        bytes_freed=total_freed_bytes,
        dry_run=dry_run,
        details=details,
    )


def trim_memory(
    processes: list[ProcessSample],
    adapter: PlatformAdapter,
    dry_run: bool = False,
) -> ActionResult:
    """Trim working sets for non-protected processes with RSS >= 50 MB (Windows only).

    Measures available memory before and 2s after.
    Gains are temporary as memory will be paged back in on demand.
    """
    if not adapter.is_trim_memory_supported():
        return ActionResult(
            action="trim_memory",
            success=False,
            message="Working set memory trimming is not supported on this platform.",
            dry_run=dry_run,
        )

    candidates = [
        p for p in processes if not p.is_protected and p.rss_bytes >= 50 * 1024 * 1024 and p.pid > 4
    ]

    if not candidates:
        return ActionResult(
            action="trim_memory",
            success=True,
            message="No non-protected processes with RSS >= 50 MB found.",
            dry_run=dry_run,
        )

    total_rss = sum(p.rss_bytes for p in candidates)
    mem_before = psutil.virtual_memory().available

    if dry_run:
        mb = total_rss / (1024 * 1024)
        msg = (
            f"[Dry-run] Would trim working set for {len(candidates)} processes "
            f"({mb:.1f} MB RSS). Note: gains are temporary."
        )
        return ActionResult(
            action="trim_memory",
            success=True,
            message=msg,
            bytes_freed=total_rss,
            dry_run=True,
            details={"candidates_count": len(candidates), "total_rss_bytes": total_rss},
        )

    trimmed_count = 0
    for p in candidates:
        if is_protected(p.name, p.pid):
            continue
        if adapter.trim_process_memory(p.pid):
            trimmed_count += 1

    # Measure available memory 2 seconds after as required by spec
    time.sleep(2.0)
    mem_after = psutil.virtual_memory().available
    freed = max(0, mem_after - mem_before)
    mb_freed = freed / (1024 * 1024)

    mb_before = mem_before / (1024 * 1024)
    mb_after = mem_after / (1024 * 1024)
    msg = (
        f"Trimmed working sets for {trimmed_count} processes. "
        f"Available memory changed by +{mb_freed:.1f} MB (before: {mb_before:.1f} MB, "
        f"after: {mb_after:.1f} MB). Note: gains are temporary."
    )

    return ActionResult(
        action="trim_memory",
        success=True,
        message=msg,
        bytes_freed=freed,
        dry_run=False,
        details={
            "trimmed_count": trimmed_count,
            "mem_before_bytes": mem_before,
            "mem_after_bytes": mem_after,
        },
    )
