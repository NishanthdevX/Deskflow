"""Recommendation engine: maps diagnostic findings and evidence to safe remediation actions."""

from dataclasses import dataclass, field
from typing import Any

from deskflow.collector.models import ProcessSample, StorageReport
from deskflow.platform.base import PlatformAdapter
from deskflow.remediation.apps import find_idle_apps, get_deskflow_process_tree_pids

CLEAN_TEMP_MIN_BYTES = 100 * 1024 * 1024


@dataclass(frozen=True)
class RecommendedAction:
    """A specific suggested remediation action backed by measured system evidence."""

    action_id: str
    title: str
    description: str
    evidence: str
    warning: str | None = None
    params: dict[str, Any] = field(default_factory=dict)


def recommend_fixes(
    top_processes: list[ProcessSample],
    storage_report: StorageReport | None,
    issue_tag: str,
    is_anomaly: bool,
    adapter: PlatformAdapter,
) -> list[RecommendedAction]:
    """Generate prioritized remediation suggestions with concrete measured evidence.

    Never promises hypothetical performance gains; states exact measured evidence.
    """
    recs: list[RecommendedAction] = []

    # 1. Clean Temporary Files
    # Check cleanable temp/cache bytes from storage report (files > 24h old)
    cleanable_bytes = 0
    total_scanned_bytes = 0
    if storage_report and storage_report.entries:
        safe_roots = {
            loc.path.lower() for loc in adapter.temp_and_cache_dirs() if loc.safe_to_clean
        }
        for entry in storage_report.entries:
            if entry.kind in ("temp", "cache") and entry.path.lower() in safe_roots:
                cleanable_bytes += getattr(entry, "cleanable_bytes", entry.bytes)
                total_scanned_bytes += entry.bytes

    if cleanable_bytes >= CLEAN_TEMP_MIN_BYTES:
        mb = cleanable_bytes / (1024 * 1024)
        evidence = (
            f"{mb:.1f} MB in cleanable files (>24h old) scanned"
            if cleanable_bytes > 0
            else "No cleanable temporary files measured"
        )
        recs.append(
            RecommendedAction(
                action_id="clean_temp",
                title="Clean temporary and cache files",
                description="Deletes files older than 24 hours in user temporary directories.",
                evidence=evidence,
            )
        )

    # 2. Close Idle Applications (unified with scan Background apps)
    idle_groups = find_idle_apps(top_processes, adapter=adapter)

    if idle_groups:
        total_procs = sum(len(g.pids) for g in idle_groups)
        total_rss = sum(g.total_rss_bytes for g in idle_groups)
        gb = total_rss / (1024**3)
        names_str = ", ".join(g.name for g in idle_groups)
        ev_text = (
            f"{total_procs} idle processes across {len(idle_groups)} app(s) hold {gb:.1f} GB of RAM"
        )
        recs.append(
            RecommendedAction(
                action_id="close_idle_apps",
                title="Close idle applications",
                description=f"Closes idle processes holding significant memory ({names_str}).",
                evidence=ev_text,
                warning="Unsaved work in those applications may be lost.",
            )
        )

    # 3. End Busy Processes
    deskflow_tree = get_deskflow_process_tree_pids()
    never_offer = {n.lower() for n in adapter.never_offer_names()}
    protected_names = {n.lower() for n in adapter.protected_process_names()}
    busy_candidates = [
        p
        for p in top_processes
        if not p.is_protected
        and p.cpu_percent >= 50.0
        and p.pid > 4
        and p.pid not in deskflow_tree
        and p.name.lower() not in never_offer
        and p.name.lower() not in protected_names
    ]
    if busy_candidates:
        top_busy = sorted(busy_candidates, key=lambda p: p.cpu_percent, reverse=True)[:3]
        desc = ", ".join(f"{p.name} ({p.cpu_percent:.0f}% CPU)" for p in top_busy)
        recs.append(
            RecommendedAction(
                action_id="end_busy_process",
                title="End high-CPU process",
                description="Terminates non-protected user processes utilizing excessive CPU.",
                evidence=f"Active high-load processes: {desc}",
                warning="Terminating active processes may cause unsaved work to be lost.",
            )
        )

    # 4. Trim Memory Working Sets (Windows only)
    if adapter.is_trim_memory_supported():
        heavy_procs = [
            p
            for p in top_processes
            if not p.is_protected and p.rss_bytes >= 50 * 1024 * 1024 and p.pid > 4
        ]
        if heavy_procs and (
            issue_tag in ("memory_pressure", "combo") or is_anomaly or len(recs) == 0
        ):
            total_rss = sum(p.rss_bytes for p in heavy_procs)
            gb = total_rss / (1024**3)
            recs.append(
                RecommendedAction(
                    action_id="trim_memory",
                    title="Trim working set memory",
                    description=(
                        "Reclaims working set pages to pagefile for idle process allocations."
                    ),
                    evidence=f"{len(heavy_procs)} processes hold {gb:.1f} GB of working set memory",
                    warning="Gains are temporary as memory is paged back into RAM on demand.",
                )
            )

    return recs
