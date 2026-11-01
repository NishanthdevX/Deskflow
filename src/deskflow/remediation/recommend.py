"""Recommendation engine: maps diagnostic findings and evidence to safe remediation actions."""

from dataclasses import dataclass, field
from typing import Any

from deskflow.collector.models import ProcessSample, StorageReport
from deskflow.platform.base import PlatformAdapter


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
    # Check cleanable temp/cache bytes from storage report
    cleanable_bytes = 0
    if storage_report and storage_report.entries:
        # Check roots from adapter that are marked safe_to_clean
        safe_roots = {
            loc.path.lower() for loc in adapter.temp_and_cache_dirs() if loc.safe_to_clean
        }
        for entry in storage_report.entries:
            if entry.kind in ("temp", "cache") and entry.path.lower() in safe_roots:
                cleanable_bytes += entry.bytes

    # If storage report wasn't available or had entries >= 50 MB, or disk_pressure/anomaly
    if cleanable_bytes >= 50 * 1024 * 1024 or issue_tag == "disk_pressure" or is_anomaly:
        mb = cleanable_bytes / (1024 * 1024)
        evidence = (
            f"{mb:.1f} MB in temporary/cache files scanned"
            if cleanable_bytes > 0
            else "Temporary file accumulation detected"
        )
        recs.append(
            RecommendedAction(
                action_id="clean_temp",
                title="Clean temporary and cache files",
                description="Deletes files older than 24 hours in user temporary directories.",
                evidence=evidence,
            )
        )

    # 2. Close Idle Applications
    idle_candidates = [
        p
        for p in top_processes
        if not p.is_protected and p.cpu_percent < 1.0 and p.age_s > 600.0 and p.pid > 4
    ]
    groups: dict[str, list[ProcessSample]] = {}
    for p in idle_candidates:
        groups.setdefault(p.name, []).append(p)

    large_idle_groups = {
        name: procs
        for name, procs in groups.items()
        if (sum(p.rss_bytes for p in procs) / (1024 * 1024)) >= 300.0
    }

    if large_idle_groups:
        total_procs = sum(len(procs) for procs in large_idle_groups.values())
        total_rss = sum(sum(p.rss_bytes for p in procs) for procs in large_idle_groups.values())
        gb = total_rss / (1024**3)
        names_str = ", ".join(large_idle_groups.keys())
        recs.append(
            RecommendedAction(
                action_id="close_idle_apps",
                title="Close idle applications",
                description=f"Closes idle processes holding significant memory ({names_str}).",
                evidence=f"{total_procs} idle processes hold {gb:.1f} GB of RAM",
                warning="Unsaved work in those applications may be lost.",
            )
        )

    # 3. End Busy Processes
    busy_candidates = [
        p for p in top_processes if not p.is_protected and p.cpu_percent >= 50.0 and p.pid > 4
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
