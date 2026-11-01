"""Feature extraction, target alignment, and dataset row formulation."""

import logging
import math
from datetime import datetime
from typing import Any

import numpy as np

from deskflow.baseline import Baseline
from deskflow.collector.models import Record
from deskflow.features.registry import (
    SLOW_THRESHOLD,
)
from deskflow.features.window import (
    compute_drop,
    compute_max,
    compute_mean,
    compute_min,
    compute_slope,
    compute_std,
)

logger = logging.getLogger(__name__)


def _parse_utc_iso(ts_str: str) -> float:
    """Parse ISO UTC string to POSIX timestamp seconds."""
    dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
    return dt.timestamp()


def _estimate_sampling_interval(records: list[Record]) -> float:
    """Estimate nominal telemetry interval in seconds from record deltas."""
    if len(records) < 2:
        return 2.0
    deltas = [
        records[i + 1].sample.monotonic_s - records[i].sample.monotonic_s
        for i in range(min(50, len(records) - 1))
    ]
    med = float(np.median(deltas)) if deltas else 2.0
    return max(0.5, med)


def _estimate_probe_interval(probe_records: list[Record]) -> float:
    """Estimate nominal probe interval in seconds from probe record deltas."""
    if len(probe_records) < 2:
        return 4.0
    deltas = [
        probe_records[i + 1].sample.monotonic_s - probe_records[i].sample.monotonic_s
        for i in range(min(20, len(probe_records) - 1))
    ]
    med = float(np.median(deltas)) if deltas else 4.0
    return max(1.0, med)


def determine_issue_label(
    record: Record,
    meta: dict[str, Any] | None,
) -> str | None:
    """Determine ground truth label 'issue_now' for a record."""
    raw_label = record.label

    if raw_label in ("idle", "normal_use"):
        return "healthy"

    if meta is None:
        if raw_label.startswith("lab:"):
            parts = raw_label.split(":")
            if len(parts) >= 2:
                scen = parts[1]
                if "+" in scen or scen == "combo":
                    return "combo"
                if scen in ("cpu_pressure", "memory_pressure", "disk_pressure", "background_load"):
                    return scen
                if scen == "idle":
                    return "healthy"
        return "healthy" if raw_label in ("unlabeled", "") else None

    # Aborted runs are excluded
    if meta.get("aborted"):
        return None

    actual = meta.get("actual", {})
    inj_start_str = actual.get("inject_start")
    inj_end_str = actual.get("inject_end")

    if not inj_start_str or not inj_end_str:
        return None

    rec_time = _parse_utc_iso(record.sample.ts_utc)
    inj_start = _parse_utc_iso(inj_start_str)
    inj_end = _parse_utc_iso(inj_end_str)

    scenario_name = str(meta.get("scenario", ""))

    if rec_time < inj_start:
        return "healthy"
    if rec_time < inj_start + 10.0:
        return "transition"
    if rec_time <= inj_end:
        if "+" in scenario_name or scenario_name == "combo":
            return "combo"
        if scenario_name in ("cpu_pressure", "memory_pressure", "disk_pressure", "background_load"):
            return scenario_name
        if scenario_name == "idle":
            return "healthy"
        return scenario_name
    if rec_time < inj_end + 20.0:
        return "transition"
    return "healthy"


def extract_run_features(
    records: list[Record],
    baseline: Baseline,
    meta: dict[str, Any] | None = None,
    horizon_s: float = 30.0,
    keep_transition: bool = False,
) -> list[dict[str, Any]]:
    """Extract features, targets, and metadata rows from a single run series.

    Computes features exclusively from data at or before each decision point t (no lookahead).
    """
    if not records:
        return []

    # Sort strictly by monotonic time
    records = sorted(records, key=lambda r: r.sample.monotonic_s)
    probe_records = [r for r in records if r.probe is not None]

    if not probe_records:
        return []

    sampling_interval = _estimate_sampling_interval(records)
    probe_interval = _estimate_probe_interval(probe_records)
    gap_limit = 3.0 * sampling_interval
    target_tolerance = (probe_interval / 2.0) + 1.0

    run_start_mono = records[0].sample.monotonic_s
    rows: list[dict[str, Any]] = []

    # Pre-index probe monotonic timestamps
    probe_monos = np.array([r.sample.monotonic_s for r in probe_records])

    for _i, cur_rec in enumerate(probe_records):
        t = cur_rec.sample.monotonic_s

        # 1. Warm-up drop: require at least 60s of prior history
        if t - run_start_mono < 60.0:
            continue

        # 2. Window isolation: strictly data at or before t
        history_60s = [r for r in records if t - 60.0 <= r.sample.monotonic_s <= t]
        if not history_60s:
            continue

        # Check gap rule inside 60s window
        has_gap = False
        for k in range(len(history_60s) - 1):
            if (
                history_60s[k + 1].sample.monotonic_s - history_60s[k].sample.monotonic_s
                > gap_limit
            ):
                has_gap = True
                break
        if has_gap:
            continue

        # 3. Target alignment at t + horizon_s
        target_t = t + horizon_s
        diffs = np.abs(probe_monos - target_t)
        best_idx = int(np.argmin(diffs))
        if diffs[best_idx] > target_tolerance:
            # No matching probe within tolerance horizon
            continue

        target_rec = probe_records[best_idx]
        target_probe = target_rec.probe
        if target_probe is None:
            continue

        # 4. Issue label assignment
        issue = determine_issue_label(cur_rec, meta)
        if issue is None:
            continue
        if issue == "transition" and not keep_transition:
            continue

        # Extract features across 10s, 30s, 60s windows
        history_30s = [r for r in history_60s if r.sample.monotonic_s >= t - 30.0]
        history_10s = [r for r in history_60s if r.sample.monotonic_s >= t - 10.0]
        window_map = {"10s": history_10s, "30s": history_30s, "60s": history_60s}

        row: dict[str, Any] = {}

        for w_name, w_recs in window_map.items():
            ts_list = [r.sample.monotonic_s for r in w_recs]

            # 1. CPU percent mean
            cpu_means = [r.sample.cpu_percent_mean for r in w_recs]
            row[f"cpu_percent_mean_mean_{w_name}"] = compute_mean(cpu_means)
            row[f"cpu_percent_mean_std_{w_name}"] = compute_std(cpu_means)
            row[f"cpu_percent_mean_max_{w_name}"] = compute_max(cpu_means)
            row[f"cpu_percent_mean_slope_{w_name}"] = compute_slope(ts_list, cpu_means)

            # 2. CPU max core
            cpu_cores = [r.sample.cpu_percent_max_core for r in w_recs]
            row[f"cpu_percent_max_core_mean_{w_name}"] = compute_mean(cpu_cores)

            # 3. CPU freq ratio
            freq_ratios: list[float] = []
            for r in w_recs:
                if (
                    r.sample.cpu_freq_reliable
                    and r.sample.cpu_freq_max_mhz
                    and r.sample.cpu_freq_max_mhz > 0
                    and r.sample.cpu_freq_current_mhz is not None
                ):
                    freq_ratios.append(r.sample.cpu_freq_current_mhz / r.sample.cpu_freq_max_mhz)
                else:
                    freq_ratios.append(float("nan"))

            row[f"cpu_freq_ratio_mean_{w_name}"] = compute_mean(freq_ratios)
            row[f"cpu_freq_ratio_min_{w_name}"] = compute_min(freq_ratios)
            row[f"cpu_freq_ratio_drop_{w_name}"] = compute_drop(freq_ratios)

            # 4. Memory percent
            mem_pcts = [r.sample.mem_percent for r in w_recs]
            row[f"mem_percent_last_{w_name}"] = mem_pcts[-1] if mem_pcts else float("nan")
            row[f"mem_percent_mean_{w_name}"] = compute_mean(mem_pcts)
            row[f"mem_percent_slope_{w_name}"] = compute_slope(ts_list, mem_pcts) * 60.0

            # 5. Memory available GB
            mem_gbs = [r.sample.mem_available_bytes / (1024**3) for r in w_recs]
            last_gb = mem_gbs[-1] if mem_gbs else float("nan")
            slope_gb_s = compute_slope(ts_list, mem_gbs)
            row[f"mem_available_gb_last_{w_name}"] = last_gb
            row[f"mem_available_gb_slope_{w_name}"] = slope_gb_s * 60.0

            if not math.isnan(last_gb) and slope_gb_s < 0:
                headroom = last_gb / max(-slope_gb_s, 1e-6)
            else:
                headroom = 3600.0
            row[f"mem_headroom_s_{w_name}"] = min(3600.0, max(0.0, headroom))

            # 6. Swap percent
            swap_pcts = [r.sample.swap_percent for r in w_recs]
            row[f"swap_percent_last_{w_name}"] = swap_pcts[-1] if swap_pcts else float("nan")
            row[f"swap_percent_slope_{w_name}"] = compute_slope(ts_list, swap_pcts) * 60.0

            # 7. Disk I/O
            disk_reads = [
                (r.sample.disk_read_bytes_per_s / (1024**2))
                if r.sample.disk_read_bytes_per_s is not None
                else 0.0
                for r in w_recs
            ]
            row[f"disk_read_mb_s_mean_{w_name}"] = compute_mean(disk_reads)
            row[f"disk_read_mb_s_max_{w_name}"] = compute_max(disk_reads)

            disk_writes = [
                (r.sample.disk_write_bytes_per_s / (1024**2))
                if r.sample.disk_write_bytes_per_s is not None
                else 0.0
                for r in w_recs
            ]
            row[f"disk_write_mb_s_mean_{w_name}"] = compute_mean(disk_writes)
            row[f"disk_write_mb_s_max_{w_name}"] = compute_max(disk_writes)

            disk_busies = [
                r.sample.disk_busy_ms_per_s
                if r.sample.disk_busy_ms_per_s is not None
                else float("nan")
                for r in w_recs
            ]
            row[f"disk_busy_ms_per_s_mean_{w_name}"] = compute_mean(disk_busies)

            # 8. Network errors and drops
            net_errs = [
                r.sample.net_errors_per_s if r.sample.net_errors_per_s is not None else 0.0
                for r in w_recs
            ]
            row[f"net_errors_per_s_mean_{w_name}"] = compute_mean(net_errs)

            net_drops = [
                r.sample.net_drops_per_s if r.sample.net_drops_per_s is not None else 0.0
                for r in w_recs
            ]
            row[f"net_drops_per_s_mean_{w_name}"] = compute_mean(net_drops)

            # 9. Process and thread counts
            p_counts = [float(r.sample.process_count) for r in w_recs]
            row[f"process_count_last_{w_name}"] = p_counts[-1] if p_counts else float("nan")
            row[f"process_count_slope_{w_name}"] = compute_slope(ts_list, p_counts) * 60.0

            th_counts = [float(r.sample.thread_count) for r in w_recs]
            row[f"thread_count_last_{w_name}"] = th_counts[-1] if th_counts else float("nan")
            row[f"thread_count_slope_{w_name}"] = compute_slope(ts_list, th_counts) * 60.0

            # 10. Context switches and interrupts
            ctx_list = [
                r.sample.ctx_switches_per_s
                if r.sample.ctx_switches_per_s is not None
                else float("nan")
                for r in w_recs
            ]
            row[f"ctx_switches_per_s_mean_{w_name}"] = compute_mean(ctx_list)

            intr_list = [
                r.sample.interrupts_per_s if r.sample.interrupts_per_s is not None else float("nan")
                for r in w_recs
            ]
            row[f"interrupts_per_s_mean_{w_name}"] = compute_mean(intr_list)

        # 11. Battery status at t
        row["battery_percent"] = (
            cur_rec.sample.battery_percent
            if cur_rec.sample.battery_percent is not None
            else float("nan")
        )
        if cur_rec.sample.battery_plugged is True:
            row["battery_plugged"] = 1.0
        elif cur_rec.sample.battery_plugged is False:
            row["battery_plugged"] = 0.0
        else:
            row["battery_plugged"] = float("nan")

        # 12. Top process metrics at t
        top_procs = cur_rec.top_processes
        tot_ram = cur_rec.sample.mem_total_bytes
        if top_procs and tot_ram > 0:
            top1_rss = max(p.rss_bytes for p in top_procs)
            row["top1_rss_share"] = top1_rss / tot_ram
            top1_cpu = max(p.cpu_percent for p in top_procs)
            row["top1_cpu_percent"] = top1_cpu
            busy_cnt = sum(1 for p in top_procs if p.cpu_percent > 20.0)
            row["busy_process_count"] = float(busy_cnt)
        else:
            row["top1_rss_share"] = 0.0
            row["top1_cpu_percent"] = 0.0
            row["busy_process_count"] = 0.0

        idle_apps = [
            p for p in top_procs if not p.is_protected and p.cpu_percent < 1.0 and p.age_s > 600.0
        ]
        row["idle_app_rss_gb"] = sum(p.rss_bytes for p in idle_apps) / (1024**3)
        row["idle_app_count"] = float(len(idle_apps))

        # 13. Probe ratios at t
        cur_probe = cur_rec.probe
        assert cur_probe is not None  # probe_records filters for probe is not None
        p_cpu_ratio = cur_probe.cpu_ns / max(1.0, baseline.cpu_ns)
        p_mem_ratio = cur_probe.mem_ns / max(1.0, baseline.mem_ns)
        p_disk_ratio = cur_probe.disk_ns / max(1.0, baseline.disk_ns)
        p_tot_ratio = cur_probe.total_ns / max(1.0, baseline.total_ns)

        row["probe_cpu_ratio"] = p_cpu_ratio
        row["probe_mem_ratio"] = p_mem_ratio
        row["probe_disk_ratio"] = p_disk_ratio
        row["probe_total_ratio"] = p_tot_ratio

        # Probe ratios in past 30s and 60s windows
        probes_30s = [
            r.probe.total_ns / max(1.0, baseline.total_ns)
            for r in history_30s
            if r.probe is not None
        ]
        row["probe_total_ratio_mean_30s"] = compute_mean(probes_30s)
        row["probe_total_ratio_max_30s"] = compute_max(probes_30s)

        probes_60s = [
            r.probe.total_ns / max(1.0, baseline.total_ns)
            for r in history_60s
            if r.probe is not None
        ]
        ts_probes_60s = [r.sample.monotonic_s for r in history_60s if r.probe is not None]
        row["probe_total_ratio_mean_60s"] = compute_mean(probes_60s)
        row["probe_total_ratio_max_60s"] = compute_max(probes_60s)
        row["probe_total_ratio_slope_60s"] = compute_slope(ts_probes_60s, probes_60s) * 60.0

        # 14. Throttling signature
        cpu_10s_mean = row["cpu_percent_mean_mean_10s"]
        cpu_factor = 1.0 - (cpu_10s_mean / 100.0) if not math.isnan(cpu_10s_mean) else 1.0
        row["slow_while_idle"] = max(0.0, p_cpu_ratio - 1.0) * max(0.0, cpu_factor)

        # 15. Targets
        target_tot_ratio = target_probe.total_ns / max(1.0, baseline.total_ns)
        row["target_total_ratio"] = target_tot_ratio
        row["target_cpu_ratio"] = target_probe.cpu_ns / max(1.0, baseline.cpu_ns)
        row["target_mem_ratio"] = target_probe.mem_ns / max(1.0, baseline.mem_ns)
        row["target_disk_ratio"] = target_probe.disk_ns / max(1.0, baseline.disk_ns)
        row["target_slow"] = bool(target_tot_ratio >= SLOW_THRESHOLD)

        # 16. Label and metadata
        row["issue_now"] = issue
        row["run_id"] = cur_rec.run_id
        row["session_id"] = meta.get("session_id") if meta else f"day-{cur_rec.sample.ts_utc[:10]}"
        row["machine_id"] = (
            meta.get("machine_id", baseline.machine_id) if meta else baseline.machine_id
        )
        row["scenario"] = meta.get("scenario", cur_rec.label) if meta else cur_rec.label
        row["intensity"] = meta.get("intensity", "none") if meta else "none"
        row["pattern"] = meta.get("pattern", "none") if meta else "none"
        row["ts_utc"] = cur_rec.sample.ts_utc
        row["horizon_s"] = float(horizon_s)
        row["baseline_total_ms"] = baseline.total_ns / 1_000_000.0

        rows.append(row)

    return rows
