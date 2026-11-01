"""Feature extraction engine: transforms raw telemetry and probes into ML features."""

import math
from typing import Any

import numpy as np

from deskflow.baseline import Baseline
from deskflow.collector.models import Record
from deskflow.features.registry import DEFAULT_HORIZON_S, MIN_HISTORY_S, SLOW_THRESHOLD
from deskflow.features.window import (
    compute_max,
    compute_mean,
    compute_min,
    compute_slope,
)


def _estimate_sampling_interval(records: list[Record]) -> float:
    """Estimate median sampling interval across records in seconds."""
    if len(records) < 2:
        return 2.0
    ts = [r.sample.monotonic_s for r in records]
    diffs = np.diff(ts)
    valid_diffs = diffs[diffs > 0.0]
    if len(valid_diffs) == 0:
        return 2.0
    return float(np.median(valid_diffs))


def _estimate_probe_interval(probe_records: list[Record]) -> float:
    """Estimate median interval between probes in seconds."""
    if len(probe_records) < 2:
        return 4.0
    ts = [r.sample.monotonic_s for r in probe_records]
    diffs = np.diff(ts)
    valid_diffs = diffs[diffs > 0.0]
    if len(valid_diffs) == 0:
        return 4.0
    return float(np.median(valid_diffs))


def _resolve_issue_label(
    label: str,
    meta: dict[str, Any] | None,
    t: float,
    inject_start_mono: float | None,
    inject_end_mono: float | None,
    keep_transition: bool = False,
) -> str | None:
    """Determine the ground-truth issue label for decision point t."""
    if meta is not None and inject_start_mono is not None and inject_end_mono is not None:
        scenario = str(meta.get("scenario", "healthy"))
        if scenario == "idle":
            scenario = "healthy"
        if t < inject_start_mono:
            return "healthy"
        if t < inject_end_mono:
            return "combo" if "+" in scenario else scenario
        if t >= inject_end_mono + 10.0:
            return "healthy"
        return "transition" if keep_transition else None

    if label.startswith("lab:"):
        parts = label.split(":")
        if len(parts) >= 2:
            scen = parts[1]
            if scen == "idle":
                return "healthy"
            return "combo" if "+" in scen else scen
    if label == "idle":
        return "healthy"
    return label


def build_row(
    records_up_to_t: list[Record],
    baseline: Baseline,
    current_record: Record | None = None,
) -> dict[str, float]:
    """Extract the canonical 27 features from records up to decision point t.

    This function is the single source of truth for feature calculation, shared
    between live scanning / inference and offline dataset building.

    Args:
        records_up_to_t: Telemetry records at or before decision point t.
        baseline: Baseline calibration for the active power state.
        current_record: The record at decision point t with a completed probe.
            If None, the last record in records_up_to_t is used.

    Returns:
        Dictionary mapping all 27 canonical feature names to float values.
    """
    if current_record is None:
        if not records_up_to_t:
            raise ValueError("records_up_to_t cannot be empty when current_record is None")
        current_record = records_up_to_t[-1]

    if current_record.probe is None:
        raise ValueError("current_record must contain a completed probe result")

    t = current_record.sample.monotonic_s

    # Filter to 12s window strictly at or before t
    history_12s = [r for r in records_up_to_t if t - 12.0 <= r.sample.monotonic_s <= t]
    history_6s = [r for r in history_12s if r.sample.monotonic_s >= t - 6.0]
    ts_12s = [r.sample.monotonic_s for r in history_12s]

    row: dict[str, float] = {}

    # --- 1. CPU Features (6s and 12s) ---
    cpu_means_6s = [r.sample.cpu_percent_mean for r in history_6s]
    cpu_means_12s = [r.sample.cpu_percent_mean for r in history_12s]
    row["cpu_percent_mean_mean_6s"] = compute_mean(cpu_means_6s)
    row["cpu_percent_mean_mean_12s"] = compute_mean(cpu_means_12s)
    row["cpu_percent_mean_max_12s"] = compute_max(cpu_means_12s)
    row["cpu_percent_mean_slope_12s"] = compute_slope(ts_12s, cpu_means_12s)

    # --- 2. CPU Frequency Ratios ---
    freq_ratios: list[float] = []
    for r in history_12s:
        if (
            r.sample.cpu_freq_reliable
            and r.sample.cpu_freq_max_mhz
            and r.sample.cpu_freq_max_mhz > 0
            and r.sample.cpu_freq_current_mhz is not None
        ):
            freq_ratios.append(r.sample.cpu_freq_current_mhz / r.sample.cpu_freq_max_mhz)
        else:
            freq_ratios.append(float("nan"))
    row["cpu_freq_ratio_mean_12s"] = compute_mean(freq_ratios)
    row["cpu_freq_ratio_min_12s"] = compute_min(freq_ratios)

    # --- 3. Memory Slope and Headroom ---
    mem_pcts_12s = [r.sample.mem_percent for r in history_12s]
    row["mem_percent_mean_12s"] = compute_mean(mem_pcts_12s)
    row["mem_percent_slope_12s"] = compute_slope(ts_12s, mem_pcts_12s) * 60.0
    row["mem_available_bytes"] = float(current_record.sample.mem_available_bytes)

    # --- 4. Swap Utilization ---
    row["swap_percent"] = float(current_record.sample.swap_percent)

    # --- 5. Disk Throughput ---
    disk_reads = [
        (r.sample.disk_read_bytes_per_s / (1024**2))
        if r.sample.disk_read_bytes_per_s is not None
        else 0.0
        for r in history_12s
    ]
    disk_writes = [
        (r.sample.disk_write_bytes_per_s / (1024**2))
        if r.sample.disk_write_bytes_per_s is not None
        else 0.0
        for r in history_12s
    ]
    row["disk_read_mb_s_mean_12s"] = compute_mean(disk_reads)
    row["disk_write_mb_s_mean_12s"] = compute_mean(disk_writes)

    # --- 6. Context Switches ---
    ctx_list = [
        r.sample.ctx_switches_per_s if r.sample.ctx_switches_per_s is not None else float("nan")
        for r in history_12s
    ]
    row["ctx_switches_per_s_mean_12s"] = compute_mean(ctx_list)

    # --- 7. Process and Thread Counts ---
    row["process_count"] = float(current_record.sample.process_count)
    row["thread_count"] = (
        float(current_record.sample.thread_count)
        if current_record.sample.thread_count is not None
        else float("nan")
    )

    # --- 8. Battery Plugged Status ---
    if current_record.sample.battery_plugged is True:
        row["battery_plugged"] = 1.0
    elif current_record.sample.battery_plugged is False:
        row["battery_plugged"] = 0.0
    else:
        row["battery_plugged"] = float("nan")

    # --- 9. Top-25 Process Metrics ---
    top_procs = current_record.top_processes
    tot_ram = current_record.sample.mem_total_bytes
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

    # --- 10. Probe Ratios Relative to Active Baseline ---
    cur_probe = current_record.probe
    p_cpu_ratio = cur_probe.cpu_ns / max(1.0, baseline.cpu_ns)
    p_mem_ratio = cur_probe.mem_ns / max(1.0, baseline.mem_ns)
    p_disk_ratio = cur_probe.disk_ns / max(1.0, baseline.disk_ns)
    p_tot_ratio = cur_probe.total_ns / max(1.0, baseline.total_ns)

    row["probe_cpu_ratio"] = p_cpu_ratio
    row["probe_mem_ratio"] = p_mem_ratio
    row["probe_disk_ratio"] = p_disk_ratio
    row["probe_total_ratio"] = p_tot_ratio

    probes_12s = [
        r.probe.total_ns / max(1.0, baseline.total_ns) for r in history_12s if r.probe is not None
    ]
    row["probe_total_ratio_mean_12s"] = compute_mean(probes_12s)

    # --- 11. Throttling Signature ---
    cpu_mean = row["cpu_percent_mean_mean_12s"]
    cpu_factor = 1.0 - (cpu_mean / 100.0) if not math.isnan(cpu_mean) else 1.0
    row["slow_while_idle"] = max(0.0, p_cpu_ratio - 1.0) * max(0.0, cpu_factor)

    return row


def extract_run_features(
    records: list[Record],
    baseline: Baseline | dict[str, Baseline | None],
    horizon_s: float = DEFAULT_HORIZON_S,
    meta: dict[str, Any] | None = None,
    keep_transition: bool = False,
    drop_stats: dict[str, int] | None = None,
) -> list[dict[str, Any]]:
    """Extract features, targets, and metadata rows from a single run series.

    Feature Set v2: Windows are 6s and 12s only. MIN_HISTORY is 12s.
    Computes features exclusively from data at or before each decision point t (no lookahead).
    """
    if not records:
        return []

    # Resolve per-state baselines mapping
    if isinstance(baseline, Baseline):
        baselines_by_state: dict[str, Baseline | None] = {
            "ac": baseline,
            "battery": baseline,
        }
    else:
        baselines_by_state = dict(baseline)

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

    # Injection window detection if metadata timestamps are present
    inject_start_mono: float | None = None
    inject_end_mono: float | None = None
    if meta is not None:
        if meta.get("aborted"):
            return []
        if "actual" in meta:
            actual = meta.get("actual") or {}
            inj_s = actual.get("inject_start")
            inj_e = actual.get("inject_end")
            pre_s_str = actual.get("pre_start") or inj_s
            if not (inj_s and inj_e and pre_s_str):
                return []
            from datetime import datetime

            t0_epoch = datetime.fromisoformat(pre_s_str).timestamp()
            t_inj_s_epoch = datetime.fromisoformat(inj_s).timestamp()
            t_inj_e_epoch = datetime.fromisoformat(inj_e).timestamp()
            inject_start_mono = run_start_mono + (t_inj_s_epoch - t0_epoch)
            inject_end_mono = run_start_mono + (t_inj_e_epoch - t0_epoch)
        elif "planned" in meta:
            pre_s = float(meta["planned"].get("pre_s", 20.0))
            hold_s = float(meta["planned"].get("hold_s", 90.0))
            inject_start_mono = run_start_mono + pre_s
            inject_end_mono = inject_start_mono + hold_s

    for cur_rec in probe_records:
        t = cur_rec.sample.monotonic_s

        # 1. Warm-up drop: require at least MIN_HISTORY_S (12s) of prior history
        if t - run_start_mono < MIN_HISTORY_S:
            if drop_stats is not None:
                drop_stats["warm-up"] = drop_stats.get("warm-up", 0) + 1
            continue

        # 2. Window isolation: strictly data at or before t (12s max window)
        history_12s = [r for r in records if t - 12.0 <= r.sample.monotonic_s <= t]
        if not history_12s:
            continue

        # Check gap rule inside 12s window
        has_gap = False
        first_idx = records.index(history_12s[0])
        records_to_check = [records[first_idx - 1]] + history_12s if first_idx > 0 else history_12s
        for k in range(len(records_to_check) - 1):
            if (
                records_to_check[k + 1].sample.monotonic_s - records_to_check[k].sample.monotonic_s
                > gap_limit
                and records_to_check[k + 1].sample.monotonic_s > t - 12.0
            ):
                has_gap = True
                break
        if has_gap:
            if drop_stats is not None:
                drop_stats["gap"] = drop_stats.get("gap", 0) + 1
            continue

        # 3. Target alignment at t + horizon_s
        target_t = t + horizon_s
        diffs = np.abs(probe_monos - target_t)
        best_idx = int(np.argmin(diffs))
        if diffs[best_idx] > target_tolerance:
            if drop_stats is not None:
                drop_stats["no target probe"] = drop_stats.get("no target probe", 0) + 1
            continue

        target_rec = probe_records[best_idx]
        target_probe = target_rec.probe
        if target_probe is None:
            if drop_stats is not None:
                drop_stats["no target probe"] = drop_stats.get("no target probe", 0) + 1
            continue

        # 4. Resolve issue label
        issue = _resolve_issue_label(
            label=cur_rec.label,
            meta=meta,
            t=t,
            inject_start_mono=inject_start_mono,
            inject_end_mono=inject_end_mono,
            keep_transition=keep_transition,
        )
        if issue is None:
            if drop_stats is not None:
                drop_stats["transition"] = drop_stats.get("transition", 0) + 1
            continue

        # 5. Determine active baseline by power state at time t
        cur_bat_plugged = cur_rec.sample.battery_plugged
        row_state = "battery" if (cur_bat_plugged is not None and not cur_bat_plugged) else "ac"
        active_baseline = baselines_by_state.get(row_state)
        if active_baseline is None:
            # Fallback to 'ac' if only 'ac' exists on desktop systems without battery
            active_baseline = baselines_by_state.get("ac")
            if active_baseline is None:
                raise ValueError(
                    f"Missing baseline for power state '{row_state}'. "
                    f"Run 'deskflow baseline calibrate' while on {row_state} power."
                )

        # Build the 27 canonical features via single source of truth
        features = build_row(
            records_up_to_t=history_12s,
            baseline=active_baseline,
            current_record=cur_rec,
        )
        row: dict[str, Any] = dict(features)

        # --- Targets at horizon_s ---
        target_tot_ratio = target_probe.total_ns / max(1.0, active_baseline.total_ns)
        row["target_total_ratio"] = target_tot_ratio
        row["target_cpu_ratio"] = target_probe.cpu_ns / max(1.0, active_baseline.cpu_ns)
        row["target_mem_ratio"] = target_probe.mem_ns / max(1.0, active_baseline.mem_ns)
        row["target_disk_ratio"] = target_probe.disk_ns / max(1.0, active_baseline.disk_ns)
        # pyrefly: ignore[unnecessary-type-conversion]
        row["target_slow"] = bool(target_tot_ratio >= SLOW_THRESHOLD)

        # --- Metadata ---
        row["issue_now"] = issue
        row["baseline_state"] = row_state
        row["run_id"] = cur_rec.run_id
        row["session_id"] = meta.get("session_id") if meta else f"day-{cur_rec.sample.ts_utc[:10]}"
        row["machine_id"] = (
            meta.get("machine_id", active_baseline.machine_id)
            if meta
            else active_baseline.machine_id
        )
        row["scenario"] = meta.get("scenario", cur_rec.label) if meta else cur_rec.label
        row["intensity"] = meta.get("intensity", "none") if meta else "none"
        row["pattern"] = meta.get("pattern", "none") if meta else "none"
        row["ts_utc"] = cur_rec.sample.ts_utc
        row["horizon_s"] = float(horizon_s)
        row["baseline_total_ms"] = active_baseline.total_ns / 1_000_000.0

        rows.append(row)

    return rows
