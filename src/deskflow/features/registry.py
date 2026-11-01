"""Feature registry defining all extractable features and versioning."""

from dataclasses import dataclass

FEATURE_SET_VERSION = 1
SLOW_THRESHOLD = 1.5


@dataclass(frozen=True)
class FeatureSpec:
    """Metadata specification for a single engineered feature."""

    name: str
    unit: str
    window: str | None
    description: str


# Canonical registry of all features for FEATURE_SET_VERSION = 1
FEATURE_SPECS: list[FeatureSpec] = []


def _register(name: str, unit: str, window: str | None, description: str) -> None:
    FEATURE_SPECS.append(FeatureSpec(name=name, unit=unit, window=window, description=description))


# 1. CPU Percent Mean: mean, std, max, slope
for w in ("10s", "30s", "60s"):
    _register(f"cpu_percent_mean_mean_{w}", "%", w, f"Mean CPU utilization across {w} window.")
    _register(
        f"cpu_percent_mean_std_{w}",
        "%",
        w,
        f"Standard deviation of CPU utilization across {w} window.",
    )
    _register(f"cpu_percent_mean_max_{w}", "%", w, f"Maximum CPU utilization across {w} window.")
    _register(
        f"cpu_percent_mean_slope_{w}",
        "%/s",
        w,
        f"Least-squares slope of CPU utilization over {w} window.",
    )

# 2. CPU Percent Max Core: mean
for w in ("10s", "30s", "60s"):
    _register(
        f"cpu_percent_max_core_mean_{w}",
        "%",
        w,
        f"Mean single-core peak CPU utilization over {w} window.",
    )

# 3. CPU Frequency Ratio: mean, min, drop
for w in ("10s", "30s", "60s"):
    _register(
        f"cpu_freq_ratio_mean_{w}",
        "ratio",
        w,
        f"Mean CPU frequency ratio (current/max) over {w} window.",
    )
    _register(
        f"cpu_freq_ratio_min_{w}",
        "ratio",
        w,
        f"Minimum CPU frequency ratio over {w} window.",
    )
    _register(
        f"cpu_freq_ratio_drop_{w}",
        "ratio",
        w,
        f"Drop in CPU frequency ratio (max - last) over {w} window.",
    )

# 4. Memory Percent: last, mean, slope
for w in ("10s", "30s", "60s"):
    _register(f"mem_percent_last_{w}", "%", w, f"Most recent virtual memory percent in {w} window.")
    _register(f"mem_percent_mean_{w}", "%", w, f"Mean virtual memory percent over {w} window.")
    _register(
        f"mem_percent_slope_{w}",
        "%/min",
        w,
        f"Least-squares memory percent slope (%/min) over {w} window.",
    )

# 5. Memory Available GB: last, slope, headroom
for w in ("10s", "30s", "60s"):
    _register(
        f"mem_available_gb_last_{w}",
        "GB",
        w,
        f"Most recent available memory in GB in {w} window.",
    )
    _register(
        f"mem_available_gb_slope_{w}",
        "GB/min",
        w,
        f"Available memory rate of change (GB/min) over {w} window.",
    )
    _register(
        f"mem_headroom_s_{w}",
        "s",
        w,
        f"Estimated time to memory exhaustion capped at 3600s based on {w} window.",
    )

# 6. Swap Percent: last, slope
for w in ("10s", "30s", "60s"):
    _register(
        f"swap_percent_last_{w}", "%", w, f"Most recent swap utilization percent in {w} window."
    )
    _register(
        f"swap_percent_slope_{w}",
        "%/min",
        w,
        f"Swap utilization slope (%/min) over {w} window.",
    )

# 7. Disk I/O: mean, max, busy
for w in ("10s", "30s", "60s"):
    _register(
        f"disk_read_mb_s_mean_{w}", "MB/s", w, f"Mean disk read rate in MB/s over {w} window."
    )
    _register(
        f"disk_read_mb_s_max_{w}",
        "MB/s",
        w,
        f"Maximum disk read rate in MB/s over {w} window.",
    )
    _register(
        f"disk_write_mb_s_mean_{w}",
        "MB/s",
        w,
        f"Mean disk write rate in MB/s over {w} window.",
    )
    _register(
        f"disk_write_mb_s_max_{w}",
        "MB/s",
        w,
        f"Maximum disk write rate in MB/s over {w} window.",
    )
    _register(
        f"disk_busy_ms_per_s_mean_{w}",
        "ms/s",
        w,
        f"Mean disk busy time ms/s over {w} window.",
    )

# 8. Network: errors, drops
for w in ("10s", "30s", "60s"):
    _register(
        f"net_errors_per_s_mean_{w}",
        "err/s",
        w,
        f"Mean network packet error rate over {w} window.",
    )
    _register(
        f"net_drops_per_s_mean_{w}",
        "drop/s",
        w,
        f"Mean network packet drop rate over {w} window.",
    )

# 9. Process and thread counts: last, slope
for w in ("10s", "30s", "60s"):
    _register(f"process_count_last_{w}", "count", w, f"Most recent process count in {w} window.")
    _register(
        f"process_count_slope_{w}",
        "count/min",
        w,
        f"Process count slope over {w} window.",
    )
    _register(
        f"thread_count_last_{w}", "count", w, f"Most recent system thread count in {w} window."
    )
    _register(
        f"thread_count_slope_{w}",
        "count/min",
        w,
        f"System thread count slope over {w} window.",
    )

# 10. Context switches and interrupts
for w in ("10s", "30s", "60s"):
    _register(
        f"ctx_switches_per_s_mean_{w}",
        "ctx/s",
        w,
        f"Mean context switches per second over {w} window.",
    )
    _register(
        f"interrupts_per_s_mean_{w}",
        "intr/s",
        w,
        f"Mean hardware interrupts per second over {w} window.",
    )

# 11. Battery status
_register("battery_percent", "%", None, "Current battery level percentage (NaN if no battery).")
_register(
    "battery_plugged",
    "flag",
    None,
    "Battery plugged in flag (1.0 = yes, 0.0 = no, NaN = unavailable).",
)

# 12. Top process statistics (at time t)
_register(
    "top1_rss_share", "ratio", None, "Largest single process RSS divided by total system RAM."
)
_register(
    "idle_app_rss_gb",
    "GB",
    None,
    "Total RSS in GB of non-protected idle apps (cpu < 1%, age > 600s).",
)
_register("idle_app_count", "count", None, "Count of non-protected idle applications.")
_register("busy_process_count", "count", None, "Count of top processes with cpu_percent > 20%.")
_register(
    "top1_cpu_percent", "%", None, "Peak CPU utilization percent among visible top processes."
)

# 13. Probe ratios (relative to baseline)
_register(
    "probe_cpu_ratio", "ratio", None, "CPU probe latency divided by baseline CPU median at t."
)
_register(
    "probe_mem_ratio", "ratio", None, "Memory probe latency divided by baseline memory median at t."
)
_register(
    "probe_disk_ratio", "ratio", None, "Disk probe latency divided by baseline disk median at t."
)
_register(
    "probe_total_ratio", "ratio", None, "Total probe latency divided by baseline total median at t."
)

_register("probe_total_ratio_mean_30s", "ratio", "30s", "Mean probe total ratio over 30s window.")
_register("probe_total_ratio_max_30s", "ratio", "30s", "Max probe total ratio over 30s window.")
_register("probe_total_ratio_mean_60s", "ratio", "60s", "Mean probe total ratio over 60s window.")
_register("probe_total_ratio_max_60s", "ratio", "60s", "Max probe total ratio over 60s window.")
_register(
    "probe_total_ratio_slope_60s",
    "ratio/min",
    "60s",
    "Least-squares slope of probe total ratio over 60s window.",
)

# 14. Throttling signature
_register(
    "slow_while_idle",
    "score",
    None,
    "Throttling signature: max(0, probe_cpu_ratio - 1) * (1 - cpu_percent_mean_10s / 100).",
)

FEATURE_NAMES: list[str] = [f.name for f in FEATURE_SPECS]
FEATURE_NAME_SET: frozenset[str] = frozenset(FEATURE_NAMES)
