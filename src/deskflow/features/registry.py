"""Feature registry defining all extractable features and versioning (Feature Set v2)."""

from dataclasses import dataclass

FEATURE_SET_VERSION = 2
SLOW_THRESHOLD = 1.5
MIN_HISTORY_S = 12.0
DEFAULT_HORIZON_S = 20.0


@dataclass(frozen=True)
class FeatureSpec:
    """Metadata specification for a single engineered feature."""

    name: str
    unit: str
    window: str | None
    description: str


# Canonical registry of all features for FEATURE_SET_VERSION = 2 (max 30 features)
FEATURE_SPECS: list[FeatureSpec] = [
    # 1. CPU Utilization across short windows (6s and 12s)
    FeatureSpec("cpu_percent_mean_mean_6s", "%", "6s", "Mean CPU utilization over 6s window."),
    FeatureSpec("cpu_percent_mean_mean_12s", "%", "12s", "Mean CPU utilization over 12s window."),
    FeatureSpec(
        "cpu_percent_mean_max_12s",
        "%",
        "12s",
        "Maximum single-tick CPU utilization over 12s window.",
    ),
    FeatureSpec(
        "cpu_percent_mean_slope_12s",
        "%/s",
        "12s",
        "Least-squares slope of CPU utilization over 12s window.",
    ),
    # 2. CPU Frequency ratio (throttling indicators)
    FeatureSpec(
        "cpu_freq_ratio_mean_12s",
        "ratio",
        "12s",
        "Mean CPU frequency ratio (current/max) over 12s window.",
    ),
    FeatureSpec(
        "cpu_freq_ratio_min_12s", "ratio", "12s", "Minimum CPU frequency ratio over 12s window."
    ),
    # 3. Memory slope and headroom
    FeatureSpec(
        "mem_percent_mean_12s", "%", "12s", "Mean RAM utilization percentage over 12s window."
    ),
    FeatureSpec(
        "mem_percent_slope_12s", "%/s", "12s", "Least-squares slope of RAM percent over 12s window."
    ),
    FeatureSpec(
        "mem_available_bytes",
        "bytes",
        None,
        "Instantaneous available physical memory headroom in bytes.",
    ),
    # 4. Swap utilization
    FeatureSpec("swap_percent", "%", None, "Instantaneous swap/pagefile utilization percent."),
    # 5. Disk throughput
    FeatureSpec(
        "disk_read_mb_s_mean_12s",
        "MB/s",
        "12s",
        "Mean disk read throughput in MB/s over 12s window.",
    ),
    FeatureSpec(
        "disk_write_mb_s_mean_12s",
        "MB/s",
        "12s",
        "Mean disk write throughput in MB/s over 12s window.",
    ),
    # 6. Context switches
    FeatureSpec(
        "ctx_switches_per_s_mean_12s",
        "ops/s",
        "12s",
        "Mean OS context switch rate over 12s window.",
    ),
    # 7. Process and thread counts
    FeatureSpec("process_count", "count", None, "System-wide total process count at time t."),
    FeatureSpec("thread_count", "count", None, "System-wide total thread count at time t."),
    # 8. Power state indicator
    FeatureSpec(
        "battery_plugged",
        "flag",
        None,
        "Device connected to external AC power (1.0 = yes, 0.0 = no).",
    ),
    # 9. Top-25 process features (evaluated over top 25 processes sorted by RSS)
    FeatureSpec(
        "top1_rss_share", "ratio", None, "Largest single process RSS divided by total system RAM."
    ),
    FeatureSpec(
        "top1_cpu_percent", "%", None, "Peak CPU utilization percent among top 25 processes."
    ),
    FeatureSpec(
        "busy_process_count", "count", None, "Count of top 25 processes with cpu_percent > 20%."
    ),
    FeatureSpec(
        "idle_app_rss_gb", "GB", None, "Total RSS in GB of non-protected idle applications."
    ),
    FeatureSpec("idle_app_count", "count", None, "Count of non-protected idle applications."),
    # 10. Probe latency ratios relative to active power-state baseline
    FeatureSpec(
        "probe_cpu_ratio", "ratio", None, "CPU probe latency divided by baseline CPU median at t."
    ),
    FeatureSpec(
        "probe_mem_ratio",
        "ratio",
        None,
        "Memory probe latency divided by baseline memory median at t.",
    ),
    FeatureSpec(
        "probe_disk_ratio",
        "ratio",
        None,
        "Disk probe latency divided by baseline disk median at t.",
    ),
    FeatureSpec(
        "probe_total_ratio",
        "ratio",
        None,
        "Total probe latency divided by baseline total median at t.",
    ),
    FeatureSpec(
        "probe_total_ratio_mean_12s", "ratio", "12s", "Mean probe total ratio over 12s window."
    ),
    # 11. Throttling signature
    FeatureSpec(
        "slow_while_idle",
        "score",
        None,
        "Throttling indicator: max(0, probe_cpu_ratio - 1) * (1 - cpu_mean / 100).",
    ),
]

FEATURE_NAMES: list[str] = [f.name for f in FEATURE_SPECS]
FEATURE_NAME_SET: frozenset[str] = frozenset(FEATURE_NAMES)
