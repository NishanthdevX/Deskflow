# Data Schema Reference

Deskflow persists all continuous monitoring runs as JSON Lines (`.jsonl`), where each line is a serialized `Record` adhering to `SCHEMA_VERSION = 1`.

## Record Structure

| Field | Type | Nullable | Description |
|---|---|---|---|
| `schema_version` | integer | No | Schema version number (currently `1`). |
| `run_id` | string | No | Unique run identifier in format `YYYYMMDDTHHMMSSZ_xxxxxx`. |
| `label` | string | No | Dataset tag or fault-injection label (e.g. `healthy`, `cpu_pressure`). |
| `sample` | SystemSample | No | Hardware telemetry snapshot and delta rates. |
| `probe` | ProbeResult | Yes | Measured execution times of micro-workloads; None when probe not scheduled. |
| `top_processes` | list[ProcessSample] | No | Up to 10 processes consuming the highest resident set size (RSS). |
| `storage` | StorageReport | Yes | Storage consumption in temporary and cache directories. |

---

## Sub-Schemas

### SystemSample

| Field | Type | Unit | Nullable | Platform Notes |
|---|---|---|---|---|
| `ts_utc` | string | ISO 8601 UTC | No | Sample timestamp. |
| `monotonic_s` | float | seconds | No | Steady monotonic clock for computing delta rates. |
| `cpu_percent_mean` | float | percent (0-100) | No | Mean CPU utilization across all cores. |
| `cpu_percent_max_core` | float | percent (0-100) | No | Maximum CPU utilization observed on any single core. |
| `cpu_freq_current_mhz` | float | MHz | Yes | Current clock frequency. |
| `cpu_freq_max_mhz` | float | MHz | Yes | Maximum rated clock frequency. |
| `cpu_freq_reliable` | boolean | bool | Yes | `False` if OS reports static base clock (common on Windows / Apple Silicon). |
| `ctx_switches_per_s` | float | count / s | Yes | Context switch rate; `None` on first sample. |
| `interrupts_per_s` | float | count / s | Yes | Hardware interrupt rate; `None` on first sample. |
| `mem_total_bytes` | integer | bytes | No | Total physical RAM installed. |
| `mem_available_bytes` | integer | bytes | No | Usable memory available for new applications. |
| `mem_percent` | float | percent (0-100) | No | Memory usage percentage. |
| `swap_used_bytes` | integer | bytes | No | Active swap space or pagefile consumption. |
| `swap_percent` | float | percent (0-100) | No | Swap space percentage. |
| `disk_read_bytes_per_s` | float | bytes / s | Yes | Storage read throughput; `None` on first sample. |
| `disk_write_bytes_per_s` | float | bytes / s | Yes | Storage write throughput; `None` on first sample. |
| `disk_busy_ms_per_s` | float | ms / s | Yes | Disk active time; `None` when OS/virtual driver does not expose busy time. |
| `net_recv_bytes_per_s` | float | bytes / s | Yes | Total network received throughput. |
| `net_sent_bytes_per_s` | float | bytes / s | Yes | Total network transmitted throughput. |
| `net_errors_per_s` | float | count / s | Yes | Combined inbound and outbound network interface errors per second. |
| `net_drops_per_s` | float | count / s | Yes | Inbound and outbound dropped packets per second. |
| `process_count` | integer | count | No | Total running system processes. |
| `thread_count` | integer | count | No | Total running threads across all enumerated processes. |
| `battery_percent` | float | percent (0-100) | Yes | Remaining battery level; `None` on desktop or VM. |
| `battery_plugged` | boolean | bool | Yes | `True` if connected to AC power; `None` if unreadable. |
| `max_temperature_c` | float | Celsius | Yes | Highest sensor reading; `None` if sensors unavailable without elevation. |
| `uptime_s` | float | seconds | No | System uptime in seconds since last boot. |

### ProbeResult

| Field | Type | Unit | Nullable | Description |
|---|---|---|---|---|
| `ts_utc` | string | ISO 8601 UTC | No | Timestamp when the probe was completed. |
| `cpu_ns` | integer | nanoseconds | No | Duration of pure-Python integer LCG loop (120,000 iterations). |
| `mem_ns` | integer | nanoseconds | No | Duration of 8 MiB strided memory allocation and access. |
| `disk_ns` | integer | nanoseconds | No | Duration of 1 MiB write and fsync. |
| `total_ns` | integer | nanoseconds | No | Sum of CPU, memory, and disk workload durations. |

### ProcessSample

| Field | Type | Unit | Nullable | Description |
|---|---|---|---|---|
| `pid` | integer | PID | No | Process ID. |
| `name` | string | name | No | Process executable name. |
| `rss_bytes` | integer | bytes | No | Resident Set Size memory consumption. |
| `cpu_percent` | float | percent | No | CPU utilization (0.0 on initial sample pass). |
| `num_threads` | integer | count | No | Thread count owned by process. |
| `age_s` | float | seconds | No | Process lifespan since creation time. |
| `is_protected` | boolean | bool | No | Whether process is system-critical, parent terminal, or non-user owned. |

*Privacy note: Process command lines and usernames are never stored.*

### StorageReport

| Field | Type | Unit | Nullable | Description |
|---|---|---|---|---|
| `ts_utc` | string | ISO 8601 UTC | No | Timestamp of scan. |
| `entries` | list[StorageEntry] | - | No | Per-directory breakdown: `path`, `kind` (`temp`/`cache`/`trash`), `bytes`. |
| `truncated` | boolean | bool | No | `True` if the 3.0s time budget or 200,000 entry cap was hit. |
| `duration_s` | float | seconds | No | Elapsed wall time of the storage scan. |
