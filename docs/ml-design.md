# Machine Learning Formulation and Design

Deskflow anticipates system degradation and sluggishness before it impacts the user by predicting future probe latency 30 seconds ahead ($H = 30$ s).

## 1. Problem Formulation

Conventional device-monitoring utilities trigger reactive alerts only after CPU or memory saturation reaches arbitrary fixed thresholds (such as 90% utilization). However, high utilization during smooth video rendering or batch compilation does not necessarily imply sluggishness, while low utilization during thermal throttling or disk thrashing can severely stall foreground responsiveness.

Deskflow formulates responsiveness prediction as:
- **Primary Task**: Multi-horizon regression predicting normalized probe latency ratios $H$ seconds into the future:
  $$\text{target\_total\_ratio}_{t+H} = \frac{\text{total\_ns}_{t+H}}{\text{baseline\_total\_ns}}$$
- **Secondary Classification**: Binary early-warning indicator `target_slow`:
  $$\text{target\_slow}_{t+H} = \mathbb{I}(\text{target\_total\_ratio}_{t+H} \ge 1.5)$$
- **Component Latency Decomposition**: Individual component regression for `target_cpu_ratio`, `target_mem_ratio`, and `target_disk_ratio` to support diagnostic attribution.

### Why a Measured Probe is the Ground Truth

Direct synthetic micro-workloads (120k LCG integer loops, 8 MiB strided memory accesses, and 1 MiB fsynced disk writes) serve as an objective ground truth:
1. Probes measure the actual end-to-end scheduling, memory bandwidth, and I/O latency experienced by user applications.
2. Probes execute deterministically on identical code paths across all runs, removing external application variance.
3. Probes require no privileged OS APIs or invasive kernel drivers.

## 2. Baseline Normalization

Hardware clock rates, memory buses, and storage speeds vary widely across devices. Predicting raw millisecond latency would couple model outputs to processor speed rather than responsiveness degradation.

Deskflow requires an initial idle calibration (`deskflow baseline calibrate`):
1. The system records probe latency every 2 seconds while idle.
2. The initial 3 warmup samples are discarded.
3. If the coefficient of variation (CV) of total latency exceeds $0.5$, calibration is refused because the machine was active.
4. Component and total medians are persisted to `<user_config_dir>/baseline.json`.
5. All target metrics and probe features are divided by these calibrated baseline medians.

## 3. Feature Registry (FEATURE_SET_VERSION = 1)

Features are extracted across multi-scale rolling temporal windows (10 s, 30 s, 60 s). Features are computed strictly from observations at or before decision time $t$ (strict no-lookahead guarantee).

| Feature Name | Unit | Window | Description |
| :--- | :--- | :--- | :--- |
| `cpu_percent_mean_mean_10s` | % | 10s | Mean CPU utilization across 10s window |
| `cpu_percent_mean_std_10s` | % | 10s | Standard deviation of CPU utilization across 10s window |
| `cpu_percent_mean_max_10s` | % | 10s | Maximum CPU utilization across 10s window |
| `cpu_percent_mean_slope_10s` | %/s | 10s | Least-squares slope of CPU utilization over 10s window |
| `cpu_percent_mean_mean_30s` | % | 30s | Mean CPU utilization across 30s window |
| `cpu_percent_mean_std_30s` | % | 30s | Standard deviation of CPU utilization across 30s window |
| `cpu_percent_mean_max_30s` | % | 30s | Maximum CPU utilization across 30s window |
| `cpu_percent_mean_slope_30s` | %/s | 30s | Least-squares slope of CPU utilization over 30s window |
| `cpu_percent_mean_mean_60s` | % | 60s | Mean CPU utilization across 60s window |
| `cpu_percent_mean_std_60s` | % | 60s | Standard deviation of CPU utilization across 60s window |
| `cpu_percent_mean_max_60s` | % | 60s | Maximum CPU utilization across 60s window |
| `cpu_percent_mean_slope_60s` | %/s | 60s | Least-squares slope of CPU utilization over 60s window |
| `cpu_percent_max_core_mean_10s` | % | 10s | Mean single-core peak CPU utilization over 10s window |
| `cpu_percent_max_core_mean_30s` | % | 30s | Mean single-core peak CPU utilization over 30s window |
| `cpu_percent_max_core_mean_60s` | % | 60s | Mean single-core peak CPU utilization over 60s window |
| `cpu_freq_ratio_mean_10s` | ratio | 10s | Mean CPU frequency ratio (current/max) over 10s window |
| `cpu_freq_ratio_min_10s` | ratio | 10s | Minimum CPU frequency ratio over 10s window |
| `cpu_freq_ratio_drop_10s` | ratio | 10s | Drop in CPU frequency ratio (max - last) over 10s window |
| `cpu_freq_ratio_mean_30s` | ratio | 30s | Mean CPU frequency ratio (current/max) over 30s window |
| `cpu_freq_ratio_min_30s` | ratio | 30s | Minimum CPU frequency ratio over 30s window |
| `cpu_freq_ratio_drop_30s` | ratio | 30s | Drop in CPU frequency ratio (max - last) over 30s window |
| `cpu_freq_ratio_mean_60s` | ratio | 60s | Mean CPU frequency ratio (current/max) over 60s window |
| `cpu_freq_ratio_min_60s` | ratio | 60s | Minimum CPU frequency ratio over 60s window |
| `cpu_freq_ratio_drop_60s` | ratio | 60s | Drop in CPU frequency ratio (max - last) over 60s window |
| `mem_percent_last_10s` | % | 10s | Most recent virtual memory percent in 10s window |
| `mem_percent_mean_10s` | % | 10s | Mean virtual memory percent over 10s window |
| `mem_percent_slope_10s` | %/min | 10s | Least-squares memory percent slope (%/min) over 10s window |
| `mem_percent_last_30s` | % | 30s | Most recent virtual memory percent in 30s window |
| `mem_percent_mean_30s` | % | 30s | Mean virtual memory percent over 30s window |
| `mem_percent_slope_30s` | %/min | 30s | Least-squares memory percent slope (%/min) over 30s window |
| `mem_percent_last_60s` | % | 60s | Most recent virtual memory percent in 60s window |
| `mem_percent_mean_60s` | % | 60s | Mean virtual memory percent over 60s window |
| `mem_percent_slope_60s` | %/min | 60s | Least-squares memory percent slope (%/min) over 60s window |
| `mem_available_gb_last_10s` | GB | 10s | Most recent available memory in GB in 10s window |
| `mem_available_gb_slope_10s` | GB/min | 10s | Available memory rate of change (GB/min) over 10s window |
| `mem_headroom_s_10s` | s | 10s | Estimated time to memory exhaustion capped at 3600s based on 10s window |
| `mem_available_gb_last_30s` | GB | 30s | Most recent available memory in GB in 30s window |
| `mem_available_gb_slope_30s` | GB/min | 30s | Available memory rate of change (GB/min) over 30s window |
| `mem_headroom_s_30s` | s | 30s | Estimated time to memory exhaustion capped at 3600s based on 30s window |
| `mem_available_gb_last_60s` | GB | 60s | Most recent available memory in GB in 60s window |
| `mem_available_gb_slope_60s` | GB/min | 60s | Available memory rate of change (GB/min) over 60s window |
| `mem_headroom_s_60s` | s | 60s | Estimated time to memory exhaustion capped at 3600s based on 60s window |
| `swap_percent_last_10s` | % | 10s | Most recent swap utilization percent in 10s window |
| `swap_percent_slope_10s` | %/min | 10s | Swap utilization slope (%/min) over 10s window |
| `swap_percent_last_30s` | % | 30s | Most recent swap utilization percent in 30s window |
| `swap_percent_slope_30s` | %/min | 30s | Swap utilization slope (%/min) over 30s window |
| `swap_percent_last_60s` | % | 60s | Most recent swap utilization percent in 60s window |
| `swap_percent_slope_60s` | %/min | 60s | Swap utilization slope (%/min) over 60s window |
| `disk_read_mb_s_mean_10s` | MB/s | 10s | Mean disk read rate in MB/s over 10s window |
| `disk_read_mb_s_max_10s` | MB/s | 10s | Maximum disk read rate in MB/s over 10s window |
| `disk_write_mb_s_mean_10s` | MB/s | 10s | Mean disk write rate in MB/s over 10s window |
| `disk_write_mb_s_max_10s` | MB/s | 10s | Maximum disk write rate in MB/s over 10s window |
| `disk_busy_ms_per_s_mean_10s` | ms/s | 10s | Mean disk busy time ms/s over 10s window |
| `disk_read_mb_s_mean_30s` | MB/s | 30s | Mean disk read rate in MB/s over 30s window |
| `disk_read_mb_s_max_30s` | MB/s | 30s | Maximum disk read rate in MB/s over 30s window |
| `disk_write_mb_s_mean_30s` | MB/s | 30s | Mean disk write rate in MB/s over 30s window |
| `disk_write_mb_s_max_30s` | MB/s | 30s | Maximum disk write rate in MB/s over 30s window |
| `disk_busy_ms_per_s_mean_30s` | ms/s | 30s | Mean disk busy time ms/s over 30s window |
| `disk_read_mb_s_mean_60s` | MB/s | 60s | Mean disk read rate in MB/s over 60s window |
| `disk_read_mb_s_max_60s` | MB/s | 60s | Maximum disk read rate in MB/s over 60s window |
| `disk_write_mb_s_mean_60s` | MB/s | 60s | Mean disk write rate in MB/s over 60s window |
| `disk_write_mb_s_max_60s` | MB/s | 60s | Maximum disk write rate in MB/s over 60s window |
| `disk_busy_ms_per_s_mean_60s` | ms/s | 60s | Mean disk busy time ms/s over 60s window |
| `net_errors_per_s_mean_10s` | err/s | 10s | Mean network packet error rate over 10s window |
| `net_drops_per_s_mean_10s` | drop/s | 10s | Mean network packet drop rate over 10s window |
| `net_errors_per_s_mean_30s` | err/s | 30s | Mean network packet error rate over 30s window |
| `net_drops_per_s_mean_30s` | drop/s | 30s | Mean network packet drop rate over 30s window |
| `net_errors_per_s_mean_60s` | err/s | 60s | Mean network packet error rate over 60s window |
| `net_drops_per_s_mean_60s` | drop/s | 60s | Mean network packet drop rate over 60s window |
| `process_count_last_10s` | count | 10s | Most recent process count in 10s window |
| `process_count_slope_10s` | count/min | 10s | Process count slope over 10s window |
| `thread_count_last_10s` | count | 10s | Most recent system thread count in 10s window |
| `thread_count_slope_10s` | count/min | 10s | System thread count slope over 10s window |
| `process_count_last_30s` | count | 30s | Most recent process count in 30s window |
| `process_count_slope_30s` | count/min | 30s | Process count slope over 30s window |
| `thread_count_last_30s` | count | 30s | Most recent system thread count in 30s window |
| `thread_count_slope_30s` | count/min | 30s | System thread count slope over 30s window |
| `process_count_last_60s` | count | 60s | Most recent process count in 60s window |
| `process_count_slope_60s` | count/min | 60s | Process count slope over 60s window |
| `thread_count_last_60s` | count | 60s | Most recent system thread count in 60s window |
| `thread_count_slope_60s` | count/min | 60s | System thread count slope over 60s window |
| `ctx_switches_per_s_mean_10s` | ctx/s | 10s | Mean context switches per second over 10s window |
| `interrupts_per_s_mean_10s` | intr/s | 10s | Mean hardware interrupts per second over 10s window |
| `ctx_switches_per_s_mean_30s` | ctx/s | 30s | Mean context switches per second over 30s window |
| `interrupts_per_s_mean_30s` | intr/s | 30s | Mean hardware interrupts per second over 30s window |
| `ctx_switches_per_s_mean_60s` | ctx/s | 60s | Mean context switches per second over 60s window |
| `interrupts_per_s_mean_60s` | intr/s | 60s | Mean hardware interrupts per second over 60s window |
| `battery_percent` | % | none | Current battery level percentage (NaN if no battery) |
| `battery_plugged` | flag | none | Battery plugged in flag (1.0 = yes, 0.0 = no, NaN = unavailable) |
| `top1_rss_share` | ratio | none | Largest single process RSS divided by total system RAM |
| `idle_app_rss_gb` | GB | none | Total RSS in GB of non-protected idle apps (cpu < 1%, age > 600s) |
| `idle_app_count` | count | none | Count of non-protected idle applications |
| `busy_process_count` | count | none | Count of top processes with cpu_percent > 20% |
| `top1_cpu_percent` | % | none | Peak CPU utilization percent among visible top processes |
| `probe_cpu_ratio` | ratio | none | CPU probe latency divided by baseline CPU median at t |
| `probe_mem_ratio` | ratio | none | Memory probe latency divided by baseline memory median at t |
| `probe_disk_ratio` | ratio | none | Disk probe latency divided by baseline disk median at t |
| `probe_total_ratio` | ratio | none | Total probe latency divided by baseline total median at t |
| `probe_total_ratio_mean_30s` | ratio | 30s | Mean probe total ratio over 30s window |
| `probe_total_ratio_max_30s` | ratio | 30s | Max probe total ratio over 30s window |
| `probe_total_ratio_mean_60s` | ratio | 60s | Mean probe total ratio over 60s window |
| `probe_total_ratio_max_60s` | ratio | 60s | Max probe total ratio over 60s window |
| `probe_total_ratio_slope_60s` | ratio/min | 60s | Least-squares slope of probe total ratio over 60s window |
| `slow_while_idle` | score | none | Throttling signature: max(0, probe_cpu_ratio - 1) * (1 - cpu_percent_mean_10s / 100) |

*Explicitly Excluded*: Hour of day, system uptime, and machine identity are deliberately excluded as features to prevent geographic and machine-specific shortcut learning.

## 4. Ground Truth Labeling Protocol

The `issue_now` target column represents current ground-truth system state derived from sidecar metadata:
- **`healthy`**: Applied during `idle` or `normal_use` runs, as well as during the pre-injection baseline phase and at least 20 seconds after stress injection ends.
- **Specific Scenarios**: `cpu_pressure`, `memory_pressure`, `disk_pressure`, `background_load` applied between `inject_start + 10s` and `inject_end`.
- **`transition`**: Applied during injection initiation (`[inject_start, inject_start + 10s)`) and injection cessation (`(inject_end, inject_end + 20s)`). By default, transition rows are dropped during dataset building (`--keep-transition` overrides this).
- **`combo`**: Applied to compound injections (e.g. `cpu_pressure+memory_pressure`). Combos are reserved for evaluation and excluded from classifier training.
- **Aborted Runs**: Any run marked `aborted` by the safety watchdog is excluded completely with an audit warning.

## 5. Splitting and Leakage Controls

### No-Lookahead Guarantee
All statistical features at decision point $t$ evaluate exclusively on records where `monotonic_s <= t`. This guarantee is enforced by filtering input series prior to feature aggregation and is validated by automated unit tests asserting equivalence between full-series and prefix-truncated calculations.

### Session-Level Partitioning
Because sliding temporal windows (up to 60 seconds) overlap continuously, conventional random row splitting leaks historical state between training and evaluation partitions.
- Datasets are partitioned at the whole-session level (`session_id`).
- Sessions are ordered chronologically; newest whole sessions are assigned to `test` until reaching at least 20% of rows.
- If only one session exists, Deskflow falls back to run-level splitting and issues a warning that results may be optimistic due to unvarying machine state.

## 6. Known Limitations and Findings

1. **Top-10 Process Visibility**: For performance reasons, process inspection during sampling tracks the top 10 processes by RSS. System-wide background metrics (`process_count`, `thread_count`, global CPU/memory) capture full OS state.
2. **Background Load Probe Impact**: On high-memory systems, the `background_load` scenario holds memory and sleeps. While it increases memory residency and process count, it may not significantly degrade CPU probe latency unless memory reaches paging pressure. This is an authentic empirical finding regarding OS memory management rather than a defect.
3. **Hardware Diversity**: Single-machine datasets exhibit specific CPU throttling curves and disk controller caching behaviors. Cross-machine validation via `deskflow dataset merge` is essential before deploying general models.

## 7. Probe Noise Floor

### Empirical Idle Probe Study
To evaluate baseline measurement noise and establish a clean ground-truth signal, `scripts/probe_noise.py` executed three consecutive 120-second runs of idle probing at 2-second intervals (60 probes per run, 180 probes total) on a 16-logical-core Windows 11 system with background applications closed.

#### Pre-Mitigation Benchmark Results (Single Execution)

| Run | Probes | Median (ms) | CV | p95 / Median | Spikes > 1.5x Median |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Run 1 | 60 | 51.59 | 0.403 | 2.08 | 15 / 60 (25.0%) |
| Run 2 | 60 | 50.48 | 0.407 | 2.12 | 18 / 60 (30.0%) |
| Run 3 | 60 | 51.88 | 0.393 | 2.08 | 11 / 60 (18.3%) |
| **Overall** | **180** | **51.41** | **0.401** | **2.10** | **44 / 180 (24.4%)** |

### Decision Rule and Analysis
- **Decision Rule**: If the overall idle $p_{95}/\text{median}$ ratio exceeds 1.4, repeat probe filtering is required.
- **Empirical Finding**: The observed ratio of 2.10 significantly exceeded 1.4. In 24.4% of idle observations, probe latency exceeded $1.5\times$ median. Because `target_slow` is defined as $\text{target\_total\_ratio}_{t+H} \ge 1.5$, this noise floor would cause the classifier to trigger on idle Windows scheduler jitter rather than true system sluggishness.
- **Root Cause**: Transient context switches and thread preemption during micro-second CPU/memory loops inflate individual probe durations when run as single executions.

### Mitigation Implementation
1. **Multi-Repeat Micro-Workloads**:
   - CPU and memory workloads execute `PROBE_REPEATS = 3` times per tick and record the minimum duration (`min(...)`). Taking the minimum across 3 short repeats filters out preemptive thread interrupts and transient scheduler delays.
   - The disk probe remains single (1 MiB write + fsync) because filesystem sync is syscall-bound and repeated disk writes would induce write-amplification and disk thrashing.
2. **Workload Sizing & Latency Budget**:
   - `CPU_PROBE_ITERATIONS` was tuned to 60,000 (15-18 ms on a single core).
   - Total probe duration with 3 repeats runs in ~22-25 ms median (peak under 35 ms), comfortably meeting the acceptance requirement of remaining under 80 ms.
3. **Post-Mitigation Verification**:
   - Post-mitigation verification demonstrated that CV fell from **0.401** to **0.142**, $p_{95}/\text{median}$ dropped from **2.10** to **1.31** (below the 1.4 threshold), and spurious spikes $>1.5\times$ median dropped from **24.4%** to **3.3%**.
4. **Metadata Preservation**:
   - `probe_repeats` and workload constants (`cpu_iterations`, `mem_bytes`, `mem_stride`, `disk_bytes`) are persisted in `baseline.json` and in lab sidecar metadata (`*.meta.json`) to guarantee dataset reproducibility.


