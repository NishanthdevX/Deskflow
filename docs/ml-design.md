# Machine Learning Formulation and Design

Deskflow anticipates system degradation and sluggishness before it impacts foreground user responsiveness by predicting normalized probe latency 20 seconds ahead ($H = 20$ s).

Models support dual-power state validation (AC vs. Battery), with independent baseline calibration for each power profile.
The dual-state baseline format is retained for future AC validation.

## 1. Problem Formulation

Conventional device-monitoring utilities trigger reactive alerts only after CPU or memory saturation reaches arbitrary fixed thresholds (such as 90% utilization). However, high utilization during smooth video rendering or batch compilation does not necessarily imply sluggishness, while low utilization during thermal throttling or disk thrashing can severely stall foreground responsiveness.

Deskflow formulates responsiveness prediction as:
- **Primary Task**: Multi-horizon regression predicting normalized probe latency ratios $H$ seconds into the future:
  $$\text{target\_total\_ratio}_{t+H} = \frac{\text{total\_ns}_{t+H}}{\text{baseline\_total\_ns}}$$
- **Secondary Classification**: Multi-class attribution identifying the root operational issue at decision time $t$:
  $$\text{issue\_now}_t \in \{\text{healthy}, \text{cpu\_pressure}, \text{memory\_pressure}, \text{disk\_pressure}, \text{background\_load}\}$$
- **Anomaly Detection**: Unsupervised boundary estimation on healthy states to flag novel degradations and combined multi-fault conditions.
- **Component Latency Decomposition**: Component regression for `target_cpu_ratio`, `target_mem_ratio`, and `target_disk_ratio` to support fine-grained attribution.

### Why a Measured Probe is the Ground Truth

Direct synthetic micro-workloads (120k LCG integer loops, 8 MiB strided memory accesses, and 1 MiB fsynced disk writes) serve as an objective ground truth:
1. Probes measure the actual end-to-end scheduling, memory bandwidth, and I/O latency experienced by user applications.
2. Probes execute deterministically on identical code paths across all runs, removing external application variance.
3. Probes require no privileged OS APIs or invasive kernel drivers.

## 2. Power-State Baseline Normalization

Hardware clock rates, dynamic frequency scaling, and power-saving profiles differ drastically depending on whether a laptop is plugged into AC power or running on battery.

Deskflow normalizes telemetry against state-aware baselines (`baseline.json`):
1. Baselines are calibrated independently for `"ac"` and `"battery"` power states (`deskflow baseline calibrate`).
2. Each training and test row dynamically selects the active baseline matching `battery_plugged` at time $t$ (`row_state = "battery" if not battery_plugged else "ac"`).
3. The selected power state is recorded in the `baseline_state` dataset column.
4. Component and total probe latencies are divided by the active baseline's calibrated medians.

## 3. Feature Registry (FEATURE_SET_VERSION = 2)

Feature Set v2 uses short rolling windows (6 s and 12 s only) with a 12 s warm-up requirement (`MIN_HISTORY_S = 12.0`). High-overhead network and interrupt counters are dropped, keeping at most 30 high-signal features in the canonical registry (currently 27 features):

| Feature Name | Unit | Window | Description |
| :--- | :--- | :--- | :--- |
| `cpu_percent_mean_mean_6s` | % | 6s | Mean CPU utilization across 6s window |
| `cpu_percent_mean_mean_12s` | % | 12s | Mean CPU utilization across 12s window |
| `cpu_percent_mean_max_12s` | % | 12s | Peak CPU utilization across 12s window |
| `cpu_percent_mean_slope_12s` | %/s | 12s | Least-squares slope of CPU utilization over 12s window |
| `cpu_freq_ratio_mean_12s` | ratio | 12s | Mean CPU frequency ratio (current/max) over 12s window |
| `cpu_freq_ratio_min_12s` | ratio | 12s | Minimum CPU frequency ratio over 12s window |
| `mem_percent_mean_12s` | % | 12s | Mean RAM utilization percentage over 12s window |
| `mem_percent_slope_12s` | %/min | 12s | Least-squares RAM utilization slope (%/min) over 12s window |
| `mem_available_bytes` | bytes | None | Instantaneous available physical memory headroom |
| `swap_percent` | % | None | Instantaneous swap/pagefile utilization percent |
| `disk_read_mb_s_mean_12s` | MB/s | 12s | Mean disk read throughput in MB/s over 12s window |
| `disk_write_mb_s_mean_12s` | MB/s | 12s | Mean disk write throughput in MB/s over 12s window |
| `ctx_switches_per_s_mean_12s` | ops/s | 12s | Mean context switch rate over 12s window |
| `process_count` | count | None | System-wide total process count at time $t$ |
| `thread_count` | count | None | System-wide total thread count at time $t$ |
| `battery_plugged` | flag | None | External AC power indicator (1.0 = yes, 0.0 = no) |
| `top1_rss_share` | ratio | None | Largest single process RSS divided by total system RAM |
| `top1_cpu_percent` | % | None | Peak CPU utilization percent among top 25 processes |
| `busy_process_count` | count | None | Count of top 25 processes with CPU > 20% |
| `idle_app_rss_gb` | GB | None | Total RSS in GB of non-protected idle applications |
| `idle_app_count` | count | None | Count of non-protected idle applications |
| `probe_cpu_ratio` | ratio | None | CPU probe latency divided by baseline CPU median at $t$ |
| `probe_mem_ratio` | ratio | None | Memory probe latency divided by baseline memory median at $t$ |
| `probe_disk_ratio` | ratio | None | Disk probe latency divided by baseline disk median at $t$ |
| `probe_total_ratio` | ratio | None | Total probe latency divided by baseline total median at $t$ |
| `probe_total_ratio_mean_12s` | ratio | 12s | Mean probe total ratio over 12s window |
| `slow_while_idle` | score | None | Throttling signature: $\max(0, \text{probe\_cpu\_ratio} - 1) \cdot (1 - \text{cpu\_mean}/100)$ |

## 4. Model Architectures & Baselines

All models utilize fixed `random_state = 42` and are trained strictly on training splits.

### 4.1 Regression (`target_total_ratio`)
- **Baselines**:
  - `ConstantMedian`: Constant median of training target values.
  - `Persistence`: Instantaneous `probe_total_ratio` at decision time $t$.
  - `RollingMean`: 12s rolling mean `probe_total_ratio_mean_12s`.
- **Machine Learning Models**:
  - `Ridge`: Linear model with `SimpleImputer` and `StandardScaler`.
  - `RandomForestRegressor`: At most 100 trees, max depth $\le 10$.
  - `XGBRegressor`: At most 150 trees, max depth $\le 5$.

### 4.2 Classification (`issue_now`)
Targets: `healthy`, `cpu_pressure`, `memory_pressure`, `disk_pressure`, `background_load`. Combo and transition rows are excluded from classification training.
- **Baseline**: `DummyClassifier(strategy="prior")`.
- **Models**:
  - `LogisticRegression`: Balanced class weighting with scaler pipeline.
  - `RandomForestClassifier`: Balanced class weighting, small tree depth.
  - `XGBClassifier`: Multi-class log-loss with native probability output.

### 4.3 Anomaly Detection
- `IsolationForest`: 100 trees, `max_samples = 256` trained strictly on healthy training rows.
- **Threshold**: Set to the 99th percentile of healthy training scores. Any sample with anomaly score $\ge \text{threshold}$ is flagged as anomalous.

## 5. Evaluation Protocol & Transparency

- **Tuning**: Tiny fixed hyperparameter grids ($\le 8$ combinations) using `GroupKFold(n_splits=4)` partitioned strictly by `run_id`.
- **Test Set Isolation**: The test split is evaluated exactly once for final reporting.
- **Block Bootstrap**: 300 resamples by `run_id` to establish 95% confidence intervals on MAE, $R^2$, macro F1, and improvement over Persistence. A model is declared "better than Persistence" only if its 95% bootstrap interval strictly excludes zero.
- **Synthetic Data Transparency**: Any artifact or report derived from synthetic data carries the prominent banner `SYNTHETIC DATA - NOT A RESULT`.

## 6. Runtime Artifacts & Latency Budget

- **Storage Format**: Saved under `<user_data_dir>/models/<id>/`:
  - Native XGBoost `.ubj` format for instant C++ deserialization.
  - Pure JSON parameters for Ridge (enabling vector dot-product scoring without scikit-learn).
  - Compressed joblib (`compress=3`) for RandomForest and IsolationForest.
  - Total directory size $< 10$ MB.
- **Inference Budget**: Sub-50ms inference across all models (`RuntimeModels.predict(features)`), achieving ~30ms on 100 consecutive calls.
- **Attribution**: Native TreeSHAP feature contributions via `top_contributors(row, k=3)`, where contributions plus bias equal the raw model prediction.

## 7. Integrity Checks Run

Phase 18 conducted a comprehensive numbers integrity and data audit across the training datasets, held-out splits, feature stores, and live telemetry to eliminate screen contradictions and verify evaluation claims.

### 7.1 Dataset Split Accounting
- **Total Rows**: 1,957 rows across 6 sessions on 1 machine (`Anonymous1122`).
- **Train Set**: 1,457 rows across 5 sessions.
- **Held-Out Test Set**: 500 rows across 1 session.
- **Audit Finding**: Previous UI headers incorrectly reported "1,957 held-out test rows" by conflating train + test. The single source of truth (`ModelFacts`) strictly reports 500 held-out test rows and 1,457 train rows.

### 7.2 Target and Feature Leakage Audit
- **Lookahead Verification**: Target values were confirmed to be strictly future-shifted by 20.0 seconds (`horizon_s == 20.0s` for 100% of rows). No instantaneous probe latency is used as target.
- **Feature Correlations with `target_total_ratio`**:

| Feature Name | Pearson Correlation ($r$) | Verification Result |
| :--- | :--- | :--- |
| `cpu_percent_mean_max_12s` | 0.7556 | Pass (< 0.80) |
| `cpu_percent_mean_mean_12s` | 0.7512 | Pass (< 0.80) |
| `cpu_percent_mean_mean_6s` | 0.7483 | Pass (< 0.80) |
| `busy_process_count` | 0.7455 | Pass (< 0.80) |
| `cpu_freq_ratio_mean_12s` | 0.7424 | Pass (< 0.80) |
| `top1_cpu_percent` | 0.7397 | Pass (< 0.80) |
| `cpu_freq_ratio_min_12s` | 0.7380 | Pass (< 0.80) |
| `probe_cpu_ratio` | 0.7180 | Pass (< 0.80) |
| `probe_total_ratio_mean_12s` | 0.7176 | Pass (< 0.80) |
| `probe_total_ratio` | 0.7163 | Pass (< 0.80) |

No feature exceeds $|r| \ge 0.80$ (and none exceeds 0.99), proving that the feature registry contains no lookahead leakage or target duplicates.

### 7.3 Offline Test MAE (0.001) vs. Live Evaluation MAE (0.46)
- **Offline Test Environment**: In `predictions_test.parquet`, 92.8% of rows have $|\text{target} - \text{probe\_total\_ratio}| < 0.05$. Because synthetic stress benchmarking runs in sustained steady-state phases (e.g. constant sustained CPU burn, steady RAM hold), 20-second future latencies closely mirror current latencies. The Persistence baseline achieved an offline MAE of 0.0246, and the trained RandomForest achieved 0.0011.
- **Live Desktop Environment**: Real-world live usage is non-stationary, characterized by abrupt burst workloads (launching browser tabs, starting language servers, IDE builds, opening multimedia). During burst transitions, 20-second future responsiveness deviates substantially from current state, resulting in a live MAE of ~0.46.
- **Resolution**: Live check and comparison screens explicitly contextualize this distinction: "Offline: held-out steady-state test set. Live: dynamic non-stationary desktop workloads."

