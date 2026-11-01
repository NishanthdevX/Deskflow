# 4. Baseline Normalization for Performance Targets

Date: 2026-11-01

## Status

Accepted

## Context

Raw performance probe measurements (CPU arithmetic loop, memory strided writes, fsynced disk writes) vary drastically across diverse client hardware. For example, a modern high-end desktop workstation executes the CPU probe in 12 ms, while an ultra-low-power laptop or older processor requires 45 ms under completely idle conditions.

If models predict raw millisecond latency, a model trained on fast machines would falsely diagnose healthy operation on slower laptops as severe degradation, while models trained on slow machines would miss significant sluggishness on fast workstations.

## Decision

We introduce an idle calibration routine (`deskflow baseline calibrate`) and express all latency features and prediction targets as normalized ratios relative to machine-calibrated baseline medians:

1. A unique machine identity (`machine_id`, stored in `<user_config_dir>/machine.json` as a random UUID4) is decoupled from hardware fingerprints and hostnames.
2. An idle calibration procedure samples probes while idle, discards the initial warmup readings, and requires the coefficient of variation (CV) of total probe latency to be $\le 0.5$. If $CV > 0.5$, calibration is refused because the machine was not sufficiently idle.
3. Medians for CPU, memory, disk, and total latency are persisted in `<user_config_dir>/baseline.json`.
4. Both features (e.g. `probe_total_ratio`, `probe_cpu_ratio`) and regression targets (e.g. `target_total_ratio = probe_total_ns / baseline_total_ns`) are normalized by these calibrated medians.
5. The binary classification target `target_slow` is defined by a relative threshold: `target_total_ratio >= 1.5` (50% latency increase above idle).

## Consequences

- Models learn relative degradation signatures rather than hardware clock speeds.
- Enables cross-machine dataset merging and unified model training.
- Baseline calibration is required before building datasets or evaluating predictions.

