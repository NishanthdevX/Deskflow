# Architectural and Design Decisions

This document is the append-only record of technical and architectural decisions for DESKFLOW. Each entry is a self-contained summary explaining the rationale and technical trade-offs.

## 0001: Psapi GetPerformanceInfo and NtQuerySystemInformation

To reduce Windows collector CPU overhead from 35% of a core down to under 2%, thread counting and process enumeration were decoupled from `CreateToolhelp32Snapshot` and `psutil.process_iter`. Thread and process totals are fetched via `GetPerformanceInfo` in psapi (<0.2 ms), while process snapshots utilize `NtQuerySystemInformation(SystemProcessInformation)` on a 10-second cadence.

## 0002: Multi-Repeat Micro-Probe Workloads

Idle probing on multicore desktop systems displayed sporadic system interrupts resulting in p95/median noise ratios above 1.4. To guarantee stable latency measurements without artificial noise, CPU and memory micro-workloads execute 3 repeats and record the minimum duration, while sequential disk writes execute once. The entire probe is budgeted and measured under 80 ms, and repeats are stored in `baseline.json` and lab metadata sidecars.

## 0003: Dual Power-State Baselines

Performance characteristics and CPU throttling policies differ fundamentally between AC power and battery power on laptops. `baseline.json` stores dual baselines keyed by power state (`ac` and `battery`), with desktop systems defaulting to AC. Calibration measures the active power state and merges into the store, and feature extraction matches against the active state.

## 0004: Standardized Lab Plan Matrix v2

The automated fault-injection lab plan standardizes on 14 runs for the full suite (4 scenarios across medium-constant, high-ramp, and high-burst, plus 2 idle reference runs) and 5 runs for the quick suite (one medium-constant run per scenario plus one idle run). Durations are fixed to 20s pre-injection, 90s hold, 30s post-injection, and 20s cooldown to balance thoroughness with reasonable completion times.

## 0005: Two-Pass Process RSS Filter and Network Cadence

Process telemetry collection is scoped to the top 25 processes sorted by resident set size (RSS). Memory consumption is read across all processes, while CPU percentages and thread counts are evaluated only for the top 25, caching ownership per `(pid, create_time)`. Network I/O counters (`psutil.net_io_counters`) are sampled on a 10-second cadence to eliminate syscall latency spikes during 2-second collection ticks.

## 0006: Feature Set v2 and Lightweight Model Architectures

Feature Set v2 replaces long 60-second temporal windows with short 6-second and 12-second windows, lowering the run warm-up requirement from 60 seconds to 12 seconds (`MIN_HISTORY_S = 12.0`) and targeting a 20-second prediction horizon ($H = 20.0$ s). The feature registry is capped at 27 features, dropping high-overhead network, interrupt, and battery percentage metrics while retaining high-signal indicators (`slow_while_idle`, frequency ratio, top-25 RSS shares, probe latency ratios, and `battery_plugged`). Model architectures prioritize ultra-low inference latency: Ridge (stored as JSON for vector dot products), RandomForest (at most 100 trees), XGBoost (stored in native `.ubj` format), and IsolationForest (100 trees on healthy training rows). GroupKFold on `run_id` prevents cross-split data leakage, and artifacts serialize under 1.5 MB with sub-50ms inference.

## 0007: Device Scan, Health Score Derivation, and Safe Remediation

Phase 4 introduces live device scanning (`deskflow scan`) and user-confirmed safe remediation (`deskflow fix`):
1. **Feature Extraction Parity:** Refactored feature engineering to use a unified `build_row` function shared identically between offline dataset building and live telemetry scanning.
2. **Deterministic Health Score (0-100):** Health score is derived mathematically from predicted total probe slowdown ratio relative to the machine's calibrated baseline. Ratios <= 1.0 map to 100 (Good), falling linearly to 0 at ratio 3.0 (Poor). Only Performance and Issue Tag are ML-derived; Memory, Storage, and Background Apps categories reflect direct hardware measurements.
3. **Primary Model CV Selection:** `model_card.json` records primary models chosen strictly by cross-validation scores on the training split only (never the test split). Heavy ML libraries (scikit-learn, XGBoost) are lazily imported only after the telemetry measurement phase completes to avoid contaminating probe latency.
4. **Safety & Remediation Protocol:** All remediation actions (`clean_temp`, `close_idle_apps`, `end_busy_process`, `trim_memory`) support `--dry-run`, require explicit user confirmation, strictly exclude protected system and parent processes, log all executions to `fixlog.jsonl`, and verify efficacy through an automated 3-second post-fix re-scan. Windows-specific memory trimming (`EmptyWorkingSet`) is abstracted behind the platform adapter.
