# Dataset Collection Protocol

This guide outlines the standardized protocol for recording telemetry runs and compiling machine learning training datasets for Deskflow. Follow these steps sequentially to generate clean, labeled, multi-session data.

---

## Prerequisites

1. Ensure Python 3.12 is active and dependencies are synchronized:
   ```bash
   uv sync
   ```
2. Verify system health and available disk space:
   ```bash
   uv run deskflow doctor
   ```
   Confirm that at least 10 GiB of free disk space is available.

---

## Step 1: Baseline Calibration

Before running injections or collections, calibrate your system's idle responsiveness baseline:

1. Close all user-facing applications (web browsers, IDEs, media players, background sync clients).
2. Leave the computer completely untouched and idle.
3. Run the calibration command:
   ```bash
   uv run deskflow baseline calibrate --duration 120
   ```
4. Verify the baseline:
   ```bash
   uv run deskflow baseline show
   ```
   If calibration fails due to high variance (coefficient of variation $> 0.5$), check for background indexing, cloud sync, or antivirus scans, let the machine settle, and re-run.

---

## Step 2: Smoke Test (Quick Plan)

Run the quick plan to verify injector operation and watchdog monitoring:

```bash
uv run deskflow lab session --plan quick --yes
```

This runs 5 short sequential injections (CPU pressure, memory pressure, disk pressure, background load, and idle reference) with a 45-second cooldown between runs.

---

## Step 3: Fault-Injection Sessions (Standard Plan)

Run the complete 20-run standard plan twice across distinct machine operational states:

1. **Session 1 (e.g. AC Power / Plugged In)**:
   ```bash
   uv run deskflow lab session --plan standard --yes
   ```
   *Expected duration: ~2 hours.*

2. **Session 2 (e.g. Battery Power or Different Day)**:
   ```bash
   uv run deskflow lab session --plan standard --yes
   ```
   *Expected duration: ~2 hours.*

*Note on Resumption*: If a session is interrupted (e.g. system sleep or manual abort), resume it without re-running finished injections:
```bash
uv run deskflow lab session --plan standard --resume <SESSION_ID> --yes
```

---

## Step 4: Normal Field Telemetry Collection

Record three separate 1-hour sessions of genuine daily computer usage (coding, browsing, meetings, document editing):

```bash
uv run deskflow collect --label normal_use --duration 3600
```

Execute this three times across different days or working sessions. The resulting records represent realistic foreground application mixes and natural multi-tasking patterns.

---

## Step 5: Dataset Compilation and Validation

Compile all recorded runs into an ML-ready parquet dataset:

1. **Build Dataset**:
   ```bash
   uv run deskflow dataset build --runs C:\Users\<USER>\AppData\Local\deskflow\deskflow\runs --out dataset.parquet
   ```
   *(On macOS, default path is `~/Library/Application Support/deskflow/runs`)*.

2. **Inspect Statistics and Label Distribution**:
   ```bash
   uv run deskflow dataset describe dataset.parquet
   ```
   Verify row counts, session coverage, and target slow percentages.

3. **Validate Schema and Integrity**:
   ```bash
   uv run deskflow dataset validate dataset.parquet
   ```
   Confirms zero duplicate timestamp keys, zero target NaNs, and complete feature coverage against the registry.

