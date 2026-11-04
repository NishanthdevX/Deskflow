# Dataset Collection Protocol

This guide outlines the standardized protocol for recording telemetry runs and compiling machine learning training datasets for Deskflow. Follow these steps sequentially to generate clean, labeled, multi-session data.

---

## Operational Rules

Strictly adhere to the following environment rules during data collection:

1. **Power Configuration**:
   - For battery sessions, ensure battery capacity is above 60% before starting.
   - Use the **Balanced** power plan or energy profile.
   - Ensure Windows Battery Saver / Energy Saver is turned **OFF**.
   - **Never change power state mid-run** (e.g., do not plug in or unplug the charger during an active run). Runs with mid-run power state transitions are detected and dropped by the dataset builder.
2. **Machine Isolation**:
   - Leave the machine completely alone and untouched during active lab injection sessions.
   - Close all background synchronisation clients (OneDrive, Google Drive, Dropbox), browsers, and antivirus on-demand scanners.

---

## Step 1: Baseline Calibration (v2)

Deskflow requires calibrated idle baselines for each power state (AC and Battery). Baselines use the median of the fastest 50% of probe totals to ensure robustness against transient background spikes.

1. Close all user-facing applications.
2. While plugged into AC power:
   ```bash
   uv run deskflow baseline calibrate --duration 90
   ```
3. Disconnect power (running on battery above 60%):
   ```bash
   uv run deskflow baseline calibrate --duration 90
   ```
4. Verify both baselines:
   ```bash
   uv run deskflow baseline show
   ```
   Calibration requires a minimum 60-second duration (default 90s), mean CPU utilisation under 15%, and a spread of the fastest 50% under 0.25.

---

## Step 2: Fault-Injection Protocol

Collect four distinct lab sessions across operational states and plan sizes:

### Session A: Standard Plan on AC Power
Connect to AC power:
```bash
uv run deskflow lab session --plan standard --yes
```
*Expected duration: ~2 hours (20 runs with 45s cooldowns).*

### Session B: Standard Plan on Battery
Ensure battery is > 60% and power cord is disconnected:
```bash
uv run deskflow lab session --plan standard --yes
```
*Expected duration: ~2 hours.*

### Session C: Quick Plan on AC Power
Connect to AC power:
```bash
uv run deskflow lab session --plan quick --yes
```
*Expected duration: ~25 minutes (5 runs).*

### Session D: Quick Plan on Battery
Disconnect power (battery > 60%):
```bash
uv run deskflow lab session --plan quick --yes
```
*Expected duration: ~25 minutes.*

*Note on Resumption*: If a session is interrupted, resume it without re-running finished injections:
```bash
uv run deskflow lab session --plan standard --resume <SESSION_ID> --yes
```

---

## Step 3: Normal Field Telemetry Collection

Record three separate 20-minute sessions of genuine daily computer usage (coding, browsing, meetings, document editing):

```bash
uv run deskflow collect --label normal_use --interval 2 --probe-interval 4 --duration 1200
```

Execute this three times across different working periods. These sessions provide realistic foreground application mixes, diverse scheduling states, and idle periods without artificial fault injection.

---

## Step 4: Dataset Compilation and Validation

Compile all recorded runs into an ML-ready parquet dataset:

1. **Build Dataset**:
   ```bash
   uv run deskflow dataset build --runs C:\Users\<USER>\AppData\Local\deskflow\deskflow\runs --out dataset.parquet
   ```
   *(On macOS, default path is `~/Library/Application Support/deskflow/runs`)*.

   The dataset builder derives per-session idle baselines (`baseline_source = "session_idle"`) when at least 20 idle probes exist within a session, and automatically discards any runs where the power state changed mid-run.

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
