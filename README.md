# deskflow

Command-line device-care tool for Windows 10/11 and macOS 12+.

## Description

Deskflow samples local Windows and macOS telemetry, runs a measured performance probe, and uses
trained models to predict probe slowdown 20 seconds ahead. It reports likely pressure, offers
confirmed safe fixes, and records data locally for offline evaluation.

Status: Version 0.1.0 release packaging and one-line installers complete. See `docs/STATUS.md`.

## Quick Start

Windows PowerShell (no administrator rights required):

```powershell
irm https://raw.githubusercontent.com/NishanthdevX/Deskflow/main/install.ps1 | iex
```

macOS Terminal (no administrator rights required):

```bash
curl -fsSL https://raw.githubusercontent.com/NishanthdevX/Deskflow/main/install.sh | bash
```

These scripts verify `uv`, provision an isolated runtime without administrator privileges or permanent PATH modifications, and open the Deskflow interactive home menu. See [docs/INSTALL.md](docs/INSTALL.md) for verification details, offline usage, and uninstallation.

## Installation

Development installation via [uv](https://docs.astral.sh/uv/):

```bash
git clone <repo-url>
cd deskflow
uv sync
```

## Usage

### Platform Health & Capability Check (`doctor`)

Inspect platform sensor availability, cache directories, and baseline probe execution times:

```bash
deskflow doctor
```

### Instant System Snapshot (`snapshot`)

Sample hardware metrics over a 1-second delta, run an inline performance probe, and display a formatted table:

```bash
deskflow snapshot
```

To output raw telemetry as machine-readable JSON:

```bash
deskflow snapshot --json
```

### Continuous Telemetry Recording (`collect`)

Stream telemetry samples and periodic performance probes into an append-only JSONL run dataset:

```bash
# Record for 10 minutes (default), probing every 5 seconds
deskflow collect --duration 600 --interval 2 --probe-interval 5 --label baseline

# Record a quick smoke test run
deskflow collect --duration 20 --interval 1 --probe-interval 5 --label smoke --out ./data/smoke.jsonl
```

### Performance Baseline Calibration (`baseline`)

Calibrate and inspect your machine's idle latency baseline:

```bash
# Calibrate idle baseline over 120 seconds (samples probe every 2s, discards first 3)
deskflow baseline calibrate --duration 120

# View current calibrated baseline
deskflow baseline show
```

### Controlled Fault-Injection Lab (`lab`)

Generate labeled system stress workloads under safety watchdog supervision:

```bash
# List all scenarios, intensities, and patterns
deskflow lab list

# Inspect execution plan schedule
deskflow lab plan --plan standard

# Execute a single injection run with confirmation
deskflow lab run --scenario cpu_pressure --intensity medium --pattern constant --pre 30 --hold 180 --post 60

# Run an automated session across plans
deskflow lab session --plan quick --yes
```

### Dataset Building & Validation (`dataset`)

Compile raw JSONL run recordings and sidecar metadata into Parquet datasets:

```bash
# Build parquet dataset with session-level splits
deskflow dataset build --runs ./runs --out dataset.parquet

# Inspect dataset metrics, distributions, and missing features
deskflow dataset describe dataset.parquet

# Validate schema, target integrity, and duplicate freedom
deskflow dataset validate dataset.parquet

# Merge datasets from multiple machines
deskflow dataset merge machine1.parquet machine2.parquet --out combined.parquet
```

### Machine Learning Model Training (`train`)

Train regression, classification, and anomaly detection models on engineered datasets:

```bash
# Fast training mode skipping hyperparameter tuning
deskflow train --dataset dataset.parquet --quick

# Full GroupKFold hyperparameter tuning over parameter grids
deskflow train --dataset dataset.parquet
```

### Model Evaluation & Diagnostic Reports (`evaluate`)

Evaluate trained models on the test split with 300-sample block bootstrap intervals:

```bash
# Generate Markdown evaluation report
deskflow evaluate --model-set latest --out report.md

# Generate diagnostic PNG plots (requires 'train' extra: uv sync --extra train)
deskflow evaluate --model-set latest --plots --out report.md
```

### Model Set Inspection (`models`)

List and inspect serialized model sets and metadata cards:

```bash
# List all trained model sets
deskflow models list

# Inspect detailed model card, thresholds, and test performance
deskflow models show latest
```

### System Scan & Health Assessment (`scan`)

Measure live telemetry and performance probes, evaluate trained ML models, compute health score, and diagnose issues:

```bash
# Live 14-second scan
deskflow scan

# Scan with timing latency breakdown
deskflow scan --timing

# Replay a recorded JSONL run
deskflow scan --replay run.jsonl

# Output structured JSON schema
deskflow scan --json
```

### Remediation & Post-Fix Verification (`fix`)

Execute user-confirmed safe remediation actions and re-scan system to verify performance improvements:

```bash
# Preview actions without making changes
deskflow fix --dry-run

# Interactive selection and execution
deskflow fix

# Execute specific action directly
deskflow fix --action clean_temp --yes
```

## How it works

1. The collector samples psutil telemetry on a short interval.
2. The feature pipeline builds rolling-window vectors (6s and 12s) from raw samples (Feature Set v2).
3. Regression models predict probe latency 20 seconds ahead.
4. A classifier tags the likely cause (cpu_pressure, memory_pressure, disk_pressure,
   background_load, or healthy).
5. An Isolation Forest flags anomalies not seen during healthy training.
6. A 0-100 health score is derived from predicted slowdown vs. this machine's own baseline.
7. Optional safe fixes are offered; each requires explicit confirmation and supports --dry-run.

## Scope and limitations

- Windows 10/11 and macOS 12+ only. Linux is not supported.
- No malware scanning or security hardening. This tool observes and reports; it does not
  protect against threats.
- All data is stored locally. No network access occurs at runtime.
- Models are trained and validated on battery power only. Results on AC power are unvalidated.
- Current datasets are collected from one machine and may not generalize to other hardware.
- Some metrics require administrator elevation on certain platforms. Those are opt-in and
  documented individually.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

MIT. See [LICENSE](LICENSE).
