# Deskflow

[![CI](https://github.com/NishanthdevX/Deskflow/actions/workflows/ci.yml/badge.svg)](https://github.com/NishanthdevX/Deskflow/actions/workflows/ci.yml)
[![Release](https://github.com/NishanthdevX/Deskflow/actions/workflows/release.yml/badge.svg)](https://github.com/NishanthdevX/Deskflow/actions/workflows/release.yml)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![Platform Windows](https://img.shields.io/badge/platform-Windows%2010%2F11-0078D6.svg)](https://microsoft.com/windows)
[![Platform macOS](https://img.shields.io/badge/platform-macOS%2012%2B-000000.svg)](https://apple.com/macos)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Deskflow is a device care system and telemetry analyzer for Windows 10/11 and macOS 12+ that measures system performance and forecasts sluggishness 20 seconds ahead using trained machine learning models. It continuously monitors CPU, memory, and disk telemetry alongside micro-benchmark probes to compute a deterministic health score and pinpoint root causes. When performance degrades, Deskflow provides safe, confirmed remediation actions and measures actual before-and-after resource recovery.

---

## Architecture Overview

```
 [ Hardware Sensors ]      [ Micro-Benchmark Probe ]
   CPU / RAM / Disk            Latency Micro-Tests
          |                            |
          +--------------+-------------+
                         |
                 [ Sampler Engine ]
              12s Telemetry Rolling Window
                         |
             [ Feature Extraction (v2) ]
              Rolling Means, Trends, Latency
                         |
          +--------------+--------------+
          |                             |
 [ Regression Ensemble ]     [ Classification & Anomaly ]
  XGBoost / Random Forest     Issue Attribution & Outliers
  20s Slowdown Forecast       (CPU / RAM / Disk / Background)
          |                             |
          +--------------+--------------+
                         |
             [ Fix Center & Recovery ]
           Safe & Confirmed Remediation
          Before/After Delta Verification
```

---

## 1. Quick One-Liners (No Administrator Rights Required)

Deskflow installs and runs in an isolated user environment with zero system pollution.

### Windows 10 / 11 (PowerShell)

Run directly through `irm` (Invoke-RestMethod):

```powershell
irm https://raw.githubusercontent.com/NishanthdevX/Deskflow/main/install.ps1 | iex
```

To run an instant non-downloading dry-run check:

```powershell
powershell -ExecutionPolicy Bypass -Command "& { `$env:DESKFLOW_DRYRUN='1'; irm https://raw.githubusercontent.com/NishanthdevX/Deskflow/main/install.ps1 | iex }"
```

### macOS 12+ (Terminal)

```bash
curl -fsSL https://raw.githubusercontent.com/NishanthdevX/Deskflow/main/install.sh | bash
```

Both installers automatically verify `uv`, download release wheels with cryptographic `SHA256SUMS` validation, verify file integrity, and launch Deskflow in an isolated user environment. For advanced configuration, manual checksum verification, and uninstallation, see [docs/INSTALL.md](docs/INSTALL.md).

---

## 2. Quick Start

```powershell
# Interactive graphical terminal dashboard
deskflow

# Run health scan and live accuracy verification on battery power
deskflow battery

# Run health scan and live accuracy verification on AC wall power
deskflow plugged

# Run comprehensive 10-minute stress test across all models
deskflow full

# Render rich animated telemetry, training, and inference pipeline
deskflow pipeline

# Inspect Fix Center remediation tools and reclaim resources
deskflow fix

# Display high-density hardware specs, GPU, DPI, and volume usage
deskflow info

# Advanced diagnostic, laboratory, and verification tools catalog
deskflow tools

# Cryptographically audit model artifacts against model card digests
deskflow tools verify
```

---

## 3. Primary Command Surface

Deskflow features a focused, standardized 7-command root interface. All advanced diagnostic, laboratory, dataset, and training tools remain accessible via `deskflow tools <command>`.

| Command | Action | Key Verification Performed |
| :--- | :--- | :--- |
| `deskflow battery` | Battery Health Check | Power state preflight, 14s health scan, and live 20s ML accuracy verification on battery. |
| `deskflow plugged` | Plugged Health Check | Power state preflight, 14s health scan, and live 20s ML accuracy on AC power (alias: `ac`). |
| `deskflow full` | End-to-End Stress Test | 10-minute automated stress suite with moving block bootstrap accuracy across every model. |
| `deskflow pipeline` | Architectural Pipeline | Live visualizer tracking data collection, feature generation, model training, and inference. |
| `deskflow fix` | Fix Center Remediation | Safe, confirmed remediation actions with before-and-after resource recovery delta audits. |
| `deskflow info` | System Specifications | High-density hardware readout: CPU model, cores, RAM, GPU, DPI resolution, disk storage. |
| `deskflow tools` | Advanced Tools Catalog | Developer, laboratory, dataset, and diagnostic tool catalog. |

Legacy and administrative commands (`scan`, `watch`, `check`, `report`, `prep`, `baseline`, `collect`, `dataset`, `lab`, `models`, `train`, `doctor`, `snapshot`, `bugreport`, `privacy`, `data`, `uninstall`) remain fully functional directly and via `deskflow tools <command>`.

---

## 4. Truth Contract Guarantee

Deskflow enforces a strict truth contract between ML models and user-facing metrics:

- **Synthetic Transparency**: When bundled or active models are trained on synthetic simulation data (`synthetic: true` in `model_card.json`), user-facing interfaces (`scan`, `battery`, `plugged`, `report`) suppress all synthetic accuracy numbers (MAE, $R^2$, $F_1$, confusion matrices, feature weights).
- **Truth Banner**: In place of synthetic metrics, Deskflow displays:
  > **NO REAL DATA YET**: this model was trained on synthetic data. Run deskflow full to measure real accuracy on this machine.
- **Cryptographic Model Verification**: `deskflow tools verify` audits on-disk model artifacts against SHA-256 digests in `model_card.json`. Production releases are guarded to prevent non-pre-release releases containing synthetic models.

---

## 5. How It Works

1. **Micro-Benchmark Probe**:
   Deskflow executes a calibrated, synthetic probe measuring the exact latency (nanoseconds) required to perform memory bandwidth copies, CPU arithmetic, and small disk writes.
2. **Dual-Power Baselines**:
   Hardware latencies vary significantly between AC power and battery power. Deskflow enforces calibrated baselines for each power state (`ac` and `battery`). A slowdown ratio is computed as:
   $$\text{Ratio} = \frac{\text{Measured Probe Latency (ms)}}{\text{Calibrated Baseline Latency (ms)}}$$
   A ratio of 1.0 represents nominal idle speed; 1.5 represents a 50% slowdown.
3. **20-Second Ahead Prediction**:
   Using a 12-second rolling window of hardware metrics (CPU utilization, context switch rates, memory commit, disk queue depth, process counts), Deskflow's primary regression models forecast the expected slowdown ratio 20 seconds into the future.
4. **Root Cause Classification & Anomaly Detection**:
   If sluggishness is predicted, a multiclass classifier attributes the cause (`cpu_pressure`, `memory_pressure`, `disk_pressure`, or `background_load`). Concurrently, an IsolationForest anomaly detector detects atypical patterns even when a specific cause cannot be isolated.

---

## 6. Safety and Fix Center Policies

Deskflow categorizes all remediation tools into strict risk tiers:

### SAFE Tools (Automated or Confirmed Execution)

| Tool ID | Action Taken | Boundaries & Guarantees |
| :--- | :--- | :--- |
| `clean_temp_files` | Clean User Temp | Deletes temporary files older than 24 hours in user `%TEMP%` / `/tmp`. Never touches recent files or system roots. |
| `clean_caches` | Clean App Caches | Purges browser and application disk caches. Does not touch cookies, login credentials, or browser history. |
| `trim_memory` | Trim Process Memory | Calls OS APIs (`SetProcessWorkingSetSize` / `task_policy`) to return unreferenced pages to the standby list. |

### CAUTION Tools (Strict User Confirmation Required)

| Tool ID | Action Taken | Boundaries & Guarantees |
| :--- | :--- | :--- |
| `empty_recycle_bin` | Empty Recycle Bin | Permanently purges items from Recycle Bin / Trash. Irreversible. |
| `close_idle_apps` | Close Idle Applications | Sends graceful `WM_CLOSE` / `SIGTERM` signals to idle windowed applications with no input for > 15 minutes. Never sends hard kill signals. |
| `end_heavy_tasks` | Terminate Heavy Tasks | Displays an itemized table of heavy processes (PID, CPU %, RSS memory) and requires explicit task selection. Never terminates tasks indiscriminately. |

### Protection Guarantees

- **No Silent CAUTION Execution**: Combining `--yes` and `--include-caution` skips CAUTION tools unless explicit confirmation is provided via `--confirm-caution <ID>`.
- **Protected Processes**: Terminals, IDEs, code editors, and core operating system processes are hardcoded as protected and are never terminated under any circumstances.

---

## 7. Privacy: 100% Offline and Local

Deskflow is an entirely offline application:

- **Zero Remote Telemetry**: Contains zero tracking scripts, network telemetry collectors, or background daemons.
- **Local Disk Storage**: All baselines, telemetry runs, model weights, and audit logs remain strictly under the user application directory:
  - Windows: `%LOCALAPPDATA%\deskflow`
  - macOS: `~/Library/Application Support/deskflow`
- **Redacted Bug Reports**: Running `deskflow bugreport` strips home folder paths, usernames, and hardware GUIDs before writing the report locally.

---

## 8. Accuracy and Model Verification

<!-- ACCURACY_START -->

**Hardware Verification Dataset**: Tested across **1,957 real rows** (1,457 train, 500 test) across **6 sessions** on 1 physical machine(s) in both **ac, battery** power states (dataset `61af1146`, commit `b6459476`).

### Held-Out Test Set Leaderboard (20s Prediction Horizon)

| Model | Type | Test MAE (95% CI) | Test RMSE | R² | vs Persistence (95% CI) | Certified Superior? |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **XGBoost** | Model | 0.0008 `[0.0007, 0.0008]` | 0.0010 | 1.0000 | `[+0.0223, +0.0256]` | **Yes (Superior)** |
| **RandomForest** | Model | 0.0011 `[0.0010, 0.0012]` | 0.0016 | 1.0000 | `[+0.0220, +0.0252]` | **Yes (Superior)** |
| **Ridge** | Model | 0.0076 `[0.0070, 0.0082]` | 0.0100 | 0.9992 | `[+0.0158, +0.0185]` | **Yes (Superior)** |
| **RollingMean** | Baseline | 0.0186 `[0.0175, 0.0199]` | 0.0239 | 0.9953 | `[+0.0047, +0.0073]` | **Yes (Superior)** |
| **Persistence** | Baseline | 0.0246 `[0.0231, 0.0264]` | 0.0311 | 0.9921 | Baseline | N/A |

### Multi-Class Issue Classification Performance

| Classifier | Balanced Accuracy | Macro F1 | Status |
| :--- | :--- | :--- | :--- |
| **Dummy** | 0.2000 | 0.0689 | Verified |
| **LogisticRegression** | 1.0000 | 1.0000 | Verified |
| **RandomForest** | 1.0000 | 1.0000 | Verified |
| **XGBoost** | 1.0000 | 1.0000 | Verified |

<!-- ACCURACY_END -->

- **Statistical Superiority Criterion**: A model is certified as better than the Persistence baseline only when the 95% bootstrap confidence interval of its error reduction strictly excludes zero.
- **Instructor Verification Report**: Run `deskflow report` to inspect the full 11-section terminal analysis, or `deskflow report --html report.html` to generate a publication-ready report featuring inline SVG charts, confusion matrices, and feature contributions.
- **Live Verification**: Run `deskflow watch` to evaluate live predictions against ground-truth measured probe latencies in real time.

---

## 9. Captured Terminal Outputs

Text captures of real commands executed with `--plain` are available in `docs/screens/`:

- [Interactive Home Dashboard](docs/screens/home.txt)
- [System Specifications (`deskflow info`)](docs/screens/info.txt)
- [Health Scan (`deskflow scan`)](docs/screens/scan.txt)
- [Fix Center Inventory (`deskflow fix --list`)](docs/screens/fix_list.txt)
- [Pipeline Visualization (`deskflow pipeline`)](docs/screens/pipeline.txt)
- [Comprehensive Verification Report (`deskflow report`)](docs/screens/report.txt)
- [Guided 5-Step Check (`deskflow check`)](docs/screens/check.txt)

---

## 10. Technical Limitations

- **Platform Scope**: Supported exclusively on Windows 10/11 (x86_64) and macOS 12+ (Apple Silicon & Intel). Linux is unsupported due to non-standard sensor APIs across distributions.
- **Hardware Telemetry Scope**: Telemetry probes monitor CPU, memory bandwidth, and disk I/O latency. Workloads constrained entirely on dedicated GPUs are not directly measured.
- **Security Scope**: Deskflow is a performance and resource health tool; it is not an anti-virus or malware scanner.
- **Single-Machine Baselines**: Baselines and training splits are unique to host hardware; datasets from different physical architectures should not be combined without normalization.

---

## 11. Contributing and License

- **Contributing**: Please review [CONTRIBUTING.md](CONTRIBUTING.md) and [SECURITY.md](SECURITY.md). All pull requests must pass `ruff`, `mypy`, and `pytest`.
- **License**: MIT License. See [LICENSE](LICENSE) for details.
