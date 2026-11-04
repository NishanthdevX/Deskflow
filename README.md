# Deskflow

Deskflow is a command-line device care tool for Windows 10/11 and macOS 12+ that measures system performance and predicts sluggishness 20 seconds ahead using trained machine learning models. It continuously monitors CPU, memory, and disk telemetry alongside micro-benchmark probes to compute a deterministic health score and pinpoint root causes. When performance degrades, Deskflow offers verified, safe remediation tools and measures actual before-and-after resource recovery.

---

## 1. Quick One-Liners (No Administrator Rights Required)

### Windows (PowerShell)
```powershell
irm https://raw.githubusercontent.com/NishanthdevX/Deskflow/main/install.ps1 | iex
```

### macOS (Terminal)
```bash
curl -fsSL https://raw.githubusercontent.com/NishanthdevX/Deskflow/main/install.sh | bash
```

These scripts verify `uv`, download the release wheel and cryptographic `SHA256SUMS`, verify the package integrity, and launch Deskflow in an isolated user environment. See [docs/INSTALL.md](docs/INSTALL.md) for checksum verification, uninstallation, and troubleshooting.

---

## 2. Quick Start

- `deskflow`: Opens the interactive terminal launcher dashboard.
- `deskflow check`: Executes a guided 5-step health evaluation (info, scan, fix inventory, live accuracy watch, and summary report with HTML export).
- `deskflow scan`: Runs a fast 14-second device health scan, computes health score (0-100), and outputs predicted slowdown.
- `deskflow fix --list`: Displays the Fix Center tools inventory and cleanable bytes.
- `deskflow info`: Displays high-density host specifications, DPI-aware display resolution, and volume usage.

---

## 3. Command Reference

| Command | Summary | Description |
| :--- | :--- | :--- |
| `deskflow check` | Guided full check | 5-step workflow: Info -> Scan -> Fix Center -> Live Watch -> Summary Report. |
| `deskflow scan` | Quick health scan | Samples 13s telemetry, runs probe, evaluates ML models, and scores health. |
| `deskflow fix` | Fix Center | Safe, confirmed remediation actions with before/after re-measurement. |
| `deskflow info` | System info | High-density hardware, GPU, DPI-aware resolution, and disk breakdown. |
| `deskflow watch` | Live accuracy watch | Compares live predictions at time $t$ against probe latency measured at $t+20\text{s}$. |
| `deskflow report` | Verification report | Renders 11 terminal sections, Markdown, JSON, and standalone HTML with SVGs. |
| `deskflow prep` | Readiness checklist | Verifies power states, baselines, and models for demonstrations. |
| `deskflow doctor` | Health diagnostics | Checks host platform capabilities, permissions, and baseline status. |
| `deskflow snapshot`| Quick snapshot | Takes a 1-second telemetry sample and inline micro-benchmark probe. |
| `deskflow baseline`| Baseline calibration| Calibrates and manages idle hardware latency baselines per power state. |
| `deskflow lab` | Fault injection | Controlled synthetic workloads under safety watchdog supervision. |
| `deskflow dataset` | Dataset engineering | Builds, validates, and merges feature datasets from raw telemetry runs. |
| `deskflow models` | Model management | Inspects trained model sets, hyperparameters, and active pointers. |
| `deskflow train` | Model training | Trains regression, classification, and anomaly models on engineered datasets. |
| `deskflow bugreport`| Sanitize bug report | Generates local Markdown report with personal paths and IDs redacted. |
| `deskflow privacy` | Privacy policy | Displays offline data storage policies and local directory paths. |
| `deskflow data` | Storage usage | Inspects local disk space consumed by telemetry, baselines, and models. |
| `deskflow uninstall`| Data removal | Purges local Deskflow configuration and history after typed confirmation. |

---

## 4. How It Works

1. **Micro-Benchmark Probe**:
   Deskflow executes a synthetic, calibrated probe measuring the exact latency (nanoseconds) required to perform memory bandwidth copies, CPU arithmetic, and small disk writes.
2. **Dual-Power Baselines**:
   Hardware latencies vary significantly between AC power and battery power. Deskflow enforces calibrated baselines for each power state (`ac` and `battery`). A slowdown ratio is computed as:
   $$\text{Ratio} = \frac{\text{Measured Probe Latency (ms)}}{\text{Calibrated Baseline Latency (ms)}}$$
   A ratio of 1.0 represents nominal idle speed; 1.5 represents a 50% slowdown.
3. **20-Second Ahead Prediction**:
   Using a 12-second rolling window of hardware metrics (CPU utilization, context switch rates, memory commit, disk queue depth, process counts), Deskflow's primary regression models forecast the expected slowdown ratio 20 seconds into the future.
4. **Root Cause Classification & Anomaly Detection**:
   If sluggishness is predicted, a multiclass classifier attributes the cause (`cpu_pressure`, `memory_pressure`, `disk_pressure`, or `background_load`). Concurrently, an IsolationForest anomaly detector detects atypical patterns even when a specific cause cannot be isolated.

---

## 5. Safety and Fix Center Policies

Deskflow categorizes all remediation tools into strict risk tiers:

### SAFE Tools
- `clean_temp_files`: Deletes temporary files older than 24 hours in user `%TEMP%` / `/tmp`. Never touches recent files or system roots.
- `clean_caches`: Purges browser and application disk caches. Does not touch cookies, login credentials, or browser history.
- `trim_memory`: Calls OS APIs (`SetProcessWorkingSetSize` / `task_policy`) to return unreferenced pages to the standby list.

### CAUTION Tools
- `empty_recycle_bin`: Permanently purges items from Recycle Bin / Trash. Irreversible.
- `close_idle_apps`: Sends graceful `WM_CLOSE` / `SIGTERM` signals to idle windowed applications with no active user input for over 15 minutes. Never sends hard kill signals (`SIGKILL` / `TerminateProcess`).
- `end_heavy_tasks`: Displays an itemized table of heavy processes (PID, CPU %, RSS memory) and requires the user to select specific task numbers. Never terminates tasks indiscriminately.

### CAUTION Policy & Protection
- **No Silent CAUTION Execution**: Combining `--yes` and `--include-caution` skips CAUTION tools unless explicit confirmation is provided via `--confirm-caution <ID>`.
- **Protected Processes**: Terminals, IDEs, code editors, and core operating system processes are hardcoded as protected and are never terminated under any circumstances.

---

## 6. Privacy: 100% Offline and Local

Deskflow is an entirely offline application:
- **Zero Remote Telemetry**: Contains zero tracking scripts, network telemetry collectors, or background daemons.
- **Local Disk Storage**: All baselines, telemetry runs, model weights, and audit logs remain strictly under the user application directory:
  - Windows: `%LOCALAPPDATA%\deskflow`
  - macOS: `~/Library/Application Support/deskflow`
- **Redacted Bug Reports**: Running `deskflow bugreport` strips home folder paths, usernames, and hardware GUIDs before writing the report locally.

---

## 7. Accuracy and Model Verification

Deskflow's accuracy claims are grounded in statistical verification:
- **Statistical Superiority Criterion**: A model is certified as better than the Persistence baseline only when the 95% bootstrap confidence interval of its error reduction strictly excludes zero.
- **Instructor Verification Report**: Run `deskflow report` to inspect the full 11-section terminal analysis, or `deskflow report --html report.html` to generate a publication-ready report featuring inline SVG charts, confusion matrices, and feature contributions.
- **Live Verification**: Run `deskflow watch` to evaluate live predictions against ground-truth measured probe latencies in real time.

---

## 8. Technical Limitations

- **Platform Scope**: Supported exclusively on Windows 10/11 (x86_64) and macOS 12+ (Apple Silicon & Intel). Linux is unsupported due to non-standard sensor APIs across distributions.
- **Hardware Telemetry Scope**: Telemetry probes monitor CPU, memory bandwidth, and disk I/O latency. Workloads constrained entirely on dedicated GPUs are not directly measured.
- **Security Scope**: Deskflow is a performance and resource health tool; it is not an anti-virus or malware scanner.
- **Single-Machine Baselines**: Baselines and training splits are unique to host hardware; datasets from different physical architectures should not be combined without normalization.

---

## 9. Captured Terminal Outputs

Text captures of real commands executed with `--plain` are available in `docs/screens/`:
- [Interactive Home Menu](docs/screens/home.txt)
- [System Specifications (`deskflow info`)](docs/screens/info.txt)
- [Health Scan (`deskflow scan`)](docs/screens/scan.txt)
- [Fix Center Inventory (`deskflow fix --list`)](docs/screens/fix_list.txt)
- [Guided 5-Step Check (`deskflow check`)](docs/screens/check.txt)

---

## 10. Contributing and License

- **Contributing**: Please review [CONTRIBUTING.md](CONTRIBUTING.md) and [SECURITY.md](SECURITY.md). All pull requests must pass `ruff`, `mypy`, and `pytest`.
- **License**: MIT License. See [LICENSE](LICENSE) for details.
