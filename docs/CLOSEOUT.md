# Deskflow Closeout Report

## 1. Deliverables Inventory

| Deliverable | Status | Evidence / Verification |
|---|:---:|---|
| `battery` command | **PASS** | `deskflow battery` runs power preflight, 14s health scan, and 20s ML accuracy verification on battery. |
| `plugged` command | **PASS** | `deskflow plugged` (alias `ac`) verifies AC power state and computes live 20s model accuracy. |
| `full` command | **PASS** | `deskflow full` runs 10-minute multi-stage stress test suite with moving block bootstrap evaluation. |
| `fix` command | **PASS** | `deskflow fix` lists and executes SAFE/CAUTION remediation tools with before-and-after recovery deltas. |
| `pipeline` command | **PASS** | `deskflow pipeline` renders animated data collection, feature generation, training, and inference pipeline. |
| `info` command | **PASS** | `deskflow info` execution latency measured at 15 ms (< 400 ms threshold) in automated self-test. |
| Home menu loop | **PASS** | `show_home_menu()` loop in `src/deskflow/cli.py` handles interactive choices and non-TTY execution cleanly. |
| Truth contract | **PASS** | `deskflow tools selftest` verifies `synthetic: false`, real model card `61af1146`, SHA-256 digests validated. |
| Fix Center tools | **PASS** | `clean_temp_files`, `clean_caches`, `trim_memory`, `empty_recycle_bin`, `close_idle_apps`, `end_heavy_tasks`. |
| Full scan | **PASS** | Verified in `deskflow full` and test suites; captures real hardware metrics and probe latencies. |
| Pipeline animation | **PASS** | ASCII/Rich live progress animation verified in both color and `--plain` modes. |
| `build-model` and gates | **PASS** | Verifies statistical superiority against persistence baseline via 95% bootstrap confidence intervals. |
| Installers with SHA-256 | **PASS** | `install.ps1` and `install.sh` dry-runs exit code 0; mismatch abort test verified in `test_standards_v2.py`. |
| `release.yml` | **PASS** | Confirmed `permissions: contents: write`, builds wheel and sdist, generates `SHA256SUMS`. |
| `pypi.yml` | **PASS** | Present in `.github/workflows/pypi.yml` and disabled by default (`workflow_dispatch`). |
| Pages workflow & `site/` | **PASS** | `.github/workflows/pages.yml` and `docs/site/` present; builds and deploys static demo site. |
| README accuracy markers | **PASS** | Held-Out Test Set Leaderboard and Multi-Class Performance markers match verified real dataset `61af1146`. |
| `docs/DEMO.md` | **PASS** | Present (9,554 bytes), details instructor walkthroughs and evaluation procedures. |
| `docs/screens/` | **PASS** | All 7 plain terminal screen captures present (`home.txt`, `info.txt`, `scan.txt`, `fix_list.txt`, `pipeline.txt`, `report.txt`, `check.txt`). |
| `docs/PERFORMANCE.md` | **PASS** | Present (3,867 bytes), details probe latency baselines and memory constraints. |

---

## 2. Model Provenance & Verification Status

- **Status**: **REAL HARDWARE MODEL (Certified)**
- **Synthetic**: `false`
- **Dataset SHA-256**: `61af1146ea1afcc9069a86bec29658028193fef1cd520520f2bd0e54936daf7f`
- **Git Commit**: `b64594768037410fff25d2d8026861fc685bdb5e`
- **Dataset Scale**: 1,957 rows (1,457 train / 500 test) across 6 sessions on physical host AMD Ryzen 7 6800HS with Radeon Graphics.
- **Power States**: Dual-calibrated for both `ac` and `battery`.
- **Statistical Superiority**: Certified superior to Persistence baseline (95% bootstrap CI `[+0.0223, +0.0256]` MAE reduction).
- **Cryptographic Hashes**: All 8 bundled model weight files (`.joblib`, `.ubj`, `.json`) verified against `model_card.json`.

---

## 3. Automated Self-Test Timings & Test Count

- **`deskflow tools selftest`**: 12 passed, 0 warnings, 0 skipped, 0 failed (exit code 0).
  - Info latency: 15 ms (< 400 ms)
  - Probe stability: median 19.76 ms, p95 21.42 ms, ratio 1.08
  - Baselines freshness: AC 1.6 days, Battery 4.6 days
- **Pytest Suite**: 195 passed, 1 skipped (0 failures).

---

## 4. Known Limitations

- **Platform Scope**: Supported exclusively on Windows 10/11 (x86_64) and macOS 12+ (Apple Silicon & Intel).
- **Hardware Telemetry Scope**: Micro-benchmark probes monitor CPU, memory bandwidth, and disk I/O latency; workloads constrained exclusively on dedicated GPUs are not directly measured.
- **Single-Host Baselines**: Baselines and training splits are calibrated to the specific host hardware.

