# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.1] - 2026-11-07

### Console Redesign, Fix Center Repair & Numbers Integrity (Phase 18)

- **Numbers Integrity Audit & Single Source of Truth (`ModelFacts`)**:
  - Eliminated hardcoded statistics and conflicting numbers across screens.
  - Implemented immutable `ModelFacts` (`deskflow.ml.facts` and `deskflow.truth`), deriving dataset splits, held-out test counts, MAE, confidence intervals, and training provenance directly from `model_card.json` and `report_data.json`.
  - Corrected test set split accounting: explicitly separated 500 held-out test rows from 1,457 training rows (1,957 total across 6 sessions on 1 machine).
  - Clarified offline test set steady-state MAE (0.001) vs. live dynamic workload MAE (0.46) with explicit contextual explanations.
  - Verified no lookahead feature leakage ($r \le 0.7556$) and strict 20.0s target alignment across 100% of rows.
- **Rich Console Architecture & Theme**:
  - Designed official `DESKFLOW_THEME` (`df.title`, `df.good`, `df.warn`, `df.bad`, `df.accent`, `df.muted`, `df.value`, `df.border.*`).
  - Added smooth color blending functions (`blend_health_color`, `blend_load_color`).
  - Implemented responsive layout system with dynamic width breakpoints ($\ge 120$ wide, 100-119 normal, 80-99 compact, $<80$ stacked) and height tiers ($\ge 34$ full 6-row banner, 26-33 wordmark, $<26$ plain title).
  - Added 6-row ASCII art banner with complete leg serif strokes, eliminating cropped character art.
- **Screen Redesigns & Defect Fixes (D1–D22)**:
  - **Home Screen**: Responsive two-column hardware status card and Model Facts card; clean two-column command menu; instant `?` glossary overlay; Enter key cleanly reprompts without scrolling.
  - **Check Flow**: Eliminated nested Live flicker and duplicated preflight borders; replaced dead progress bar with real-time measuring dashboard (sparklines, gauges, countdown, prediction tracking); 3-second verdict sentence; dynamic multi-class attribution with confidence check; anomaly gauge with threshold marker; clean timings footer; post-check key navigation bar.
  - **Fix Center**: Fixed JSON dump bug; added clear human-readable delta indicators (`+15 (better)`, `-0.10x (better)`); dimmed empty tools with "nothing to do"; preserved system processes and active IDEs; recorded audit log paths.
  - **System Info Dashboard**: Replaced raw JSON dumping with fastfetch-style 3-panel display (`HARDWARE`, `MEMORY AND DISKS`, `POWER AND DISPLAY`); restricted JSON to explicit `--json` flag.
  - **Pipeline Tour**: Aligned stage boxes to equal width; bound display strictly to `ModelFacts`; enabled interactive autoplay with Space, Arrow keys, Restart, and Quit controls.
  - **Model Comparison**: Enhanced with dedicated `BASELINE` badge column and proportional whisker bars; strictly honors truth contract for synthetic and real models.
- **Keyboard-First Event Engine**:
  - Non-blocking keyboard input via `deskflow.platform.base.read_key()` supporting Windows (`msvcrt`) and POSIX (`termios`).
  - Standardized `Screen` lifecycle with 10 Hz event loop and graceful non-interactive TTY fallback.
- **Visual Artifacts**:
  - Regenerated SVG and plain-text captures in `docs/screens/` via `scripts/update_screens.py`.

## [0.2.0] - 2026-11-05

### Added
- One-command automated model build: `deskflow tools build-model --runs DIR [DIR ...] [--export]` integrating dataset build with per-session baselines, dataset validation, describe, full hyperparameter tuning grids, and moving block bootstrap evaluation.
- Hard quality gates (`deskflow.ml.gates`): enforced thresholds for training rows (>=1,000), test rows (>=300), test sessions (>=2), per-class balance (>=100 each), dual power states, validation purity, and prediction sanity ([0.0, 20.0]). Stops before export if any gate fails.
- Real-number documentation generators:
  - `scripts/update_readme_accuracy.py`: rewrites README Accuracy section between markers from verified `model_card.json` and `report_data.json`.
  - `scripts/update_screens.py`: regenerates plain screen captures in `docs/screens/` using real CLI outputs.
- Comprehensive 6-minute demonstration script (`docs/DEMO.md`) with measured durations, verbatim presenter text, and fallback workflows.
- 10-minute system stress test: `deskflow full [--yes] [--plain]` executing 5 distinct workload stages (idle, CPU ramp, memory ramp, disk saturation, background load) with live 20s prediction resolution and moving block bootstrap evaluation.
- Pipeline visualization: `deskflow pipeline [--fast] [--static] [--plain]` rendering dual-lane 8 fps architecture animations in terminal.
- Automated 12-check self-test suite: `deskflow tools selftest [--plain]` covering truth contract, model hashes, baseline freshness, probe jitter, CLI commands, responsive layout, zero emoji, JSON purity, exit codes, execution latencies, installer dry runs, and offline socket verification.
- Phase 14 Five-Command Interface v3 (`battery`, `plugged`, `full`, `fix`, `pipeline`, `info`, and `tools`).
- Hardware preflight checks in `deskflow battery` and `deskflow plugged`: verifies matching power state (aborts with code `5` if mismatched), checks baseline age and drift (prompts for 40s calibration), and executes a 6-probe stability check.
- Truth contract (`src/deskflow/truth.py`): tracks `ModelProvenance` loaded from `model_card.json` (`synthetic`, `dataset_hash`, `train_rows`, `test_rows`, `sessions_count`, `machines_count`, `power_states`, `git_commit`, `model_hashes`).
- `deskflow tools verify`: cryptographically verifies all serialized model artifacts against card SHA-256 hashes.
- Standardized exit codes (`0` ok, `1` error/fail-on, `2` usage, `3` unsupported, `4` baseline missing, `5` declined/blocked/wrong power state).
- Section 12 `[ FULL SCAN ]` integrated into terminal, Markdown, and HTML reports.

## [0.1.0] - 2026-11-04

### Added
- Feature Set v2 with 27 canonical features, 6s and 12s rolling windows, 12s warm-up rule, and 20s prediction horizon.
- Lightweight regression models (Ridge, RandomForest, XGBoost) and baselines (ConstantMedian, Persistence, RollingMean).
- Multi-class issue classifiers (LogisticRegression, RandomForest, XGBoost) and Dummy baseline.
- Anomaly detection via Isolation Forest trained on healthy data with 99th percentile thresholding.
- Fast runtime inference (`RuntimeModels`) with sub-50ms latency and TreeSHAP explainability.
- CLI training (`deskflow train`), offline evaluation (`deskflow evaluate`), and model management (`deskflow models`).
- Live system scanning (`deskflow scan`) with 14-second measuring, lazy ML imports, and rich progress display.
- Deterministic 0-100 health score derived from predicted slowdown ratio vs baseline.
- Categories evaluation: Performance (ML), Memory, Storage, and Background Apps with direct measurements.
- Remediation engine (`src/deskflow/remediation/`) supporting `clean_temp`, `close_idle_apps`, `end_busy_process`, and `trim_memory` (Windows `EmptyWorkingSet`).
- Post-fix verification re-scan, before/after impact comparison table, and audit logging to `fixlog.jsonl`.
- Parity test guaranteeing identical feature vectors between live scanning and dataset building.

## [0.1.0.dev2] - 2026-11-01

### Added
- Machine identity module (`src/deskflow/baseline.py`) with random UUID4 generation decoupled from hardware/hostnames.
- System performance baseline calibration (`deskflow baseline calibrate` and `deskflow baseline show`) with 0.5 CV idle verification.
- Fault-injection lab framework (`src/deskflow/lab/`) with `CpuPressureScenario`, `MemoryPressureScenario`, `DiskPressureScenario`, `BackgroundLoadScenario`, `IdleScenario`, and compound `ComboScenario`.
- Injection patterns support: `constant`, `ramp`, and `burst` duty cycles.
- Background safety watchdog (`LabWatchdog`) terminating runs if RAM available falls below safety floor (`max(2GB, 15% RAM)`) or free disk falls below 5 GiB.
- Process tracking lock file (`lab_children.lock`) and leftover scratch cleanup on lab startup and exit.
- Multi-run lab execution plans (`plans.py`) and CLI commands: `deskflow lab list`, `deskflow lab plan`, `deskflow lab run`, and `deskflow lab session` with resumption support.
- Sidecar metadata persistence (`<run_id>.meta.json`) tracking precise planned/actual injection phase timelines and worker allocations.
- Feature engineering registry (`src/deskflow/features/registry.py`) with `FEATURE_SET_VERSION = 1` defining 48 time-windowed statistical features (10s, 30s, 60s).
- Feature extraction pipeline (`extract.py` and `window.py`) with strict no-lookahead enforcement, 60s history warm-up rule, 3x interval gap drop rule, and target alignment.
- Dataset building, validation, description, and multi-machine merging (`builder.py`) outputting Parquet files and sibling JSON manifests with SHA-256 verification.
- Session-level splitting (`splits.py`) preventing rolling window data leakage between train and test splits, with run-level fallback warning.
- CLI dataset commands: `deskflow dataset build`, `deskflow dataset describe`, `deskflow dataset validate`, and `deskflow dataset merge`.
- Architecture decisions: `0002-add-pyarrow-dependency.md`, `0003-session-level-splitting.md`, `0004-baseline-normalization.md`.
- Documentation: `docs/ml-design.md`, `docs/data-collection.md`, and measured collector overhead in `docs/architecture.md`.
- Overhead measurement script: `scripts/measure_overhead.py`.

## [0.1.0.dev1] - 2026-11-01

### Added
- Platform abstraction layer (`src/deskflow/platform/`) with `WindowsAdapter`, `MacOSAdapter`, and `get_adapter()`.
- Telemetry data models with `SCHEMA_VERSION = 1`: `SystemSample`, `ProbeResult`, `ProcessSample`, `StorageEntry`, `StorageReport`, and `Record` (`to_dict` and `from_dict`).
- Telemetry sampler (`Sampler`) measuring CPU, memory, swap, context switches, interrupts, disk I/O, network I/O, battery, temperatures, and process snapshots with delta rates.
- Micro-benchmark performance probe (`run_probe`) measuring integer arithmetic LCG, 8 MiB strided memory access, and 1 MiB fsynced disk writes.
- Storage scanner (`StorageScanner`) with explicit stack traversal, symlink guards, 3.0s time budget, 200k entry cap, and TTL caching.
- Run recorder (`RunRecorder`) writing atomic JSONL files with UTC+hex `run_id`, plus `load_run` reader with schema validation.
- CLI commands: `deskflow snapshot` (compact table or JSON), `deskflow collect` (continuous monitoring), and `deskflow doctor` (platform capability audit).
- Process protection rules (`safety.is_protected`) protecting system PIDs (0/1/4), shell ancestors, critical names, and non-user processes.
- Comprehensive unit and integration test suites (41 tests, 100% passing).
- Data schema reference (`docs/data-schema.md`).

## [0.1.0.dev0] - 2026-11-01

### Added
- Repository foundation: pyproject.toml with hatchling build backend, ruff, mypy strict,
  pytest with coverage, and uv lock.
- Package skeleton: src/deskflow with cli, collector, features, ml, health, remediation,
  evaluation, and platform subpackages.
- CLI entry point with --version and --help.
- README, LICENSE (MIT), CONTRIBUTING, CODE_OF_CONDUCT, SECURITY, CHANGELOG.
- GitHub Actions CI: matrix over windows-latest/macos-latest x Python 3.11/3.12.
- GitHub issue templates and pull request template.
- docs/architecture.md module map.
- docs/decisions/0001-record-architecture-decisions.md (ADR format adopted).
- .gitignore, .editorconfig, .gitattributes.
