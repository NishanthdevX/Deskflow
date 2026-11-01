# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0.dev3] - 2026-11-01

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
