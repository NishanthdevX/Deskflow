# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
