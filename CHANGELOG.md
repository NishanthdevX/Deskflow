# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0.dev0] - 2026-11-01

### Added
- Repository foundation: pyproject.toml with hatchling build backend, ruff, mypy strict,
  pytest with coverage, and uv lock.
- Package skeleton: src/deskflow with cli, collector, features, ml, health, remediation,
  evaluation, and platform subpackages (stubs only; implementation begins in Phase 1).
- CLI entry point with --version and --help.
- README, LICENSE (MIT), CONTRIBUTING, CODE_OF_CONDUCT, SECURITY, CHANGELOG.
- GitHub Actions CI: matrix over windows-latest/macos-latest x Python 3.11/3.12.
- GitHub issue templates and pull request template.
- docs/architecture.md module map.
- docs/decisions/0001-record-architecture-decisions.md (ADR format adopted).
- .gitignore, .editorconfig, .gitattributes.
