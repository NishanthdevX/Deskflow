# Release Policy and Procedures

This document outlines the versioning, changelog, and release workflow for Deskflow.

---

## 1. Semantic Versioning (SemVer)

Deskflow strictly follows [Semantic Versioning 2.0.0](https://semver.org/):
`vMAJOR.MINOR.PATCH`

- **MAJOR (`X.0.0`)**: Incompatible API or CLI changes (e.g., changes to exit codes, changes to telemetry schema that break existing datasets, removal of commands or flags).
- **MINOR (`0.Y.0`)**: Backwards-compatible new features, added commands, model architecture enhancements, or new platform adapters.
- **PATCH (`0.0.Z`)**: Backwards-compatible bug fixes, performance optimizations, documentation updates, and test improvements.

Pre-releases are tagged with `-rc.N` or `-dev` (e.g. `v0.1.0-rc.1`).

---

## 2. Changelog Policy

All changes are documented in `CHANGELOG.md` following [Keep a Changelog](https://keepachangelog.com/en/1.0.0/):
- `Added` for new features and commands.
- `Changed` for modifications in existing functionality.
- `Deprecated` for soon-to-be removed features.
- `Removed` for removed features.
- `Fixed` for any bug fixes.
- `Security` in case of vulnerabilities.

---

## 3. Release Checklist

1. **Verify Quality Gates**:
   Ensure all automated checks pass locally:
   ```bash
   uv run ruff check .
   uv run ruff format --check .
   uv run mypy src
   uv run pytest
   ```
2. **Update Version**:
   - Update `version` in `pyproject.toml` and `src/deskflow/__init__.py`.
   - Update `VERSION` in `install.ps1` and `install.sh`.
   - Update `CHANGELOG.md` with release notes and date.
3. **Commit and Tag**:
   ```bash
   git commit -m "chore: release v0.2.0"
   git tag -a v0.2.0 -m "Release v0.2.0"
   git push origin main --tags
   ```
4. **GitHub Release & Assets**:
   - GitHub Actions (`.github/workflows/release.yml`) builds the distribution wheel and generates `SHA256SUMS`.
   - A pre-release or release draft is populated on GitHub with verified SHA-256 sums.
5. **PyPI Deployment**:
   - To deploy to PyPI, set repository variable `PUBLISH_TO_PYPI = "true"` and publish the release.
   - Trusted Publishing executes via `.github/workflows/pypi.yml`.

