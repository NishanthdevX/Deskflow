Read AGENTS.md fully. Implement Phase 0 only: repository foundation. Do not implement any
telemetry or ML code yet.

1. Project metadata (pyproject.toml)
- Build backend: hatchling. Project name "deskflow" (make the distribution name a single
  value so it can be renamed later), version 0.1.0.dev0, requires-python ">=3.11",
  license MIT, readme README.md.
- Console script: deskflow = deskflow.cli:app
- Dependencies exactly as listed in AGENTS.md section 3; dev dependencies in a
  [dependency-groups] dev group. Use current stable releases and let uv lock them.
- Tool config in pyproject.toml: ruff (line length 100, rules E,F,I,B,UP,SIM,PL minus
  noisy ones you justify), mypy strict with the src layout, pytest (testpaths = tests,
  markers: integration), coverage.

2. Package skeleton
- src/deskflow/__init__.py exposing __version__ (read from package metadata).
- src/deskflow/cli.py: a Typer app with --version and a root help text. Commands for later
  phases are not registered yet.
- Empty but importable subpackages: collector, features, ml, health, remediation,
  evaluation, platform, each with __init__.py containing only a one-line module docstring.
- src/deskflow/py.typed.

3. Repository files
- README.md: project name, one-paragraph description from AGENTS.md section 1, status
  line stating "Early development", planned install and usage sections marked as not yet
  available, a Limitations section (Windows and macOS only, no malware scanning), license.
  No badges.
- LICENSE (MIT, copyright holder as a placeholder line I will edit), CONTRIBUTING.md
  (dev setup with uv, run checks, commit style, how to add a platform capability),
  CODE_OF_CONDUCT.md (Contributor Covenant 2.1 reference text), SECURITY.md (local-only
  data, no network, how to report issues), CHANGELOG.md, .gitignore (Python, data/, models/
  artifacts, IDE files), .editorconfig, .gitattributes (LF line endings).
- .github/ISSUE_TEMPLATE/bug_report.yml and feature_request.yml,
  .github/pull_request_template.md.
- docs/architecture.md (module map from the layout in AGENTS.md, no detail yet),
  docs/decisions/0001-record-architecture-decisions.md (ADR format).

4. CI (.github/workflows/ci.yml)
- Triggers: push and pull_request.
- Matrix: os [windows-latest, macos-latest], python [3.11, 3.12].
- Steps: checkout, install uv, uv sync, ruff check, ruff format --check, mypy, pytest with
  coverage. Cache uv.

5. Tests
- tests/test_cli.py: invoking `deskflow --version` exits 0 and prints the version;
  `deskflow --help` exits 0.
- tests/test_package.py: every subpackage imports.

6. Verification (run all and paste real output in the report)
uv sync
uv run ruff check .
uv run ruff format --check .
uv run mypy src
uv run pytest
uv run deskflow --version

Acceptance criteria
- All commands above succeed with zero warnings.
- Fresh clone plus `uv sync` works with no manual steps.
- No emojis or decorative characters in any file.
- Git history has Conventional Commit messages; working tree clean.