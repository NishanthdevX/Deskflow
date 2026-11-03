# DESKFLOW: project brief for the coding agent

## 1. What this is
DESKFLOW is an open-source command-line device-care tool for Windows 10/11 and macOS 12+.
It samples system telemetry, runs a short measured performance probe, and uses trained ML
models to predict how sluggish the machine will be 30 seconds from now. It tags the likely
cause (CPU pressure or throttling, memory pressure, disk pressure, background app load),
offers safe fixes, and re-measures after a fix. It is also a college ML project, so model
training, offline evaluation and live accuracy comparison are first-class features.

Package name: deskflow. Command: deskflow. Python 3.11 or newer.

## 2. Style rules (strict)
- No emojis anywhere: code, comments, CLI output, docs, commit messages.
- No decorative unicode (box art, arrows, stars). Plain text, simple tables via rich.
- No marketing language ("blazing fast", "AI-powered magic"). Describe what the code does.
- No invented claims, fake badges, fake benchmarks, placeholder statistics or lorem ipsum.
- CLI output is short, factual, aligned. Colour only for status (ok, warning, error).
- Comments explain why, not what. Docstrings on every public function and class.
- README and docs read like a mature open-source project: install, usage, how it works,
  limitations, contributing.

## 3. Engineering rules
- src layout: src/deskflow/. Tests in tests/. Scripts in scripts/. Docs in docs/.
- Full type hints; mypy strict must pass. ruff check and ruff format must pass.
- Dependencies are declared in pyproject.toml only. Use uv for environments and locking.
- Runtime dependencies: psutil, numpy, pandas, scikit-learn, xgboost, shap, typer, rich,
  textual, joblib, platformdirs. Dev: pytest, pytest-cov, ruff, mypy. Add nothing else
  without recording an ADR (docs/decisions/NNNN-title.md).
- Platform-specific code lives only in src/deskflow/platform/ behind a common interface.
  Nothing outside that package may import ctypes, winreg or check sys.platform.
- Never crash on a missing capability. Return None and expose a capability flag instead.
- No network access at runtime. All data stays local. State this in README and SECURITY.md.
- No sudo or administrator elevation by default. Anything needing it is opt-in and documented.
- Use the logging module (never print) for diagnostics; rich only for user-facing output.
- User data goes under platformdirs locations (user_data_dir, user_config_dir).
- Every module has unit tests. Tests that touch real system APIs are marked
  @pytest.mark.integration and must pass on both windows-latest and macos-latest in CI.
- Commits use Conventional Commits (feat:, fix:, docs:, test:, chore:, refactor:).
  One logical change per commit. Do not push.

## 4. Safety rules for any code that changes the system (Phase 6, but design for it now)
- Never act without explicit user confirmation. Provide --dry-run for every action.
- Never terminate: processes not owned by the current user, protected system processes,
  the DESKFLOW process, or its parent terminal and shell.
- Cleanup actions delete only inside an allowlist of temp and cache directories, and never
  follow symlinks out of them.
- Every action logs what it did, and reports bytes or MB freed as measured, not estimated.

## 5. ML design summary (implemented in Phases 2 to 7)
- Probe: fixed CPU, memory and disk micro-workloads timed with perf_counter_ns. This is the
  measured ground truth for how sluggish the machine is.
- Regression: predict probe latency 30 seconds ahead from rolling telemetry features.
  Compare Linear, Ridge, Random Forest, Gradient Boosting and XGBoost. Metrics: MAE,
  RMSE, R2. Time-based train/test split only; never a random split.
- Classification: tag the issue (healthy, cpu_pressure, memory_pressure, disk_pressure,
  background_load) using labels from controlled fault injection. Metrics: precision,
  recall, F1, confusion matrix.
- Anomaly detection: Isolation Forest trained on healthy data only.
- Health score 0 to 100: derived from predicted probe slowdown relative to this machine's
  own healthy baseline. It is never an arbitrary hand-written formula.
- Live accuracy: each prediction is stored, then scored against the probe measured at the
  predicted time. All models run side by side.

## 6. Working agreement
- Do only the current phase. Do not start later phases or add unrequested features.
- Make decisions yourself. If something is ambiguous, choose the simplest option that
  satisfies this brief and record it in docs/decisions/. Do not stop to ask questions
  unless the phase prompt cannot be completed at all.
- Run every verification command listed in the phase and fix failures before finishing.
- Finish with a completion report: files created, commands run with their real output,
  acceptance criteria checked one by one, deviations and why, and known limitations.
- Update CHANGELOG.md (Keep a Changelog format) at the end of each phase.

## 7. Phases
0 Foundation. 1 Collector and probe. 2 Dataset and features. 3 Training and offline
evaluation. 4 Inference and health score. 5 CLI and dashboard. 6 Fixes and re-validation.
7 Live accuracy view. 8 Packaging and release. 9 Report and demo.