Read AGENTS.md. WORK RULES still apply: speed over ceremony, run only relevant tests while
developing, one full verification at the end, batch edits, no progress narration, no new
documents except those named here. Implement Phase 5 only.

0. AGENTS.md amendments (apply them first)
- Style: rich panels, box-drawing and block-bar characters are allowed. Still no emojis, no
  ASCII-art banners (one plain title line is fine), no marketing words ("excellent",
  "ultimate", "cyber"), and no self-referential text in the UI (never mention agents,
  AGENTS.md, compliance or phases). Any rating label must come from documented numeric
  thresholds in code.
- Scope: there is no daemon, tray icon, background service or GUI. Phase list: 5 stabilize
  and demo app and dashboard style; 6 live accuracy (watch) and report; 7 packaging, release,
  final audit.

1. Truth audit (cap at about 30 minutes)
- A prior "audit report" in the repo makes claims that contradict earlier specs and agent
  reports. Generate docs/STATUS.md by reading the code, not memory. For each item state the
  actual value found in code: models in ml/models.py (spec: Ridge, RandomForest, XGBoost; any
  LightGBM?), lab scenarios and classifier classes (spec: cpu_pressure, memory_pressure,
  disk_pressure, background_load, idle, combo; any "lock"?), feature windows, MIN_HISTORY and
  horizon, scan seconds, model artifact files, explanation method (spec: XGBoost native
  contributions), CLI command list from `deskflow --help`, and EXACTLY which inputs affect the
  health score.
- The health score must depend only on the predicted slowdown ratio (formula in the Phase 4
  spec). If memory, swap or anomaly terms are in the score, remove them; those stay as
  separate displayed categories. Any other deviation from spec: revert if it weakens the ML
  claims, otherwise document it in STATUS.md with the reason.
- Delete the old audit report file, or replace its content with a pointer to STATUS.md.
  README and docs must contain only facts confirmed in STATUS.md.

2. Measurement integrity
a) Overhead. The last result (0.016 s CPU in 66 s) is impossible for a process that runs
   probes, so it measured the wrong process (probably the uv launcher). Rewrite
   scripts/measure_overhead.py: start `sys.executable -m deskflow collect ...` directly (never
   through uv), then measure CPU time of that process plus all children with psutil, and
   cross-check by sampling the same PIDs from outside once per second. Print the process tree
   and PIDs measured and both numbers. Expected range is roughly 0.5 to 3 percent of one core.
b) Probe stability. Observed on an idle machine on battery: consecutive scans had probe
   latency of 20 ms, 81, 73, 72, 20. This is a measurement problem and must be fixed before
   any training data is collected.
   - Extend scripts/probe_noise.py: 5 minutes of continuous probing (one probe every 2 s) on
     the current power state; for each probe record cpu frequency, cpu utilisation, seconds
     since the previous probe, and each repeat's raw timing. Print a histogram, median, p95 /
     median, count above 1.5 x median, and the correlation of slow probes with frequency and
     with time since the previous probe.
   - Test these hypotheses and report evidence for each: (1) cold-core effect (a short probe
     right after sleep runs on a parked or low-frequency core); (2) OpenMP or XGBoost threads
     spinning after model load in scripts/score_stability.py (set nthread=1, OMP_NUM_THREADS=1
     and OMP_WAIT_POLICY=PASSIVE); (3) background activity such as antivirus or indexing; (4)
     the minimum-of-repeats logic (verify repeats are back to back and the minimum is used).
   - If cold-core is the cause, add a constant PROBE_WARMUP_MS (about 10 ms busy loop right
     before the timed CPU workload), applied identically in collect, lab, calibrate and scan,
     and keep total probe cost under 80 ms.
   - Change score_stability.py so each scan runs in a fresh subprocess; it reports probe ms
     and score per run. The model is still synthetic, so the gate below applies to the probe
     only.
   - GATE: over 5 minutes idle, on battery and again plugged in, p95 / median <= 1.4 and fewer
     than 3 percent of probes above 1.5 x median. If the gate cannot be met after a reasonable
     attempt, stop and report the diagnostics; do not relax the gate.
   - Probe settings are stored in baseline.json; after any probe change, baselines become
     invalid and must be recalibrated (refuse with a clear message on mismatch).

3. Fix-flow corrections
- Recommend clean_temp only when cleanable bytes >= 100 MB (named constant); never recommend
  an action while its evidence reads zero. Show a "truncated" marker in the storage line when
  the scan hit its time budget or entry cap.
- Storage totals: run `deskflow doctor` and `deskflow scan` back to back; print both totals,
  roots scanned, entries counted and truncated flags; explain any difference with evidence and
  fix the scanner if the two disagree on roots or skipped directories.
- Browsers (chrome, msedge, firefox, brave, opera, safari) are CAUTION apps: shown with the
  warning "closes all windows and tabs of this browser" and their window count; never
  included by --yes; only with an explicit --include-caution flag plus a typed confirmation.

4. Demo app, so the demo has a real, safe, fixable problem
- `deskflow lab demo-app [--memory-mb N] [--cpu-percent P] [--foreground]`: opens a small
  tkinter window titled "DESKFLOW Demo App (safe to close)" showing its settings. It
  allocates and write-touches N MiB (default 2048, capped by the same memory floor as the lab
  scenarios) and optionally burns P percent CPU on a duty-cycled worker thread. Closing the
  window (WM_CLOSE) frees the memory and exits. By default it launches detached and prints
  its PID; --foreground keeps it attached.
- find_idle_apps and end_busy_process must be able to discover it through the normal rules
  (visible titled window, not in NEVER_OFFER). It is not special-cased anywhere in
  remediation code.
- If tkinter is unavailable, print a clear message and exit with a non-zero code.
- Tests: memory cap logic; classification as idle app and as busy process using a fake
  adapter; integration test skipped when no display is available.

5. Dashboard style
- Create src/deskflow/ui/theme.py and ui/widgets.py: bordered titled panels, key-value
  tables, status colours (green, yellow, red, dim only), a score bar and a comparison bar.
- At startup on Windows reconfigure stdout to UTF-8 with errors="replace". If the console
  encoding is not UTF-8, the terminal is dumb, NO_COLOR is set or --plain is passed, fall
  back to ASCII boxes and ASCII bars.
- Apply to scan, fix (before and after) and doctor. Scan layout: a title line with short
  machine id, power state and scan duration; a row of three panels (Health score with bar and
  label; Predicted slowdown in 20 s; Issue and status); a panel of top factors; a category
  table with status colours and a small bar for memory use; a panel of recommended fixes;
  a footer. Width up to 100 columns, stacking panels down to 80.
- Tests: render each view with Console(record=True) at widths 80 and 100 without exceptions;
  no emoji in output; --plain output is pure ASCII.

6. Verify once and paste real output
Full suite, ruff, mypy; docs/STATUS.md discrepancy list (what was wrong and what you
changed); new measure_overhead.py output including the process tree; probe_noise.py for 5
minutes on battery and again plugged in, with hypothesis findings; score_stability.py
(subprocess version); `deskflow scan` at 100 columns; `deskflow doctor`;
`deskflow lab demo-app --memory-mb 1024` followed by `deskflow fix --dry-run` showing the demo
app offered.

Acceptance criteria
- STATUS.md matches the code; health score depends on slowdown ratio only.
- Overhead measured on the correct process tree.
- Probe gate passed on both power states, or a clear failure report.
- Demo app discoverable and closable through the normal fix flow.
- No emojis; no print() outside CLI and ui modules; no platform checks outside the platform
  package.