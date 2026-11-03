Read AGENTS.md fully. Phases 0 and 1 are complete. Implement Phase 2 only: baseline
calibration, the fault-injection lab, feature engineering, labeling, dataset building and
splitting. Do not train models and do not add inference code.

0. Pre-flight fixes
a) Add .python-version containing 3.12 and re-run `uv sync`. Add tests/test_ml_imports.py
   that imports numpy, pandas, sklearn, xgboost, shap and pyarrow and builds a tiny
   IsolationForest and XGBRegressor on random data. If any of these fail on 3.12, record an
   ADR and fix the dependency pins.
b) The Phase 1 overhead figure (0.00%) is not credible. Write scripts/measure_overhead.py:
   run `deskflow collect --duration 60 --interval 2 --probe-interval 5` as a subprocess,
   read the child's total CPU time (user + system) with psutil when it exits, and print it
   as a percentage of one core. Run it, record the real number and machine in
   docs/architecture.md. Target: under 3 percent. If it is higher, report it honestly.
c) Add pyarrow as a runtime dependency with an ADR (parquet is used for datasets).

1. Machine identity and baseline calibration
- src/deskflow/baseline.py. machine_id: random UUID4 created once and stored in
  <user_config_dir>/machine.json. Never derive it from hostname, username or hardware.
- Baseline: JSON at <user_config_dir>/baseline.json with machine_id, created_utc,
  platform, probe medians for cpu_ns, mem_ns, disk_ns, total_ns, p10 and p90 of total_ns,
  sample count, and deskflow version.
- Command `deskflow baseline calibrate [--duration 120] [--out PATH]`: tell the user to
  close heavy applications and leave the machine idle, show a countdown, take one probe
  every 2 seconds, discard the first 3, store medians. Refuse to save and explain if the
  coefficient of variation of total_ns exceeds 0.5 (machine was not idle).
- Command `deskflow baseline show`.
- Baseline load/save is pure and unit tested.

2. Fault-injection lab (src/deskflow/lab/)
Purpose: create controlled, labeled system stress with known ground truth. All injectors
must be safe on a normal user laptop.

2.1 Scenarios (one module each, common base class Scenario with start(), stop(), name):
- cpu_pressure: worker processes (multiprocessing, spawn context, daemon) running busy
  loops. Intensity low = 25 percent of logical cores (minimum 1), medium = 50 percent,
  high = 100 percent. Option --max-cores N caps workers.
- memory_pressure: allocate and write-touch 64 MiB bytearray chunks in a worker process.
  Target = fraction of memory available at start: low 25 percent, medium 50 percent, high
  70 percent, but never allocate below a floor of max(2 GiB, 15 percent of total RAM)
  remaining available.
- disk_pressure: writer threads in a worker process writing 4 MiB blocks to a scratch
  file with periodic fsync, overwriting a fixed 512 MiB file cyclically (so disk usage
  never grows past 512 MiB per writer). Writers: low 1, medium 2, high 4. Scratch dir is
  <user_data_dir>/lab-scratch/. Abort if free disk space on that volume falls below 5 GiB.
- background_load: spawn idle subprocesses (subprocess.Popen with sys.executable and a
  tiny script) each holding and touching N MiB and sleeping. low = 10 x 50 MiB,
  medium = 25 x 100 MiB, high = 40 x 150 MiB, automatically scaled down so the memory
  floor above is respected.
- idle: does nothing (healthy reference). normal_use is recorded separately with
  `deskflow collect --label normal_use`.
2.2 Patterns, applied to every scenario where meaningful: constant (full intensity for the
  hold period), ramp (intensity grows linearly from near zero to full over the hold
  period; this is what gives prediction lead time), burst (20 s on, 20 s off).
2.3 Safety requirements (all must have tests):
- A watchdog thread checks every 1 second; if memory available < floor or free disk <
  5 GiB, stop all injection immediately and mark the run "aborted" in its metadata.
- Cleanup on normal exit, Ctrl+C, SIGTERM and unexpected exceptions (try/finally plus
  signal handlers plus atexit). Child processes exit if the parent process disappears
  (poll parent PID each second).
- On start, delete leftover scratch files and terminate leftover lab child processes from
  a previous crashed run (track child PIDs in a lock file in the scratch dir).
- Never require administrator rights. Print a plain-text warning describing what will
  happen and require confirmation (typer.confirm) unless --yes is passed.
2.4 Run orchestration (lab/runner.py):
- A lab run = pre phase (baseline, default 30 s, no injection), inject phase (default
  180 s), post phase (recovery, default 60 s). Collector runs in the main process for the
  whole duration with --interval 2 and --probe-interval 4 defaults (must stay a multiple of
  the interval). Record.label = "lab:<scenario>:<intensity>:<pattern>".
- Write a sidecar <run_id>.meta.json next to the JSONL: scenario, intensity, pattern,
  machine_id, session_id, deskflow version, planned and actual UTC timestamps for
  pre_start, inject_start, inject_end, post_end, aborted flag and reason, and the actual
  worker counts and bytes allocated.
- Commands: `deskflow lab list`; `deskflow lab run --scenario S --intensity I --pattern P
  [--pre 30 --hold 180 --post 60] [--yes]`; `deskflow lab plan [--plan standard|quick]`
  (prints the table of runs and total time); `deskflow lab session --plan standard|quick
  [--yes]` which runs the plan sequentially with a 45 second cooldown between runs and a
  single shared session_id (format lab-<UTC timestamp>), resumable: skip runs already
  completed in this session if --resume SESSION_ID is given.
- quick plan: one run per scenario (cpu, memory, disk, background) at medium constant, plus
  one idle run. standard plan: for each of the four scenarios: low-constant,
  medium-constant, medium-ramp, high-constant, high-burst, plus two idle runs (20 runs
  total, about 2 hours).
- Combination scenarios: accept `--scenario cpu_pressure+memory_pressure`. These are for
  evaluation only; label them "combo" and exclude them from classifier training labels.

3. Feature engineering (src/deskflow/features/)
Requirements apply to every feature: computed only from data at or before decision time t
(no lookahead), based on time windows in seconds (not sample counts), and independent of
the sampling interval.
3.1 Decision points: every probe record time t in a run. A row is dropped when fewer than
  60 s of history exist, or any gap larger than 3 x the sampling interval lies inside the
  60 s window (sleep, pause), or the baseline is missing.
3.2 Windows: 10 s, 30 s, 60 s. Per window and metric compute only the statistics listed:
- cpu_percent_mean: mean, std, max, slope (percent per second, least squares)
- cpu_percent_max_core: mean
- cpu_freq_ratio (current/max): mean, min, and drop = window max minus last value; NaN when
  cpu_freq_reliable is false
- mem_percent: last, mean, slope (percent per minute); mem_available_gb: last, slope (GB
  per minute); mem_headroom_s = mem_available / max(-slope, eps), capped at 3600
- swap_percent: last, slope
- disk_read_mb_s and disk_write_mb_s: mean and max; disk_busy_ms_per_s mean (NaN when
  unavailable)
- net_errors_per_s and net_drops_per_s: mean
- process_count and thread_count: last, slope
- ctx_switches_per_s and interrupts_per_s: mean
- battery_percent, battery_plugged (1, 0, or NaN)
- From top_processes: top1_rss_share (largest RSS over total RAM), idle_app_rss_gb (sum
  of RSS of non-protected processes with cpu_percent under 1 and age over 600 s), 
  idle_app_count, busy_process_count (cpu_percent over 20), top1_cpu_percent. Document
  that only the top 10 processes by RSS are visible.
- Probe features (relative to baseline): probe_{cpu,mem,disk,total}_ratio at t, mean and
  max of probe_total_ratio over the 30 s and 60 s windows, slope of probe_total_ratio over
  60 s.
- slow_while_idle = max(0, probe_cpu_ratio - 1) * (1 - cpu_percent_mean_10s / 100): the
  throttling signature (slow although not busy). Document it.
- Do not use hour of day, uptime or machine identity as model features.
3.3 Feature registry: features/registry.py lists every feature with name, unit, window,
  description, and FEATURE_SET_VERSION = 1. A test asserts the dataset columns equal the
  registry.

4. Targets and labels
4.1 Regression targets (horizon H, default 30 s, --horizon option): find the probe record
  whose time is closest to t + H; require |difference| <= half the probe interval + 1 s,
  else drop the row. Columns: target_total_ratio, target_cpu_ratio, target_mem_ratio,
  target_disk_ratio (each = probe latency divided by the baseline median for that
  component) and target_slow = target_total_ratio >= SLOW_THRESHOLD (constant 1.5,
  documented, configurable).
4.2 Issue label issue_now: runs labeled "idle" or "normal_use" give "healthy". Lab runs use
  the sidecar timeline: rows inside [inject_start + 10 s, inject_end] get the scenario
  name (cpu_pressure, memory_pressure, disk_pressure, background_load); rows in the pre
  phase and rows at least 20 s after inject_end get "healthy"; rows in the first 10 s of
  injection and the 20 s after it ends get "transition" and are excluded by default
  (--keep-transition to keep). Combos get "combo". Aborted runs are excluded entirely with
  a warning. Unknown labels are skipped with a warning.
4.3 Extra columns: run_id, session_id, machine_id, scenario, intensity, pattern, ts_utc,
  horizon_s, baseline_total_ms.
  session_id: lab runs use the lab session id; other runs use "day-<UTC date>".

5. Dataset building, splitting and CLI
- `deskflow dataset build --runs DIR [DIR ...] [--baseline FILE] [--horizon 30]
  [--out FILE.parquet] [--keep-transition]`: writes parquet plus a sibling
  <name>.manifest.json (feature_set_version, deskflow version, horizon, row counts,
  source run ids, baseline used, build UTC time, SHA-256 of the parquet). Building twice
  from the same inputs must produce identical data (deterministic ordering).
- `deskflow dataset describe FILE`: rows; rows per issue_now, scenario, intensity, pattern;
  time span; sessions; missing fraction per feature; target_total_ratio min, median, p95,
  max; share of target_slow; split sizes.
- `deskflow dataset validate FILE`: checks columns against the registry, no NaN in targets,
  no duplicate (run_id, ts_utc), reports problems with exit code 1.
- `deskflow dataset merge A B [...] --out FILE`: concatenates datasets from different
  machines; requires equal FEATURE_SET_VERSION and horizon; keeps machine_id.
- Splitting (features/splits.py, pure and tested), written into a `split` column:
  Strategy "session" (default when two or more sessions exist): order sessions by start
  time, then assign whole sessions to test starting from the newest until at least 20
  percent of rows are test; the rest is train. Strategy "run" (fallback when only one
  session exists): assign whole runs to test starting from the newest until 20 percent;
  print a warning that results will be optimistic. A row of a test run never appears in
  train. Document the reasoning (random row splits leak because neighbouring windows
  overlap) in docs/ml-design.md.

6. Tests (all new modules, target at least 85 percent coverage of new code)
- Window statistics: linear memory growth gives the expected slope; constant series gives
  zero slope; irregular timestamps handled; windows use time not count.
- No-lookahead test: for several t, features computed from the full series equal features
  computed from the series truncated at t.
- Gap rule, warm-up drop, target alignment (correct probe chosen, tolerance drop).
- Labeling from sidecar timelines including transition and aborted runs.
- Splitting: no run_id in both train and test; fallback warning; 20 percent rule.
- Lab: scenario lifecycle with tiny parameters (one worker for 2 s, 64 MiB memory, 8 MiB
  disk); watchdog abort with monkeypatched memory and disk values; scratch dir and child
  process cleanup; plan generation arithmetic (standard plan has 20 runs).
- End-to-end: generate a synthetic run directory (use the recorder to write fake samples
  and probes with a known ramp) and sidecar, build a dataset, validate and describe it.
- Integration (@pytest.mark.integration, tiny values so CI runners are safe):
  `deskflow lab run --scenario cpu_pressure --intensity low --pattern constant --pre 8
  --hold 10 --post 8 --yes`, then dataset build and validate on that run, on Windows and
  macOS.

7. Documentation
- docs/ml-design.md: problem statement (predict probe latency H seconds ahead), why a
  measured probe is the target, baseline normalization, feature table generated from the
  registry, label rules, split protocol, leakage controls, known limitations (single or
  few machines, scenarios that may not slow the probe, such as background_load, which
  should be reported as a finding rather than tuned away).
- docs/data-collection.md: the exact step-by-step protocol for collecting a dataset
  (see section 8) written for a contributor.
- ADRs: pyarrow, session-level splitting, baseline normalization.
- README: add the baseline, lab and dataset commands. CHANGELOG: Phase 2 entry.

8. Data collection protocol (document it; I will run it after you finish)
1) deskflow baseline calibrate  2) deskflow lab session --plan quick (smoke test)
3) deskflow lab session --plan standard, twice on different days or machine states (for
example once plugged in, once on battery)  4) deskflow collect --label normal_use
--duration 3600, three times on different days while working normally  5) deskflow dataset
build, describe, validate.

9. Verification (run and paste real output)
uv run ruff check . && uv run ruff format --check . && uv run mypy src && uv run pytest
uv run deskflow lab plan --plan standard
uv run deskflow lab run --scenario cpu_pressure --intensity low --pattern constant
  --pre 10 --hold 15 --post 10 --yes
uv run deskflow baseline calibrate --duration 30   (report whether it was accepted)
uv run deskflow dataset build --runs <the runs directory> --out test.parquet
uv run deskflow dataset describe test.parquet
uv run deskflow dataset validate test.parquet
python scripts/measure_overhead.py

Acceptance criteria
- All checks pass on Python 3.12; CI green on Windows and macOS.
- Lab runs leave no scratch files or child processes behind, including after Ctrl+C
  (test it manually and report).
- The no-lookahead test exists and passes.
- docs/ml-design.md is complete enough for an instructor to understand the whole ML
  formulation without reading code.
- No emojis, no print() outside CLI output modules, no sys.platform outside
  src/deskflow/platform/.