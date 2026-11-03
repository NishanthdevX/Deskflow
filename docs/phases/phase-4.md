Read AGENTS.md and obey the WORK RULES from earlier (speed, minimal docs, batch edits, one
verification run). Phases 0 to 3 are done. Implement Phase 4 only: scan, health score, fix
and re-validation. No watch command and no packaging yet.

1. One source of truth for features
- Refactor so the live path and the dataset builder call the same function
  (features.extract.build_row or equivalent). Parity test: for a synthetic recorded run, the
  row the dataset builder produces at time t equals (numpy allclose) the row the live path
  produces from only the records up to t.

2. deskflow scan [--seconds 14] [--json] [--no-storage] [--timing] [--replay RUN.jsonl] [--yes]
a) Resolve the baseline for the current power state. If missing, explain, and with
   confirmation (or --yes) run an automatic 40 s calibration, then continue. Refuse with a
   clear message if baseline probe settings differ from the model card.
b) Measure with the same Sampler and probe code as `collect` (sample 2 s, probe 4 s,
   processes at start and at 10 s) for SCAN_SECONDS (default 14, minimum 13) with a rich
   progress bar. Do not import sklearn, xgboost or load any model until measuring is
   finished (imports would disturb the probe).
c) Then scan storage (3 s budget, skippable), load RuntimeModels, predict.
d) Primary models: model_card.json names the primary model for each task, chosen by
   cross-validation score on the training split only (never the test split). Update the
   training code to record this. Scan uses those.
e) Health score in health/score.py with documented constants: predicted slowdown ratio
   clipped to [0.5, 5]; score 100 when ratio <= 1.0, falling linearly to 0 at ratio 3.0,
   rounded to an integer. Labels: Good >= 80, Fair 50 to 79, Poor < 50.
f) Categories shown: Performance (the ML score), Memory (available memory percent and
   headroom), Storage (cleanable temp and cache bytes), Background apps (idle app count and
   RSS). Only the Performance score and the issue tag are ML-derived; the other rows are
   measurements with simple documented thresholds. State this in the output footer and in
   docs.
g) Issue tag: classifier argmax when its probability >= 0.5, otherwise "none"; plus the
   Isolation Forest anomaly flag. Reasons: top 3 contributors from the regressor
   (explain.top_contributors) in plain words using registry descriptions.
h) Recommended fixes from remediation/recommend.py (tag to actions table), each with
   measured evidence, for example "5 idle apps hold 2.1 GB" or "1.4 GB of temp files older
   than 24 hours". Never promise gains.
i) If the model card has synthetic: true, print a one-line "DEMO MODEL trained on synthetic
   data" notice at the top.
j) Output: compact, no emojis, fits 80 columns: title line with machine and power state;
   score and label; predicted slowdown in 20 s; issue tag with confidence; anomaly flag;
   reasons; category table; recommended fixes; footer with scan duration. --timing prints
   per-stage seconds. --json prints a stable documented schema (docs/cli.md).
k) --replay builds the scan result from a recorded run's records (no live measuring).

3. Remediation (src/deskflow/remediation/), every action supports dry-run
- clean_temp: roots from the platform adapter with safe_to_clean true; delete only regular
  files older than 24 h; never follow symlinks; skip locked or in-use files without error;
  remove empty directories; report bytes actually deleted (measured).
- close_idle_apps: candidates are non-protected processes owned by the user with cpu under
  1 percent, age over 10 minutes; group by process name; offer groups whose total RSS is at
  least 300 MB with name, count and total RSS. Terminate gracefully (terminate, wait 5 s);
  kill only with --force. Warn that unsaved work in those apps may be lost.
- end_busy_process: up to 3 non-protected user processes using at least 50 percent CPU at
  scan time, same termination rules.
- trim_memory (Windows only, EmptyWorkingSet via ctypes inside platform/windows.py): for
  non-protected user processes with RSS of at least 50 MB; report available memory before
  and 2 s after, as measured. Hidden on macOS. Document that gains are temporary.
- Only the top 25 processes by RSS are visible to scan; document this limit.
- Safety rules from AGENTS.md section 4 apply to all actions: confirmation required, never
  protected processes, never other users' processes, never this process or its parents.
- deskflow fix [--action NAME ...] [--dry-run] [--yes] [--no-rescan]: scan, list
  recommended actions with evidence, choose by number (default none), confirm, execute,
  wait 3 s, re-scan, print a before and after table (score, predicted ratio, measured probe
  ms, available memory, temp bytes), append a record to <data_dir>/fixlog.jsonl. If the
  score did not improve, say so plainly.

4. Tests (concise)
- Feature parity test; score mapping bounds and monotonicity; tag mapping; scan --replay on
  a synthetic run; JSON schema; clean_temp on tmp dirs (old file, new file, symlink,
  locked-file simulation, dry-run changes nothing); protected processes never offered;
  confirmation required; fixlog written.

5. Verify once (paste real output)
full suite, ruff, mypy; `deskflow scan --timing` on this machine (report total seconds and
the per-stage breakdown); `deskflow scan --json | python -m json.tool`;
`deskflow fix --dry-run`.

Acceptance criteria
- scan completes in under 20 seconds on this machine (report the real number).
- Fix actions are previewable, confirmed, logged, and re-validated by a re-scan.
- No emojis, no print() outside CLI modules, no platform checks outside the platform package.