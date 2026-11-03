Read AGENTS.md (with the Phase 5 amendments). WORK RULES apply. Implement Phase 6 only: live
accuracy comparison and the instructor report. No packaging yet.

1. deskflow watch [--seconds N] [--demo SCENARIO] [--no-live] [--yes] [--plain]
- Loop with the same Sampler and probe code as collect (2 s tick, probe every 4 s, process
  pass every 10 s). At every probe time t with at least 12 s of history, build the feature row
  with the shared function used by scan and the dataset builder, then predict the slowdown 20 s
  ahead with every regression model: Persistence, RollingMean, Ridge, RandomForest, XGBoost.
  Keep each prediction pending.
- Resolution: when a probe at about t + 20 s arrives (nearest probe within 2 s, using the same
  target alignment function as the dataset builder), compute the actual ratio against the
  baseline of the power state at that time and resolve all pending predictions for t.
- Per model: rolling (last 30 resolved) and cumulative MAE, tolerance accuracy (within 0.25),
  resolved count, and difference from Persistence. For the primary model also cumulative
  slow-event precision and recall (predicted slow versus actually slow).
- Show the current health score, current probe ratio, predicted ratio, issue tag and anomaly
  flag.
- Display with rich Live at 1 Hz using the ui widgets: header panel (score bar, tag), a model
  comparison table with Persistence clearly marked as the baseline row and a bar column for
  rolling MAE, and the last 5 resolved predictions (time, predicted, actual, error) for the
  primary model. Fits 80x24. --no-live prints one status line every 4 s. Ctrl+C exits cleanly.
- Persist every prediction and resolution to <data_dir>/live/<session_id>.jsonl and write a
  summary json at exit.
- --demo cpu_pressure|memory_pressure|disk_pressure|background_load runs a timeline with the
  lab scenarios (reuse lab code and its watchdog): 40 s idle, 80 s ramp at medium intensity,
  30 s recovery. Requires confirmation unless --yes. During the ramp also compute live issue
  tag accuracy because the true scenario is known.
- Measure watch overhead excluding injected load; it must stay under 3 percent of one core
  (use the corrected measurement method).
- If the model card has synthetic: true, show the "DEMO MODEL trained on synthetic data"
  banner on every watch view.

2. deskflow report [--out FILE.md]
- Generated only from artifacts (model card, metrics.json, predictions_test.parquet, live
  session summaries, fixlog). No hard-coded numbers anywhere.
- Sections: overview; table "ML concept, where used (file path), what it does, measured
  result"; dataset description (rows, sessions, machines, scenarios, split protocol);
  offline leaderboards per task with baselines as marked rows and confidence intervals;
  aggregate live accuracy over all watch sessions; fix effectiveness from fixlog records where
  dry_run is false only (count, median score before and after, memory and disk freed);
  limitations generated from facts (number of machines and sessions, synthetic flag, scenarios
  that did not slow the probe, anomaly false-alarm rate on healthy test rows).
- Wording rule: a model is called better than a baseline only when its confidence interval of
  the improvement excludes zero; ratings come from documented thresholds.
- Synthetic model: banner "SYNTHETIC DATA - NOT A RESULT" on the report.
- Plots (predicted versus actual, residuals, model comparison, confusion matrix, feature
  importance) when the train extra is installed; otherwise skipped with a note.

3. Demo script (docs/DEMO.md, about 5 minutes)
Exact commands, expected seconds per step, what to say, and a fallback if the live demo fails:
1) deskflow doctor 2) deskflow scan (healthy) 3) deskflow lab demo-app --memory-mb 3000
4) deskflow scan (shows the problem) 5) deskflow fix (closes the demo app) with the automatic
re-scan showing recovery 6) deskflow watch --demo cpu_pressure (live predictions versus actual
against the baselines) 7) deskflow report. Fallback: `deskflow scan --replay <fixture run>`.

4. Tests (concise)
Prediction and resolution bookkeeping with a fake clock (alignment tolerance, missing probe,
power state change), metric calculations, jsonl and summary files, report generation from a
synthetic model set including the banner, report ignoring dry-run fix records, view rendering
at widths 80 and 100.

5. Verify once and paste real output
Full suite, ruff, mypy; `deskflow watch --seconds 40`; `deskflow watch --demo cpu_pressure
--yes`; `deskflow report --out report.md`; measured watch overhead.

Acceptance criteria
- Every model appears against Persistence with resolved counts, live.
- Report builds from artifacts only and contains no hard-coded results.
- Demo steps in DEMO.md run exactly as written, with timings measured on this machine.
- No emojis; no print() outside CLI and ui modules; no platform checks outside the platform
  package.