# Deskflow terminal design

Deskflow uses dense, dark-terminal-friendly output. Colour carries status only: cyan for ML,
green for good health, yellow for warnings and recommendations, red for poor/error, magenta for
prediction tags, and grey for system facts. Rating words come only from documented thresholds.

Panels use uppercase bracketed titles, rounded outer borders, and square or horizontal tables.
Scan shows a five-field status strip, four KPI cards, factors/categories, recommendations, and a
timing footer. Fix shows actions and before/after deltas. Doctor uses a capability table.

Unicode block bars and sparklines are used only when UTF-8 output is available. `--plain`,
`NO_COLOR`, dumb terminals, and non-UTF-8 streams use ASCII bars and boxes with no colour.
At widths below 100, panel groups stack. The home screen alone may show the cyan Deskflow banner;
it becomes a plain title below 80 columns or in plain mode.

## 9. Five-command interface (v3)

Primary commands: `battery`, `plugged` (alias `ac`), `full`, `fix`, `pipeline`. Also `info`. Everything else is
under `deskflow tools` and hidden from the default help.

### HOME (loops until Q)
- Header banner, subtitle, version
- `[ THIS MACHINE ]`: OS | CPU | RAM | POWER (battery XX% or AC) | MODEL (real data, N rows | synthetic demo)
- `[1] Battery check     about 75 s   health score, prediction, model comparison`
- `[2] Plugged-in check  about 75 s   same, on AC power`
- `[3] Full scan         about 10 min stress test with live accuracy of every model`
- `[4] Fix Center        clean cache, free memory, close heavy tasks`
- `[5] How it works      animated pipeline: data, models, training`
- `[6] System info       CPU, GPU, memory, disks`
- `[Q] Quit`

### BATTERY / PLUGGED CHECK (steps panel, then results)
- **Preflight Steps Panel**: Power state match check, baseline age & drift validation, probe stability (CV < 15%).
- **Measuring**: 14s telemetry & probe baseline sample + live continuous prediction tracking.
- **Results Dashboard**:
  - Dense status strip
  - KPI row: Health (0-100), Next 20s slowdown ratio, Likely cause with confidence, Unusual anomaly score.
  - Split `[ WHY ]` top factors and `[ RESOURCES ]` telemetry sparklines.
  - `[ HEAVY TASKS ]` top non-protected background processes pass.
  - `[ ACTIONS ]` Fix Center inventory with measured potential.
  - `[ MODEL COMPARISON ]`:
    - **Held-Out Test**: (real models only) Model MAE, 95% bootstrap confidence interval, whisker bar.
    - **Truth Contract**: For synthetic models, suppresses all metric numbers and renders:
      `NO REAL DATA YET: this model was trained on synthetic data. Run deskflow full to measure real accuracy on this machine.`
    - **Live, This Check**: P prediction pairs evaluated live against real 20s future probe latency.
    - **Verdict sentence**: Rule-based verdict sentence indicating whether any model reliably beats Persistence.

