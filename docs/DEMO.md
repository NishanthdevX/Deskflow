# Deskflow 6-Minute Demonstration Script

This document provides a timed, step-by-step demonstration walkthrough for Deskflow. It is designed
for live presentations, technical interviews, and evaluations.

- **Total Duration**: Exactly 6 minutes (360 seconds).
- **Core Theme**: Offline predictive resource health, 20-second slowdown horizon forecasts, and
  measured, non-destructive remediation without synthetic numbers.
- **Hardware Profile**: AMD Ryzen 7 6800HS / Intel Core, Windows 10/11 or macOS 12+.

---

## Demonstration Timeline Overview

| Phase | Elapsed | Action / Command | Primary Focus |
| :--- | :--- | :--- | :--- |
| **Intro** | 0:00 - 0:30 | Setup and context | Offline, zero-cloud architecture; problem statement |
| **Step 1** | 0:30 - 1:30 | `deskflow pipeline --fast` | Architecture walkthrough: telemetry lane and training lane |
| **Step 2** | 1:30 - 3:00 | `deskflow plugged` | Hardware preflight, baseline drift, 20s horizon forecast |
| **Step 3** | 3:00 - 4:30 | `deskflow fix` | Safe remediation, demo app teardown, re-scan verification |
| **Step 4** | 4:30 - 5:30 | `deskflow report` | Full scan stress test review, bootstrap 95% CIs |
| **Wrap-up**| 5:30 - 6:00 | Summary & Q&A | Statistical superiority criteria, one-line installer |

---

## Detailed Step-by-Step Script

### Introduction (0:00 - 0:30, Duration: 30s)

**Objective**: Establish context, offline guarantee, and transparency.

**What to Say**:
> "Traditional PC cleanup utilities rely on aggressive registry cleaning or background daemons
> that collect telemetry and consume resources. Deskflow is an offline, research-grade resource
> health tool built in Python. It runs without administrator privileges, makes zero network
> connections, and uses lightweight machine learning models to forecast slowdowns 20 seconds
> before they impact user experience."

---

### Step 1: Pipeline Architecture Walkthrough (0:30 - 1:30, Duration: 60s)

**Command**:
```bash
uv run deskflow pipeline --fast
```

**Measured Duration**: ~12 seconds animation, 48 seconds explanation.

**What to Say**:
> "Let us look at how Deskflow processes system signals in real time.
>
> On the upper lane, Deskflow samples OS telemetry every 2 seconds—tracking CPU core saturation,
> memory bandwidth, context switch velocity, and disk queue depth. Every 4 seconds, it runs a
> micro-benchmark probe directly against host hardware to measure physical CPU, RAM, and disk
> execution latency. These signals are fed into a 27-feature engineering window.
>
> Two models run in parallel: a regressor predicting future slowdown ratio, and a multi-class
> classifier isolating the root cause—whether CPU contention, memory exhaustion, or disk thrashing.
>
> On the lower lane, you see the offline training pipeline: trained using grouped cross-validation
> by session, evaluated against a Persistence baseline using moving block bootstrap intervals, and
> exported with cryptographic SHA-256 provenance."

**Expected Output**:
- Rich 8 fps dual-lane animation rendering live telemetry metrics and model state.
- `[ MODELS USED ]` and `[ HOW IT IS TRAINED ]` structured specification tables.

**Fallback Plan**:
If presenting in a terminal that does not support ANSI 8 fps redraws or non-interactive recording:
```bash
uv run deskflow pipeline --static
```
or
```bash
uv run deskflow pipeline --plain
```

---

### Step 2: Live Hardware Health Assessment (1:30 - 3:00, Duration: 90s)

**Command**:
```bash
uv run deskflow plugged
```
*(If presenting on battery power without AC adapter connected, run `uv run deskflow battery`)*.

**Measured Duration**: 75 seconds execution.

**What to Say**:
> "Now let us evaluate live hardware health. Notice that before touching the ML models, Deskflow
> runs hardware preflight checks:
> 1. It confirms the power state matches our calibrated baseline.
> 2. It checks baseline freshness and drift—here our AC baseline was calibrated recently.
> 3. It runs a 6-probe stability test to guarantee timer accuracy.
>
> Over the next 75 seconds, Deskflow continuously measures probe response times against our host's
> calibrated physical baseline of 12 milliseconds. It projects our performance ratio 20 seconds into
> the future.
>
> Notice the health score gauge: a deterministic 0-100 score derived from predicted slowdown.
> Below it, category audits breakdown Performance, Memory, Storage, and Background load."

**Expected Output**:
- Hardware preflight status: Power State (AC), Baseline Freshness (PASS), Stability Ratio (PASS).
- Live 75-second evaluation loop with real-time countdown.
- Final Health Score summary panel with category breakdown and recommended remediations.

**Fallback Plan**:
If power state mismatches (e.g., unplugged during `deskflow plugged`):
The preflight will abort with exit code `5`. Explain:
> "Deskflow enforces strict power-state isolation because CPU power governor profiles on battery
> differ substantially from AC power. We switch to `deskflow battery`."
Then execute:
```bash
uv run deskflow battery
```

---

### Step 3: Non-Destructive Remediation with Fix Center (3:00 - 4:30, Duration: 90s)

**Setup Command** (run in background terminal beforehand or inline):
```bash
uv run deskflow lab demo-app
```
*(Demo app opens, allocates memory and generates simulated background thread contention)*.

**Command**:
```bash
uv run deskflow fix
```

**Measured Duration**: ~45 seconds.

**What to Say**:
> "Deskflow's Fix Center addresses performance degradation safely. Unlike commercial cleaners
> that claim gigabytes of phantom disk savings, Deskflow strictly distinguishes between real
> reclaimable file bytes—such as temp caches and user Recycle Bin—and application working sets.
>
> Notice our inventory table: safe tools like `clean_temp` or `close_idle_apps` are marked SAFE,
> while terminating active tasks requires typed confirmation.
>
> Let us remediate the background memory pressure. We choose option 2 (`close_idle_apps`).
> Notice what happens next: Deskflow gracefully terminates the idle processes, measures the exact
> bytes released, and immediately triggers an automatic verification re-scan.
>
> The delta table confirms the remediation: available memory increased by +0.52 GB, and probe
> latency dropped back to baseline. The entire audit trail is logged locally to `fixlog.jsonl`."

**Expected Output**:
- Fix inventory table listing registered remediations with Measured Potential.
- Confirmation prompt.
- Remediation execution report with bytes freed.
- Post-fix verification re-scan comparison table (Before vs After vs Delta).

**Fallback Plan**:
If unwilling to terminate live processes during an interactive session, run fix inventory in
read-only inspection mode:
```bash
uv run deskflow fix --list
```
This prints the entire diagnostic inventory without altering system state.

---

### Step 4: Verification & Statistical Superiority (4:30 - 5:30, Duration: 60s)

**Command**:
```bash
uv run deskflow report
```

**Measured Duration**: ~30 seconds.

**What to Say**:
> "How do we prove our models actually work and beat simple heuristics?
>
> Deskflow provides `deskflow full`—a 10-minute stress test running 5 stages: idle, CPU ramp,
> memory ramp, disk saturation, and background load. Rather than waiting 10 minutes right now,
> `deskflow report` loads the verified outcomes of our latest full scan in Section 12.
>
> Deskflow enforces the **Statistical Superiority Criterion**: a model is certified as better
> than Persistence only when the 95% moving block bootstrap confidence interval of its error
> reduction strictly excludes zero.
>
> Looking at our held-out test leaderboard: XGBoost achieved a test MAE of 0.0008, and its
> improvement over Persistence has a 95% confidence interval of `[+0.0223, +0.0256]`. Because zero
> is strictly excluded, the model is statistically certified as superior on real hardware."

**Expected Output**:
- 12-section terminal verification report:
  - Section 1: Executive Summary & Health Gauge
  - Section 2-8: Diagnostic breakdowns, confusion matrices, SHAP feature rankings
  - Section 11: Model Provenance & Quality Gates (synthetic: false, commit hash, dataset hash)
  - Section 12: Full Scan 5-stage stress test telemetry and block bootstrap intervals

**Fallback Plan**:
If visual graphics or HTML export are requested:
```bash
uv run deskflow report --html report.html
```
Open `report.html` in default browser to showcase self-contained SVG charts (<500 KB, 0 external CDN).

---

### Wrap-Up & Conclusion (5:30 - 6:00, Duration: 30s)

**What to Say**:
> "To summarize:
> 1. Deskflow runs 100% offline with zero cloud telemetry and zero administrator privileges.
> 2. Predictions look 20 seconds into the future to alert users before physical lag spikes occur.
> 3. Remediations are verified with immediate before/after re-scans and audit logging.
> 4. All accuracy claims are statistically validated with 95% bootstrap intervals.
>
> Installation requires a single zero-dependency command on Windows (`irm ... | iex`) or
> macOS (`curl ... | sh`), or direct execution via `uvx deskflow`."

---

## Presenter Preparation Checklist

Before starting the live demonstration:
1. Ensure laptop is plugged in (for `deskflow plugged`) or unplugged (for `deskflow battery`).
2. Calibrate hardware baseline if older than 7 days: `uv run deskflow prep --calibrate`.
3. Verify test suite and self-test pass cleanly: `uv run deskflow tools selftest --plain`.
4. Maximize terminal window to at least 100 columns by 30 rows.

