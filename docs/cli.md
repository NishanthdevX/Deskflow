# Deskflow CLI Reference and Schemas

This document describes the CLI commands introduced in Deskflow and the structured JSON output schema for automated scripting and integration.

## Primary Commands (v3 Interface)

The default `deskflow --help` presents 7 primary commands tailored for immediate device-care and live accuracy verification:

### `deskflow battery`

Executes device-care health scan and live accuracy verification on battery power.

```bash
deskflow battery [OPTIONS]
```

#### Preflight Checks
1. Power State Verification: Verifies machine is running on battery power; aborts with exit code `5` if plugged in.
2. Baseline & Drift: Validates battery baseline exists, is under 7 days old, and within drift threshold (< 20%); prompts for 40s calibration if needed.
3. Probe Stability: Measures 6 micro-probes and verifies CV < 15%.

#### Options
- `--quick`: Fast ~25s health scan without waiting for live prediction horizons.
- `--long`: Extended ~135s check with ~26 resolved prediction pairs.
- `--fail-on <fair|poor>`: Exit 1 if health score reaches the specified threshold.
- `--yes`, `-y`: Automatically confirm baseline calibration without prompting.
- `--json`: Output full structured JSON results.
- `--plain`: Render pure ASCII without ANSI colors.

---

### `deskflow plugged` (alias `deskflow ac`)

Executes device-care health scan and live accuracy verification on AC wall power.

```bash
deskflow plugged [OPTIONS]
# or:
deskflow ac [OPTIONS]
```

#### Preflight Checks
1. Power State Verification: Verifies AC power is connected; aborts with exit code `5` if running on battery.
2. Baseline & Drift: Validates AC baseline exists, is under 7 days old, and drift is within threshold.
3. Probe Stability: Measures 6 micro-probes and checks stability.

Shares identical options (`--quick`, `--long`, `--fail-on`, `--yes`, `--json`, `--plain`) with `battery`.

---

### `deskflow full`

Full 10-minute stress test with live accuracy evaluation across every model (stub for Phase 15 release).

---

### `deskflow pipeline`

Animated visual data, model, and inference pipeline running live on current telemetry (stub for Phase 15 release).

---

### `deskflow fix`

Executes safe, user-confirmed remediation actions and measures real-world impact with a post-fix re-scan.

```bash
deskflow fix [OPTIONS]
```

---

### `deskflow info`

Fast hardware and operating system summary including CPU, GPU, memory, and disks.

---

### `deskflow tools`

Advanced diagnostic, laboratory, and developer tools. Invoking `deskflow tools` prints the tools catalog with one-line descriptions.

```bash
deskflow tools [COMMAND]
```

#### Key Subcommand: `deskflow tools verify`
Cryptographically verifies all registered model files against SHA-256 hashes in `model_card.json`.

```bash
deskflow tools verify [--plain]
```

---

## Exit Codes

| Code | Name | Description |
|---|---|---|
| `0` | `EXIT_OK` | Command completed successfully. |
| `1` | `EXIT_ERROR` | Runtime error, hardware check failure, or `--fail-on` condition met. |
| `2` | `EXIT_USAGE` | Invalid command line arguments, flags, or durations. |
| `3` | `EXIT_UNSUPPORTED` | Operating system or architecture is not supported. |
| `4` | `EXIT_BASELINE_MISSING` | Required baseline for target power state has not been calibrated. |
| `5` | `EXIT_BLOCKED_OR_DECLINED` | Action declined by user, blocked by safety policy, or power state mismatch. |

> Note: Health status is never encoded in exit codes unless explicitly requested via `--fail-on fair` or `--fail-on poor`.

---

## Compatibility / Advanced Tools (hidden from default `--help`)

All legacy commands remain fully functional and callable directly or via `deskflow tools <cmd>`:

- `deskflow scan`: Instantaneous 14s health scan and model prediction.
- `deskflow watch`: Live prediction accuracy horizon evaluation.
- `deskflow check`: Guided 5-step interactive health evaluation.
- `deskflow report`: Generate verification report across terminal, Markdown, HTML, and JSON.
- `deskflow prep`: Hardware readiness checklist and calibration.
- `deskflow baseline`: Manage and calibrate baseline profiles.
- `deskflow collect`: Record background telemetry datasets.
- `deskflow dataset`: Build and validate parquet training datasets.
- `deskflow lab`: Controlled fault injection suite.
- `deskflow models`: Inspect serialized model registry and metadata.
- `deskflow doctor`: Platform capabilities and environment health.
- `deskflow snapshot`: Instantaneous JSON telemetry capture.
- `deskflow bugreport`: Redacted diagnostic report generator.
- `deskflow privacy`: Privacy commitments and data directory paths.
- `deskflow data`: Local disk space utilization inspector.
- `deskflow uninstall`: Purge local data directories.

#### Options

- `--seconds`, `-s` `INTEGER`: Duration in seconds to measure live telemetry and probes (default: 14, minimum: 13).
- `--json`: Output raw structured JSON schema instead of human-readable tables.
- `--no-storage`: Skip filesystem storage scan stage.
- `--timing`: Print stage latency breakdown (measuring, storage, model load, inference, total).
- `--replay` `PATH`: Replay a previously recorded run JSONL file instead of live measuring.
- `--yes`, `-y`: Automatically confirm any required baseline calibration.

#### JSON Output Schema (`deskflow scan --json`)

```json
{
  "machine_id": "string",
  "power_state": "ac" | "battery",
  "scan_duration_s": 14.52,
  "synthetic_model": false,
  "health": {
    "score": 85,
    "label": "Good" | "Fair" | "Poor",
    "predicted_slowdown_ratio": 1.15,
    "horizon_s": 20.0
  },
  "issue": {
    "tag": "healthy" | "cpu_pressure" | "memory_pressure" | "disk_pressure" | "background_load" | "none",
    "confidence": 0.95,
    "anomaly": false,
    "anomaly_score": 0.42
  },
  "reasons": [
    "High average CPU load over 12s (+0.25)",
    "Elevated probe total ratio (+0.12)"
  ],
  "categories": {
    "performance": {
      "status": "Good" | "Fair" | "Poor",
      "score": 85,
      "is_ml": true
    },
    "memory": {
      "status": "Good" | "Fair" | "Poor",
      "available_gb": 8.4,
      "available_percent": 52.5,
      "is_ml": false
    },
    "storage": {
      "status": "Good" | "Fair" | "Poor",
      "cleanable_gb": 1.2,
      "is_ml": false
    },
    "background_apps": {
      "status": "Good" | "Fair" | "Poor",
      "idle_count": 4,
      "idle_rss_gb": 1.5,
      "is_ml": false
    }
  },
  "recommendations": [
    {
      "action_id": "clean_temp" | "close_idle_apps" | "end_busy_process" | "trim_memory",
      "title": "string",
      "description": "string",
      "evidence": "string",
      "warning": "string | null"
    }
  ],
  "timings": {
    "measuring_s": 14.02,
    "storage_s": 0.45,
    "model_load_s": 0.08,
    "inference_s": 0.0215,
    "total_s": 14.57
  }
}
```

### `deskflow fix`

Executes safe, user-confirmed remediation actions and measures real-world impact with a post-fix re-scan.

```bash
deskflow fix [OPTIONS]
```

#### Options

- `--action`, `-a` `TEXT`: One or more specific action IDs to execute directly (`clean_temp`, `close_idle_apps`, `end_busy_process`, `trim_memory`).
- `--dry-run`: Simulate actions and print expected impact without making changes.
- `--force`: Allow terminating processes if graceful closing times out (requires explicit second confirmation).
- `--yes`, `-y`: Bypass interactive selection and confirmation prompts.
- `--no-rescan`: Skip post-fix verification re-scan.

#### Audit Log (`fixlog.jsonl`)

Every execution of `deskflow fix` appends a record to `<user_data_dir>/fixlog.jsonl`.

- **Non-dry-run records (`"dry_run": false`)**: Contains actual measured `bytes_freed` and the `after` verification state block.
- **Dry-run records (`"dry_run": true`)**: Estimated impact goes into `estimated_bytes_freed` and `estimated_details`. `bytes_freed` and `after` are strictly `null` (never storing unmeasured estimates as actuals).

##### Non-Dry-Run Schema Example
```json
{
  "ts_utc": "2026-11-01T12:00:00Z",
  "actions": ["clean_temp"],
  "dry_run": false,
  "results": [
    {
      "action": "clean_temp",
      "success": true,
      "message": "Deleted 142 files (145.2 MB) older than 24 hours.",
      "bytes_freed": 152254464
    }
  ],
  "before": {
    "score": 75,
    "predicted_ratio": 1.50,
    "probe_ms": 78.4,
    "available_memory_bytes": 4294967296,
    "temp_bytes": 152254464
  },
  "after": {
    "score": 85,
    "predicted_ratio": 1.15,
    "probe_ms": 62.1,
    "available_memory_bytes": 4447221760,
    "temp_bytes": 0
  }
}
```

##### Dry-Run Schema Example
```json
{
  "ts_utc": "2026-11-01T12:00:00Z",
  "actions": ["close_idle_apps"],
  "dry_run": true,
  "results": [
    {
      "action": "close_idle_apps",
      "success": true,
      "message": "[Dry-run] Would close 2 idle processes across 1 app groups holding 620.0 MB.",
      "bytes_freed": null,
      "estimated_bytes_freed": 650117120,
      "estimated_details": {
        "spotify.exe": {
          "count": 2,
          "rss_bytes": 650117120,
          "pids": [1234, 1235]
        }
      }
    }
  ],
  "before": {
    "score": 75,
    "predicted_ratio": 1.50,
    "probe_ms": 78.4,
    "available_memory_bytes": 4294967296,
    "temp_bytes": 152254464
  },
  "after": null
}
```

---

## Standard Process Exit Codes

Deskflow standardizes process exit codes across all commands:

| Code | Identifier | Description |
| :---: | :--- | :--- |
| `0` | `EXIT_OK` | Command completed successfully with no critical errors. |
| `1` | `EXIT_ERROR` | Runtime error, unexpected exception, or internal fault. |
| `2` | `EXIT_USAGE` | Invalid CLI options, mutually exclusive arguments, or syntax error. |
| `3` | `EXIT_UNSUPPORTED_PLATFORM` | Operating system or architecture unsupported (e.g. Linux). |
| `4` | `EXIT_BASELINE_MISSING` | Required baseline for current power state has not been calibrated. |
| `5` | `EXIT_BLOCKED_OR_DECLINED` | Action blocked by safety policy or declined by user. |

---

## Global Options

- `--plain`: Emit monochrome ASCII-only output. Honored alongside the `NO_COLOR` environment variable.
- `--no-input`: Never prompt for user interaction; exit immediately with code `5` or continue non-interactively where safe.
- `--verbose`, `-v`: Enable verbose diagnostic logging to `<data_dir>/logs/deskflow.log`.
- `--debug`: Print complete exception tracebacks directly to standard error.

---

## Additional Commands

### `deskflow check`
Guided 5-step health evaluation: Info -> Scan -> Fix Center -> Live Watch -> Summary Report with HTML export.

### `deskflow report`
Generates comprehensive verification report across terminal, Markdown, pure JSON (`--json`), and publication-ready HTML (`--html FILE`).

### `deskflow info`
High-density hardware and operating system specification dashboard.

### `deskflow bugreport`
Generates a sanitized local Markdown diagnostic report with usernames, home paths, and machine GUIDs redacted. Never uploads anything.

### `deskflow privacy`
Displays Deskflow offline privacy commitments and local data storage locations.

### `deskflow data`
Inspects local storage usage across baselines, telemetry, models, and audit logs.

### `deskflow uninstall --data`
Purges local Deskflow configuration, baselines, and logs after typed confirmation.


