# deskflow

Command-line device-care tool for Windows 10/11 and macOS 12+.

## Description

Deskflow samples system telemetry, runs a short measured performance probe, and uses trained
ML models to predict how sluggish the machine will be 30 seconds from now. It tags the likely
cause (CPU pressure or throttling, memory pressure, disk pressure, background application load),
offers safe fixes, and re-measures after a fix.

It is also a college ML project: model training, offline evaluation, and live accuracy
comparison are first-class features.

Status: Early development. No functionality is available yet.

## Installation

Not yet available. Will be published to PyPI once Phase 8 is complete.

## Usage

Not yet available.

## How it works

1. The collector samples psutil telemetry on a short interval.
2. The feature pipeline builds rolling-window vectors from raw samples.
3. Regression models predict probe latency 30 seconds ahead.
4. A classifier tags the likely cause (cpu_pressure, memory_pressure, disk_pressure,
   background_load, or healthy).
5. An Isolation Forest flags anomalies not seen during healthy training.
6. A 0-100 health score is derived from predicted slowdown vs. this machine's own baseline.
7. Optional safe fixes are offered; each requires explicit confirmation and supports --dry-run.
8. Predictions are stored and scored against the probe measured at the predicted time.

## Limitations

- Windows 10/11 and macOS 12+ only. Linux is not supported.
- No malware scanning or security hardening. This tool observes and reports; it does not
  protect against threats.
- All data is stored locally. No network access occurs at runtime.
- Some metrics require administrator elevation on certain platforms. Those are opt-in and
  documented individually.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

MIT. See [LICENSE](LICENSE).
