# Architecture

## Module map

```
src/deskflow/
    __init__.py          Package version (read from installed metadata).
    cli.py               Typer application; root entry point for the deskflow command.
    py.typed             PEP 561 marker.

    platform/            OS-specific code behind a common abstract interface.
        __init__.py      Public platform API. Callers import only from here.
        base.py          Abstract base class defining the platform contract. (Phase 1)
        windows.py       Windows implementation using ctypes and winreg. (Phase 1)
        macos.py         macOS implementation using IOKit and sysctl. (Phase 1)

    collector/           Telemetry sampling.
        __init__.py
        sampler.py       Calls platform/* and psutil; returns a TelemetrySample. (Phase 1)
        probe.py         Fixed CPU/memory/disk micro-workloads timed with
                         perf_counter_ns. Ground truth for ML training. (Phase 1)

    features/            Feature engineering.
        __init__.py
        window.py        Rolling-window aggregation over TelemetrySample sequences. (Phase 2)
        schema.py        Typed FeatureVector definition. (Phase 2)

    ml/                  Machine-learning models.
        __init__.py
        train.py         Time-based train/test split; fits all models. (Phase 3)
        evaluate.py      MAE, RMSE, R2 for regression; P/R/F1 for classification. (Phase 3)
        infer.py         Loads saved models; returns predictions + SHAP attributions. (Phase 4)
        models.py        Model definitions (Linear, Ridge, RF, GB, XGBoost, IF). (Phase 3)

    health/              Health score.
        __init__.py
        score.py         Derives 0-100 score from predicted vs. baseline probe latency. (Phase 4)

    remediation/         Safe system fixes.
        __init__.py
        actions.py       Fix implementations, each requiring explicit confirmation. (Phase 6)
        safety.py        Allowlist checks; symlink guard. (Phase 6)

    evaluation/          Live accuracy tracking.
        __init__.py
        tracker.py       Stores predictions; scores them on ground-truth probe arrival. (Phase 7)
        report.py        Side-by-side model accuracy table. (Phase 7)
```

## Data flow (high level)

```
Collector (psutil + platform) --> TelemetrySample
TelemetrySample (rolling window) --> FeatureVector
FeatureVector --> Regression model --> predicted probe latency (t+30s)
FeatureVector --> Classifier --> pressure label
FeatureVector --> IsolationForest --> anomaly flag
predicted latency + baseline --> Health score (0-100)
Health score + label --> CLI / dashboard display
stored (prediction, timestamp) --> [30 s later] ground-truth probe --> live accuracy
```

## Design decisions

See `docs/decisions/` for Architecture Decision Records.
