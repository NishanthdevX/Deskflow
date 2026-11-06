# Deskflow status

This is a code-derived status record for the current tree (Deskflow 0.2.0).

- Primary command surface is standardized to 7 root commands: `battery`, `plugged`, `full`, `pipeline`, `fix`, `info`, and `tools`. All diagnostic and laboratory utilities remain available directly and under `deskflow tools <command>`.
- Regression models are Ridge, RandomForest, and XGBoost. No LightGBM dependency or model is present.
- Regression baselines are ConstantMedian, Persistence, and RollingMean.
- Classifiers are LogisticRegression, RandomForest, and XGBoost; supported labels are healthy, cpu_pressure, memory_pressure, disk_pressure, and background_load.
- Anomaly detection is performed using IsolationForest on telemetry features.
- Feature windows are 6 and 12 seconds. `MIN_HISTORY_S` is 12 seconds and the prediction horizon is 20 seconds.
- Live scan defaults to 14 seconds. Model artifacts are XGBoost UBJ files, RandomForest joblib files, Ridge JSON parameters, classifiers, Isolation Forest, model card, metrics, and test predictions.
- Active bundled model is verified on real hardware: 1,957 rows across 6 sessions (ac, battery) on host AMD Ryzen 7 6800HS, commit `b6459476`, dataset `61af1146`. All gates passed.
- Truth contract guarantees: if models are synthetic, benchmarks are suppressed with a clear warning; if real, cryptographic SHA-256 verification confirms artifact integrity.
- Packaging provides bundled default models in wheel data, zero-dependency one-line installers (`install.ps1` and `install.sh`) with SHA-256 validation, and an automated GitHub Actions release workflow.
- Health score inputs are only predicted slowdown ratio. It is 100 at ratio <= 1, falls linearly to 0 at ratio >= 3, and is clipped to [0, 100].
