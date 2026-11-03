# Deskflow status

This is a code-derived status record for the current tree.

- Regression models are Ridge, RandomForest, and XGBoost. No LightGBM dependency or model is present.
- Regression baselines are ConstantMedian, Persistence, and RollingMean. Classifiers are LogisticRegression, RandomForest, and XGBoost; supported labels are healthy, cpu_pressure, memory_pressure, disk_pressure, and background_load.
- Lab scenarios are cpu_pressure, memory_pressure, disk_pressure, background_load, idle, and plus-separated combinations. There is no lock scenario in the current lab command surface.
- Feature windows are 6 and 12 seconds. `MIN_HISTORY_S` is 12 seconds and the prediction horizon is 20 seconds.
- Live scan defaults to 14 seconds. Model artifacts are XGBoost UBJ files, RandomForest joblib files, Ridge JSON parameters, classifiers, Isolation Forest, model card, metrics, and test predictions when training extras are installed.
- Runtime explanations use the existing XGBoost contribution path where available; non-XGBoost models do not claim native contribution explanations.
- The CLI commands are train, evaluate, scan, fix, snapshot, collect, doctor, baseline, lab, dataset, and models.
- Health score inputs are only predicted slowdown ratio. It is 100 at ratio <= 1, falls linearly to 0 at ratio >= 3, and is clipped to the documented range.

Known historical discrepancies corrected by this phase: the prior implementation summary overstated completed work and referred to unsupported daemon/tray behavior. The canonical current status is this file.
