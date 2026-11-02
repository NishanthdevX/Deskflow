"""Pytest configuration and shared fixtures."""

import os
from collections.abc import Generator
from pathlib import Path

import pytest


@pytest.fixture(scope="session", autouse=True)
def isolate_deskflow_dirs(
    tmp_path_factory: pytest.TempPathFactory,
) -> Generator[tuple[Path, Path], None, None]:
    """Isolate DESKFLOW user data and configuration directories during testing."""
    test_data_dir = tmp_path_factory.mktemp("deskflow_test_data")
    test_config_dir = tmp_path_factory.mktemp("deskflow_test_config")

    old_data = os.environ.get("DESKFLOW_DATA_DIR")
    old_config = os.environ.get("DESKFLOW_CONFIG_DIR")

    os.environ["DESKFLOW_DATA_DIR"] = str(test_data_dir)
    os.environ["DESKFLOW_CONFIG_DIR"] = str(test_config_dir)

    yield test_data_dir, test_config_dir

    if old_data is not None:
        os.environ["DESKFLOW_DATA_DIR"] = old_data
    else:
        os.environ.pop("DESKFLOW_DATA_DIR", None)

    if old_config is not None:
        os.environ["DESKFLOW_CONFIG_DIR"] = old_config
    else:
        os.environ.pop("DESKFLOW_CONFIG_DIR", None)


@pytest.fixture
def mock_fix_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Setup mock baseline and models for fix testing."""
    import numpy as np
    import xgboost as xgb
    from sklearn.ensemble import IsolationForest
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    from deskflow.baseline import Baseline, save_baseline
    from deskflow.ml.artifacts import save_model_set
    from deskflow.ml.models import TrainedAnomalyDetector

    data_dir = tmp_path / "data"
    config_dir = tmp_path / "config"
    monkeypatch.setenv("DESKFLOW_DATA_DIR", str(data_dir))
    monkeypatch.setenv("DESKFLOW_CONFIG_DIR", str(config_dir))

    # Baseline
    baseline = Baseline(
        machine_id="test-fix-machine",
        created_utc="2026-11-01T00:00:00Z",
        platform="windows",
        cpu_ns=20_000_000,
        mem_ns=20_000_000,
        disk_ns=20_000_000,
        total_ns=60_000_000,
        p10_total_ns=55_000_000,
        p90_total_ns=65_000_000,
        sample_count=20,
        deskflow_version="0.1.0",
        cv=0.03,
        power_state="ac",
    )
    save_baseline(baseline, power_state="ac")
    save_baseline(baseline, power_state="battery")

    # Minimal models
    X_dummy = np.ones((50, 27))
    y_dummy_reg = np.ones(50) * 1.1
    y_dummy_clf = np.zeros(50, dtype=int)

    xgb_reg = xgb.XGBRegressor(n_estimators=2, max_depth=2, random_state=42)
    xgb_reg.fit(X_dummy, y_dummy_reg)

    xgb_clf = xgb.XGBClassifier(n_estimators=2, max_depth=2, random_state=42)
    xgb_clf.fit(X_dummy, y_dummy_clf)

    ridge = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=1.0, random_state=42)),
        ]
    )
    ridge.fit(X_dummy, y_dummy_reg)

    raw_if = IsolationForest(n_estimators=2, random_state=42).fit(X_dummy)
    anom = TrainedAnomalyDetector(raw_if, threshold=0.5, feature_names=["f"] * 27)

    save_model_set(
        regressors={"XGBoost": xgb_reg, "Ridge": ridge},
        classifiers={"XGBoost": xgb_clf},
        anomaly_detector=anom,
        classes=["healthy", "cpu_pressure", "memory_pressure", "disk_pressure", "background_load"],
        cv_metrics={
            "regression": {"XGBoost": 0.05, "Ridge": 0.06},
            "classification": {"XGBoost": 0.99},
        },
        test_metrics={},
        dataset_sha256="dummy-sha",
        synthetic=True,
    )

    return data_dir
