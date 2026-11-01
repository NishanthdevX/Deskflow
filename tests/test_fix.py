"""Tests for deskflow fix command, dry-run mode, confirmation, and log persistence."""

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from deskflow.baseline import Baseline, save_baseline
from deskflow.cli import app
from deskflow.ml.artifacts import save_model_set
from deskflow.ml.models import TrainedAnomalyDetector

runner = CliRunner()


@pytest.fixture
def mock_fix_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Setup mock baseline and models for fix testing."""
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
    import numpy as np
    import xgboost as xgb
    from sklearn.ensemble import IsolationForest
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

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


def test_fix_dry_run(mock_fix_environment: Path) -> None:
    """Test deskflow fix --dry-run prints plan and does not modify system."""
    res = runner.invoke(app, ["fix", "--action", "clean_temp", "--dry-run", "--yes"])
    assert res.exit_code == 0, f"Command failed: {res.output}"
    assert "Diagnosing current system state" in res.output
    assert "[Dry-run]" in res.output or "clean_temp" in res.output

    # Check that fixlog.jsonl was created with dry_run: true
    log_file = mock_fix_environment / "fixlog.jsonl"
    assert log_file.exists()
    lines = log_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["dry_run"] is True
    assert "clean_temp" in record["actions"]


def test_fix_execution_with_rescan(mock_fix_environment: Path) -> None:
    """Test deskflow fix with an action executes and performs re-scan."""
    res = runner.invoke(app, ["fix", "--action", "clean_temp", "--yes"])
    assert res.exit_code == 0, f"Command failed: {res.output}"
    assert "Executing actions" in res.output
    assert "Remediation Verification" in res.output
    assert "Health Score" in res.output

    # Check audit log
    log_file = mock_fix_environment / "fixlog.jsonl"
    assert log_file.exists()
    lines = log_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) >= 1
    record = json.loads(lines[-1])
    assert record["dry_run"] is False
    assert record["before"] is not None
    assert record["after"] is not None
