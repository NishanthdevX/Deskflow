"""Tests for deskflow scan command, --replay mode, and JSON output schema."""

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from deskflow.baseline import Baseline, save_baseline
from deskflow.cli import app
from deskflow.collector.models import (
    ProbeResult,
    ProcessSample,
    Record,
    SystemSample,
)
from deskflow.ml.artifacts import save_model_set
from deskflow.ml.models import TrainedAnomalyDetector

runner = CliRunner()


@pytest.fixture
def mock_scan_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Setup mock baseline, trained model set, and synthetic run JSONL."""
    monkeypatch.setenv("DESKFLOW_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("DESKFLOW_CONFIG_DIR", str(tmp_path / "config"))

    # 1. Baseline
    baseline = Baseline(
        machine_id="test-scan-machine",
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

    # 2. Minimal trained models for RuntimeModels to load
    import numpy as np
    import xgboost as xgb
    from sklearn.ensemble import IsolationForest
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    X_dummy = np.ones((50, 27))
    y_dummy_reg = np.ones(50) * 1.2
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

    # Anomaly detector
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

    # 3. Create synthetic run JSONL for --replay
    run_file = tmp_path / "test_run.jsonl"
    with open(run_file, "w", encoding="utf-8") as f:
        for i in range(10):
            t = float(i * 2)
            has_probe = i % 2 == 0
            sample = SystemSample(
                ts_utc=f"2026-11-01T12:00:{int(t):02d}Z",
                monotonic_s=t,
                cpu_percent_mean=20.0,
                cpu_percent_max_core=25.0,
                cpu_freq_current_mhz=2500.0,
                cpu_freq_max_mhz=3000.0,
                cpu_freq_reliable=True,
                ctx_switches_per_s=1000.0,
                interrupts_per_s=500.0,
                mem_total_bytes=16 * (1024**3),
                mem_available_bytes=8 * (1024**3),
                mem_percent=50.0,
                swap_used_bytes=1024**2,
                swap_percent=1.0,
                disk_read_bytes_per_s=1024.0,
                disk_write_bytes_per_s=2048.0,
                disk_busy_ms_per_s=10.0,
                net_sent_bytes_per_s=100.0,
                net_recv_bytes_per_s=200.0,
                net_errors_per_s=0.0,
                net_drops_per_s=0.0,
                process_count=150,
                thread_count=1200,
                battery_percent=100.0,
                battery_plugged=True,
                max_temperature_c=45.0,
                uptime_s=5000.0,
            )
            probe = (
                ProbeResult(
                    ts_utc=f"2026-11-01T12:00:{int(t):02d}Z",
                    cpu_ns=20_000_000,
                    mem_ns=20_000_000,
                    disk_ns=20_000_000,
                    total_ns=60_000_000,
                )
                if has_probe
                else None
            )
            rec = Record(
                schema_version=1,
                run_id="run-replay",
                label="idle",
                sample=sample,
                probe=probe,
                top_processes=[
                    ProcessSample(
                        pid=200,
                        name="test_worker.exe",
                        rss_bytes=400 * 1024 * 1024,
                        cpu_percent=0.2,
                        num_threads=4,
                        age_s=800.0,
                        is_protected=False,
                    )
                ],
                storage=None,
            )
            f.write(json.dumps(rec.to_dict()) + "\n")

    return run_file


def test_scan_replay_human_output(mock_scan_environment: Path) -> None:
    """Test deskflow scan --replay produces expected sections and output."""
    res = runner.invoke(app, ["scan", "--replay", str(mock_scan_environment), "--no-storage"])
    assert res.exit_code == 0, f"Command failed: {res.output}"
    assert "DESKFLOW SCAN" in res.output
    assert "HEALTH" in res.output
    assert "Performance" in res.output
    assert "Memory" in res.output
    assert "Background apps" in res.output


def test_scan_replay_json_schema(mock_scan_environment: Path) -> None:
    """Test deskflow scan --json validates against required JSON schema structure."""
    res = runner.invoke(
        app, ["scan", "--replay", str(mock_scan_environment), "--json", "--no-storage"]
    )
    assert res.exit_code == 0, f"Command failed: {res.output}"
    data = json.loads(res.output)

    # Validate top-level schema fields
    assert "machine_id" in data
    assert "power_state" in data
    assert "health" in data
    assert "score" in data["health"]
    assert "label" in data["health"]
    assert "predicted_slowdown_ratio" in data["health"]

    assert "issue" in data
    assert "tag" in data["issue"]
    assert "confidence" in data["issue"]
    assert "anomaly" in data["issue"]

    assert "categories" in data
    assert "performance" in data["categories"]
    assert "memory" in data["categories"]
    assert "storage" in data["categories"]
    assert "background_apps" in data["categories"]

    assert "recommendations" in data
    assert "timings" in data
    assert isinstance(data["recommendations"], list)
