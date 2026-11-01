"""Unit and integration tests for Phase 3 machine learning models, artifacts, and runtime."""

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from deskflow.features.registry import FEATURE_NAMES
from deskflow.ml.artifacts import (
    compute_directory_size_bytes,
    save_model_set,
)
from deskflow.ml.explain import compute_prediction_contributions, top_contributors
from deskflow.ml.metrics import (
    compute_classification_metrics,
)
from deskflow.ml.models import (
    VALID_ISSUE_CLASSES,
    train_anomaly_detector,
    train_classifiers,
    train_regressors,
    validate_training_dataset,
)
from deskflow.ml.runtime import RuntimeModels

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.make_synthetic_dataset import generate_synthetic_dataset


def test_synthetic_end_to_end_and_performance(tmp_path: Path) -> None:
    """Verify synthetic end-to-end training runs in <40s, beats persistence, and macro F1 > 0.9."""
    syn_parquet = tmp_path / "synthetic.parquet"
    t_start = time.perf_counter()
    generate_synthetic_dataset(out_path=syn_parquet, n_rows=1200, seed=42)

    df = pd.read_parquet(syn_parquet)
    df_train = df[df["split"] == "train"].reset_index(drop=True)
    df_test = df[df["split"] == "test"].reset_index(drop=True)

    X_train = df_train[FEATURE_NAMES].to_numpy()
    y_train_reg = df_train["target_total_ratio"].to_numpy()
    groups_train = df_train["run_id"].to_numpy()

    X_test = df_test[FEATURE_NAMES].to_numpy()
    y_test_reg = df_test["target_total_ratio"].to_numpy()

    # Train regressors with quick=True
    reg_res = train_regressors(X_train, y_train_reg, groups_train, quick=True)
    reg_models = reg_res["models"]

    # Train classifiers with quick=True
    train_clf_mask = df_train["issue_now"].isin(VALID_ISSUE_CLASSES).to_numpy()
    X_train_clf = X_train[train_clf_mask]
    y_train_clf = df_train.loc[train_clf_mask, "issue_now"].to_numpy()
    groups_train_clf = groups_train[train_clf_mask]

    clf_res = train_classifiers(
        X_train_clf,
        y_train_clf,
        groups_train_clf,
        classes=VALID_ISSUE_CLASSES,
        quick=True,
    )
    clf_models = clf_res["models"]

    # Train anomaly
    healthy_mask = (df_train["issue_now"] == "healthy").to_numpy()
    anom_model = train_anomaly_detector(X_train[healthy_mask], feature_names=FEATURE_NAMES)
    anom_scores = anom_model.score(X_test)
    assert len(anom_scores) == len(X_test)

    total_time = time.perf_counter() - t_start
    assert total_time < 40.0, f"Training took {total_time:.2f}s, expected < 40s"

    # 1. Regressors must beat Persistence on test split
    pers_preds = reg_models["Persistence"].predict(X_test)
    pers_mae = float(np.mean(np.abs(y_test_reg - pers_preds)))

    for m_name in ["Ridge", "RandomForest", "XGBoost"]:
        p = reg_models[m_name].predict(X_test)
        m_mae = float(np.mean(np.abs(y_test_reg - p)))
        assert m_mae < pers_mae, (
            f"{m_name} MAE ({m_mae:.4f}) failed to beat Persistence ({pers_mae:.4f})"
        )

    # 2. Classification macro F1 > 0.9 on separable classes
    test_clf_mask = df_test["issue_now"].isin(VALID_ISSUE_CLASSES).to_numpy()
    X_test_clf = X_test[test_clf_mask]
    y_test_clf = df_test.loc[test_clf_mask, "issue_now"].to_numpy()

    for m_name in ["LogisticRegression", "RandomForest", "XGBoost"]:
        p_raw = clf_models[m_name].predict(X_test_clf)
        p_str = (
            np.array([VALID_ISSUE_CLASSES[i] for i in p_raw])
            if np.issubdtype(p_raw.dtype, np.integer)
            else p_raw
        )
        metrics = compute_classification_metrics(y_test_clf, p_str, classes=VALID_ISSUE_CLASSES)
        assert metrics["macro_f1"] > 0.9, f"{m_name} macro F1 {metrics['macro_f1']} <= 0.9"


def test_no_test_leakage_in_preprocessing() -> None:
    """Verify that preprocessing scaler and imputer statistics are computed strictly from train."""
    rng = np.random.RandomState(42)
    X_train = rng.normal(loc=10.0, scale=2.0, size=(300, 27))
    y_train = rng.uniform(1.0, 3.0, size=300)
    groups_train = np.array([f"run_{i % 5}" for i in range(300)])

    # Test data has a drastically different distribution (mean 500.0)
    X_test = rng.normal(loc=500.0, scale=10.0, size=(100, 27))

    reg_res = train_regressors(X_train, y_train, groups_train, quick=True)
    ridge_pipe = reg_res["models"]["Ridge"]
    test_preds = ridge_pipe.predict(X_test)
    assert len(test_preds) == 100

    scaler = ridge_pipe.named_steps["scaler"]
    # Scaler mean must match X_train mean within numerical tolerance, untouched by X_test
    np.testing.assert_allclose(scaler.mean_, np.mean(X_train, axis=0), rtol=1e-5)


def test_group_kfold_no_overlap() -> None:
    """Verify that GroupKFold cross-validation groups never overlap between train and validation."""
    from sklearn.model_selection import GroupKFold

    groups = np.array(["run_A", "run_A", "run_B", "run_B", "run_C", "run_C", "run_D", "run_D"])
    X = np.ones((8, 2))
    y = np.ones(8)

    gkf = GroupKFold(n_splits=4)
    for tr_idx, val_idx in gkf.split(X, y, groups=groups):
        tr_groups = set(groups[tr_idx])
        val_groups = set(groups[val_idx])
        assert tr_groups.isdisjoint(val_groups)


def test_training_refusals_and_warnings() -> None:
    """Verify dataset validation raises clear errors on small data and warns on single machine."""
    # 1. Fewer than 200 rows raises ValueError
    small_df = pd.DataFrame(
        {
            "issue_now": ["healthy"] * 100,
            "run_id": ["r1"] * 100,
            "machine_id": ["m1"] * 100,
        }
    )
    with pytest.raises(ValueError, match="minimum 200 rows required"):
        validate_training_dataset(small_df)

    # 2. Class with fewer than 30 rows raises ValueError
    rows = []
    for cls in VALID_ISSUE_CLASSES:
        cnt = 10 if cls == "background_load" else 50
        rows.extend([{"issue_now": cls, "run_id": "r1", "machine_id": "m1"}] * cnt)
    df_imbalanced = pd.DataFrame(rows)
    with pytest.raises(ValueError, match="Class 'background_load' has only 10 training rows"):
        validate_training_dataset(df_imbalanced)

    # 3. Single machine produces warning
    valid_rows = []
    for cls in VALID_ISSUE_CLASSES:
        valid_rows.extend([{"issue_now": cls, "run_id": "r1", "machine_id": "single-box"}] * 45)
    df_valid_single_m = pd.DataFrame(valid_rows)
    warnings = validate_training_dataset(df_valid_single_m)
    assert any("only 1 machine_id" in w for w in warnings)


def test_reproducibility(tmp_path: Path) -> None:
    """Verify that training twice with fixed seed 42 produces identical test predictions."""
    syn_parquet = tmp_path / "synthetic_repro.parquet"
    generate_synthetic_dataset(out_path=syn_parquet, n_rows=1200, seed=42)
    df = pd.read_parquet(syn_parquet)

    df_train = df[df["split"] == "train"]
    df_test = df[df["split"] == "test"]

    X_train = df_train[FEATURE_NAMES].to_numpy()
    y_train = df_train["target_total_ratio"].to_numpy()
    groups = df_train["run_id"].to_numpy()
    X_test = df_test[FEATURE_NAMES].to_numpy()

    # Fit 1
    res1 = train_regressors(X_train, y_train, groups, quick=True)
    preds1 = res1["models"]["XGBoost"].predict(X_test)

    # Fit 2
    res2 = train_regressors(X_train, y_train, groups, quick=True)
    preds2 = res2["models"]["XGBoost"].predict(X_test)

    np.testing.assert_array_equal(preds1, preds2)


def test_artifact_round_trip_and_size(tmp_path: Path) -> None:
    """Verify saving models, verifying folder size < 10MB, and loading runtime models."""
    syn_parquet = tmp_path / "synthetic_artifacts.parquet"
    generate_synthetic_dataset(out_path=syn_parquet, n_rows=1200, seed=42)
    df = pd.read_parquet(syn_parquet)

    df_train = df[df["split"] == "train"]
    df_test = df[df["split"] == "test"]

    X_train = df_train[FEATURE_NAMES].to_numpy()
    y_train_reg = df_train["target_total_ratio"].to_numpy()
    groups = df_train["run_id"].to_numpy()

    reg_res = train_regressors(X_train, y_train_reg, groups, quick=True)

    train_clf_mask = df_train["issue_now"].isin(VALID_ISSUE_CLASSES).to_numpy()
    clf_res = train_classifiers(
        X_train[train_clf_mask],
        df_train.loc[train_clf_mask, "issue_now"].to_numpy(),
        groups[train_clf_mask],
        classes=VALID_ISSUE_CLASSES,
        quick=True,
    )

    healthy_mask = (df_train["issue_now"] == "healthy").to_numpy()
    anom = train_anomaly_detector(X_train[healthy_mask], feature_names=FEATURE_NAMES)

    model_dir = save_model_set(
        regressors=reg_res["models"],
        classifiers=clf_res["models"],
        anomaly_detector=anom,
        classes=VALID_ISSUE_CLASSES,
        cv_metrics={},
        test_metrics={},
        dataset_sha256="abc123",
        synthetic=True,
        models_root=tmp_path / "models",
        model_id="test_model_set",
    )

    # Size check
    size_bytes = compute_directory_size_bytes(model_dir)
    assert size_bytes < 10 * 1024 * 1024

    # Load runtime
    runtime = RuntimeModels.load(model_dir)
    assert runtime.feature_order == FEATURE_NAMES

    # Check predictions match in-memory model
    X_test = df_test[FEATURE_NAMES].to_numpy()
    rt_preds = runtime.predict(X_test)

    # Regressor predictions match
    xgb_direct = reg_res["models"]["XGBoost"].predict(X_test)
    np.testing.assert_allclose(rt_preds["slowdown_ratio"]["XGBoost"], xgb_direct, rtol=1e-5)

    # Anomaly scores match
    anom_direct = anom.score(X_test)
    np.testing.assert_allclose(rt_preds["anomaly"]["score"], anom_direct, atol=1e-4)


def test_predict_speed_under_50ms(tmp_path: Path) -> None:
    """Verify that RuntimeModels.predict takes < 50ms per call averaged over 100 calls."""
    syn_parquet = tmp_path / "synthetic_speed.parquet"
    generate_synthetic_dataset(out_path=syn_parquet, n_rows=1200, seed=42)
    df = pd.read_parquet(syn_parquet)

    df_train = df[df["split"] == "train"]
    X_train = df_train[FEATURE_NAMES].to_numpy()
    y_train = df_train["target_total_ratio"].to_numpy()
    groups = df_train["run_id"].to_numpy()

    reg_res = train_regressors(X_train, y_train, groups, quick=True)
    clf_mask = df_train["issue_now"].isin(VALID_ISSUE_CLASSES).to_numpy()
    clf_res = train_classifiers(
        X_train[clf_mask],
        df_train.loc[clf_mask, "issue_now"].to_numpy(),
        groups[clf_mask],
        classes=VALID_ISSUE_CLASSES,
        quick=True,
    )
    healthy_mask = (df_train["issue_now"] == "healthy").to_numpy()
    anom = train_anomaly_detector(X_train[healthy_mask], feature_names=FEATURE_NAMES)

    model_dir = save_model_set(
        regressors=reg_res["models"],
        classifiers=clf_res["models"],
        anomaly_detector=anom,
        classes=VALID_ISSUE_CLASSES,
        cv_metrics={},
        test_metrics={},
        dataset_sha256="abc",
        models_root=tmp_path / "models",
    )

    runtime = RuntimeModels.load(model_dir)
    sample_row = X_train[0]

    # Benchmark 100 calls (disable sys.gettrace during timing to avoid coverage profiler overhead)
    import sys

    old_trace = sys.gettrace()
    sys.settrace(None)
    try:
        t0 = time.perf_counter()
        for _ in range(100):
            runtime.predict(sample_row)
        elapsed = time.perf_counter() - t0
    finally:
        sys.settrace(old_trace)

    avg_ms = (elapsed / 100.0) * 1000.0

    assert avg_ms < 50.0, f"Average predict time {avg_ms:.2f}ms exceeded 50ms"


def test_xgboost_explain_contributions() -> None:
    """Verify that XGBoost native pred_contribs sum to the raw model output."""
    import xgboost as xgb

    rng = np.random.RandomState(42)
    X = rng.normal(size=(100, 27))
    y = rng.uniform(1.0, 3.0, size=100)

    model = xgb.XGBRegressor(n_estimators=30, max_depth=3, random_state=42)
    model.fit(X, y)

    row = X[0]
    contribs_dict, bias, raw_pred = compute_prediction_contributions(
        row, model, feature_names=FEATURE_NAMES
    )

    total_sum = sum(contribs_dict.values()) + bias
    assert np.isclose(total_sum, raw_pred, atol=1e-5)

    # Top contributors returns k elements sorted descending by magnitude
    top3 = top_contributors(row, model, feature_names=FEATURE_NAMES, k=3)
    assert len(top3) == 3
    assert abs(top3[0][1]) >= abs(top3[1][1]) >= abs(top3[2][1])


def test_version_mismatch_warning(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """Verify that RuntimeModels.load logs a warning when feature_set_version differs."""
    model_dir = tmp_path / "mismatch_model"
    model_dir.mkdir(parents=True)

    card = {
        "model_id": "mismatch",
        "feature_set_version": 999,  # Mismatched version
        "feature_order": FEATURE_NAMES,
        "classes": VALID_ISSUE_CLASSES,
    }
    with open(model_dir / "model_card.json", "w", encoding="utf-8") as f:
        json.dump(card, f)

    # Create dummy isolation forest file
    import joblib
    from sklearn.ensemble import IsolationForest

    from deskflow.ml.models import TrainedAnomalyDetector

    iso = IsolationForest(n_estimators=10, random_state=42).fit(np.zeros((10, 27)))
    detector = TrainedAnomalyDetector(model=iso, threshold=0.5, feature_names=FEATURE_NAMES)
    joblib.dump(detector, model_dir / "isolation_forest.joblib")

    with caplog.at_level(logging.WARNING):
        RuntimeModels.load(model_dir)

    assert any("feature_set_version 999" in rec.message for rec in caplog.records)
