"""Tests for Phase 9 model input guards and robustness."""

import numpy as np

from deskflow.features.registry import FEATURE_NAMES
from deskflow.ml.artifacts import resolve_model_dir
from deskflow.ml.models import train_regressors
from deskflow.ml.runtime import RuntimeModels


def test_feature_at_1e6_is_clipped() -> None:
    """Extreme feature value at 1e6 must be clipped and predicted ratio bounded."""
    model_dir, _ = resolve_model_dir()
    runtime = RuntimeModels.load(model_dir)

    # Base feature array of zeros
    x = np.zeros(len(FEATURE_NAMES), dtype=np.float64)
    # Set thread_count to 1,000,000
    idx_thread = FEATURE_NAMES.index("thread_count")
    x[idx_thread] = 1_000_000.0

    res = runtime.predict(x)
    slowdown = res["slowdown_ratio"]
    for m_name, val in slowdown.items():
        assert 0.3 <= val <= 10.0, f"Model {m_name} output {val} out of bounds [0.3, 10.0]"

    # If bounds exist in model card, verify clipping happened
    t_pct = runtime.card.get("training_percentiles", {})
    if t_pct.get("bounds_upper"):
        assert res["clipped_count"] >= 1


def test_multiple_extreme_features_triggers_out_of_range() -> None:
    """More than 3 features outside training bounds sets out_of_range flag."""
    model_dir, _ = resolve_model_dir()
    runtime = RuntimeModels.load(model_dir)

    x = np.full(len(FEATURE_NAMES), 1e8, dtype=np.float64)
    res = runtime.predict(x)

    t_pct = runtime.card.get("training_percentiles", {})
    if t_pct.get("bounds_upper"):
        assert res["out_of_range"] is True
        assert res["clipped_count"] > 3

    for m_name, val in res["slowdown_ratio"].items():
        assert 0.3 <= val <= 10.0, f"Model {m_name} output {val} out of bounds [0.3, 10.0]"


def test_flipped_power_flag() -> None:
    """Flipping battery_plugged flag does not cause invalid outputs."""
    model_dir, _ = resolve_model_dir()
    runtime = RuntimeModels.load(model_dir)

    x_ac = np.ones(len(FEATURE_NAMES), dtype=np.float64)
    idx_pwr = FEATURE_NAMES.index("battery_plugged")
    x_ac[idx_pwr] = 1.0

    x_bat = np.copy(x_ac)
    x_bat[idx_pwr] = 0.0

    res_ac = runtime.predict(x_ac)
    res_bat = runtime.predict(x_bat)

    for val in res_ac["slowdown_ratio"].values():
        assert 0.3 <= val <= 10.0
    for val in res_bat["slowdown_ratio"].values():
        assert 0.3 <= val <= 10.0


def test_constant_feature_scaling() -> None:
    """Features with zero variance must be zeroed out in Ridge and not blow up."""
    n_samples = 40
    n_feats = len(FEATURE_NAMES)
    rng = np.random.RandomState(42)

    X_train = rng.randn(n_samples, n_feats)
    # Make feature 5 strictly constant
    X_train[:, 5] = 42.0

    y_train = rng.uniform(0.8, 2.0, size=n_samples)
    groups = np.array([f"run-{i // 10}" for i in range(n_samples)])

    res = train_regressors(X_train, y_train, groups, quick=True)
    ridge_pipe = res["models"]["Ridge"]

    # Verify coef on constant feature is zero
    r_model = ridge_pipe.named_steps["ridge"]
    r_scaler = ridge_pipe.named_steps["scaler"]
    assert r_model.coef_[5] == 0.0
    assert r_scaler.scale_[5] == 1.0

    # Predict on test sample where feature 5 has a massive value
    X_test = rng.randn(5, n_feats)
    X_test[:, 5] = 1e9

    preds = ridge_pipe.predict(X_test)
    assert not np.isnan(preds).any()
    assert not np.isinf(preds).any()
    assert np.all(preds < 100.0)
