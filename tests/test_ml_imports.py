"""Pre-flight ML dependency imports and sanity checks for Python 3.12."""

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
import xgboost as xgb
from sklearn.ensemble import IsolationForest

shap = pytest.importorskip("shap", reason="SHAP is an optional training extra")


def test_ml_imports_and_basic_fits() -> None:
    """Verify that all core ML libraries load and fit on synthetic data."""
    assert np.__version__ is not None
    assert pd.__version__ is not None
    assert pa.__version__ is not None
    assert shap.__version__ is not None
    assert xgb.__version__ is not None

    rng = np.random.RandomState(42)
    x = rng.randn(30, 4)
    y = rng.randn(30)

    # Test IsolationForest fit
    iso = IsolationForest(n_estimators=10, random_state=42)
    iso.fit(x)
    preds = iso.predict(x)
    assert len(preds) == 30

    # Test XGBRegressor fit
    reg = xgb.XGBRegressor(n_estimators=10, max_depth=3, random_state=42)
    reg.fit(x, y)
    y_pred = reg.predict(x)
    assert len(y_pred) == 30
