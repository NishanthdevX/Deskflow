"""Feature importance and prediction explanation via native XGBoost tree SHAP contributions."""

import logging
from typing import Any

import numpy as np
import xgboost as xgb

from deskflow.features.registry import FEATURE_NAMES

logger = logging.getLogger(__name__)


def compute_prediction_contributions(
    row: np.ndarray,
    model: Any,
    feature_names: list[str] | None = None,
) -> tuple[dict[str, float], float, float]:
    """Compute exact TreeSHAP contributions and bias term using native XGBoost pred_contribs.

    Returns:
        (feature_contributions_dict, bias_term, total_raw_prediction)
    """
    names = feature_names or FEATURE_NAMES
    arr = np.asarray(row, dtype=np.float64).reshape(1, -1)

    booster = model.get_booster() if hasattr(model, "get_booster") else model
    dmat = xgb.DMatrix(arr, feature_names=names)

    # booster.predict with pred_contribs=True returns array of shape (1, num_features + 1)
    # The last value in each row is the expected value (bias)
    contribs = booster.predict(dmat, pred_contribs=True)[0]

    bias = float(contribs[-1])
    raw_pred = float(booster.predict(dmat, output_margin=True)[0])

    feat_contribs: dict[str, float] = {}
    for name, c in zip(names, contribs[:-1], strict=False):
        feat_contribs[name] = float(c)

    return feat_contribs, bias, raw_pred


def top_contributors(
    row: np.ndarray,
    model: Any,
    feature_names: list[str] | None = None,
    k: int = 3,
) -> list[tuple[str, float]]:
    """Return top-k features contributing most strongly to prediction by SHAP magnitude."""
    feat_contribs, _, _ = compute_prediction_contributions(row, model, feature_names=feature_names)

    # Sort descending by absolute contribution
    sorted_items = sorted(feat_contribs.items(), key=lambda kv: abs(kv[1]), reverse=True)
    return [(k_name, round(val, 4)) for k_name, val in sorted_items[:k]]
