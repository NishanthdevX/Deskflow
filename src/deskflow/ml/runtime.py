"""Low-latency inference runtime for production scoring and anomaly detection."""

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from deskflow.features.registry import FEATURE_NAMES, FEATURE_SET_VERSION
from deskflow.ml.artifacts import load_model_card

logger = logging.getLogger(__name__)


class RuntimeModels:
    """Production runtime container loading serialized models for sub-50ms inference."""

    def __init__(
        self,
        model_dir: Path,
        card: dict[str, Any],
        xgb_reg: Any | None,
        rf_reg: Any | None,
        ridge_params: dict[str, Any] | None,
        xgb_clf: Any | None,
        rf_clf: Any | None,
        logreg_clf: Any | None,
        iso_detector: Any,
        classes: list[str],
    ) -> None:
        self.model_dir = model_dir
        self.card = card
        self.xgb_reg = xgb_reg
        self.rf_reg = rf_reg
        self.ridge_params = ridge_params
        self.xgb_clf = xgb_clf
        self.rf_clf = rf_clf
        self.logreg_clf = logreg_clf
        self.iso_detector = iso_detector
        self.classes = classes
        self.feature_order = card.get("feature_order", FEATURE_NAMES)

        # Precompute persistence and rolling mean feature indices
        self.idx_probe_total_ratio = (
            self.feature_order.index("probe_total_ratio")
            if "probe_total_ratio" in self.feature_order
            else 25
        )
        self.idx_probe_total_ratio_12s = (
            self.feature_order.index("probe_total_ratio_mean_12s")
            if "probe_total_ratio_mean_12s" in self.feature_order
            else 26
        )

    @classmethod
    def load(cls, model_dir: Path) -> "RuntimeModels":
        """Load trained models and configuration from disk with lazy imports."""
        import joblib
        import xgboost as xgb

        card = load_model_card(model_dir)

        # Check version compatibility warning
        feat_ver = card.get("feature_set_version", 1)
        if feat_ver != FEATURE_SET_VERSION:
            logger.warning(
                "Model was trained with feature_set_version %d, current runtime is %d.",
                feat_ver,
                FEATURE_SET_VERSION,
            )

        classes = card.get(
            "classes",
            ["healthy", "cpu_pressure", "memory_pressure", "disk_pressure", "background_load"],
        )

        # 1. Regressors
        xgb_reg = None
        xgb_reg_path = model_dir / "regressor_xgb.ubj"
        if xgb_reg_path.exists():
            xgb_reg = xgb.XGBRegressor(n_jobs=1)
            xgb_reg.load_model(xgb_reg_path)

        rf_reg = None
        rf_reg_path = model_dir / "regressor_rf.joblib"
        if rf_reg_path.exists():
            rf_reg = joblib.load(rf_reg_path)
            if hasattr(rf_reg, "n_jobs"):
                rf_reg.n_jobs = 1

        ridge_params = None
        ridge_path = model_dir / "regressor_ridge.json"
        if ridge_path.exists():
            with open(ridge_path, encoding="utf-8") as f:
                ridge_params = json.load(f)

        # 2. Classifiers
        xgb_clf = None
        xgb_clf_path = model_dir / "classifier_xgb.ubj"
        if xgb_clf_path.exists():
            xgb_clf = xgb.XGBClassifier(n_jobs=1)
            xgb_clf.load_model(xgb_clf_path)

        rf_clf = None
        rf_clf_path = model_dir / "classifier_rf.joblib"
        if rf_clf_path.exists():
            rf_clf = joblib.load(rf_clf_path)
            if hasattr(rf_clf, "n_jobs"):
                rf_clf.n_jobs = 1

        logreg_clf = None
        logreg_path = model_dir / "classifier_logreg.joblib"
        if logreg_path.exists():
            logreg_clf = joblib.load(logreg_path)

        # 3. Anomaly Detector
        iso_detector = joblib.load(model_dir / "isolation_forest.joblib")
        if hasattr(iso_detector, "model") and hasattr(iso_detector.model, "n_jobs"):
            iso_detector.model.n_jobs = 1

        return cls(
            model_dir=model_dir,
            card=card,
            xgb_reg=xgb_reg,
            rf_reg=rf_reg,
            ridge_params=ridge_params,
            xgb_clf=xgb_clf,
            rf_clf=rf_clf,
            logreg_clf=logreg_clf,
            iso_detector=iso_detector,
            classes=classes,
        )

    def _predict_ridge(self, X: np.ndarray) -> np.ndarray:
        """Fast vectorized inference for Ridge using JSON parameters."""
        if not self.ridge_params:
            return np.ones(len(X))
        imputer_stats = self.ridge_params.get("imputer_statistics")
        scaler_mean = self.ridge_params.get("scaler_mean")
        scaler_scale = self.ridge_params.get("scaler_scale")
        coef = np.asarray(self.ridge_params.get("coef", []))
        intercept = float(self.ridge_params.get("intercept", 1.0))

        X_work = np.copy(X)
        if imputer_stats:
            stats_arr = np.asarray(imputer_stats)
            inds = np.where(np.isnan(X_work))
            X_work[inds] = np.take(stats_arr, inds[1])

        if scaler_mean and scaler_scale:
            X_work = (X_work - np.asarray(scaler_mean)) / np.asarray(scaler_scale)

        raw_pred = np.dot(X_work, coef) + intercept
        return np.asarray(raw_pred, dtype=np.float64)

    def predict(self, features: np.ndarray) -> dict[str, Any]:
        """Perform sub-50ms inference across all models for input feature array."""
        arr = np.asarray(features, dtype=np.float64)
        is_single = arr.ndim == 1
        if is_single:
            arr = arr.reshape(1, -1)

        # 1. Regression predictions
        slowdown: dict[str, Any] = {}
        if self.idx_probe_total_ratio < arr.shape[1]:
            slowdown["Persistence"] = (
                float(arr[0, self.idx_probe_total_ratio])
                if is_single
                else arr[:, self.idx_probe_total_ratio]
            )
        if self.idx_probe_total_ratio_12s < arr.shape[1]:
            slowdown["RollingMean"] = (
                float(arr[0, self.idx_probe_total_ratio_12s])
                if is_single
                else arr[:, self.idx_probe_total_ratio_12s]
            )
        if self.ridge_params:
            r_pred = self._predict_ridge(arr)
            slowdown["Ridge"] = float(r_pred[0]) if is_single else r_pred
        if self.rf_reg:
            rf_pred = self.rf_reg.predict(arr)
            slowdown["RandomForest"] = float(rf_pred[0]) if is_single else rf_pred
        if self.xgb_reg:
            xgb_pred = self.xgb_reg.predict(arr)
            slowdown["XGBoost"] = float(xgb_pred[0]) if is_single else xgb_pred

        # 2. Classification predictions
        probs: dict[str, Any] = {}
        predicted_classes: dict[str, Any] = {}

        if self.xgb_clf:
            xgb_p = self.xgb_clf.predict_proba(arr)
            if is_single:
                p0 = xgb_p[0]
                probs["XGBoost"] = {c: float(v) for c, v in zip(self.classes, p0, strict=False)}
                predicted_classes["XGBoost"] = self.classes[int(np.argmax(p0))]
            else:
                probs["XGBoost"] = [
                    {cls_name: float(p) for cls_name, p in zip(self.classes, row_p, strict=False)}
                    for row_p in xgb_p
                ]
                predicted_classes["XGBoost"] = [
                    self.classes[int(np.argmax(row_p))] for row_p in xgb_p
                ]

        if self.rf_clf:
            rf_p = self.rf_clf.predict_proba(arr)
            if is_single:
                p0 = rf_p[0]
                probs["RandomForest"] = {
                    c: float(v) for c, v in zip(self.classes, p0, strict=False)
                }
                predicted_classes["RandomForest"] = self.classes[int(np.argmax(p0))]
            else:
                probs["RandomForest"] = [
                    {cls_name: float(p) for cls_name, p in zip(self.classes, row_p, strict=False)}
                    for row_p in rf_p
                ]
                predicted_classes["RandomForest"] = [
                    self.classes[int(np.argmax(row_p))] for row_p in rf_p
                ]

        if self.logreg_clf:
            lr_p = self.logreg_clf.predict_proba(arr)
            if is_single:
                p0 = lr_p[0]
                probs["LogisticRegression"] = {
                    c: float(v) for c, v in zip(self.classes, p0, strict=False)
                }
                predicted_classes["LogisticRegression"] = self.classes[int(np.argmax(p0))]
            else:
                probs["LogisticRegression"] = [
                    {cls_name: float(p) for cls_name, p in zip(self.classes, row_p, strict=False)}
                    for row_p in lr_p
                ]
                predicted_classes["LogisticRegression"] = [
                    self.classes[int(np.argmax(row_p))] for row_p in lr_p
                ]

        # 3. Anomaly scoring
        anomaly_scores = self.iso_detector.score(arr)
        threshold = float(self.iso_detector.threshold)

        if is_single:
            score_val = float(anomaly_scores[0])
            return {
                "slowdown_ratio": slowdown,
                "issue_probabilities": probs,
                "predicted_issue": predicted_classes,
                "anomaly": {
                    "score": round(score_val, 4),
                    "is_anomaly": bool(score_val >= threshold),
                    "threshold": round(threshold, 4),
                },
            }

        anomalies = anomaly_scores >= threshold
        return {
            "slowdown_ratio": slowdown,
            "issue_probabilities": probs,
            "predicted_issue": predicted_classes,
            "anomaly": {
                "score": [round(float(s), 4) for s in anomaly_scores],
                "is_anomaly": [bool(a) for a in anomalies],
                "threshold": round(threshold, 4),
            },
        }
