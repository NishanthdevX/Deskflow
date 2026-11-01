"""Model definitions, baselines, training pipelines, and validation rules."""

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import xgboost as xgb
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import IsolationForest, RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from deskflow.features.registry import FEATURE_NAMES

logger = logging.getLogger(__name__)

VALID_ISSUE_CLASSES: list[str] = [
    "healthy",
    "cpu_pressure",
    "memory_pressure",
    "disk_pressure",
    "background_load",
]


class ConstantMedianRegressor:
    """Baseline regressor predicting the constant median of training targets."""

    def __init__(self) -> None:
        self.median_: float = 1.0

    def fit(self, X: Any, y: np.ndarray, **kwargs: Any) -> "ConstantMedianRegressor":
        self.median_ = float(np.median(y))
        return self

    def predict(self, X: Any) -> np.ndarray:
        n = len(X) if hasattr(X, "__len__") else 1
        return np.full(n, self.median_, dtype=np.float64)


class PersistenceRegressor:
    """Baseline regressor predicting current probe latency ratio probe_total_ratio."""

    def __init__(self, feature_idx: int | None = None) -> None:
        self.feature_idx = (
            feature_idx if feature_idx is not None else FEATURE_NAMES.index("probe_total_ratio")
        )

    def fit(self, X: Any, y: Any = None, **kwargs: Any) -> "PersistenceRegressor":
        return self

    def predict(self, X: Any) -> np.ndarray:
        if hasattr(X, "iloc") or (hasattr(X, "__getitem__") and hasattr(X, "columns")):
            return np.asarray(X["probe_total_ratio"], dtype=np.float64)
        arr = np.asarray(X)
        return np.asarray(arr[:, self.feature_idx], dtype=np.float64)


class RollingMeanRegressor:
    """Baseline regressor predicting 12s mean probe latency ratio probe_total_ratio_mean_12s."""

    def __init__(self, feature_idx: int | None = None) -> None:
        self.feature_idx = (
            feature_idx
            if feature_idx is not None
            else FEATURE_NAMES.index("probe_total_ratio_mean_12s")
        )

    def fit(self, X: Any, y: Any = None, **kwargs: Any) -> "RollingMeanRegressor":
        return self

    def predict(self, X: Any) -> np.ndarray:
        if hasattr(X, "iloc") or (hasattr(X, "__getitem__") and hasattr(X, "columns")):
            return np.asarray(X["probe_total_ratio_mean_12s"], dtype=np.float64)
        arr = np.asarray(X)
        return np.asarray(arr[:, self.feature_idx], dtype=np.float64)


@dataclass
class TrainedAnomalyDetector:
    """Anomaly detector wrapping IsolationForest with 99th percentile threshold."""

    model: IsolationForest
    threshold: float
    feature_names: list[str]

    def score(self, X: Any) -> np.ndarray:
        """Compute anomaly score where higher values mean more anomalous."""
        arr = np.asarray(X)
        # IsolationForest.score_samples returns opposite of anomaly score (lower is more abnormal)
        raw_scores = -self.model.score_samples(arr)
        return np.asarray(raw_scores, dtype=np.float64)

    def predict(self, X: Any) -> np.ndarray:
        """Return boolean mask indicating anomaly condition (score >= threshold)."""
        scores = self.score(X)
        return scores >= self.threshold


def validate_training_dataset(df_train: Any, df_test: Any | None = None) -> list[str]:
    """Validate dataset rows and class distributions against minimum requirements.

    Raises ValueError if training constraints are violated. Returns warning messages.
    """
    warnings: list[str] = []

    # 1. Minimum training rows refusal
    if len(df_train) < 200:
        raise ValueError(
            f"Training dataset has only {len(df_train)} rows; minimum 200 rows required to train."
        )

    # 2. Minimum class counts refusal
    clf_train = df_train[df_train["issue_now"].isin(VALID_ISSUE_CLASSES)]
    class_counts = clf_train["issue_now"].value_counts().to_dict()
    for cls in VALID_ISSUE_CLASSES:
        cnt = class_counts.get(cls, 0)
        if cnt < 30:
            raise ValueError(
                f"Class '{cls}' has only {cnt} training rows; minimum 30 rows required per class."
            )

    # 3. Hardware variety warning
    if "machine_id" in df_train.columns:
        machine_ids = df_train["machine_id"].unique()
        if len(machine_ids) == 1:
            warnings.append(
                f"Warning: Dataset contains data from only 1 machine_id ({machine_ids[0]}); "
                "models may not generalize across different hardware."
            )

    # 4. Test split warnings
    if df_test is not None:
        if len(df_test) < 200:
            warnings.append(f"Warning: Test split has fewer than 200 rows ({len(df_test)} rows).")
        if "run_id" in df_test.columns:
            n_runs = df_test["run_id"].nunique()
            if n_runs < 4:
                warnings.append(
                    f"Warning: Test split has fewer than 4 unique runs ({n_runs} runs)."
                )

    return warnings


def _tune_model(
    model_factory: Any,
    param_grid: list[dict[str, Any]],
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    scoring_func: Any,
    higher_is_better: bool = False,
) -> tuple[Any, dict[str, Any], float]:
    """Perform GroupKFold grid search tuning over tiny parameter combinations."""
    unique_groups = np.unique(groups)
    n_splits = min(4, len(unique_groups))
    gkf = GroupKFold(n_splits=n_splits)

    best_score = -float("inf") if higher_is_better else float("inf")
    best_params = param_grid[0] if param_grid else {}

    for params in param_grid:
        cv_scores: list[float] = []
        for train_idx, val_idx in gkf.split(X, y, groups=groups):
            X_tr, y_tr = X[train_idx], y[train_idx]
            X_val, y_val = X[val_idx], y[val_idx]

            m = model_factory(**params)
            m.fit(X_tr, y_tr)
            preds = m.predict(X_val)
            score = scoring_func(y_val, preds)
            cv_scores.append(score)

        mean_score = float(np.mean(cv_scores))
        if higher_is_better:
            if mean_score > best_score:
                best_score = mean_score
                best_params = params
        elif mean_score < best_score:
            best_score = mean_score
            best_params = params

    # Fit final model on all training data with best params
    final_model = model_factory(**best_params)
    final_model.fit(X, y)
    return final_model, best_params, best_score


def train_regressors(
    X_train: np.ndarray,
    y_train: np.ndarray,
    groups_train: np.ndarray,
    quick: bool = False,
) -> dict[str, Any]:
    """Train regression baselines and models on target_total_ratio."""
    models: dict[str, Any] = {}
    cv_metrics: dict[str, float] = {}

    # 1. Baselines
    med = ConstantMedianRegressor().fit(X_train, y_train)
    models["ConstantMedian"] = med
    cv_metrics["ConstantMedian"] = float(np.mean(np.abs(y_train - med.predict(X_train))))

    pers = PersistenceRegressor().fit(X_train, y_train)
    models["Persistence"] = pers
    cv_metrics["Persistence"] = float(np.mean(np.abs(y_train - pers.predict(X_train))))

    roll = RollingMeanRegressor().fit(X_train, y_train)
    models["RollingMean"] = roll
    cv_metrics["RollingMean"] = float(np.mean(np.abs(y_train - roll.predict(X_train))))

    def mae_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(np.mean(np.abs(y_true - y_pred)))

    # 2. Ridge Regression
    def make_ridge(alpha: float = 1.0) -> Pipeline:
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("ridge", Ridge(alpha=alpha, random_state=42)),
            ]
        )

    if quick:
        m_ridge = make_ridge(alpha=1.0)
        m_ridge.fit(X_train, y_train)
        models["Ridge"] = m_ridge
        cv_metrics["Ridge"] = mae_score(y_train, m_ridge.predict(X_train))
    else:
        grid_ridge = [{"alpha": a} for a in [0.01, 0.1, 1.0, 10.0, 100.0]]
        m_ridge, _, score = _tune_model(
            make_ridge,
            grid_ridge,
            X_train,
            y_train,
            groups_train,
            mae_score,
            higher_is_better=False,
        )
        models["Ridge"] = m_ridge
        cv_metrics["Ridge"] = score

    # 3. Random Forest Regressor (at most 100 trees, max_depth at most 10)
    def make_rf(
        n_estimators: int = 100, max_depth: int = 10, min_samples_leaf: int = 2
    ) -> RandomForestRegressor:
        return RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=42,
            n_jobs=-1,
        )

    if quick:
        m_rf = make_rf(n_estimators=50, max_depth=8, min_samples_leaf=2)
        m_rf.fit(X_train, y_train)
        models["RandomForest"] = m_rf
        cv_metrics["RandomForest"] = mae_score(y_train, m_rf.predict(X_train))
    else:
        grid_rf = [
            {"n_estimators": n, "max_depth": d, "min_samples_leaf": leaf}
            for n in [50, 100]
            for d in [6, 10]
            for leaf in [2, 4]
        ]  # 8 combinations
        m_rf, _, score = _tune_model(
            make_rf, grid_rf, X_train, y_train, groups_train, mae_score, higher_is_better=False
        )
        models["RandomForest"] = m_rf
        cv_metrics["RandomForest"] = score

    # 4. XGBoost Regressor (at most 150 trees, max_depth at most 5)
    def make_xgb(
        n_estimators: int = 100, max_depth: int = 4, learning_rate: float = 0.08
    ) -> xgb.XGBRegressor:
        return xgb.XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=42,
            n_jobs=-1,
            objective="reg:squarederror",
            verbosity=0,
        )

    if quick:
        m_xgb = make_xgb(n_estimators=50, max_depth=3, learning_rate=0.1)
        m_xgb.fit(X_train, y_train)
        models["XGBoost"] = m_xgb
        cv_metrics["XGBoost"] = mae_score(y_train, m_xgb.predict(X_train))
    else:
        grid_xgb = [
            {"n_estimators": n, "max_depth": d, "learning_rate": lr}
            for n in [50, 100]
            for d in [3, 5]
            for lr in [0.05, 0.1]
        ]  # 8 combinations
        m_xgb, _, score = _tune_model(
            make_xgb, grid_xgb, X_train, y_train, groups_train, mae_score, higher_is_better=False
        )
        models["XGBoost"] = m_xgb
        cv_metrics["XGBoost"] = score

    return {"models": models, "cv_metrics": cv_metrics}


def train_classifiers(
    X_train: np.ndarray,
    y_train: np.ndarray,
    groups_train: np.ndarray,
    classes: list[str],
    quick: bool = False,
) -> dict[str, Any]:
    """Train classification baseline and models on issue_now."""
    models: dict[str, Any] = {}
    cv_metrics: dict[str, float] = {}

    # Map class strings to integers
    class_to_idx = {c: i for i, c in enumerate(classes)}
    y_encoded = np.array([class_to_idx[val] for val in y_train], dtype=np.int32)

    # 1. Dummy Baseline
    dummy = DummyClassifier(strategy="prior", random_state=42)
    dummy.fit(X_train, y_encoded)
    models["Dummy"] = dummy
    from sklearn.metrics import f1_score

    def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    cv_metrics["Dummy"] = macro_f1(y_encoded, dummy.predict(X_train))

    # 2. Logistic Regression (balanced)
    def make_logreg(C: float = 1.0) -> Pipeline:
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "logreg",
                    LogisticRegression(
                        C=C, class_weight="balanced", random_state=42, max_iter=1000
                    ),
                ),
            ]
        )

    if quick:
        m_lr = make_logreg(C=1.0)
        m_lr.fit(X_train, y_encoded)
        models["LogisticRegression"] = m_lr
        cv_metrics["LogisticRegression"] = macro_f1(y_encoded, m_lr.predict(X_train))
    else:
        grid_lr = [{"C": c} for c in [0.01, 0.1, 1.0, 10.0]]  # 4 combinations
        m_lr, _, score = _tune_model(
            make_logreg,
            grid_lr,
            X_train,
            y_encoded,
            groups_train,
            macro_f1,
            higher_is_better=True,
        )
        models["LogisticRegression"] = m_lr
        cv_metrics["LogisticRegression"] = score

    # 3. Random Forest Classifier (balanced, small)
    def make_rf(
        n_estimators: int = 100, max_depth: int = 8, min_samples_leaf: int = 2
    ) -> RandomForestClassifier:
        return RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )

    if quick:
        m_rf = make_rf(n_estimators=50, max_depth=6, min_samples_leaf=2)
        m_rf.fit(X_train, y_encoded)
        models["RandomForest"] = m_rf
        cv_metrics["RandomForest"] = macro_f1(y_encoded, m_rf.predict(X_train))
    else:
        grid_rf = [
            {"n_estimators": n, "max_depth": d, "min_samples_leaf": leaf}
            for n in [50, 100]
            for d in [6, 10]
            for leaf in [2, 4]
        ]  # 8 combinations
        m_rf, _, score = _tune_model(
            make_rf,
            grid_rf,
            X_train,
            y_encoded,
            groups_train,
            macro_f1,
            higher_is_better=True,
        )
        models["RandomForest"] = m_rf
        cv_metrics["RandomForest"] = score

    # 4. XGBoost Classifier
    def make_xgb(
        n_estimators: int = 100, max_depth: int = 4, learning_rate: float = 0.08
    ) -> xgb.XGBClassifier:
        return xgb.XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=42,
            n_jobs=-1,
            eval_metric="mlogloss",
            verbosity=0,
        )

    if quick:
        m_xgb = make_xgb(n_estimators=50, max_depth=3, learning_rate=0.1)
        m_xgb.fit(X_train, y_encoded)
        models["XGBoost"] = m_xgb
        cv_metrics["XGBoost"] = macro_f1(y_encoded, m_xgb.predict(X_train))
    else:
        grid_xgb = [
            {"n_estimators": n, "max_depth": d, "learning_rate": lr}
            for n in [50, 100]
            for d in [3, 5]
            for lr in [0.05, 0.1]
        ]  # 8 combinations
        m_xgb, _, score = _tune_model(
            make_xgb,
            grid_xgb,
            X_train,
            y_encoded,
            groups_train,
            macro_f1,
            higher_is_better=True,
        )
        models["XGBoost"] = m_xgb
        cv_metrics["XGBoost"] = score

    return {"models": models, "cv_metrics": cv_metrics, "class_to_idx": class_to_idx}


def train_anomaly_detector(
    X_healthy_train: np.ndarray,
    feature_names: list[str],
) -> TrainedAnomalyDetector:
    """Train IsolationForest anomaly detector on healthy training rows only."""
    n_samples = len(X_healthy_train)
    max_samples = min(256, n_samples) if n_samples > 0 else 256
    iso = IsolationForest(
        n_estimators=100,
        max_samples=max_samples,
        random_state=42,
        n_jobs=-1,
    )
    iso.fit(X_healthy_train)

    # Threshold is the 99th percentile of training healthy scores
    healthy_scores = -iso.score_samples(X_healthy_train)
    threshold = float(np.percentile(healthy_scores, 99.0))

    return TrainedAnomalyDetector(
        model=iso,
        threshold=threshold,
        feature_names=feature_names,
    )
