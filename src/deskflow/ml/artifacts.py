"""Artifact persistence, serialization, directory size constraints, and metadata cards."""

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib

from deskflow import __version__
from deskflow.features.registry import (
    DEFAULT_HORIZON_S,
    FEATURE_NAMES,
    FEATURE_SET_VERSION,
    SLOW_THRESHOLD,
)
from deskflow.ml.models import (
    TrainedAnomalyDetector,
)
from deskflow.paths import get_data_dir

logger = logging.getLogger(__name__)

MAX_ARTIFACT_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB


def compute_directory_size_bytes(directory: Path) -> int:
    """Calculate total byte size of all files in a directory recursively."""
    total = 0
    for p in directory.rglob("*"):
        if p.is_file():
            total += p.stat().st_size
    return total


def save_model_set(
    regressors: dict[str, Any],
    classifiers: dict[str, Any],
    anomaly_detector: TrainedAnomalyDetector,
    classes: list[str],
    cv_metrics: dict[str, Any],
    test_metrics: dict[str, Any],
    dataset_sha256: str,
    df_test_predictions: Any | None = None,
    synthetic: bool = False,
    model_id: str | None = None,
    models_root: Path | None = None,
    horizon_s: float = DEFAULT_HORIZON_S,
    primary_models: dict[str, str] | None = None,
    X_train: Any | None = None,
    training_percentiles: dict[str, Any] | None = None,
) -> Path:
    """Serialize all runtime-ready models, cards, and predictions to user_data_dir."""
    root = models_root or (get_data_dir() / "models")
    root.mkdir(parents=True, exist_ok=True)

    mid = model_id or datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    dest_dir = root / mid
    dest_dir.mkdir(parents=True, exist_ok=True)

    # 1. XGBoost Regressor (native .ubj)
    if "XGBoost" in regressors and hasattr(regressors["XGBoost"], "save_model"):
        regressors["XGBoost"].save_model(dest_dir / "regressor_xgb.ubj")

    # 2. XGBoost Classifier (native .ubj)
    if "XGBoost" in classifiers and hasattr(classifiers["XGBoost"], "save_model"):
        classifiers["XGBoost"].save_model(dest_dir / "classifier_xgb.ubj")

    # 3. Ridge Regressor (JSON format with coefficients and scaler params)
    if "Ridge" in regressors and hasattr(regressors["Ridge"], "named_steps"):
        ridge_pipe = regressors["Ridge"]
        # Extract steps
        imputer = ridge_pipe.named_steps.get("imputer")
        scaler = ridge_pipe.named_steps.get("scaler")
        ridge_model = ridge_pipe.named_steps.get("ridge")

        ridge_data = {
            "feature_names": FEATURE_NAMES,
            "intercept": float(ridge_model.intercept_),
            "coef": [float(c) for c in ridge_model.coef_],
            "imputer_statistics": [float(s) for s in imputer.statistics_] if imputer else None,
            "scaler_mean": [float(m) for m in scaler.mean_] if scaler else None,
            "scaler_scale": [float(s) for s in scaler.scale_] if scaler else None,
        }
        with open(dest_dir / "regressor_ridge.json", "w", encoding="utf-8") as f:
            json.dump(ridge_data, f, indent=2)

    # 4. Random Forest Regressor (joblib compress=3)
    if "RandomForest" in regressors:
        joblib.dump(regressors["RandomForest"], dest_dir / "regressor_rf.joblib", compress=3)

    # 5. Classifiers: LogisticRegression and RandomForest (joblib compress=3)
    if "LogisticRegression" in classifiers:
        joblib.dump(
            classifiers["LogisticRegression"],
            dest_dir / "classifier_logreg.joblib",
            compress=3,
        )
    if "RandomForest" in classifiers:
        joblib.dump(classifiers["RandomForest"], dest_dir / "classifier_rf.joblib", compress=3)

    # 6. IsolationForest Anomaly Detector (joblib compress=3)
    joblib.dump(anomaly_detector, dest_dir / "isolation_forest.joblib", compress=3)

    # 7. Predictions test dataframe
    if df_test_predictions is not None:
        pred_parquet_path = dest_dir / "predictions_test.parquet"
        df_test_predictions.to_parquet(pred_parquet_path, index=False, engine="pyarrow")

    # 8. Model Card
    if X_train is not None and training_percentiles is None:
        import numpy as np

        p005 = np.percentile(X_train, 0.5, axis=0)
        p995 = np.percentile(X_train, 99.5, axis=0)
        span = p995 - p005
        margin = 0.05 * np.maximum(span, 1e-5)
        training_percentiles = {
            "p005": [float(x) for x in p005],
            "p995": [float(x) for x in p995],
            "bounds_lower": [float(x) for x in (p005 - margin)],
            "bounds_upper": [float(x) for x in (p995 + margin)],
        }

    card = {
        "model_id": mid,
        "feature_order": FEATURE_NAMES,
        "deskflow_version": __version__,
        "feature_set_version": FEATURE_SET_VERSION,
        "horizon_s": horizon_s,
        "thresholds": {
            "slow_threshold": SLOW_THRESHOLD,
            "anomaly_threshold": round(float(anomaly_detector.threshold), 4),
        },
        "primary_models": primary_models
        or {
            "regressor": "XGBoost" if "XGBoost" in regressors else "Ridge",
            "classifier": "XGBoost" if "XGBoost" in classifiers else "LogisticRegression",
        },
        "dataset_sha256": dataset_sha256,
        "seeds": {"random_state": 42},
        "classes": classes,
        "cv_metrics": cv_metrics,
        "test_metrics": test_metrics,
        "synthetic": synthetic,
        "training_percentiles": training_percentiles or {},
        "created_utc": datetime.now(UTC).isoformat(),
    }

    card_path = dest_dir / "model_card.json"
    with open(card_path, "w", encoding="utf-8") as f:
        json.dump(card, f, indent=2)

    # 9. Update 'latest' pointer
    latest_path = root / "latest"
    with open(latest_path, "w", encoding="utf-8") as f:
        f.write(mid.strip())

    # 10. Check folder size (< 10 MB constraint)
    size_bytes = compute_directory_size_bytes(dest_dir)
    if size_bytes > MAX_ARTIFACT_SIZE_BYTES:
        raise ValueError(
            f"Artifact size {size_bytes / (1024 * 1024):.2f} MB exceeds maximum allowed 10 MB."
        )

    return dest_dir


def load_model_card(model_dir: Path) -> dict[str, Any]:
    """Load model_card.json from specified model directory."""
    card_path = model_dir / "model_card.json"
    if not card_path.exists():
        raise FileNotFoundError(f"Model card not found at {card_path}")
    with open(card_path, encoding="utf-8") as f:
        data: dict[str, Any] = json.load(f)
        return data


def get_latest_model_id(models_root: Path | None = None) -> str | None:
    """Return model ID referenced by the 'latest' pointer file."""
    root = models_root or (get_data_dir() / "models")
    latest_file = root / "latest"
    if latest_file.exists():
        mid = latest_file.read_text(encoding="utf-8").strip()
        if mid and (root / mid).exists():
            return mid
    return None


def get_bundled_model_dir() -> Path:
    """Return path to bundled default models directory."""
    return Path(__file__).resolve().parent.parent / "data" / "models" / "default"


def resolve_model_dir(models_root: Path | None = None) -> tuple[Path, bool]:
    """Resolve model directory: user's latest trained set first, otherwise bundled default.

    Returns:
        tuple[Path, bool]: (model_dir, is_bundled) where is_bundled is True if using
        the bundled default models set.

    Raises:
        FileNotFoundError: If neither user model nor bundled default model exists.
    """
    root = models_root or (get_data_dir() / "models")
    latest_id = get_latest_model_id(root)
    if latest_id:
        user_dir = root / latest_id
        if user_dir.exists() and (user_dir / "model_card.json").exists():
            return user_dir, False

    bundled = get_bundled_model_dir()
    if bundled.exists() and (bundled / "model_card.json").exists():
        return bundled, True

    raise FileNotFoundError(
        "No trained model set found. No user model exists and bundled default is missing."
    )


def list_model_sets(models_root: Path | None = None) -> list[dict[str, Any]]:
    """List all available trained model sets with summary metadata."""
    root = models_root or (get_data_dir() / "models")
    if not root.exists():
        return []

    latest_id = get_latest_model_id(root)
    results: list[dict[str, Any]] = []

    for d in sorted(root.iterdir()):
        if d.is_dir() and (d / "model_card.json").exists():
            try:
                card = load_model_card(d)
                results.append(
                    {
                        "model_id": card.get("model_id", d.name),
                        "created_utc": card.get("created_utc", ""),
                        "deskflow_version": card.get("deskflow_version", "unknown"),
                        "synthetic": card.get("synthetic", False),
                        "is_latest": card.get("model_id") == latest_id,
                        "path": d,
                        "card": card,
                    }
                )
            except Exception as e:
                logger.warning("Failed to parse card in %s: %s", d, e)

    return results
