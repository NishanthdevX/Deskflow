"""Training CLI command: model training, cross-validation, and artifact creation."""

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from deskflow.features.builder import merge_datasets
from deskflow.features.registry import DEFAULT_HORIZON_S, FEATURE_NAMES
from deskflow.ml.artifacts import compute_directory_size_bytes, save_model_set
from deskflow.ml.metrics import (
    compute_anomaly_metrics,
    compute_classification_metrics,
    compute_regression_metrics,
)
from deskflow.ml.models import (
    VALID_ISSUE_CLASSES,
    train_anomaly_detector,
    train_classifiers,
    train_regressors,
    validate_training_dataset,
)

logger = logging.getLogger(__name__)
console = Console()


def _is_synthetic(paths: list[Path]) -> bool:
    """Check if any of the dataset manifests indicate synthetic data."""
    for p in paths:
        mf = p.with_suffix(".manifest.json")
        if mf.exists():
            try:
                with open(mf, encoding="utf-8") as f:
                    data = json.load(f)
                    if data.get("synthetic"):
                        return True
            except Exception:
                pass
    return False


def _compute_dataset_sha256(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def train_command(
    datasets: list[Path] = typer.Option(
        ...,
        "--dataset",
        "-d",
        help="One or more parquet dataset paths to train on.",
    ),
    quick: bool = typer.Option(
        False,
        "--quick",
        help="Fast training mode skipping hyperparameter tuning grids.",
    ),
    out_id: str | None = typer.Option(
        None,
        "--id",
        help="Custom model set identifier.",
    ),
) -> None:
    """Train regression, classification, and anomaly models on engineered datasets."""
    import pandas as pd

    # Check synthetic banner
    synthetic = _is_synthetic(datasets)
    if synthetic:
        console.print(
            Panel.fit(
                "[bold red]SYNTHETIC DATA - NOT A RESULT[/bold red]",
                border_style="red",
            )
        )

    # 1. Load dataset
    if len(datasets) == 1:
        dataset_path = datasets[0]
        if not dataset_path.exists():
            console.print(f"[red]Error: Dataset file not found: {dataset_path}[/red]")
            raise typer.Exit(code=1)
        df = pd.read_parquet(dataset_path)
    else:
        # Merge multi-dataset
        merged_path = Path("merged_train_dataset.parquet")
        dataset_path, _ = merge_datasets(datasets, out_path=merged_path)
        df = pd.read_parquet(dataset_path)

    dataset_sha256 = _compute_dataset_sha256(dataset_path)

    # Validate split column
    if "split" not in df.columns:
        console.print("[red]Error: Dataset missing required 'split' column.[/red]")
        raise typer.Exit(code=1)

    df_train = df[df["split"] == "train"].reset_index(drop=True)
    df_test = df[df["split"] == "test"].reset_index(drop=True)

    # 2. Check validation rules & warnings
    try:
        warnings = validate_training_dataset(df_train, df_test)
    except ValueError as e:
        console.print(f"[bold red]Training Refusal:[/bold red] {e}")
        raise typer.Exit(code=1) from e

    for w in warnings:
        console.print(f"[yellow]{w}[/yellow]")

    console.print(
        f"Training set: {len(df_train)} rows ({df_train['run_id'].nunique()} runs). "
        f"Test set: {len(df_test)} rows ({df_test['run_id'].nunique()} runs)."
    )

    # 3. Prepare Arrays
    X_train = df_train[FEATURE_NAMES].to_numpy()
    y_train_reg = df_train["target_total_ratio"].to_numpy()
    groups_train = df_train["run_id"].to_numpy()

    X_test = df_test[FEATURE_NAMES].to_numpy()
    y_test_reg = df_test["target_total_ratio"].to_numpy()

    # Classification subsets (exclude combo and transition)
    train_clf_mask = df_train["issue_now"].isin(VALID_ISSUE_CLASSES).to_numpy()
    X_train_clf = X_train[train_clf_mask]
    y_train_clf = df_train.loc[train_clf_mask, "issue_now"].to_numpy()
    groups_train_clf = groups_train[train_clf_mask]

    test_clf_mask = df_test["issue_now"].isin(VALID_ISSUE_CLASSES).to_numpy()
    X_test_clf = X_test[test_clf_mask]
    y_test_clf = df_test.loc[test_clf_mask, "issue_now"].to_numpy()

    # Healthy subset for anomaly training
    healthy_mask = (df_train["issue_now"] == "healthy").to_numpy()
    X_train_healthy = X_train[healthy_mask]

    # 4. Train Regressors
    reg_results = train_regressors(X_train, y_train_reg, groups_train, quick=quick)
    regressors = reg_results["models"]
    reg_cv = reg_results["cv_metrics"]

    # 5. Train Classifiers
    clf_results = train_classifiers(
        X_train_clf,
        y_train_clf,
        groups_train_clf,
        classes=VALID_ISSUE_CLASSES,
        quick=quick,
    )
    classifiers = clf_results["models"]
    clf_cv = clf_results["cv_metrics"]

    # 6. Train Anomaly Detector
    anomaly_detector = train_anomaly_detector(X_train_healthy, feature_names=FEATURE_NAMES)

    # 7. Evaluate on Test Set
    reg_test_metrics: dict[str, dict[str, float]] = {}
    test_preds_reg: dict[str, np.ndarray] = {}

    for name, model in regressors.items():
        if hasattr(model, "predict"):
            preds = model.predict(X_test)
        else:
            preds = np.ones(len(X_test))
        test_preds_reg[name] = preds
        reg_test_metrics[name] = compute_regression_metrics(y_test_reg, preds)

    clf_test_metrics: dict[str, dict[str, Any]] = {}
    test_preds_clf: dict[str, np.ndarray] = {}

    for name, model in classifiers.items():
        preds_enc = model.predict(X_test_clf)
        # Decode integers to class strings if encoded
        if np.issubdtype(preds_enc.dtype, np.integer):
            preds_str = np.array([VALID_ISSUE_CLASSES[i] for i in preds_enc])
        else:
            preds_str = preds_enc
        test_preds_clf[name] = preds_str
        clf_test_metrics[name] = compute_classification_metrics(
            y_test_clf, preds_str, classes=VALID_ISSUE_CLASSES
        )

    # Anomaly evaluation
    anom_scores = anomaly_detector.score(X_test)
    y_true_binary = (df_test["issue_now"] != "healthy").to_numpy().astype(int)
    is_combo = (df_test["issue_now"] == "combo").to_numpy()
    anom_metrics = compute_anomaly_metrics(
        anom_scores,
        y_true_binary,
        threshold=anomaly_detector.threshold,
        is_combo_mask=is_combo,
    )

    # 8. Assemble test predictions dataframe
    df_test_preds = df_test.copy()
    for name, preds in test_preds_reg.items():
        df_test_preds[f"pred_reg_{name}"] = preds
    # Add anomaly score
    df_test_preds["pred_anomaly_score"] = anom_scores
    df_test_preds["pred_is_anomaly"] = anom_scores >= anomaly_detector.threshold

    # 9. Save Model Set Artifacts
    cv_metrics = {"regression": reg_cv, "classification": clf_cv}
    test_metrics = {
        "regression": reg_test_metrics,
        "classification": clf_test_metrics,
        "anomaly": anom_metrics,
    }

    model_dir = save_model_set(
        regressors=regressors,
        classifiers=classifiers,
        anomaly_detector=anomaly_detector,
        classes=VALID_ISSUE_CLASSES,
        cv_metrics=cv_metrics,
        test_metrics=test_metrics,
        dataset_sha256=dataset_sha256,
        df_test_predictions=df_test_preds,
        synthetic=synthetic,
        model_id=out_id,
        horizon_s=DEFAULT_HORIZON_S,
    )

    # 10. Display Leaderboards
    # Regression Leaderboard
    table_reg = Table(title="Regression Leaderboard (Target: target_total_ratio)")
    table_reg.add_column("Model", style="bold")
    table_reg.add_column("Type")
    table_reg.add_column("Test MAE", justify="right")
    table_reg.add_column("Test RMSE", justify="right")
    table_reg.add_column("Test R2", justify="right")
    table_reg.add_column("Tol Acc (<0.25)", justify="right")
    table_reg.add_column("Slow F1 (>=1.5)", justify="right")

    for name in [
        "ConstantMedian",
        "Persistence",
        "RollingMean",
        "Ridge",
        "RandomForest",
        "XGBoost",
    ]:
        if name in reg_test_metrics:
            m = reg_test_metrics[name]
            is_base = name in ("ConstantMedian", "Persistence", "RollingMean")
            m_type = "Baseline" if is_base else "Model"
            table_reg.add_row(
                name,
                m_type,
                f"{m['mae']:.4f}",
                f"{m['rmse']:.4f}",
                f"{m['r2']:.4f}",
                f"{m['tolerance_accuracy']:.4f}",
                f"{m['slow_f1']:.4f}",
            )
    console.print(table_reg)

    # Classification Leaderboard
    table_clf = Table(title="Classification Leaderboard (Target: issue_now)")
    table_clf.add_column("Model", style="bold")
    table_clf.add_column("Type")
    table_clf.add_column("Accuracy", justify="right")
    table_clf.add_column("Balanced Acc", justify="right")
    table_clf.add_column("Macro F1", justify="right")

    for name in ["Dummy", "LogisticRegression", "RandomForest", "XGBoost"]:
        if name in clf_test_metrics:
            m = clf_test_metrics[name]
            is_base = name == "Dummy"
            m_type = "Baseline" if is_base else "Model"
            table_clf.add_row(
                name,
                m_type,
                f"{m['accuracy']:.4f}",
                f"{m['balanced_accuracy']:.4f}",
                f"{m['macro_f1']:.4f}",
            )
    console.print(table_clf)

    # Anomaly Detection Table
    table_anom = Table(title="Anomaly Detection (IsolationForest)")
    table_anom.add_column("Metric", style="bold")
    table_anom.add_column("Value", justify="right")
    table_anom.add_row("Threshold (99th percentile)", f"{anom_metrics['threshold']:.4f}")
    table_anom.add_row("ROC-AUC", f"{anom_metrics['roc_auc']:.4f}")
    table_anom.add_row("Recall (anomalies)", f"{anom_metrics['recall']:.4f}")
    table_anom.add_row("False Alarm Rate (healthy)", f"{anom_metrics['false_alarm_rate']:.4f}")
    table_anom.add_row("Combo Detection Rate", f"{anom_metrics['combo_detection_rate']:.4f}")
    console.print(table_anom)

    size_mb = compute_directory_size_bytes(model_dir) / (1024 * 1024)
    console.print(
        f"\nTrained models saved to [cyan]{model_dir}[/cyan] "
        f"(Folder size: {size_mb:.2f} MB, latest pointer updated)"
    )
