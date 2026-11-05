"""One-command model build, hard quality gate validation, and release export."""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import typer
from rich import box
from rich.console import Console
from rich.table import Table

from deskflow.baseline import load_all_baselines
from deskflow.exit_codes import EXIT_ERROR
from deskflow.features.builder import (
    build_dataset,
    describe_dataset,
    validate_dataset,
)
from deskflow.features.registry import DEFAULT_HORIZON_S, FEATURE_NAMES
from deskflow.ml.artifacts import (
    get_bundled_model_dir,
    save_model_set,
)
from deskflow.ml.gates import (
    GateResult,
    evaluate_dataset_gates,
    evaluate_prediction_gates,
)
from deskflow.ml.metrics import (
    compute_anomaly_metrics,
    compute_block_bootstrap,
    compute_classification_metrics,
    compute_regression_metrics,
)
from deskflow.ml.models import (
    VALID_ISSUE_CLASSES,
    train_anomaly_detector,
    train_classifiers,
    train_regressors,
)
from deskflow.ml.report_data import build_report_data
from deskflow.truth import compute_file_sha256, get_git_commit

logger = logging.getLogger(__name__)
console = Console()


def _format_gates_table(
    gates: list[GateResult],
    plain: bool = False,
) -> None:
    """Render structured table of quality gate results."""
    title = "[ HARD QUALITY GATES ]"
    if plain:
        console.print(f"\n{title}")
        header = f"{'Gate':<22} {'Required':<25} {'Measured':<32} {'Status':<6}"
        console.print(header)
        console.print("-" * len(header))
        for g in gates:
            st = "PASS" if g.passed else "FAIL"
            meas = g.measured if len(g.measured) <= 30 else g.measured[:27] + "..."
            console.print(f"{g.name:<22} {g.required:<25} {meas:<32} {st:<6}")
            if not g.passed and g.reason:
                console.print(f"  Reason: {g.reason}")
        console.print("")
        return

    table = Table(
        title=title,
        box=box.ROUNDED,
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("Quality Gate", style="bold")
    table.add_column("Required Threshold")
    table.add_column("Measured Value")
    table.add_column("Status", justify="center")

    for g in gates:
        st_style = "bold green" if g.passed else "bold red"
        st_text = f"[{st_style}]{'PASS' if g.passed else 'FAIL'}[/{st_style}]"
        table.add_row(g.name, g.required, g.measured, st_text)

    console.print(table)


def build_model_command(
    runs: list[Path] = typer.Option(
        ...,
        "--runs",
        "-r",
        help="One or more directories or JSONL files containing run records.",
    ),
    export: bool = typer.Option(
        False,
        "--export",
        help="Export passing model set to bundled models directory (synthetic: false).",
    ),
    quick: bool = typer.Option(
        False,
        "--quick",
        help="Fast training mode skipping hyperparameter tuning grids (testing only).",
    ),
    plain: bool = typer.Option(
        False,
        "--plain",
        help="Plain ASCII output without styling or box characters.",
    ),
) -> None:
    """Build dataset, validate, train full tuning grids, evaluate, and export models."""
    import pandas as pd

    console.print(f"\nStarting model build from {len(runs)} run path(s)...")

    # 1. Dataset Build with per-session baselines
    temp_dataset_path = Path("build_model_dataset.parquet")
    try:
        build_result = build_dataset(
            run_paths=runs,
            horizon_s=DEFAULT_HORIZON_S,
            out_path=temp_dataset_path,
        )
    except Exception as exc:
        console.print(f"[red]Dataset build failed:[/red] {exc}")
        raise typer.Exit(code=EXIT_ERROR) from exc

    parquet_path = build_result.parquet_path
    manifest_path = build_result.manifest_path

    with open(manifest_path, encoding="utf-8") as f:
        manifest: dict[str, Any] = json.load(f)

    dataset_sha256 = str(manifest.get("sha256", "unknown"))
    is_dataset_synthetic = bool(manifest.get("synthetic", False))

    # 2. Dataset Validate & Describe
    val_errors = validate_dataset(parquet_path)
    _ = describe_dataset(parquet_path)

    # 3. Load Parquet into splits
    df = pd.read_parquet(parquet_path)
    if "split" not in df.columns:
        console.print("[red]Error: Dataset missing required 'split' column.[/red]")
        raise typer.Exit(code=EXIT_ERROR)

    df_train = df[df["split"] == "train"].reset_index(drop=True)
    df_test = df[df["split"] == "test"].reset_index(drop=True)

    train_rows = len(df_train)
    test_rows = len(df_test)

    # Count test sessions
    test_session_col = "session_id" if "session_id" in df_test.columns else "run_id"
    test_sessions_count = int(df_test[test_session_col].nunique())

    # Count class rows in train
    class_counts: dict[str, int] = {}
    for cls in VALID_ISSUE_CLASSES:
        class_counts[cls] = int((df_train["issue_now"] == cls).sum())

    # Power states in train
    p_col = "baseline_state" if "baseline_state" in df_train.columns else "power_state"
    if p_col in df_train.columns:
        power_states_train = sorted([str(s) for s in df_train[p_col].dropna().unique()])
    elif "battery_plugged" in df_train.columns:
        power_states_train = []
        if (df_train["battery_plugged"] == 1.0).any():
            power_states_train.append("ac")
        if (df_train["battery_plugged"] == 0.0).any():
            power_states_train.append("battery")
    else:
        power_states_train = ["ac"]

    # Baselines check
    baselines = load_all_baselines()
    has_both_baselines = (baselines.get("ac") is not None) and (
        baselines.get("battery") is not None
    )

    # 4. Evaluate preliminary dataset gates
    dataset_gates = evaluate_dataset_gates(
        train_rows=train_rows,
        test_rows=test_rows,
        test_sessions_count=test_sessions_count,
        class_counts=class_counts,
        power_states_train=power_states_train,
        has_both_baselines=has_both_baselines,
        validation_errors=val_errors,
    )

    console.print(
        f"Dataset loaded: {len(df):,} total rows "
        f"({train_rows:,} train, {test_rows:,} test across {test_sessions_count} sessions)."
    )

    # 5. Prepare Arrays for Training
    X_train = df_train[FEATURE_NAMES].to_numpy()
    y_train_reg = df_train["target_total_ratio"].to_numpy()
    groups_train = df_train["run_id"].to_numpy()

    X_test = df_test[FEATURE_NAMES].to_numpy()
    y_test_reg = df_test["target_total_ratio"].to_numpy()

    # Classification subsets
    train_clf_mask = df_train["issue_now"].isin(VALID_ISSUE_CLASSES).to_numpy()
    X_train_clf = X_train[train_clf_mask]
    y_train_clf = df_train.loc[train_clf_mask, "issue_now"].to_numpy()
    groups_train_clf = groups_train[train_clf_mask]

    test_clf_mask = df_test["issue_now"].isin(VALID_ISSUE_CLASSES).to_numpy()
    X_test_clf = X_test[test_clf_mask]
    y_test_clf = df_test.loc[test_clf_mask, "issue_now"].to_numpy()

    # Healthy subset for anomaly detector
    healthy_mask = (df_train["issue_now"] == "healthy").to_numpy()
    X_train_healthy = (
        X_train[healthy_mask] if healthy_mask.any() else X_train[: min(100, len(X_train))]
    )

    # 6. Train Regressors with full tuning grids
    console.print("Training regression models with grouped cross-validation...")
    reg_results = train_regressors(X_train, y_train_reg, groups_train, quick=quick)
    regressors = reg_results["models"]
    reg_cv = reg_results["cv_metrics"]

    # 7. Train Classifiers with full tuning grids
    console.print("Training multi-class fault classifiers with grouped cross-validation...")
    clf_results = train_classifiers(
        X_train_clf,
        y_train_clf,
        groups_train_clf,
        classes=VALID_ISSUE_CLASSES,
        quick=quick,
    )
    classifiers = clf_results["models"]
    clf_cv = clf_results["cv_metrics"]

    # 8. Train Anomaly Detector
    console.print("Training Isolation Forest anomaly detector on healthy telemetry...")
    anomaly_detector = train_anomaly_detector(X_train_healthy, feature_names=FEATURE_NAMES)

    # 9. Evaluate Regressors on Test Set
    reg_test_metrics: dict[str, dict[str, float]] = {}
    test_preds_reg: dict[str, np.ndarray] = {}

    for name, model in regressors.items():
        if hasattr(model, "predict"):
            preds = model.predict(X_test)
        else:
            preds = np.ones(len(X_test))
        test_preds_reg[name] = preds
        reg_test_metrics[name] = compute_regression_metrics(y_test_reg, preds)

    # 10. Evaluate Classifiers on Test Set
    clf_test_metrics: dict[str, dict[str, Any]] = {}
    test_preds_clf: dict[str, np.ndarray] = {}

    for name, model in classifiers.items():
        preds_enc = model.predict(X_test_clf)
        if np.issubdtype(preds_enc.dtype, np.integer):
            preds_str = np.array([VALID_ISSUE_CLASSES[i] for i in preds_enc])
        else:
            preds_str = preds_enc
        test_preds_clf[name] = preds_str
        clf_test_metrics[name] = compute_classification_metrics(
            y_test_clf, preds_str, classes=VALID_ISSUE_CLASSES
        )

    # 11. Evaluate Anomaly Detector on Test Set
    anom_scores = anomaly_detector.score(X_test)
    y_true_binary = (df_test["issue_now"] != "healthy").to_numpy().astype(int)
    is_combo = (df_test["issue_now"] == "combo").to_numpy()
    anom_metrics = compute_anomaly_metrics(
        anom_scores,
        y_true_binary,
        threshold=anomaly_detector.threshold,
        is_combo_mask=is_combo,
    )

    # 12. Evaluate Prediction Integrity Gates
    pred_gates = evaluate_prediction_gates(test_preds_reg)

    # Combine all gates
    all_gates = dataset_gates + pred_gates
    all_passed = all(g.passed for g in all_gates)

    # Render Quality Gates Table
    _format_gates_table(all_gates, plain=plain)

    # 13. Moving Block Bootstrap Evaluation vs Persistence
    run_ids_test = df_test["run_id"].to_numpy()
    bootstrap_results = compute_block_bootstrap(
        run_ids=run_ids_test,
        y_true_reg=y_test_reg,
        preds_reg=test_preds_reg,
        n_resamples=1000,
        seed=42,
    )

    # 14. Assemble Test Predictions Dataframe
    df_test_preds = df_test.copy()
    for name, preds in test_preds_reg.items():
        df_test_preds[f"pred_reg_{name}"] = preds
    df_test_preds["pred_anomaly_score"] = anom_scores
    df_test_preds["pred_is_anomaly"] = anom_scores >= anomaly_detector.threshold

    # Determine primary models via CV score on training split only
    candidate_regs = [m for m in ("Ridge", "RandomForest", "XGBoost") if m in reg_cv]
    primary_reg = min(candidate_regs, key=lambda m: reg_cv[m]) if candidate_regs else "XGBoost"

    candidate_clfs = [m for m in ("LogisticRegression", "RandomForest", "XGBoost") if m in clf_cv]
    primary_clf = max(candidate_clfs, key=lambda m: clf_cv[m]) if candidate_clfs else "XGBoost"

    # 15. Save Model Set into User Models Directory
    cv_metrics = {"regression": reg_cv, "classification": clf_cv}
    test_metrics = {
        "regression": reg_test_metrics,
        "classification": clf_test_metrics,
        "anomaly": anom_metrics,
    }

    sessions_total = (
        int(df["session_id"].nunique()) if "session_id" in df.columns else test_sessions_count + 2
    )
    machines_total = int(df["machine_id"].nunique()) if "machine_id" in df.columns else 1

    model_dir = save_model_set(
        regressors=regressors,
        classifiers=classifiers,
        anomaly_detector=anomaly_detector,
        classes=VALID_ISSUE_CLASSES,
        cv_metrics=cv_metrics,
        test_metrics=test_metrics,
        dataset_sha256=dataset_sha256,
        df_test_predictions=df_test_preds,
        synthetic=is_dataset_synthetic,
        horizon_s=DEFAULT_HORIZON_S,
        primary_models={"regressor": primary_reg, "classifier": primary_clf},
        X_train=X_train,
        train_rows=train_rows,
        test_rows=test_rows,
        sessions_count=sessions_total,
        machines_count=machines_total,
        power_states=power_states_train,
        gates_passed=all_passed,
    )

    # Build report_data.json
    build_report_data(model_dir)

    console.print(f"Model set persisted to: {model_dir}")

    # 16. Print Leaderboard Summary
    p_comp = bootstrap_results.get(primary_reg, {})
    delta_ci = p_comp.get("delta_vs_persistence_ci", (0.0, 0.0))
    beats_pers = p_comp.get("beats_persistence", False)
    status_str = (
        "Superior (Interval strictly positive)"
        if beats_pers
        else "Not distinguishable by interval rule"
    )

    ci_low = p_comp.get("mae_ci", (0, 0))[0]
    ci_high = p_comp.get("mae_ci", (0, 0))[1]
    p_mae = reg_test_metrics.get(primary_reg, {}).get("mae", 0.0)
    console.print(
        f"\nPrimary Regressor: {primary_reg} | MAE: {p_mae:.4f} "
        f"(95% CI: [{ci_low:.4f}, {ci_high:.4f}])\n"
        f"Persistence Delta CI: [{delta_ci[0]:.4f}, {delta_ci[1]:.4f}] -> {status_str}"
    )

    # 17. Export Handling
    if export:
        if not all_passed:
            console.print("\n[bold red]EXPORT REFUSAL:[/bold red] Quality gate check failed.")
            for g in all_gates:
                if not g.passed:
                    console.print(f"  - {g.name}: {g.reason}")
            console.print(
                "\nModel set was saved to user data directory but will NOT be exported "
                "to bundled defaults until all gates pass."
            )
            # Clean up temporary dataset
            if temp_dataset_path.exists():
                temp_dataset_path.unlink(missing_ok=True)
            if manifest_path.exists():
                manifest_path.unlink(missing_ok=True)
            raise typer.Exit(code=EXIT_ERROR)

        # Copy to bundled models directory
        target_dir = get_bundled_model_dir()
        target_dir.mkdir(parents=True, exist_ok=True)

        for item in model_dir.iterdir():
            if item.is_file():
                shutil.copy2(item, target_dir / item.name)

        # Update card in target_dir
        card_file = target_dir / "model_card.json"
        with open(card_file, encoding="utf-8") as f:
            card_data = json.load(f)

        card_data["synthetic"] = is_dataset_synthetic
        card_data["git_commit"] = get_git_commit()
        card_data["gates_passed"] = True

        # Also update report_data.json inside target_dir before computing hashes
        build_report_data(target_dir)

        # Recompute model file hashes
        model_hashes: dict[str, str] = {}
        for fpath in sorted(target_dir.iterdir()):
            if fpath.is_file() and fpath.name != "model_card.json":
                model_hashes[fpath.name] = compute_file_sha256(fpath)

        card_data["models_sha256"] = model_hashes

        with open(card_file, "w", encoding="utf-8") as f:
            json.dump(card_data, f, indent=2)

        console.print(
            f"\n[green]Model set successfully exported to:[/green] {target_dir}\n"
            f"Provenance: synthetic={is_dataset_synthetic}, dataset_sha256={dataset_sha256[:8]}, "
            f"train_rows={train_rows:,}, test_rows={test_rows:,}"
        )

    # Clean up temporary dataset files
    if temp_dataset_path.exists():
        temp_dataset_path.unlink(missing_ok=True)
    if manifest_path.exists():
        manifest_path.unlink(missing_ok=True)
