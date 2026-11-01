"""Evaluation CLI command: performance reporting, bootstrap intervals, and diagnostic plots."""

import logging
from pathlib import Path
from typing import Any

import numpy as np
import typer
from rich.console import Console
from rich.panel import Panel

from deskflow.ml.artifacts import get_latest_model_id, load_model_card
from deskflow.ml.metrics import compute_block_bootstrap, compute_breakdowns
from deskflow.paths import get_data_dir

logger = logging.getLogger(__name__)
console = Console()


def _resolve_model_dir(model_set: Path) -> Path:
    """Resolve model directory from path, ID, or 'latest' keyword."""
    if model_set.exists() and model_set.is_dir():
        return model_set

    # Check under get_data_dir() / models
    models_root = get_data_dir() / "models"
    if str(model_set) == "latest":
        latest_id = get_latest_model_id(models_root)
        if not latest_id:
            raise ValueError("No trained models found (latest pointer does not exist).")
        p = models_root / latest_id
        if p.exists():
            return p
        raise ValueError(f"Latest model '{latest_id}' directory not found at {p}")

    p = models_root / model_set
    if p.exists():
        return p

    raise ValueError(f"Model directory or ID '{model_set}' could not be found.")


def generate_plots(df_test: Any, model_dir: Path, card: dict[str, Any]) -> list[Path]:
    """Generate diagnostic plots using matplotlib if installed."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        console.print(
            "[yellow]Notice: 'matplotlib' not installed. Diagnostic plots skipped. "
            "Install optional extra with: uv sync --extra train[/yellow]"
        )
        return []

    plots_dir = model_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    generated: list[Path] = []

    y_true = df_test["target_total_ratio"].to_numpy()
    pred_col = (
        "pred_reg_XGBoost" if "pred_reg_XGBoost" in df_test.columns else "pred_reg_RandomForest"
    )

    if pred_col in df_test.columns:
        y_pred = df_test[pred_col].to_numpy()

        # 1. Predicted vs Actual
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(y_true, y_pred, alpha=0.5, edgecolors="none")
        min_v = min(float(np.min(y_true)), float(np.min(y_pred)))
        max_v = max(float(np.max(y_true)), float(np.max(y_pred)))
        ax.plot([min_v, max_v], [min_v, max_v], "r--", label="Identity (Perfect)")
        ax.set_xlabel("Actual Ratio")
        ax.set_ylabel("Predicted Ratio")
        ax.set_title(f"Predicted vs Actual ({pred_col})")
        ax.legend()
        p1 = plots_dir / "predicted_vs_actual.png"
        fig.tight_layout()
        fig.savefig(p1, dpi=150)
        plt.close(fig)
        generated.append(p1)

        # 2. Residuals
        fig, ax = plt.subplots(figsize=(6, 5))
        residuals = y_pred - y_true
        ax.scatter(y_pred, residuals, alpha=0.5, edgecolors="none")
        ax.axhline(0, color="r", linestyle="--")
        ax.set_xlabel("Predicted Ratio")
        ax.set_ylabel("Residual (Pred - True)")
        ax.set_title("Residual Plot")
        p2 = plots_dir / "residuals.png"
        fig.tight_layout()
        fig.savefig(p2, dpi=150)
        plt.close(fig)
        generated.append(p2)

    # 3. Model Comparison Bar Chart
    reg_metrics = card.get("test_metrics", {}).get("regression", {})
    if reg_metrics:
        models = list(reg_metrics.keys())
        maes = [reg_metrics[m]["mae"] for m in models]

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(models, maes, color="steelblue")
        ax.set_ylabel("Test MAE")
        ax.set_title("Model Comparison (Lower is Better)")
        plt.xticks(rotation=20)
        p3 = plots_dir / "model_comparison.png"
        fig.tight_layout()
        fig.savefig(p3, dpi=150)
        plt.close(fig)
        generated.append(p3)

    return generated


def evaluate_command(
    model_set: Path = typer.Option(
        Path("latest"),
        "--model-set",
        "-m",
        help="Model directory path or model set ID (default: 'latest').",
    ),
    plots: bool = typer.Option(
        False,
        "--plots",
        help="Generate PNG diagnostic plots (requires train extra with matplotlib).",
    ),
    out: Path | None = typer.Option(
        None,
        "--out",
        "-o",
        help="Path to write the Markdown evaluation report.",
    ),
) -> None:
    """Evaluate trained model sets on test split and generate verification reports."""
    import pandas as pd

    try:
        model_dir = _resolve_model_dir(model_set)
    except ValueError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(code=1) from e

    card = load_model_card(model_dir)
    is_synthetic = bool(card.get("synthetic", False))

    if is_synthetic:
        console.print(
            Panel.fit(
                "[bold red]SYNTHETIC DATA - NOT A RESULT[/bold red]",
                border_style="red",
            )
        )

    pred_parquet = model_dir / "predictions_test.parquet"
    if not pred_parquet.exists():
        console.print(f"[red]Error: Test predictions file missing at {pred_parquet}[/red]")
        raise typer.Exit(code=1)

    df_test = pd.read_parquet(pred_parquet)

    # Prepare data for bootstrap and breakdowns
    y_true_reg = df_test["target_total_ratio"].to_numpy()
    run_ids = df_test["run_id"].to_numpy()

    preds_reg: dict[str, np.ndarray] = {}
    for col in df_test.columns:
        if col.startswith("pred_reg_"):
            name = col.replace("pred_reg_", "")
            preds_reg[name] = df_test[col].to_numpy()

    # Bootstrap calculations
    bootstrap_results = compute_block_bootstrap(
        run_ids=run_ids,
        y_true_reg=y_true_reg,
        preds_reg=preds_reg,
        n_resamples=300,
        seed=42,
    )

    # Breakdowns
    if "XGBoost" in preds_reg:
        primary_pred = preds_reg["XGBoost"]
    elif "RandomForest" in preds_reg:
        primary_pred = preds_reg["RandomForest"]
    else:
        primary_pred = y_true_reg
    breakdowns = compute_breakdowns(df_test, y_true_reg=y_true_reg, y_pred_reg=primary_pred)

    # Optional plots
    plot_files: list[Path] = []
    if plots:
        plot_files = generate_plots(df_test, model_dir, card)
        if plot_files:
            console.print(f"Generated {len(plot_files)} diagnostic plots in {model_dir / 'plots'}")

    # Build Markdown Report
    test_metrics = card.get("test_metrics", {})
    reg_test = test_metrics.get("regression", {})
    clf_test = test_metrics.get("classification", {})
    anom_test = test_metrics.get("anomaly", {})

    lines: list[str] = []
    if is_synthetic:
        lines.append("> **SYNTHETIC DATA - NOT A RESULT**\n")

    lines.append(f"# Model Evaluation Report: {card.get('model_id')}\n")
    lines.append(
        f"**Generated:** {card.get('created_utc')} | "
        f"**Deskflow Version:** {card.get('deskflow_version')} | "
        f"**Feature Set Version:** {card.get('feature_set_version')}\n"
    )

    lines.append("## 1. Dataset & Protocol Summary\n")
    lines.append(f"- **Test Rows:** {len(df_test)}")
    lines.append(f"- **Unique Test Runs:** {df_test['run_id'].nunique()}")
    lines.append(f"- **Unique Machines:** {df_test['machine_id'].nunique()}")
    lines.append(f"- **Horizon:** {card.get('horizon_s', 20.0)} s")
    lines.append(f"- **Dataset SHA-256:** `{card.get('dataset_sha256')}`")
    lines.append(
        "- **Evaluation Protocol:** GroupKFold on run_id; test split evaluated once; "
        "300-sample block bootstrap by run_id for 95% confidence intervals.\n"
    )

    lines.append("## 2. Regression Leaderboard (target_total_ratio)\n")
    lines.append(
        "| Model | Type | Test MAE [95% CI] | Test RMSE | R2 [95% CI] | "
        "Tol Acc (<0.25) | Slow F1 (>=1.5) | Improvement over Persistence [95% CI] | Verdict |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|")

    reg_boot = bootstrap_results.get("regression", {})
    for name in [
        "ConstantMedian",
        "Persistence",
        "RollingMean",
        "Ridge",
        "RandomForest",
        "XGBoost",
    ]:
        if name in reg_test:
            m = reg_test[name]
            is_base = name in ("ConstantMedian", "Persistence", "RollingMean")
            m_type = "Baseline" if is_base else "Model"

            b_info = reg_boot.get(name, {})
            mae_ci = b_info.get("mae_ci", [m["mae"], m["mae"]])
            r2_ci = b_info.get("r2_ci", [m["r2"], m["r2"]])
            imp_ci = b_info.get("improvement_over_persistence_ci", [0.0, 0.0])
            better = b_info.get("better_than_persistence", False)
            verdict = "Better" if better else ("Baseline" if is_base else "No Improvement")

            lines.append(
                f"| {name} | {m_type} | {m['mae']:.4f} [{mae_ci[0]:.4f}, {mae_ci[1]:.4f}] | "
                f"{m['rmse']:.4f} | {m['r2']:.4f} [{r2_ci[0]:.4f}, {r2_ci[1]:.4f}] | "
                f"{m['tolerance_accuracy']:.4f} | {m['slow_f1']:.4f} | "
                f"[{imp_ci[0]:.4f}, {imp_ci[1]:.4f}] | {verdict} |"
            )
    lines.append("")

    lines.append("## 3. Classification Leaderboard (issue_now)\n")
    lines.append("| Model | Type | Accuracy | Balanced Accuracy | Macro F1 |")
    lines.append("|---|---|---|---|---|")
    for name in ["Dummy", "LogisticRegression", "RandomForest", "XGBoost"]:
        if name in clf_test:
            m = clf_test[name]
            is_base = name == "Dummy"
            m_type = "Baseline" if is_base else "Model"
            lines.append(
                f"| {name} | {m_type} | {m['accuracy']:.4f} | "
                f"{m['balanced_accuracy']:.4f} | {m['macro_f1']:.4f} |"
            )
    lines.append("")

    # Best classifier per-class and confusion matrix
    best_clf_name = (
        "XGBoost"
        if "XGBoost" in clf_test
        else ("RandomForest" if "RandomForest" in clf_test else None)
    )
    if best_clf_name and best_clf_name in clf_test:
        best_m = clf_test[best_clf_name]
        lines.append(f"### Per-Class Statistics ({best_clf_name})\n")
        lines.append("| Class | Precision | Recall | F1-Score | Support |")
        lines.append("|---|---|---|---|---|")
        for cls_name, stats in best_m.get("per_class", {}).items():
            lines.append(
                f"| {cls_name} | {stats['precision']:.4f} | {stats['recall']:.4f} | "
                f"{stats['f1']:.4f} | {stats['support']} |"
            )
        lines.append("")

        lines.append(f"### Confusion Matrix ({best_clf_name})\n")
        classes = best_m.get("classes", [])
        cm = best_m.get("confusion_matrix", [])
        if classes and cm:
            header = "| Actual \\ Predicted | " + " | ".join(classes) + " |"
            sep = "|---| " + " | ".join(["---"] * len(classes)) + " |"
            lines.append(header)
            lines.append(sep)
            for i, row in enumerate(cm):
                row_str = " | ".join(str(val) for val in row)
                lines.append(f"| **{classes[i]}** | {row_str} |")
            lines.append("")

    lines.append("## 4. Anomaly Detection (IsolationForest)\n")
    lines.append(f"- **Threshold (99th percentile healthy):** {anom_test.get('threshold', 'N/A')}")
    lines.append(f"- **ROC-AUC:** {anom_test.get('roc_auc', 'N/A')}")
    lines.append(f"- **Recall on Anomalies:** {anom_test.get('recall', 'N/A')}")
    lines.append(f"- **False Alarm Rate (Healthy):** {anom_test.get('false_alarm_rate', 'N/A')}")
    lines.append(f"- **Combo Detection Rate:** {anom_test.get('combo_detection_rate', 'N/A')}\n")

    lines.append("## 5. Performance Breakdowns\n")
    for category in ["scenario", "intensity", "pattern"]:
        sub = breakdowns.get(category, {})
        if sub:
            lines.append(f"### Breakdown by {category.title()}\n")
            lines.append(f"| {category.title()} | Sample Count | Regression MAE |")
            lines.append("|---|---|---|")
            for val, stats in sub.items():
                lines.append(f"| {val} | {int(stats['count'])} | {stats['mae']:.4f} |")
            lines.append("")

    lines.append("## 6. Caveats & Validity\n")
    if is_synthetic:
        lines.append(
            "- **Synthetic Data Notice:** All metrics in this report were computed on synthetic "
            "benchmarks designed for functional verification and pipeline testing. "
            "They do not represent real-world physical workload distributions."
        )
    if df_test["machine_id"].nunique() == 1:
        lines.append(
            "- **Single Machine:** The dataset includes data from only 1 hardware machine. "
            "Generalization to other CPU/memory topologies is unverified."
        )
    if len(df_test) < 200:
        lines.append(
            f"- **Sample Size:** Test split has only {len(df_test)} rows (< 200 rows). "
            "Confidence intervals may be broad."
        )
    lines.append(
        "- **Horizon:** Projections are valid strictly for H = 20.0 seconds into the future."
    )

    report_content = "\n".join(lines) + "\n"

    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            f.write(report_content)
        console.print(f"Evaluation report written to [cyan]{out}[/cyan]")
    else:
        console.print(report_content)
