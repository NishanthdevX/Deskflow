"""Models CLI command: list and inspect trained model sets."""

import logging
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from deskflow.ml.artifacts import (
    compute_directory_size_bytes,
    get_latest_model_id,
    list_model_sets,
    load_model_card,
)
from deskflow.paths import get_data_dir

logger = logging.getLogger(__name__)
console = Console()
models_app = typer.Typer(name="models", help="Manage trained machine learning model sets.")


@models_app.command("list")
def list_models() -> None:
    """List all available trained model sets with summary performance."""
    model_sets = list_model_sets()
    if not model_sets:
        console.print("No trained model sets found. Run 'deskflow train --dataset ...' first.")
        return

    table = Table(title="Trained Model Sets")
    table.add_column("Model ID", style="bold")
    table.add_column("Created (UTC)")
    table.add_column("Deskflow Ver")
    table.add_column("Synthetic")
    table.add_column("Reg MAE", justify="right")
    table.add_column("Clf Macro F1", justify="right")
    table.add_column("Latest")

    for ms in model_sets:
        card = ms.get("card", {})
        test_m = card.get("test_metrics", {})
        reg_m = test_m.get("regression", {})
        clf_m = test_m.get("classification", {})

        # Pick XGBoost or RandomForest or first available
        best_reg = (
            reg_m.get("XGBoost") or reg_m.get("RandomForest") or reg_m.get("Persistence") or {}
        )
        mae_str = f"{best_reg.get('mae', 0.0):.4f}" if "mae" in best_reg else "-"

        best_clf = (
            clf_m.get("XGBoost")
            or clf_m.get("RandomForest")
            or clf_m.get("LogisticRegression")
            or {}
        )
        f1_str = f"{best_clf.get('macro_f1', 0.0):.4f}" if "macro_f1" in best_clf else "-"

        table.add_row(
            ms["model_id"],
            ms["created_utc"][:19] if ms["created_utc"] else "-",
            ms["deskflow_version"],
            "Yes" if ms["synthetic"] else "No",
            mae_str,
            f1_str,
            "[bold green]yes[/bold green]" if ms["is_latest"] else "no",
        )

    console.print(table)


@models_app.command("show")
def show_model(
    model_id: str = typer.Argument(
        ...,
        help="Model set identifier or 'latest'",
    ),
) -> None:
    """Show detailed model card and performance statistics for a model set."""
    root = get_data_dir() / "models"
    target_id = model_id
    if target_id == "latest":
        latest = get_latest_model_id(root)
        if latest and (root / latest).exists():
            model_dir = root / latest
        else:
            from deskflow.ml.artifacts import get_bundled_model_dir

            bundled = get_bundled_model_dir()
            if bundled.exists() and (bundled / "model_card.json").exists():
                model_dir = bundled
            else:
                console.print("[red]Error: No 'latest' model set pointer exists.[/red]")
                raise typer.Exit(code=1)
    else:
        model_dir = root / target_id
    if not model_dir.exists():
        console.print(f"[red]Error: Model set '{target_id}' not found in {root}[/red]")
        raise typer.Exit(code=1)

    card = load_model_card(model_dir)
    size_mb = compute_directory_size_bytes(model_dir) / (1024 * 1024)

    console.print(f"\n[bold]Model Set ID:[/bold] {card.get('model_id')}")
    console.print(f"[bold]Path:[/bold] {model_dir} ({size_mb:.2f} MB)")
    console.print(f"[bold]Created UTC:[/bold] {card.get('created_utc')}")
    console.print(f"[bold]Deskflow Version:[/bold] {card.get('deskflow_version')}")
    console.print(f"[bold]Feature Set Version:[/bold] {card.get('feature_set_version')}")
    console.print(f"[bold]Horizon:[/bold] {card.get('horizon_s')} s")
    console.print(f"[bold]Synthetic:[/bold] {card.get('synthetic')}")
    console.print(f"[bold]Dataset SHA-256:[/bold] {card.get('dataset_sha256')}")

    thresholds = card.get("thresholds", {})
    console.print(
        f"[bold]Thresholds:[/bold] Slow ratio >= {thresholds.get('slow_threshold')}, "
        f"Anomaly score >= {thresholds.get('anomaly_threshold')}"
    )

    test_metrics = card.get("test_metrics", {})

    # Regression table
    reg_test = test_metrics.get("regression", {})
    if reg_test:
        table_reg = Table(title="Regression Test Metrics")
        table_reg.add_column("Model", style="bold")
        table_reg.add_column("MAE", justify="right")
        table_reg.add_column("RMSE", justify="right")
        table_reg.add_column("R2", justify="right")
        table_reg.add_column("Tol Acc", justify="right")
        table_reg.add_column("Slow F1", justify="right")
        for name, m in reg_test.items():
            table_reg.add_row(
                name,
                f"{m.get('mae', 0.0):.4f}",
                f"{m.get('rmse', 0.0):.4f}",
                f"{m.get('r2', 0.0):.4f}",
                f"{m.get('tolerance_accuracy', 0.0):.4f}",
                f"{m.get('slow_f1', 0.0):.4f}",
            )
        console.print(table_reg)

    # Classification table
    clf_test = test_metrics.get("classification", {})
    if clf_test:
        table_clf = Table(title="Classification Test Metrics")
        table_clf.add_column("Model", style="bold")
        table_clf.add_column("Accuracy", justify="right")
        table_clf.add_column("Balanced Acc", justify="right")
        table_clf.add_column("Macro F1", justify="right")
        for name, m in clf_test.items():
            table_clf.add_row(
                name,
                f"{m.get('accuracy', 0.0):.4f}",
                f"{m.get('balanced_accuracy', 0.0):.4f}",
                f"{m.get('macro_f1', 0.0):.4f}",
            )
        console.print(table_clf)

    # Anomaly summary
    anom_test = test_metrics.get("anomaly", {})
    if anom_test:
        table_anom = Table(title="Anomaly Detection Test Metrics")
        table_anom.add_column("Metric", style="bold")
        table_anom.add_column("Value", justify="right")
        for k, v in anom_test.items():
            table_anom.add_row(k, f"{v}")
        console.print(table_anom)


@models_app.command("export")
def export_models(
    to: Path = typer.Option(
        ...,
        "--to",
        help="Destination directory path for exported model set.",
    ),
    model_id: str = typer.Option(
        "latest",
        "--model-id",
        help="Model set ID to export, or 'latest'.",
    ),
    allow_synthetic: bool = typer.Option(
        False,
        "--allow-synthetic",
        help="Allow exporting synthetic model sets.",
    ),
) -> None:
    """Export a trained model set to a destination directory."""
    import shutil

    root = get_data_dir() / "models"
    target_id = model_id
    if target_id == "latest":
        latest = get_latest_model_id(root)
        if latest and (root / latest).exists():
            model_dir = root / latest
        else:
            from deskflow.ml.artifacts import get_bundled_model_dir

            bundled = get_bundled_model_dir()
            if bundled.exists() and (bundled / "model_card.json").exists():
                model_dir = bundled
            else:
                console.print("[red]Error: No 'latest' model set pointer exists.[/red]")
                raise typer.Exit(code=1)
    else:
        model_dir = root / target_id
    if not model_dir.exists():
        console.print(f"[red]Error: Model set '{target_id}' not found in {root}[/red]")
        raise typer.Exit(code=1)

    card = load_model_card(model_dir)
    is_synth = bool(card.get("synthetic", False))
    if is_synth and not allow_synthetic:
        console.print(
            "[red]Error: Model set is synthetic. "
            "Use --allow-synthetic to export synthetic models.[/red]"
        )
        raise typer.Exit(code=1)

    to.mkdir(parents=True, exist_ok=True)
    count = 0
    for item in model_dir.iterdir():
        if item.is_file():
            shutil.copy2(item, to / item.name)
            count += 1

    size_mb = compute_directory_size_bytes(to) / (1024 * 1024)
    console.print(f"[green]Successfully exported {count} files ({size_mb:.2f} MB) to {to}[/green]")
