"""Command-line entry point for deskflow."""

import sys

from deskflow import __version__

if any(arg in ("--version", "-V") for arg in sys.argv[1:]):
    print(f"deskflow {__version__}")
    sys.exit(0)

import typer

from deskflow.commands.baseline import app as baseline_app
from deskflow.commands.collect import app as collect_app
from deskflow.commands.dataset import app as dataset_app
from deskflow.commands.doctor import app as doctor_app
from deskflow.commands.lab import app as lab_app
from deskflow.commands.models import models_app
from deskflow.commands.snapshot import app as snapshot_app

app = typer.Typer(
    name="deskflow",
    help=(
        "Device-care tool: samples system telemetry, measures performance, "
        "predicts sluggishness 30 seconds ahead, and suggests safe fixes."
    ),
    add_completion=False,
)

# Register subcommands
app.add_typer(snapshot_app, name="snapshot")
app.add_typer(collect_app, name="collect")
app.add_typer(doctor_app, name="doctor")
app.add_typer(baseline_app, name="baseline")
app.add_typer(lab_app, name="lab")
app.add_typer(dataset_app, name="dataset")
app.add_typer(models_app, name="models")


@app.command(
    "train",
    help="Train regression, classification, and anomaly models on engineered datasets.",
)
def train(
    datasets: list[str] = typer.Option(
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
    id: str | None = typer.Option(
        None,
        "--id",
        help="Custom model set identifier.",
    ),
) -> None:
    """Train regression, classification, and anomaly models on engineered datasets."""
    from pathlib import Path

    from deskflow.commands.train import train_command

    train_command(datasets=[Path(p) for p in datasets], quick=quick, out_id=id)


@app.command(
    "evaluate",
    help="Evaluate trained model sets on test split and generate verification reports.",
)
def evaluate(
    model_set: str = typer.Option(
        "latest",
        "--model-set",
        "-m",
        help="Model directory path or model set ID (default: 'latest').",
    ),
    plots: bool = typer.Option(
        False,
        "--plots",
        help="Generate PNG diagnostic plots (requires train extra with matplotlib).",
    ),
    out: str | None = typer.Option(
        None,
        "--out",
        "-o",
        help="Path to write the Markdown evaluation report.",
    ),
) -> None:
    """Evaluate trained model sets on test split and generate verification reports."""
    from pathlib import Path

    from deskflow.commands.evaluate import evaluate_command

    evaluate_command(
        model_set=Path(model_set),
        plots=plots,
        out=Path(out) if out else None,
    )


def _version_callback(value: bool) -> None:
    """Print version and exit when --version is passed."""
    if value:
        typer.echo(__version__)
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        False,
        "--version",
        callback=_version_callback,
        is_eager=True,
        help="Print version and exit.",
    ),
) -> None:
    """deskflow: command-line device-care and performance prediction."""


if __name__ == "__main__":
    app()
