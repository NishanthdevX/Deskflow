"""Command-line entry point for deskflow."""

import typer

from deskflow import __version__
from deskflow.commands.baseline import app as baseline_app
from deskflow.commands.collect import app as collect_app
from deskflow.commands.dataset import app as dataset_app
from deskflow.commands.doctor import app as doctor_app
from deskflow.commands.lab import app as lab_app
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
