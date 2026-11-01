"""Command-line entry point for deskflow.

Commands for later phases (collect, train, infer, fix) are not registered
here yet. Only --version and the root help text are wired up.
"""

import typer

from deskflow import __version__

app = typer.Typer(
    name="deskflow",
    help=(
        "Device-care tool: samples system telemetry, measures performance, "
        "predicts sluggishness 30 seconds ahead, and suggests safe fixes."
    ),
    add_completion=False,
)


def _version_callback(value: bool) -> None:
    """Print version and exit when --version is passed."""
    if value:
        typer.echo(__version__)
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(  # noqa: A002 -- shadowing builtin is intentional here
        False,
        "--version",
        callback=_version_callback,
        is_eager=True,
        help="Print version and exit.",
    ),
) -> None:
    """deskflow: command-line device-care and performance prediction."""
