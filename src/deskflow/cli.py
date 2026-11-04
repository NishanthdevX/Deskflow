"""Command-line entry point for deskflow."""

import sys
from collections.abc import Callable

from deskflow import __version__

if any(arg in ("--version", "-V") for arg in sys.argv[1:]):
    print(f"deskflow {__version__}")
    sys.exit(0)

import typer
from rich.console import Console

from deskflow.commands.baseline import app as baseline_app
from deskflow.commands.collect import app as collect_app
from deskflow.commands.dataset import app as dataset_app
from deskflow.commands.doctor import app as doctor_app
from deskflow.commands.lab import app as lab_app
from deskflow.commands.models import models_app
from deskflow.commands.report import app as report_app
from deskflow.commands.snapshot import app as snapshot_app
from deskflow.commands.watch import app as watch_app

app = typer.Typer(
    name="deskflow",
    help=(
        "Device-care tool: samples system telemetry, measures performance, "
        "predicts sluggishness 20 seconds ahead, and suggests safe fixes."
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
app.add_typer(watch_app, name="watch")
app.add_typer(report_app, name="report")


def show_home_menu(input_fn: Callable[[str], str] = input) -> None:
    """Display the lightweight interactive launcher and run one selected action."""
    console = Console()
    console.print("[bright_cyan bold]DESKFLOW[/bright_cyan bold]")
    console.print("[dim]Device health scan with ML slowdown prediction[/dim]")
    console.print(
        "1 Scan\n2 Fix (preview first)\n3 Watch (Live Accuracy)\n"
        "4 Doctor\n5 Baseline status\n6 Quit"
    )
    choice = input_fn("Select: ").strip()
    if choice == "1":
        from deskflow.commands.scan import scan_command

        scan_command()
    elif choice == "2":
        from deskflow.commands.fix import fix_command

        fix_command(dry_run=True)
    elif choice == "3":
        from deskflow.commands.watch import watch_command

        watch_command()
    elif choice == "4":
        from deskflow.commands.doctor import doctor_command

        doctor_command()
    elif choice == "5":
        from deskflow.commands.baseline import show_command

        show_command()


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


@app.command(
    "scan",
    help="Sample system telemetry, evaluate ML models, compute health score, and suggest fixes.",
)
def scan(
    seconds: int = typer.Option(
        14,
        "--seconds",
        "-s",
        help="Measuring duration in seconds (default 14, min 13).",
    ),
    json: bool = typer.Option(
        False,
        "--json",
        help="Output raw structured JSON schema instead of human-readable tables.",
    ),
    no_storage: bool = typer.Option(
        False,
        "--no-storage",
        help="Skip filesystem storage scan stage.",
    ),
    timing: bool = typer.Option(
        False,
        "--timing",
        help="Print detailed latency breakdown per stage.",
    ),
    replay: str | None = typer.Option(
        None,
        "--replay",
        help="Replay a previously recorded run JSONL file instead of live measuring.",
    ),
    yes: bool = typer.Option(
        False,
        "--yes",
        "-y",
        help="Automatically confirm any required baseline calibration.",
    ),
    plain: bool = typer.Option(False, "--plain", help="Use ASCII-only, uncoloured output."),
) -> None:
    """Run an end-to-end device scan, evaluate health score, and recommend fixes."""
    from pathlib import Path

    from deskflow.commands.scan import scan_command

    scan_command(
        seconds=seconds,
        output_json=json,
        no_storage=no_storage,
        timing=timing,
        replay=Path(replay) if replay else None,
        yes=yes,
        plain=plain,
    )


@app.command(
    "fix",
    help="Execute safe, user-confirmed remediation actions and verify impact with re-scan.",
)
def fix(
    action: list[str] = typer.Option(
        None,
        "--action",
        "-a",
        help="One or more actions to execute directly (clean_temp, close_idle_apps, trim_memory).",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Simulate actions and print expected impact without making changes.",
    ),
    yes: bool = typer.Option(
        False,
        "--yes",
        "-y",
        help="Bypass interactive selection and confirmation prompts.",
    ),
    no_rescan: bool = typer.Option(
        False,
        "--no-rescan",
        help="Skip post-fix verification scan.",
    ),
    plain: bool = typer.Option(False, "--plain", help="Use ASCII-only, uncoloured output."),
) -> None:
    """Execute safe remediation actions and verify impact with re-scan."""
    from deskflow.commands.fix import fix_command

    fix_command(
        actions=action,
        dry_run=dry_run,
        yes=yes,
        no_rescan=no_rescan,
        plain=plain,
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
    from deskflow.ui.theme import configure_stdout

    configure_stdout()
    if len(sys.argv) != 1 or not sys.stdin.isatty():
        return
    show_home_menu()


if __name__ == "__main__":
    app()
