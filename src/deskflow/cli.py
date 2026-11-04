"""Command-line entry point for deskflow."""

import platform
import shutil
import sys
from collections.abc import Callable

import psutil
from rich import box
from rich.panel import Panel

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
from deskflow.commands.info import app as info_app
from deskflow.commands.lab import app as lab_app
from deskflow.commands.models import models_app
from deskflow.commands.prep import app as prep_app
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
app.add_typer(prep_app, name="prep")
app.add_typer(info_app, name="info")


def show_home_menu(input_fn: Callable[[str], str] = input) -> None:
    """Display the interactive launcher and loop until quit per terminal design v2."""
    from deskflow.baseline import (
        get_baseline_age_str,
        get_current_power_state,
        load_baseline,
    )
    from deskflow.platform.factory import get_adapter

    console = Console()
    adapter = get_adapter()

    while True:
        # 1. Header Banner
        cols, _ = shutil.get_terminal_size(fallback=(100, 30))
        if cols >= 80:
            console.print(
                "[bright_cyan bold]"
                "  ____  _____ ____  _  __\n"
                " |  _ \\| ____/ ___|| |/ /\n"
                " | | | |  _| \\___ \\| ' / \n"
                " | |_| | |___ ___) | . \\ \n"
                " |____/|_____|____/|_|\\_\\"
                "[/bright_cyan bold]"
            )
            console.print(
                f"[dim]DESKFLOW v{__version__} - "
                f"Device health scan with ML slowdown prediction[/dim]\n"
            )
        else:
            console.print(f"[bright_cyan bold]DESKFLOW v{__version__}[/bright_cyan bold]\n")

        # 2. THIS MACHINE panel
        pwr_state = get_current_power_state(adapter)
        bat = adapter.battery()
        pwr_info = f"Battery ({bat.percent:.0f}%)" if (bat and not bat.plugged_in) else "AC Power"
        mem = psutil.virtual_memory()
        ram_gb = mem.total / (1024**3)
        cores = psutil.cpu_count(logical=True) or 1
        os_info = f"{platform.system()} {platform.release()}"

        try:
            b = load_baseline(power_state=pwr_state)
            b_state = f"Calibrated ({get_baseline_age_str(b)})"
        except Exception:
            b_state = "Uncalibrated"

        machine_str = (
            f"OS: {os_info}  |  CPU: {cores} cores  |  "
            f"RAM: {ram_gb:.1f} GB  |  Power: {pwr_info}  |  Baseline: {b_state}"
        )
        console.print(
            Panel(
                machine_str,
                title="[THIS MACHINE]",
                box=box.ROUNDED,
                border_style="grey50",
            )
        )

        # 3. Menu Options
        console.print("[bold cyan][1][/bold cyan] Full check   scan, fixes and report in one run")
        console.print("[bold cyan][2][/bold cyan] Quick scan   14 s health score and prediction")
        console.print("[bold cyan][3][/bold cyan] Fix Center   clean cache, free RAM, close tasks")
        console.print("[bold cyan][4][/bold cyan] Live watch   predictions vs reality, comparison")
        console.print("[bold cyan][5][/bold cyan] System info  CPU, GPU, memory, disks")
        console.print("[bold cyan][6][/bold cyan] Report       graphs and model comparison")
        console.print("[bold cyan][7][/bold cyan] Prepare demo readiness checklist")
        console.print("[dim][Q] Quit[/dim]\n")

        choice = input_fn("Select: ").strip()
        if choice.upper() in ("Q", "QUIT"):
            break
        elif choice == "1":
            from deskflow.commands.fix import fix_command
            from deskflow.commands.report import report_command
            from deskflow.commands.scan import scan_command

            scan_command()
            fix_command(dry_run=True)
            report_command()
        elif choice == "2":
            from deskflow.commands.scan import scan_command

            scan_command()
        elif choice == "3":
            from deskflow.commands.fix import fix_command

            fix_command(dry_run=True)
        elif choice == "4":
            from deskflow.commands.watch import watch_command

            watch_command()
        elif choice == "5":
            from deskflow.commands.info import info_command

            info_command()
        elif choice == "6":
            from deskflow.commands.report import report_command

            report_command()
        elif choice == "7":
            from deskflow.commands.prep import prep_command

            prep_command()


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
