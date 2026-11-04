"""Command-line entry point for deskflow."""

import logging
import platform
import shutil
import sys
from collections.abc import Callable
from typing import Any

import psutil

from deskflow import __version__
from deskflow.logging_config import get_log_file_path, setup_logging

if any(arg in ("--version", "-V") for arg in sys.argv[1:]):
    print(f"deskflow {__version__}")
    sys.exit(0)

import typer
from rich.console import Console

from deskflow.commands.baseline import app as baseline_app
from deskflow.commands.check import app as check_app
from deskflow.commands.collect import app as collect_app
from deskflow.commands.dataset import app as dataset_app
from deskflow.commands.doctor import app as doctor_app
from deskflow.commands.fix import app as fix_app
from deskflow.commands.info import app as info_app
from deskflow.commands.lab import app as lab_app
from deskflow.commands.models import models_app
from deskflow.commands.prep import app as prep_app
from deskflow.commands.report import app as report_app
from deskflow.commands.snapshot import app as snapshot_app
from deskflow.commands.support import (
    app_bugreport,
    app_data,
    app_privacy,
    app_uninstall,
)
from deskflow.commands.watch import app as watch_app


class DeskflowTyper(typer.Typer):
    """Custom Typer subclass providing top-level uncaught exception handling."""

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        try:
            return super().__call__(*args, **kwargs)
        except SystemExit:
            raise
        except Exception as exc:
            if "--debug" in sys.argv:
                raise
            logger = logging.getLogger("deskflow")
            logger.exception("Uncaught exception in CLI: %s", exc)
            log_path = get_log_file_path()
            sys.stderr.write(
                f"\nError: {exc}\n"
                f"Detailed diagnostic logs written to: {log_path}\n"
                "Run 'deskflow bugreport' to generate a sanitized diagnostic report.\n"
            )
            sys.exit(1)


app = DeskflowTyper(
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
app.add_typer(check_app, name="check")
app.add_typer(fix_app, name="fix")
app.add_typer(app_bugreport, name="bugreport")
app.add_typer(app_privacy, name="privacy")
app.add_typer(app_data, name="data")
app.add_typer(app_uninstall, name="uninstall")


def show_home_menu(input_fn: Callable[[str], str] = input) -> None:
    """Display the interactive launcher and loop until quit per terminal design v2."""
    from deskflow.baseline import (
        get_baseline_age_str,
        get_current_power_state,
        load_baseline,
    )
    from deskflow.platform.factory import get_adapter
    from deskflow.ui.components import banner, domain_panel
    from deskflow.ui.theme import use_plain_output

    console = Console()
    adapter = get_adapter()
    plain = use_plain_output()

    while True:
        # 1. Header Banner
        cols, _ = shutil.get_terminal_size(fallback=(100, 30))
        console.print(banner(version=__version__, width=cols, plain=plain))

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
            domain_panel(
                "[THIS MACHINE]",
                machine_str,
                domain="system",
                plain=plain,
            )
        )

        # 3. Menu Options matching docs/DESIGN.md v2 exactly
        console.print("[bold cyan][1][/bold cyan] Full check   scan, fixes and report in one run")
        console.print("[bold cyan][2][/bold cyan] Quick scan   14 s health score and prediction")
        console.print(
            "[bold cyan][3][/bold cyan] Fix Center   clean cache, free memory, close heavy tasks"
        )
        console.print(
            "[bold cyan][4][/bold cyan] Live watch   predictions versus reality, model comparison"
        )
        console.print("[bold cyan][5][/bold cyan] System info  CPU, GPU, memory, disks")
        console.print("[bold cyan][6][/bold cyan] Report       graphs and model comparison")
        console.print("[bold cyan][7][/bold cyan] Prepare demo readiness checklist")
        console.print("[dim][Q] Quit[/dim]\n")

        try:
            choice = input_fn("Select: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if choice.upper() in ("Q", "QUIT"):
            break
        elif choice == "1":
            from deskflow.commands.check import check_command

            check_command(input_fn=input_fn)
        elif choice == "2":
            from deskflow.commands.scan import scan_command

            scan_command()
        elif choice == "3":
            from deskflow.commands.fix import fix_command

            fix_command(dry_run=False)
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
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable verbose diagnostic logging.",
    ),
    debug: bool = typer.Option(
        False,
        "--debug",
        help="Enable full stack traces on uncaught exceptions.",
    ),
    no_input: bool = typer.Option(
        False,
        "--no-input",
        help="Disable interactive user prompts across commands.",
    ),
) -> None:
    """deskflow: command-line device-care and performance prediction."""
    from deskflow.ui.theme import configure_stdout

    configure_stdout()
    setup_logging(verbose=verbose, debug=debug)
    if len(sys.argv) != 1 or not sys.stdin.isatty():
        return
    show_home_menu()


if __name__ == "__main__":
    app()
