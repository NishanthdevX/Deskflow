"""Tools subcommand group: advanced diagnostics, maintenance, and verification."""

from __future__ import annotations

from pathlib import Path

import typer
from rich import box
from rich.console import Console
from rich.table import Table

from deskflow.commands.baseline import app as baseline_app
from deskflow.commands.build_model import build_model_command
from deskflow.commands.check import app as check_app
from deskflow.commands.collect import app as collect_app
from deskflow.commands.dataset import app as dataset_app
from deskflow.commands.doctor import app as doctor_app
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
from deskflow.exit_codes import EXIT_ERROR, EXIT_OK
from deskflow.truth import load_provenance, verify_model_hashes

app = typer.Typer(
    name="tools",
    help="Advanced diagnostic, laboratory, and developer tools.",
    invoke_without_command=True,
    add_completion=False,
)

TOOLS_REGISTRY: list[tuple[str, str]] = [
    ("selftest", "Run automated end-to-end verification self-test."),
    ("build-model", "Build, validate, tune, and export models from run directories."),
    ("verify", "Verify model artifact checksums against the model card."),
    ("scan", "Sample system telemetry, evaluate models, and compute health score."),
    ("watch", "Live accuracy verification against future 20s horizons."),
    ("check", "Guided 5-step health evaluation: info, scan, fix, watch, report."),
    ("report", "Generate verification report across terminal, Markdown, HTML, and JSON."),
    ("prep", "Run hardware readiness preflight and baseline calibration."),
    ("baseline", "Calibrate and manage hardware latency baselines."),
    ("collect", "Record background telemetry and probe datasets."),
    ("dataset", "Build, inspect, and validate training datasets."),
    ("lab", "Run controlled fault injection experiments."),
    ("models", "Inspect registered model artifacts and metadata."),
    ("doctor", "Verify platform capabilities and dependency health."),
    ("snapshot", "Capture instantaneous telemetry snapshot."),
    ("bugreport", "Generate sanitized local diagnostic report."),
    ("privacy", "Display offline privacy commitments and storage paths."),
    ("data", "Inspect local disk usage across Deskflow data folders."),
    ("uninstall", "Purge local Deskflow data directories."),
]


@app.callback()
def tools_default(ctx: typer.Context) -> None:
    """List available advanced tools when invoked without subcommands."""
    if ctx.invoked_subcommand is None:
        console = Console()
        table = Table(
            title="[DESKFLOW ADVANCED TOOLS]",
            box=box.ROUNDED,
            expand=True,
        )
        table.add_column("Tool", style="bold cyan", width=14)
        table.add_column("Description")
        for name, desc in TOOLS_REGISTRY:
            table.add_row(name, desc)
        console.print(table)
        console.print(
            "\n[dim]Run 'deskflow tools <tool> --help' or 'deskflow <tool>' for details.[/dim]"
        )


@app.command("selftest", help="Run automated end-to-end verification self-test.")
def selftest(
    plain: bool = typer.Option(False, "--plain", help="Render pure ASCII without colors."),
) -> None:
    """Run automated end-to-end self-test replacing manual verification."""
    from deskflow.commands.selftest import selftest_command

    selftest_command(plain=plain)


@app.command("verify", help="Check model files on disk against card SHA-256 hashes.")
def verify_command(
    plain: bool = typer.Option(False, "--plain", help="Render pure ASCII without colors."),
) -> None:
    """Verify that model files on disk match the cryptographic hashes in the card."""
    console = Console(no_color=plain)
    prov = load_provenance()
    passed, errors = verify_model_hashes(prov.model_dir)

    if passed:
        n_files = len(prov.model_hashes)
        commit = prov.git_commit[:8] if prov.git_commit else "unknown"
        ds = prov.dataset_hash[:8] if prov.dataset_hash else "none"
        console.print(
            f"[bold green]PASS: Model files verified against card hashes[/bold green]\n"
            f"- Checked {n_files} artifacts in {prov.model_dir}\n"
            f"- Git commit: {commit}\n"
            f"- Dataset SHA-256: {ds}\n"
            f"- Synthetic demo flag: {prov.synthetic}"
        )
        raise typer.Exit(code=EXIT_OK)

    console.print("[bold red]FAIL: Model verification failed[/bold red]")
    for err in errors:
        console.print(f"- [red]{err}[/red]")
    raise typer.Exit(code=EXIT_ERROR)


@app.command(
    "scan",
    help="Sample system telemetry, evaluate ML models, compute health score, and suggest fixes.",
)
def scan_cmd(
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
    fail_on: str | None = typer.Option(
        None,
        "--fail-on",
        help="Exit 1 if health score status reaches 'fair' or 'poor'.",
    ),
) -> None:
    """Run an end-to-end device scan, evaluate health score, and recommend fixes."""
    from deskflow.commands.scan import scan_command

    scan_command(
        seconds=seconds,
        output_json=json,
        no_storage=no_storage,
        timing=timing,
        replay=Path(replay) if replay else None,
        yes=yes,
        plain=plain,
        fail_on=fail_on,
    )


# Attach remaining tools as subcommands under 'deskflow tools <tool>'
app.command("build-model")(build_model_command)
app.add_typer(watch_app, name="watch")
app.add_typer(check_app, name="check")
app.add_typer(report_app, name="report")
app.add_typer(prep_app, name="prep")
app.add_typer(baseline_app, name="baseline")
app.add_typer(collect_app, name="collect")
app.add_typer(dataset_app, name="dataset")
app.add_typer(lab_app, name="lab")
app.add_typer(models_app, name="models")
app.add_typer(doctor_app, name="doctor")
app.add_typer(snapshot_app, name="snapshot")
app.add_typer(app_bugreport, name="bugreport")
app.add_typer(app_privacy, name="privacy")
app.add_typer(app_data, name="data")
app.add_typer(app_uninstall, name="uninstall")
