"""Guided end-to-end device health check: 5-step workflow.

Steps:
1. System Info
2. Health Scan
3. Fix Center Inventory (with options to run tools)
4. Optional 60-second Live Accuracy Watch (default yes, --skip-watch to skip)
5. Summary Report (Terminal comparison charts + HTML export)
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import typer
from rich.console import Console

from deskflow.commands.fix import fix_command
from deskflow.commands.info import info_command
from deskflow.commands.report import generate_report
from deskflow.commands.scan import scan_command
from deskflow.commands.watch import watch_command
from deskflow.ui.theme import use_plain_output

app = typer.Typer(
    help="Execute guided full health check: info -> scan -> fixes -> watch -> report."
)
console = Console()


def check_command(
    plain: bool = False,
    yes: bool = False,
    skip_watch: bool = False,
    no_input: bool = False,
    html: Path | None = None,
    scan_seconds: int = 13,
    input_fn: Callable[[str], str] = input,
) -> None:
    """Execute guided 5-step workflow: info -> scan -> fix -> watch -> report."""
    plain = use_plain_output(plain)
    step_console = Console(no_color=plain)

    # 1. System Info
    step_console.print("\n[bold cyan]=== STEP 1/5: SYSTEM INFO ===[/bold cyan]")
    try:
        info_command(json_output=False, plain=plain)
    except Exception as exc:
        step_console.print(f"[yellow]Info step warning: {exc}[/yellow]")

    if not yes and not no_input:
        try:
            input_fn("\nPress Enter to continue to Health Scan...")
        except (EOFError, KeyboardInterrupt):
            return

    # 2. Quick Health Scan
    step_console.print("\n[bold cyan]=== STEP 2/5: HEALTH SCAN ===[/bold cyan]")
    try:
        scan_command(
            seconds=scan_seconds,
            output_json=False,
            no_storage=True,
            timing=False,
            replay=None,
            yes=yes,
            plain=plain,
        )
    except Exception as exc:
        step_console.print(f"[yellow]Scan step warning: {exc}[/yellow]")

    if not yes and not no_input:
        try:
            input_fn("\nPress Enter to continue to Fix Center...")
        except (EOFError, KeyboardInterrupt):
            return

    # 3. Fix Center Inventory
    step_console.print("\n[bold cyan]=== STEP 3/5: FIX CENTER INVENTORY ===[/bold cyan]")
    try:
        fix_command(
            actions=[],
            tool=[],
            dry_run=False,
            yes=yes,
            include_caution=False,
            list_tools=True,
            json_output=False,
            no_rescan=True,
            force=False,
            plain=plain,
        )
    except Exception as exc:
        step_console.print(f"[yellow]Fix step warning: {exc}[/yellow]")

    # 4. Optional Live Accuracy Watch
    step_console.print("\n[bold cyan]=== STEP 4/5: LIVE ACCURACY WATCH ===[/bold cyan]")
    run_watch = not skip_watch
    if not skip_watch and not yes and not no_input:
        try:
            prompt = "\nRun 60-second live accuracy watch? [Y/n]: "
            choice = input_fn(prompt).strip().lower()
            if choice in ("n", "no"):
                run_watch = False
        except (EOFError, KeyboardInterrupt):
            run_watch = False

    if run_watch:
        try:
            step_console.print("[dim]Starting 60s live accuracy verification session...[/dim]")
            watch_command(duration=60, plain=plain, replay=None, fast=False)
        except Exception as exc:
            step_console.print(f"[yellow]Live watch step note: {exc}[/yellow]")
    else:
        step_console.print("[dim]Live accuracy watch skipped.[/dim]")

    if not yes and not no_input:
        try:
            input_fn("\nPress Enter to generate Summary Report...")
        except (EOFError, KeyboardInterrupt):
            return

    # 5. Summary Report
    step_console.print("\n[bold cyan]=== STEP 5/5: SUMMARY REPORT ===[/bold cyan]")
    html_target = html or Path("report.html")
    try:
        generate_report(html=html_target, plain=plain)
        step_console.print(
            f"\n[spring_green2 bold]HTML verification report written to: {html_target.resolve()}"
            "[/spring_green2 bold]"
        )
    except Exception as exc:
        step_console.print(f"[yellow]Report generation error: {exc}[/yellow]")

    step_console.print("\n[spring_green2 bold]Guided full check completed.[/spring_green2 bold]\n")


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    plain: bool = typer.Option(
        False,
        "--plain",
        help="Emit monochrome ASCII output.",
    ),
    yes: bool = typer.Option(
        False,
        "--yes",
        "-y",
        help="Automatically advance through guided steps without waiting for Enter.",
    ),
    skip_watch: bool = typer.Option(
        False,
        "--skip-watch",
        help="Skip the 60-second live accuracy watch session.",
    ),
    no_input: bool = typer.Option(
        False,
        "--no-input",
        help="Never prompt for user input; proceed automatically.",
    ),
    html: Path | None = typer.Option(
        None,
        "--html",
        help="Custom output path for the written HTML report.",
    ),
) -> None:
    """Run guided full device check: info -> scan -> fix -> watch -> report."""
    if ctx.invoked_subcommand is None:
        check_command(
            plain=plain,
            yes=yes,
            skip_watch=skip_watch,
            no_input=no_input,
            html=html,
        )
