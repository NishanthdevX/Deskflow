"""Guided end-to-end device health check: info -> scan -> fixes -> report."""

from __future__ import annotations

from collections.abc import Callable

import typer
from rich.console import Console

from deskflow.commands.fix import fix_command
from deskflow.commands.info import info_command
from deskflow.commands.report import report_command
from deskflow.commands.scan import scan_command
from deskflow.ui.theme import use_plain_output

app = typer.Typer(help="Execute guided full health check: info, scan, fix inventory, and report.")
console = Console()


def check_command(
    plain: bool = False,
    yes: bool = False,
    input_fn: Callable[[str], str] = input,
) -> None:
    """Execute guided 4-step workflow: info -> scan -> fix inventory -> report."""
    plain = use_plain_output(plain)
    step_console = Console(no_color=plain)

    # 1. System Info
    step_console.print("\n[bold cyan]=== STEP 1/4: SYSTEM INFO ===[/bold cyan]")
    try:
        info_command(json_output=False, plain=plain)
    except Exception as exc:
        step_console.print(f"[yellow]Info step warning: {exc}[/yellow]")

    if not yes:
        try:
            input_fn("\nPress Enter to continue to Health Scan...")
        except (EOFError, KeyboardInterrupt):
            return

    # 2. Quick Scan
    step_console.print("\n[bold cyan]=== STEP 2/4: HEALTH SCAN ===[/bold cyan]")
    try:
        scan_command(
            seconds=13,
            output_json=False,
            no_storage=True,
            timing=False,
            replay=None,
            yes=yes,
            plain=plain,
        )
    except Exception as exc:
        step_console.print(f"[yellow]Scan step warning: {exc}[/yellow]")

    if not yes:
        try:
            input_fn("\nPress Enter to continue to Fix Center...")
        except (EOFError, KeyboardInterrupt):
            return

    # 3. Fix Center Inventory
    step_console.print("\n[bold cyan]=== STEP 3/4: FIX CENTER INVENTORY ===[/bold cyan]")
    try:
        # Display inventory table with prompt or list mode
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

    if not yes:
        try:
            input_fn("\nPress Enter to continue to Report...")
        except (EOFError, KeyboardInterrupt):
            return

    # 4. Report
    step_console.print("\n[bold cyan]=== STEP 4/4: SUMMARY REPORT ===[/bold cyan]")
    try:
        report_command(plain=plain)
    except Exception:
        # Fallback placeholder if report dependencies / models not yet populated
        step_console.print(
            "[dim][REPORT] System report summary placeholder (Phase 12 report module).[/dim]"
        )

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
) -> None:
    """Run guided full device check."""
    if ctx.invoked_subcommand is None:
        check_command(plain=plain, yes=yes)
