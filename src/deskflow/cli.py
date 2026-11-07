"""Main CLI entry point for Deskflow."""

from __future__ import annotations

import contextlib
import logging
import platform
import sys
from collections.abc import Callable
from pathlib import Path

import psutil
import typer
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from deskflow import __version__
from deskflow.baseline import get_baseline_age_str, load_baseline
from deskflow.exit_codes import (
    EXIT_USAGE,
)
from deskflow.logging_config import setup_logging

logger = logging.getLogger(__name__)

# Initialize CLI application
app = typer.Typer(
    name="deskflow",
    help=(
        "Deskflow measures real device responsiveness via lightweight micro-benchmarks, "
        "predicts sluggishness 20 seconds ahead, and suggests safe fixes."
    ),
    add_completion=False,
    no_args_is_help=False,
)

# Lazy command registration — imports intentionally placed after app definition
from deskflow.commands.fix import app as fix_app  # noqa: E402
from deskflow.commands.info import app as info_app  # noqa: E402
from deskflow.commands.support import (  # noqa: E402
    app_bugreport,
    app_data,
    app_privacy,
    app_uninstall,
)
from deskflow.commands.tools import app as tools_app  # noqa: E402

app.add_typer(fix_app, name="fix")
app.add_typer(info_app, name="info")
app.add_typer(tools_app, name="tools")
app.add_typer(app_bugreport, name="bugreport", hidden=True)
app.add_typer(app_privacy, name="privacy", hidden=True)
app.add_typer(app_data, name="data", hidden=True)
app.add_typer(app_uninstall, name="uninstall", hidden=True)


@app.command("battery")
def battery_command(
    quick: bool = typer.Option(False, "--quick", "-q", help="Quick check (~25s)."),
    long: bool = typer.Option(False, "--long", "-l", help="Extended check (~135s)."),
    force: bool = typer.Option(
        False, "--force", help="Force check even if power state mismatches."
    ),
    fail_on: str = typer.Option(None, "--fail-on", help="Exit code 1 if health is fair or poor."),
    output_json: bool = typer.Option(False, "--json", help="Emit JSON results."),
    plain: bool = typer.Option(False, "--plain", help="Pure ASCII output."),
    timing: bool = typer.Option(False, "--timing", help="Print step execution durations."),
) -> None:
    """Preflight battery state, run 14s telemetry scan, and evaluate 20s ML prediction."""
    from deskflow.commands.check_power import run_power_check

    run_power_check(
        target_state="battery",
        quick=quick,
        long=long,
        force=force,
        fail_on=fail_on,
        output_json=output_json,
        plain=plain,
        timing=timing,
    )


@app.command("plugged")
@app.command("ac")
def plugged_command(
    quick: bool = typer.Option(False, "--quick", "-q", help="Quick check (~25s)."),
    long: bool = typer.Option(False, "--long", "-l", help="Extended check (~135s)."),
    force: bool = typer.Option(
        False, "--force", help="Force check even if power state mismatches."
    ),
    fail_on: str = typer.Option(None, "--fail-on", help="Exit code 1 if health is fair or poor."),
    output_json: bool = typer.Option(False, "--json", help="Emit JSON results."),
    plain: bool = typer.Option(False, "--plain", help="Pure ASCII output."),
    timing: bool = typer.Option(False, "--timing", help="Print step execution durations."),
) -> None:
    """Preflight AC wall power, run 14s telemetry scan, and evaluate 20s ML prediction."""
    from deskflow.commands.check_power import run_power_check

    run_power_check(
        target_state="ac",
        quick=quick,
        long=long,
        force=force,
        fail_on=fail_on,
        output_json=output_json,
        plain=plain,
        timing=timing,
    )


@app.command("full")
def full_command_cli(
    yes: bool = typer.Option(False, "--yes", "-y", help="Execute without prompting."),
    quick: bool = typer.Option(False, "--quick", "-q", help="Quick 3-stage stress suite (~3 min)."),
    plain: bool = typer.Option(False, "--plain", help="Pure ASCII output."),
) -> None:
    """Run automated 10-minute multi-stage hardware stress suite with live model accuracy."""
    from deskflow.commands.full import full_command

    full_command(yes=yes, plain=plain, quick_test=quick)


@app.command("pipeline")
def pipeline_command_cli(
    static: bool = typer.Option(False, "--static", help="Print final frame without animation."),
    fast: bool = typer.Option(False, "--fast", help="Fast animation mode."),
    plain: bool = typer.Option(False, "--plain", help="Pure ASCII output."),
) -> None:
    """Render live animated architecture visualizer: data, models, and training pipeline."""
    from deskflow.commands.pipeline import pipeline_command

    pipeline_command(static=static, fast=fast, plain=plain)


@app.command("scan", hidden=True)
def scan_command(
    seconds: int = typer.Option(14, "--seconds", "-s", min=13, help="Duration in seconds."),
    output_json: bool = typer.Option(False, "--json", help="Emit JSON summary."),
    plain: bool = typer.Option(False, "--plain", help="Pure ASCII output."),
) -> None:
    """Sample system telemetry, evaluate ML models, compute health score, and suggest fixes."""
    if seconds < 13:
        typer.echo("Error: --seconds must be at least 13.", err=True)
        raise typer.Exit(code=EXIT_USAGE)
    from deskflow.commands.scan import scan_command as run_scan

    run_scan(seconds=seconds, output_json=output_json, plain=plain)


@app.command("watch", hidden=True)
def watch_command_cli(
    seconds: int = typer.Option(60, "--seconds", "-s", help="Streaming duration in seconds."),
    plain: bool = typer.Option(False, "--plain", help="Pure ASCII output."),
) -> None:
    """Continuous live streaming accuracy tracking (prediction vs probe ground truth)."""
    from deskflow.commands.watch import watch_command

    watch_command(duration=seconds, plain=plain)


@app.command("report", hidden=True)
def report_command_cli(
    html: Path | None = typer.Option(None, "--html", help="Path to write standalone HTML report."),
    output_json: bool = typer.Option(False, "--json", help="Emit JSON report."),
    plain: bool = typer.Option(False, "--plain", help="Pure ASCII output."),
) -> None:
    """Print comprehensive terminal evaluation report or generate HTML report."""
    from deskflow.commands.report import report_command

    report_command(html=html, json_output=output_json, plain=plain)


@app.command("doctor", hidden=True)
def doctor_command_cli(
    output_json: bool = typer.Option(False, "--json", help="Emit JSON report."),
    plain: bool = typer.Option(False, "--plain", help="Pure ASCII output."),
) -> None:
    """Self-check platform capabilities, sensors, and dependencies."""
    from deskflow.commands.doctor import doctor_command

    doctor_command(plain=plain)


def show_home_menu(input_fn: Callable[[str], str] = input) -> None:
    """Display responsive interactive launcher per Phase 18 Section D1."""
    from deskflow.platform.factory import get_adapter
    from deskflow.truth import load_model_facts
    from deskflow.ui.app import show_glossary
    from deskflow.ui.components import banner, bar, render_key_hints
    from deskflow.ui.layout import get_terminal_dimensions, make_card_grid
    from deskflow.ui.theme import (
        COLOR_CYAN,
        COLOR_GREEN,
        use_plain_output,
    )

    console = Console()
    adapter = get_adapter()
    plain = use_plain_output()

    while True:
        cols, rows = get_terminal_dimensions(console)

        # 1. Responsive Header Banner (6-row on tall, wordmark on mid, compact on short)
        console.print(banner(version=__version__, width=cols, height=rows, plain=plain))

        # 2. THIS MACHINE card & MODEL card side-by-side
        bat = adapter.battery()
        pwr_info = (
            f"Battery ({bat.percent:.0f}%) {bar(bat.percent / 100.0, 1.0, 8, plain)}"
            if (bat and not bat.plugged_in)
            else "AC Wall Power"
        )
        mem = psutil.virtual_memory()
        ram_gb = mem.total / (1024**3)

        static_hw = adapter.get_static_hardware()
        cpu_model = static_hw.get("cpu_model") or platform.processor() or "CPU"
        gpu_list = static_hw.get("gpus", [])
        gpu_name = gpu_list[0].get("name", "Integrated") if gpu_list else "Integrated"

        os_info = f"{platform.system()} {platform.release()}"
        facts = load_model_facts()

        # THIS MACHINE card
        m_lines = [
            f"Host / OS: {platform.node()} ({os_info})",
            f"CPU: {cpu_model}",
            f"GPU: {gpu_name} | RAM: {ram_gb:.1f} GB",
            f"Power: {pwr_info}",
        ]
        card_machine = Panel(
            "\n".join(m_lines),
            title="[ THIS MACHINE ]",
            border_style=COLOR_CYAN if not plain else "white",
            box=box.ASCII if plain else box.ROUNDED,
        )

        # MODEL card with consistent counts from ModelFacts (D1, D2 fix)
        base_age_ac = "uncalibrated"
        base_age_bat = "uncalibrated"
        with contextlib.suppress(Exception):
            base_ac = load_baseline(power_state="ac")
            base_age_ac = get_baseline_age_str(base_ac)
        with contextlib.suppress(Exception):
            base_bat = load_baseline(power_state="battery")
            base_age_bat = get_baseline_age_str(base_bat)

        mod_lines = [
            f"Status: {'REAL HARDWARE MODEL' if not facts.synthetic else 'SYNTHETIC DEMO'}",
            f"Dataset: {facts.dataset_split_summary}",
            f"Horizon: {facts.horizon_s:.0f}s ahead slowdown prediction",
            f"Baselines: AC ({base_age_ac}) | Battery ({base_age_bat})",
        ]
        card_model = Panel(
            "\n".join(mod_lines),
            title="[ MODEL FACTS ]",
            border_style=COLOR_GREEN if not plain else "white",
            box=box.ASCII if plain else box.ROUNDED,
        )

        grid_top = make_card_grid([card_machine, card_model], width=cols)
        console.print(grid_top)
        console.print()

        # 3. Fixed 2-Column Menu Grid (D18 fix)
        menu_table = Table.grid(expand=True, padding=(0, 3))
        menu_table.add_column(ratio=1)
        menu_table.add_column(ratio=1)

        c1 = (
            "[bold cyan][1][/bold cyan] Battery check     "
            "[dim]about 75s  health, prediction, accuracy[/dim]"
        )
        c2 = (
            "[bold cyan][4][/bold cyan] Fix Center        "
            "[dim]clean cache, memory, idle tasks[/dim]"
        )
        c3 = (
            "[bold cyan][2][/bold cyan] Plugged-in check  "
            "[dim]about 75s  same, on AC wall power[/dim]"
        )
        c4 = (
            "[bold cyan][5][/bold cyan] How it works      "
            "[dim]animated pipeline & architecture[/dim]"
        )
        c5 = (
            "[bold cyan][3][/bold cyan] Full scan         "
            "[dim]about 10m  automated hardware stress suite[/dim]"
        )
        c6 = (
            "[bold cyan][6][/bold cyan] System info       "
            "[dim]fastfetch CPU, GPU, memory, disks[/dim]"
        )

        menu_table.add_row(c1, c2)
        menu_table.add_row(c3, c4)
        menu_table.add_row(c5, c6)
        console.print(menu_table)
        console.print()

        # 4. KeyHints & Tip
        console.print(
            render_key_hints(
                [
                    ("1-6", "Select tool"),
                    ("?", "Glossary"),
                    ("Q", "Quit"),
                ],
                plain=plain,
            )
        )
        tip_msg = "[dim]Tip: Enter '1' for battery check or '4' to open the Fix Center.[/dim]\n"
        console.print(tip_msg)

        try:
            choice = input_fn("Select option: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not choice:
            continue

        if choice.upper() in ("Q", "QUIT"):
            break
        elif choice == "?":
            show_glossary(console, plain=plain)
        elif choice == "1":
            from deskflow.commands.check_power import run_power_check

            with contextlib.suppress(typer.Exit):
                run_power_check("battery", input_fn=input_fn)
        elif choice == "2":
            from deskflow.commands.check_power import run_power_check

            with contextlib.suppress(typer.Exit):
                run_power_check("ac", input_fn=input_fn)
        elif choice == "3":
            from deskflow.commands.full import full_command

            with contextlib.suppress(typer.Exit):
                full_command(input_fn=input_fn)
        elif choice == "4":
            from deskflow.commands.fix import fix_command

            with contextlib.suppress(typer.Exit):
                fix_command(
                    list_tools=False,
                    json_output=False,
                    dry_run=False,
                    yes=False,
                    plain=plain,
                )
        elif choice == "5":
            from deskflow.commands.pipeline import pipeline_command

            with contextlib.suppress(typer.Exit):
                pipeline_command()
        elif choice == "6":
            from deskflow.commands.info import render_info_dashboard

            render_info_dashboard(json_output=False, plain=plain)
            console.print()
            info_hints = render_key_hints(
                [("Enter", "Return to Menu"), ("Q", "Quit")],
                plain=plain,
            )
            console.print(info_hints)

        # If interactive and not quit, wait for key before redisplaying home
        if sys.stdin.isatty():
            console.print("\n[dim]Press [Enter] to return to Home or [Q] to quit...[/dim]")
            while True:
                k = adapter.read_key(0.2)
                if k is None:
                    continue
                if k in ("enter", "space", "esc", "h"):
                    break
                elif k == "q":
                    return


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(__version__)
        raise typer.Exit()


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    version: bool = typer.Option(
        False,
        "--version",
        callback=_version_callback,
        is_eager=True,
        help="Print version and exit.",
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable verbose diagnostic logging."
    ),
    debug: bool = typer.Option(False, "--debug", help="Enable debug logging and rich tracebacks."),
    plain: bool = typer.Option(
        False, "--plain", help="Force pure ASCII output without ANSI styling."
    ),
    json_output: bool = typer.Option(False, "--json", help="Pure JSON output."),
) -> None:
    """Deskflow: Hardware health scan with machine learning slowdown prediction."""
    setup_logging(verbose=verbose, debug=debug)

    if ctx.invoked_subcommand is not None:
        return

    # Non-TTY pipeline check (e.g. irm ... | iex)
    if not sys.stdin.isatty():
        show_home_menu(input_fn=lambda _: "Q")
        return

    show_home_menu()


if __name__ == "__main__":
    app()
