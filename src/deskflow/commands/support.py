"""Support, privacy, diagnostic, and data management commands for Deskflow."""

from __future__ import annotations

import getpass
import platform
import re
import shutil
import sys
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import typer
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from deskflow import __version__
from deskflow.baseline import load_all_baselines
from deskflow.logging_config import get_log_file_path
from deskflow.paths import get_data_dir
from deskflow.platform.factory import get_adapter
from deskflow.ui.theme import COLOR_CYAN, use_plain_output

app_bugreport = typer.Typer(help="Generate sanitized diagnostic bug report.")
app_privacy = typer.Typer(help="Display privacy and local storage policies.")
app_data = typer.Typer(help="Inspect local Deskflow storage usage and paths.")
app_uninstall = typer.Typer(help="Uninstall or purge local Deskflow application data.")


def sanitize_text(
    text: str,
    home_dir: str | None = None,
    username: str | None = None,
    machine_id: str | None = None,
) -> str:
    """Sanitize text by redacting sensitive paths, usernames, and machine IDs."""
    h_dir = home_dir or str(Path.home())
    u_name = username or getpass.getuser()

    # Redact home directory
    if h_dir:
        # Normalize backslashes/slashes
        text = text.replace(h_dir, "[HOME_DIR]")
        text = text.replace(h_dir.replace("\\", "/"), "[HOME_DIR]")

    # Redact username
    if u_name and len(u_name) > 2:
        text = re.sub(re.escape(u_name), "[USERNAME]", text, flags=re.IGNORECASE)

    # Redact machine ID if provided
    if machine_id:
        text = text.replace(machine_id, "[MACHINE_ID]")

    # Redact standard UUID/GUID patterns
    guid_pattern = r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"
    text = re.sub(guid_pattern, "[MACHINE_ID]", text)

    return text


@app_bugreport.callback(invoke_without_command=True)
def bugreport_command(
    out: Path | None = typer.Option(
        None, "--out", "-o", help="Custom output path for the sanitized bugreport markdown."
    ),
    plain: bool = typer.Option(False, "--plain", help="Use monochrome ASCII output."),
) -> Path:
    """Generate a fully sanitized, local markdown bug report with zero remote uploads."""
    plain = use_plain_output(plain)
    console = Console(no_color=plain)

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    report_file = out or Path(f"deskflow-bugreport-{timestamp}.md")

    adapter = get_adapter()
    static_hw = adapter.get_static_hardware()
    cpu_model = static_hw.get("cpu_model") or platform.processor() or "Unknown CPU"

    # Baselines
    try:
        baselines = load_all_baselines()
        b_summary = ", ".join(f"{k}: calibrated" for k in baselines) or "None calibrated"
    except Exception as exc:
        b_summary = f"Error loading baselines: {exc}"

    # Log lines (last 200 lines)
    log_path = get_log_file_path()
    log_lines: list[str] = []
    if log_path.exists():
        try:
            with open(log_path, encoding="utf-8", errors="replace") as f:
                all_lines = f.readlines()
                log_lines = all_lines[-200:]
        except Exception as exc:
            log_lines = [f"Failed to read deskflow.log: {exc}\n"]
    else:
        log_lines = ["No deskflow.log file found.\n"]

    raw_report = [
        "# Deskflow Sanitized Diagnostic Bug Report",
        "",
        f"- Generated UTC: {timestamp}",
        f"- Deskflow Version: {__version__}",
        f"- Python Version: {sys.version.split()[0]} ({platform.architecture()[0]})",
        f"- Operating System: {platform.system()} {platform.release()} ({platform.version()})",
        f"- Machine Architecture: {platform.machine()}",
        f"- Hardware: {cpu_model}",
        f"- Baselines Status: {b_summary}",
        "",
        "## Diagnostic Doctor Overview",
        "",
        "```",
        f"Host: {platform.node()}",
        f"Home Path: {Path.home()}",
        f"User: {getpass.getuser()}",
        f"Data Directory: {get_data_dir()}",
        "```",
        "",
        "## Recent Log History (Last 200 Lines)",
        "",
        "```text",
        "".join(log_lines).strip(),
        "```",
        "",
        "## User Notes",
        "",
        "<!-- Please describe the issue, steps to reproduce, and expected behavior below -->",
        "",
    ]

    sanitized_content = sanitize_text("\n".join(raw_report))
    report_file.write_text(sanitized_content, encoding="utf-8")

    console.print(
        Panel(
            f"[bold green]Sanitized bug report written to: {report_file.resolve()}[/bold green]\n"
            "[dim]All usernames, personal directories, and hardware IDs have been redacted. "
            "No data was uploaded.[/dim]",
            title="[DESKFLOW BUGREPORT]",
            border_style="green" if not plain else "none",
            box=box.ASCII if plain else box.ROUNDED,
        )
    )
    return report_file


@app_privacy.callback(invoke_without_command=True)
def privacy_command(
    plain: bool = typer.Option(False, "--plain", help="Use monochrome ASCII output."),
) -> None:
    """Explain Deskflow local storage and privacy commitments."""
    plain = use_plain_output(plain)
    console = Console(no_color=plain)

    table = Table(
        box=box.ASCII if plain else box.ROUNDED,
        show_header=True,
        header_style=COLOR_CYAN if not plain else None,
        expand=True,
    )
    table.add_column("Category", style="bold")
    table.add_column("Storage Location")
    table.add_column("Data Retained")
    table.add_column("Network Policy")

    d_dir = str(get_data_dir())
    table.add_row(
        "Baselines",
        f"{d_dir}/baselines.json",
        "Micro-benchmark hardware latency percentiles",
        "Strictly local (0 network calls)",
    )
    table.add_row(
        "Telemetry",
        f"{d_dir}/telemetry/",
        "Rolling 12s hardware metrics & probe samples",
        "Strictly local (0 network calls)",
    )
    table.add_row(
        "Live Watch",
        f"{d_dir}/watch/",
        "Verification session error rates and accuracy",
        "Strictly local (0 network calls)",
    )
    table.add_row(
        "Remediation",
        f"{d_dir}/fixlog.jsonl",
        "Audit log of executed user fixes and freed bytes",
        "Strictly local (0 network calls)",
    )
    table.add_row(
        "Logs",
        f"{d_dir}/logs/deskflow.log",
        "Rotating application diagnostics (max 3 MB)",
        "Strictly local (0 network calls)",
    )

    console.print("\n[bold cyan][DESKFLOW PRIVACY POLICY][/bold cyan]")
    console.print(
        "Deskflow is an entirely offline, device-local performance tool. "
        "It contains zero telemetry collectors, analytics pings, or background cloud connections. "
        "All data remains strictly on your local disk.\n"
    )
    console.print(table)
    console.print()


@app_data.callback(invoke_without_command=True)
def data_command(
    plain: bool = typer.Option(False, "--plain", help="Use monochrome ASCII output."),
) -> None:
    """Inspect local Deskflow storage usage and directory sizes."""
    plain = use_plain_output(plain)
    console = Console(no_color=plain)

    data_dir = get_data_dir()
    table = Table(
        box=box.ASCII if plain else box.ROUNDED,
        show_header=True,
        header_style=COLOR_CYAN if not plain else None,
        expand=True,
    )
    table.add_column("Component", style="bold")
    table.add_column("Path")
    table.add_column("Files", justify="right")
    table.add_column("Total Size", justify="right")

    total_bytes = 0
    total_files = 0

    sub_targets = ["baselines.json", "telemetry", "watch", "fixlog.jsonl", "models", "logs"]
    for sub in sub_targets:
        target = data_dir / sub
        if not target.exists():
            table.add_row(sub, str(target), "0", "0 B")
            continue

        if target.is_file():
            sz = target.stat().st_size
            total_bytes += sz
            total_files += 1
            table.add_row(sub, str(target), "1", f"{sz / 1024:.1f} KB")
        else:
            files = list(target.rglob("*"))
            f_count = sum(1 for f in files if f.is_file())
            sz = sum(f.stat().st_size for f in files if f.is_file())
            total_bytes += sz
            total_files += f_count
            sz_str = f"{sz / (1024**2):.2f} MB" if sz > 1024**2 else f"{sz / 1024:.1f} KB"
            table.add_row(sub, str(target), str(f_count), sz_str)

    console.print(f"\n[bold cyan][DESKFLOW LOCAL DATA USAGE: {data_dir}][/bold cyan]\n")
    console.print(table)
    if total_bytes > 1024**2:
        total_str = f"{total_bytes / (1024**2):.2f} MB"
    else:
        total_str = f"{total_bytes / 1024:.1f} KB"
    console.print(f"[bold]Total Storage Consumed:[/bold] {total_str} ({total_files} files)\n")


def purge_deskflow_data(
    data: bool = False,
    no_input: bool = False,
    input_fn: Callable[[str], str] = input,
) -> None:
    """Purge application data with typed confirmation."""
    console = Console()
    data_dir = get_data_dir()

    if not data:
        console.print(
            "To uninstall the Deskflow executable, run:\n"
            "  uv tool uninstall deskflow\n\n"
            "To delete all local benchmarks, telemetry, and logs, run:\n"
            "  deskflow uninstall --data"
        )
        return

    if not data_dir.exists():
        console.print(f"[dim]Data directory does not exist: {data_dir}[/dim]")
        return

    if no_input:
        console.print("[red]Aborted: --no-input passed and typed confirmation required.[/red]")
        raise typer.Exit(code=5)

    warn_msg = (
        f"[bold red]WARNING: Permanently deleting all Deskflow data in {data_dir}:[/bold red]"
    )
    console.print(
        f"{warn_msg}\n"
        "- All calibrated hardware baselines\n"
        "- All telemetry runs and live watch logs\n"
        "- All fix audit history\n"
    )
    try:
        confirm = input_fn("Type 'delete my data' to confirm removal: ").strip()
    except (EOFError, KeyboardInterrupt):
        raise typer.Exit(code=5) from None

    if confirm != "delete my data":
        console.print("[yellow]Confirmation text did not match. Aborting data deletion.[/yellow]")
        raise typer.Exit(code=5)

    try:
        shutil.rmtree(data_dir)
        console.print(f"[bold green]Data directory successfully deleted: {data_dir}[/bold green]")
    except Exception as exc:
        console.print(f"[red]Error deleting data directory: {exc}[/red]")
        raise typer.Exit(code=1) from exc


@app_uninstall.callback(invoke_without_command=True)
def uninstall_command(
    data: bool = typer.Option(
        False,
        "--data",
        help="Purge all local Deskflow configuration, baselines, and logs.",
    ),
    no_input: bool = typer.Option(
        False,
        "--no-input",
        help="Non-interactive mode (aborts destructive actions without explicit flag).",
    ),
) -> None:
    """Uninstall Deskflow or purge application data."""
    purge_deskflow_data(data=data, no_input=no_input)
