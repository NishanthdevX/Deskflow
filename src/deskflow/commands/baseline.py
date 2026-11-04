"""CLI commands for baseline management and calibration."""

import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from deskflow.baseline import (
    CalibrationError,
    calibrate_baseline,
    load_all_baselines,
)
from deskflow.platform import get_adapter

app = typer.Typer(help="Manage and calibrate system performance baselines.")
console = Console()


@app.command("calibrate")
def calibrate_command(
    duration: float = typer.Option(
        90.0,
        "--duration",
        help="Calibration duration in seconds (default: 90s, minimum: 60s).",
    ),
    out: Path | None = typer.Option(
        None,
        "--out",
        help="Optional custom output path for baseline.json.",
    ),
) -> None:
    """Calibrate performance baseline while the system is idle."""
    console.print(
        "Please close heavy applications and leave the system completely idle during calibration."
    )
    console.print(f"Calibration duration: {duration:.0f} seconds (sampling every 2s).")

    def _progress(idx: int, total: int, last_total_ms: float) -> None:
        sys.stdout.write(f"\rProbe {idx}/{total} - last total latency: {last_total_ms:.2f} ms")
        sys.stdout.flush()

    try:
        baseline = calibrate_baseline(
            duration_s=duration,
            interval_s=2.0,
            warmup_samples=3,
            out_path=out,
            progress_callback=_progress,
        )
    except CalibrationError as e:
        console.print()
        console.print(f"[red]Calibration rejected:[/red] {e}")
        raise typer.Exit(code=1) from e

    console.print("\n")
    console.print("[green]Baseline calibration successful.[/green]")

    table = Table(title="System Performance Baseline", show_header=True)
    table.add_column("Property", style="bold")
    table.add_column("Value")

    table.add_row("Machine ID", baseline.machine_id)
    table.add_row("Created UTC", baseline.created_utc)
    table.add_row("Platform", baseline.platform)
    table.add_row("CPU Median Latency", f"{baseline.cpu_ns / 1_000_000:.2f} ms")
    table.add_row("Memory Median Latency", f"{baseline.mem_ns / 1_000_000:.2f} ms")
    table.add_row("Disk Median Latency", f"{baseline.disk_ns / 1_000_000:.2f} ms")
    table.add_row("Total Median Latency", f"{baseline.total_ns / 1_000_000:.2f} ms")
    table.add_row("Total Latency 10th Percentile", f"{baseline.p10_total_ns / 1_000_000:.2f} ms")
    table.add_row("Total Latency 90th Percentile", f"{baseline.p90_total_ns / 1_000_000:.2f} ms")
    table.add_row("Coefficient of Variation", f"{baseline.cv:.3f}")
    table.add_row("Sample Count", str(baseline.sample_count))
    table.add_row("Deskflow Version", baseline.deskflow_version)

    console.print(table)


@app.command("show")
def show_command(
    path: Path | None = typer.Option(
        None,
        "--path",
        help="Optional custom path to baseline.json.",
    ),
) -> None:
    """Show calibrated baselines for all power states."""
    try:
        baselines = load_all_baselines(path=path)
    except FileNotFoundError as e:
        console.print(f"[red]Error:[/red] {e}")
        console.print("Run 'deskflow baseline calibrate' to generate an idle baseline.")
        raise typer.Exit(code=1) from e

    adapter = get_adapter()
    bat = adapter.battery()
    has_battery = bat is not None

    for state in ("ac", "battery"):
        state_label = "AC Power" if state == "ac" else "Battery Power"
        base = baselines.get(state)
        if base is not None:
            table = Table(title=f"System Performance Baseline ({state_label})", show_header=True)
            table.add_column("Property", style="bold")
            table.add_column("Value")

            table.add_row("Power State", state.upper())
            table.add_row("Machine ID", base.machine_id)
            table.add_row("Created UTC", base.created_utc)
            table.add_row("Platform", base.platform)
            table.add_row("CPU Median Latency", f"{base.cpu_ns / 1_000_000:.2f} ms")
            table.add_row("Memory Median Latency", f"{base.mem_ns / 1_000_000:.2f} ms")
            table.add_row("Disk Median Latency", f"{base.disk_ns / 1_000_000:.2f} ms")
            table.add_row("Total Median Latency", f"{base.total_ns / 1_000_000:.2f} ms")
            table.add_row(
                "Total Latency 10th Percentile", f"{base.p10_total_ns / 1_000_000:.2f} ms"
            )
            table.add_row(
                "Total Latency 90th Percentile", f"{base.p90_total_ns / 1_000_000:.2f} ms"
            )
            table.add_row("Coefficient of Variation", f"{base.cv:.3f}")
            if base.spread_fastest_half > 0:
                table.add_row("Spread (Fastest 50%)", f"{base.spread_fastest_half:.3f}")
            if base.mean_cpu_percent > 0:
                table.add_row("Mean CPU Utilisation", f"{base.mean_cpu_percent:.1f}%")
            table.add_row("Sample Count", str(base.sample_count))
            table.add_row("Deskflow Version", base.deskflow_version)

            console.print(table)
        elif state == "battery" and not has_battery:
            console.print("[dim]Battery Power Baseline: Not applicable (desktop system).[/dim]")
        else:
            console.print(
                f"[yellow]{state_label} Baseline: Missing. "
                f"Run 'deskflow baseline calibrate' while on {state} power.[/yellow]"
            )
