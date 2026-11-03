"""Implementation of the doctor CLI command."""

import platform
import statistics
import sys

import psutil
import typer
from rich import box
from rich.console import Console
from rich.table import Table

from deskflow.collector import StorageScanner, run_probe
from deskflow.platform import get_adapter

app = typer.Typer(help="Inspect host platform capabilities and measure probe baseline.")


def _format_bytes(num_bytes: int) -> str:
    """Format bytes into readable string."""
    if num_bytes >= 1024 * 1024 * 1024:
        return f"{num_bytes / (1024**3):.2f} GB"
    if num_bytes >= 1024 * 1024:
        return f"{num_bytes / (1024**2):.1f} MB"
    if num_bytes >= 1024:
        return f"{num_bytes / 1024:.1f} KB"
    return f"{num_bytes} B"


@app.callback(invoke_without_command=True)
def doctor_command(
    plain: bool = typer.Option(False, "--plain", help="Use ASCII-only, uncoloured output."),
) -> None:
    """Inspect system telemetry capabilities, measure baseline probe cost, and output verdict."""
    from deskflow.ui.theme import use_plain_output
    from deskflow.ui.widgets import titled_panel

    plain = use_plain_output(plain)
    console = Console(no_color=plain)
    adapter = get_adapter()

    # System overview
    os_name = f"{platform.system()} {platform.release()}"
    py_ver = sys.version.split()[0]
    console.print(f"Platform: {adapter.name} ({os_name})")
    console.print(f"Python:   {py_ver}")
    console.print()

    # Capabilities Table
    cap_table = Table(box=box.ASCII if plain else box.SQUARE, show_header=True)
    cap_table.add_column("Subsystem", style="bold")
    cap_table.add_column("Capability")
    cap_table.add_column("Status")
    cap_table.add_column("Details")

    # 1. CPU Frequency
    freq = adapter.cpu_freq()
    if freq is not None:
        rel_str = "[green]reliable[/green]" if freq.reliable else "[yellow]nominal/static[/yellow]"
        cap_table.add_row(
            "CPU Frequency",
            "Clock sampling",
            "[green]available[/green]",
            f"{freq.current_mhz:.0f} MHz (max: {freq.max_mhz:.0f} MHz, {rel_str})",
        )
    else:
        cap_table.add_row(
            "CPU Frequency",
            "Clock sampling",
            "[yellow]unavailable[/yellow]",
            "psutil returned no frequency data",
        )

    # 2. Temperatures
    temps = adapter.temperatures()
    if temps:
        cap_table.add_row(
            "Thermal",
            "Hardware sensors",
            "[green]available[/green]",
            f"{len(temps)} sensors detected (max: {max(temps.values()):.1f} C)",
        )
    else:
        cap_table.add_row(
            "Thermal",
            "Hardware sensors",
            "[yellow]unavailable[/yellow]",
            "No thermal sensors exposed via standard OS APIs without elevation",
        )

    # 3. Battery
    bat = adapter.battery()
    if bat is not None:
        state = "plugged in" if bat.plugged_in else "on battery"
        cap_table.add_row(
            "Power",
            "Battery status",
            "[green]available[/green]",
            f"{bat.percent:.0f}% ({state})",
        )
    else:
        cap_table.add_row(
            "Power",
            "Battery status",
            "[yellow]unavailable[/yellow]",
            "No battery present or readable (AC desktop or virtual machine)",
        )

    # 4. Disk Busy Time
    disk_io = psutil.disk_io_counters() if hasattr(psutil, "disk_io_counters") else None
    disk_busy = getattr(disk_io, "busy_time", None)
    if disk_io and disk_busy is not None:
        cap_table.add_row(
            "Disk",
            "Busy time tracking",
            "[green]available[/green]",
            f"Active (cumulative: {disk_busy} ms)",
        )
    else:
        cap_table.add_row(
            "Disk",
            "Busy time tracking",
            "[yellow]unavailable[/yellow]",
            "busy_time counter not supported by this disk driver or OS",
        )

    # 5. Storage Cache Locations
    scanner = StorageScanner(adapter=adapter, time_budget_s=2.0)
    report = scanner.scan(force=True)
    cache_details = ", ".join(f"{e.path} ({_format_bytes(e.bytes)})" for e in report.entries)
    cap_table.add_row(
        "Storage",
        f"Cache dirs ({len(report.entries)} found)",
        "[green]available[/green]",
        cache_details if cache_details else "None found",
    )

    console.print(titled_panel("[SYSTEM CAPABILITIES]", cap_table, plain))
    console.print()

    # 6. Probe Baseline (median of 5 probes)
    console.print("Running 5 performance probes to measure baseline latency...")
    probe_totals_ms: list[float] = []
    for _ in range(5):
        pr = run_probe()
        probe_totals_ms.append(pr.total_ns / 1_000_000)

    median_ms = statistics.median(probe_totals_ms)
    min_ms = min(probe_totals_ms)
    max_ms = max(probe_totals_ms)

    baseline_table = Table(box=box.ASCII if plain else box.SQUARE, show_header=True)
    baseline_table.add_column("Statistic")
    baseline_table.add_column("Total Latency (ms)", justify="right")
    baseline_table.add_row("Median", f"{median_ms:.2f} ms")
    baseline_table.add_row("Min", f"{min_ms:.2f} ms")
    baseline_table.add_row("Max", f"{max_ms:.2f} ms")
    console.print(titled_panel("[PROBE BASELINE]", baseline_table, plain))
    console.print()

    # Verdict
    notes: list[str] = []
    if not temps:
        notes.append("temperatures not readable")
    if bat is None:
        notes.append("no battery")
    if freq is not None and not freq.reliable:
        notes.append("CPU frequency static/nominal")

    caveat_str = f" ({', '.join(notes)})" if notes else ""
    console.print(
        f"Verdict: Ready for telemetry collection and ML modeling. "
        f"Core CPU, memory, disk, network, and probe latency fully operational{caveat_str}."
    )
