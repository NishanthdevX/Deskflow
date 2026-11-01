"""Implementation of the snapshot CLI command."""

import json
import time

import typer
from rich.console import Console
from rich.table import Table

from deskflow.collector import (
    SCHEMA_VERSION,
    Record,
    Sampler,
    StorageScanner,
    generate_run_id,
    run_probe,
)
from deskflow.platform import get_adapter

app = typer.Typer(help="Take an instant system snapshot and performance probe.")


def _format_bytes(num_bytes: float | None) -> str:
    """Format bytes into readable MiB or GiB string."""
    if num_bytes is None:
        return "n/a"
    if num_bytes >= 1024 * 1024 * 1024:
        return f"{num_bytes / (1024**3):.2f} GB"
    return f"{num_bytes / (1024**2):.1f} MB"


def _format_rate(bytes_per_s: float | None) -> str:
    """Format bytes per second into readable rate."""
    if bytes_per_s is None:
        return "n/a"
    if bytes_per_s >= 1024 * 1024:
        return f"{bytes_per_s / (1024 * 1024):.1f} MB/s"
    if bytes_per_s >= 1024:
        return f"{bytes_per_s / 1024:.1f} KB/s"
    return f"{bytes_per_s:.0f} B/s"


@app.callback(invoke_without_command=True)
def snapshot_command(
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Output raw Record object as formatted JSON.",
    ),
) -> None:
    """Take two telemetry samples 1s apart, execute one performance probe, and display results."""
    console = Console()
    adapter = get_adapter()
    sampler = Sampler(adapter=adapter)

    # First sample to prime delta counters
    sampler.sample()
    time.sleep(1.0)
    sample, processes = sampler.sample()

    probe = run_probe()
    storage = StorageScanner(adapter=adapter).scan()

    record = Record(
        schema_version=SCHEMA_VERSION,
        run_id=generate_run_id(),
        label="snapshot",
        sample=sample,
        probe=probe,
        top_processes=processes[:10],
        storage=storage,
    )

    if json_output:
        console.print_json(json.dumps(record.to_dict()))
        return

    # Compact tabular overview
    overview_table = Table(title="System Telemetry Snapshot", box=None, show_header=True)
    overview_table.add_column("Subsystem", style="bold")
    overview_table.add_column("Metric")
    overview_table.add_column("Value")

    # CPU
    cpu_freq_str = (
        f"{sample.cpu_freq_current_mhz:.0f} MHz" if sample.cpu_freq_current_mhz else "n/a"
    )
    overview_table.add_row(
        "CPU",
        "Utilization (mean / max core)",
        f"{sample.cpu_percent_mean:.1f}% / {sample.cpu_percent_max_core:.1f}%",
    )
    overview_table.add_row("CPU", "Frequency", cpu_freq_str)

    # Memory
    mem_avail_str = _format_bytes(sample.mem_available_bytes)
    mem_total_str = _format_bytes(sample.mem_total_bytes)
    overview_table.add_row(
        "Memory",
        "Utilization (available / total)",
        f"{sample.mem_percent:.1f}% ({mem_avail_str} avail of {mem_total_str})",
    )
    overview_table.add_row(
        "Swap",
        "Utilization (used)",
        f"{sample.swap_percent:.1f}% ({_format_bytes(sample.swap_used_bytes)})",
    )

    # Disk
    busy_str = (
        f"{sample.disk_busy_ms_per_s:.0f} ms/s" if sample.disk_busy_ms_per_s is not None else "n/a"
    )
    r_rate = _format_rate(sample.disk_read_bytes_per_s)
    w_rate = _format_rate(sample.disk_write_bytes_per_s)
    overview_table.add_row(
        "Disk I/O",
        "Read / Write / Busy",
        f"{r_rate} / {w_rate} / {busy_str}",
    )

    # Network
    net_err = f"{sample.net_errors_per_s:.0f}/s" if sample.net_errors_per_s is not None else "0/s"
    recv_rate = _format_rate(sample.net_recv_bytes_per_s)
    sent_rate = _format_rate(sample.net_sent_bytes_per_s)
    overview_table.add_row(
        "Network",
        "Recv / Sent / Errors",
        f"{recv_rate} / {sent_rate} / {net_err}",
    )

    # Probe Timings
    cpu_ms = probe.cpu_ns / 1_000_000
    mem_ms = probe.mem_ns / 1_000_000
    disk_ms = probe.disk_ns / 1_000_000
    total_ms = probe.total_ns / 1_000_000
    overview_table.add_row(
        "Probe",
        "Latency (CPU / Mem / Disk)",
        f"{cpu_ms:.2f} ms / {mem_ms:.2f} ms / {disk_ms:.2f} ms",
    )
    overview_table.add_row("Probe", "Total Measured Latency", f"{total_ms:.2f} ms")

    console.print(overview_table)
    console.print()

    # Top 5 Processes Table
    proc_table = Table(title="Top 5 Processes by Memory (RSS)", box=None, show_header=True)
    proc_table.add_column("PID", justify="right")
    proc_table.add_column("Name")
    proc_table.add_column("RSS", justify="right")
    proc_table.add_column("CPU %", justify="right")
    proc_table.add_column("Threads", justify="right")
    proc_table.add_column("Age", justify="right")
    proc_table.add_column("Protected")

    for proc in processes[:5]:
        proc_table.add_row(
            str(proc.pid),
            proc.name,
            _format_bytes(proc.rss_bytes),
            f"{proc.cpu_percent:.1f}%",
            str(proc.num_threads),
            f"{proc.age_s:.0f}s",
            "yes" if proc.is_protected else "no",
        )

    console.print(proc_table)
