"""Fastfetch-style system information command: deskflow info."""

import platform
import shutil
import time

import psutil
import typer
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from deskflow.platform.factory import get_adapter
from deskflow.ui.widgets import score_bar

app = typer.Typer(help="Fast hardware and operating system summary.")
console = Console()

LOGO = """
  ____  _____ ____  _  __
 |  _ \\| ____/ ___|| |/ /
 | | | |  _| \\___ \\| ' / 
 | |_| | |___ ___) | . \\ 
 |____/|_____|____/|_|\\_\\
"""


@app.callback(invoke_without_command=True)
def info_command(
    plain: bool = typer.Option(False, "--plain", help="Pure ASCII output."),
) -> None:
    """Print high-density system specifications and resource usage."""
    adapter = get_adapter()

    # 1. Host and OS
    node = platform.node() or "localhost"
    os_name = f"{platform.system()} {platform.release()} ({platform.machine()})"

    # 2. CPU
    phys_cores = psutil.cpu_count(logical=False) or 1
    total_cores = psutil.cpu_count(logical=True) or 1
    freq = psutil.cpu_freq()
    clock_mhz = f"{freq.current:.0f} MHz" if freq else "Unknown"
    cpu_str = f"{platform.processor() or 'CPU'} ({phys_cores}C/{total_cores}T @ {clock_mhz})"

    # 3. Memory
    mem = psutil.virtual_memory()
    mem_used_gb = (mem.total - mem.available) / (1024**3)
    mem_total_gb = mem.total / (1024**3)
    mem_pct = int(mem.percent)
    mem_str = (
        f"{mem_used_gb:.1f} / {mem_total_gb:.1f} GB ({mem_pct}%) {score_bar(mem_pct, 10, plain)}"
    )

    # 4. Disks
    disk_lines: list[str] = []
    try:
        parts = psutil.disk_partitions(all=False)
        seen = set()
        for p in parts[:3]:
            if p.mountpoint in seen:
                continue
            seen.add(p.mountpoint)
            try:
                du = psutil.disk_usage(p.mountpoint)
                u_gb = (du.total - du.free) / (1024**3)
                t_gb = du.total / (1024**3)
                pct = int(du.percent)
                disk_lines.append(
                    f"{p.mountpoint} {u_gb:.0f}/{t_gb:.0f} GB ({pct}%) {score_bar(pct, 8, plain)}"
                )
            except (PermissionError, OSError):
                continue
    except Exception:
        pass
    if not disk_lines:
        s_du = shutil.disk_usage(".")
        u_gb = (s_du.total - s_du.free) / (1024**3)
        t_gb = s_du.total / (1024**3)
        pct = int((u_gb / max(1.0, t_gb)) * 100)
        disk_lines.append(f". {u_gb:.0f}/{t_gb:.0f} GB ({pct}%) {score_bar(pct, 8, plain)}")

    # 5. Power
    bat = adapter.battery()
    if bat is not None:
        pwr_state_txt = "Charging/AC" if bat.plugged_in else "Discharging"
        pwr_str = f"Battery {bat.percent:.0f}% ({pwr_state_txt})"
    else:
        pwr_str = "AC Power (Desktop)"

    # 6. Uptime
    boot = psutil.boot_time()
    uptime_s = max(0, time.time() - boot)
    days = int(uptime_s // 86400)
    hours = int((uptime_s % 86400) // 3600)
    mins = int((uptime_s % 3600) // 60)
    uptime_str = f"{days}d {hours}h {mins}m" if days > 0 else f"{hours}h {mins}m"

    # Layout: 2 columns table
    grid = Table.grid(expand=True, padding=(0, 2))
    grid.add_column("Logo", justify="left")
    grid.add_column("Details", justify="left")

    logo_text = Text(LOGO.strip("\n"), style="bright_cyan bold" if not plain else "none")

    details_table = Table(box=box.ASCII if plain else box.SIMPLE, show_header=False, pad_edge=False)
    details_table.add_column("Key", style="bold cyan" if not plain else "none")
    details_table.add_column("Val")

    details_table.add_row("Host", node)
    details_table.add_row("OS", os_name)
    details_table.add_row("CPU", cpu_str)
    details_table.add_row("Memory", mem_str)
    for i, dl in enumerate(disk_lines):
        details_table.add_row("Disk" if i == 0 else "", dl)
    details_table.add_row("Power", pwr_str)
    details_table.add_row("Uptime", uptime_str)

    grid.add_row(logo_text, details_table)

    panel = Panel(
        grid,
        title="[DESKFLOW SYSTEM INFO]",
        box=box.ASCII if plain else box.ROUNDED,
        border_style="cyan" if not plain else "none",
    )
    console.print(panel)
