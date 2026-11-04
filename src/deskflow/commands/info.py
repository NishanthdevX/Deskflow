"""Fastfetch-style system information command: deskflow info."""

from __future__ import annotations

import json
import platform
import shutil
import time
from typing import Any

import psutil
import typer
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from deskflow import __version__
from deskflow.paths import get_data_dir
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


def _get_cached_static_hardware(adapter: Any) -> dict[str, Any]:
    """Retrieve static hardware info, caching in <data_dir>/hwinfo.json for 24 hours."""
    cache_path = get_data_dir() / "hwinfo.json"
    now = time.time()

    if cache_path.exists():
        try:
            with open(cache_path, encoding="utf-8") as f:
                data = json.load(f)
            cached_at = float(data.get("cached_at_s", 0.0))
            if now - cached_at < 86400.0 and "static" in data:
                return dict(data["static"])
        except Exception:
            pass

    # Cold fetch from adapter
    static_info: dict[str, Any] = {}
    try:
        static_info = dict(adapter.get_static_hardware())
    except Exception:
        static_info = {
            "host_model": "unknown",
            "os_build": "unknown",
            "cpu_model": "unknown",
            "gpus": [],
            "resolution": "unknown",
        }

    # Save cache
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump({"cached_at_s": now, "static": static_info}, f, indent=2)
    except Exception:
        pass

    return static_info


@app.callback(invoke_without_command=True)
def info_command(
    json_output: bool = typer.Option(False, "--json", help="Output specifications as JSON."),
    plain: bool = typer.Option(False, "--plain", help="Pure ASCII output."),
) -> None:
    """Print high-density system specifications and resource usage."""
    adapter = get_adapter()
    static_hw = _get_cached_static_hardware(adapter)

    # 1. Host and OS
    node = platform.node() or "localhost"
    host_model = static_hw.get("host_model") or node
    if host_model == "unknown":
        host_model = node

    os_build = static_hw.get("os_build", "")
    os_name = f"{platform.system()} {platform.release()}"
    if os_build and os_build != "unknown":
        os_str = f"{os_name} ({platform.machine()}) {os_build}"
    else:
        os_str = f"{os_name} ({platform.machine()})"

    # 2. CPU
    phys_cores = psutil.cpu_count(logical=False) or 1
    total_cores = psutil.cpu_count(logical=True) or 1
    freq = psutil.cpu_freq()
    clock_mhz = (
        f"{freq.max:.0f} MHz"
        if (freq and freq.max > 0)
        else (f"{freq.current:.0f} MHz" if freq else "Unknown")
    )
    cpu_model = static_hw.get("cpu_model") or platform.processor() or "CPU"
    cpu_str = f"{cpu_model} ({phys_cores}C/{total_cores}T @ {clock_mhz})"

    # 3. GPU(s)
    gpus_list = static_hw.get("gpus", [])
    gpu_strs: list[str] = []
    if isinstance(gpus_list, list):
        for g in gpus_list:
            if isinstance(g, dict):
                g_name = g.get("name", "GPU")
                vram = g.get("vram_bytes", 0)
                if vram > 0:
                    vram_gb = vram / (1024**3)
                    gpu_strs.append(f"{g_name} ({vram_gb:.1f} GB VRAM)")
                else:
                    gpu_strs.append(g_name)
    if not gpu_strs:
        gpu_strs.append("unknown")

    # 4. Memory
    mem = psutil.virtual_memory()
    mem_used_gb = (mem.total - mem.available) / (1024**3)
    mem_total_gb = mem.total / (1024**3)
    mem_avail_gb = mem.available / (1024**3)
    mem_pct = int(mem.percent)
    mem_str = (
        f"{mem_used_gb:.1f} / {mem_total_gb:.1f} GB "
        f"({mem_avail_gb:.1f} GB free, {mem_pct}%) {score_bar(mem_pct, 8, plain)}"
    )

    # 5. Swap / Pagefile
    swap = adapter.swap_memory()
    if swap is not None:
        swap_used_bytes, swap_pct = swap
        swap_used_gb = swap_used_bytes / (1024**3)
        swap_str = f"{swap_used_gb:.1f} GB ({swap_pct:.0f}%) {score_bar(int(swap_pct), 8, plain)}"
    else:
        pswap = psutil.swap_memory()
        swap_used_gb = pswap.used / (1024**3)
        swap_total_gb = pswap.total / (1024**3)
        swap_pct = int(pswap.percent)
        swap_str = (
            f"{swap_used_gb:.1f} / {swap_total_gb:.1f} GB ({swap_pct}%) "
            f"{score_bar(swap_pct, 8, plain)}"
        )

    # 6. Disks
    disks_info: list[dict[str, Any]] = []
    disk_lines: list[str] = []
    try:
        parts = psutil.disk_partitions(all=False)
        seen = set()
        for p in parts:
            if p.mountpoint in seen:
                continue
            seen.add(p.mountpoint)
            try:
                du = psutil.disk_usage(p.mountpoint)
                u_gb = (du.total - du.free) / (1024**3)
                t_gb = du.total / (1024**3)
                f_gb = du.free / (1024**3)
                pct = int(du.percent)
                fstype = p.fstype or "unknown"
                disk_lines.append(
                    f"{p.mountpoint} [{fstype}] {u_gb:.0f}/{t_gb:.0f} GB "
                    f"({f_gb:.0f} GB free, {pct}%) {score_bar(pct, 6, plain)}"
                )
                disks_info.append(
                    {
                        "mount": p.mountpoint,
                        "filesystem": fstype,
                        "total_bytes": du.total,
                        "used_bytes": du.total - du.free,
                        "free_bytes": du.free,
                        "percent": pct,
                    }
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
        disk_lines.append(f". {u_gb:.0f}/{t_gb:.0f} GB ({pct}%) {score_bar(pct, 6, plain)}")
        disks_info.append(
            {
                "mount": ".",
                "filesystem": "unknown",
                "total_bytes": s_du.total,
                "used_bytes": s_du.total - s_du.free,
                "free_bytes": s_du.free,
                "percent": pct,
            }
        )

    # 7. Power
    bat = adapter.battery()
    if bat is not None:
        pwr_txt = "Charging/AC" if bat.plugged_in else "Discharging"
        pwr_str = f"Battery {bat.percent:.0f}% ({pwr_txt})"
    else:
        pwr_str = "AC Power (Desktop)"

    # 8. Uptime
    boot = psutil.boot_time()
    uptime_s = max(0, time.time() - boot)
    days = int(uptime_s // 86400)
    hours = int((uptime_s % 86400) // 3600)
    mins = int((uptime_s % 3600) // 60)
    uptime_str = f"{days}d {hours}h {mins}m" if days > 0 else f"{hours}h {mins}m"

    # 9. Display resolution
    res_str = static_hw.get("resolution", "unknown")

    # JSON output mode
    if json_output:
        full_info = {
            "host_model": host_model,
            "os": os_str,
            "cpu": cpu_str,
            "gpus": gpu_strs,
            "memory": {
                "total_bytes": mem.total,
                "used_bytes": mem.total - mem.available,
                "available_bytes": mem.available,
                "percent": mem_pct,
            },
            "swap": swap_str,
            "disks": disks_info,
            "power": pwr_str,
            "uptime_seconds": uptime_s,
            "uptime_str": uptime_str,
            "display_resolution": res_str,
            "deskflow_version": f"v{__version__}",
        }
        console.print_json(json.dumps(full_info))
        return

    # Layout: side-by-side fastfetch panel
    grid = Table.grid(expand=True, padding=(0, 2))
    grid.add_column("Logo", justify="left")
    grid.add_column("Details", justify="left")

    logo_text = Text(LOGO.strip("\n"), style="bright_cyan bold" if not plain else "none")

    details_table = Table(box=box.ASCII if plain else box.SIMPLE, show_header=False, pad_edge=False)
    details_table.add_column("Key", style="bold cyan" if not plain else "none")
    details_table.add_column("Val")

    details_table.add_row("Host", host_model)
    details_table.add_row("OS", os_str)
    details_table.add_row("CPU", cpu_str)
    for i, g_str in enumerate(gpu_strs):
        details_table.add_row("GPU" if i == 0 else "", g_str)
    details_table.add_row("Memory", mem_str)
    details_table.add_row("Swap", swap_str)
    for i, dl in enumerate(disk_lines):
        details_table.add_row("Disk" if i == 0 else "", dl)
    details_table.add_row("Display", res_str)
    details_table.add_row("Power", pwr_str)
    details_table.add_row("Uptime", uptime_str)
    details_table.add_row("Version", f"v{__version__}")

    grid.add_row(logo_text, details_table)

    panel = Panel(
        grid,
        title="[DESKFLOW SYSTEM INFO]",
        box=box.ASCII if plain else box.ROUNDED,
        border_style="cyan" if not plain else "none",
    )
    console.print(panel)
