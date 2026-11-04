"""Demo readiness pre-flight checklist: deskflow prep."""

import contextlib
import shutil
import sys
import time

import numpy as np
import psutil
import typer
from rich import box
from rich.console import Console
from rich.table import Table

from deskflow.baseline import (
    CalibrationError,
    calibrate_baseline,
    check_baseline_drift,
    get_baseline_age_days,
    get_baseline_age_str,
    get_current_power_state,
    load_baseline,
    save_baseline,
)
from deskflow.collector.probe import run_probe
from deskflow.ml.artifacts import resolve_model_dir
from deskflow.ml.runtime import RuntimeModels
from deskflow.platform.factory import get_adapter

app = typer.Typer(help="Verify system readiness for live demos and evaluation.")
console = Console()


@app.callback(invoke_without_command=True)
def prep_command(
    calibrate: bool = typer.Option(
        False,
        "--calibrate",
        help="Automatically calibrate missing or drifting baselines.",
    ),
    plain: bool = typer.Option(
        False,
        "--plain",
        help="Pure ASCII output mode.",
    ),
) -> None:
    """Run an 8-point pre-flight checklist to verify demo readiness."""
    table = Table(
        title="[DESKFLOW DEMO READINESS PRE-FLIGHT CHECKLIST]",
        box=box.ASCII if plain else box.ROUNDED,
        show_header=True,
    )
    table.add_column("Check", style="bold")
    table.add_column("Status", justify="center")
    table.add_column("Details")

    adapter = get_adapter()
    power_state = get_current_power_state(adapter)
    fails: list[str] = []

    # 1. Power State
    bat = adapter.battery()
    if bat is not None and not bat.plugged_in:
        pwr_status = "OK"
        pwr_detail = f"Battery at {bat.percent:.0f}%"
    else:
        pwr_status = "OK"
        pwr_detail = "AC Connected"
    pwr_cell = f"[green]{pwr_status}[/green]" if not plain else pwr_status
    table.add_row("Power State", pwr_cell, pwr_detail)

    # 2. Baseline
    baseline = None
    with contextlib.suppress(FileNotFoundError, CalibrationError):
        baseline = load_baseline(power_state=power_state)

    if baseline is None:
        if calibrate:
            console.print(f"[cyan]Calibrating baseline for {power_state}...[/cyan]")
            try:
                baseline = calibrate_baseline(duration_s=60.0)
                save_baseline(baseline, power_state=power_state)
                b_status = "OK"
                b_detail = f"Calibrated ({baseline.total_ns / 1_000_000:.2f} ms)"
            except Exception as e:
                b_status = "FAIL"
                b_detail = f"Calibration failed: {e}"
                fails.append("Baseline calibration failed")
        else:
            b_status = "FAIL"
            b_detail = f"Missing for {power_state} (run with --calibrate)"
            fails.append("Baseline missing")
    else:
        age_str = get_baseline_age_str(baseline)
        age_days = get_baseline_age_days(baseline)
        is_drifting, drift_ratio, _ = check_baseline_drift(baseline, num_probes=3)
        if is_drifting:
            if calibrate:
                console.print(f"[cyan]Recalibrating drifting baseline for {power_state}...[/cyan]")
                try:
                    baseline = calibrate_baseline(duration_s=60.0)
                    save_baseline(baseline, power_state=power_state)
                    b_status = "OK"
                    b_detail = f"Recalibrated ({baseline.total_ns / 1_000_000:.2f} ms)"
                except Exception as e:
                    b_status = "WARN"
                    b_detail = f"Drifted ({drift_ratio:.2f}x), recalibration failed: {e}"
            else:
                b_status = "WARN"
                b_detail = f"Drifted ({drift_ratio:.2f}x, {age_str})"
        elif age_days > 7.0:
            b_status = "WARN"
            b_detail = f"Age > 7 days ({age_str})"
        else:
            b_status = "OK"
            b_detail = f"Valid ({age_str}, {baseline.total_ns / 1_000_000:.2f} ms)"

    b_style = "green" if b_status == "OK" else ("yellow" if b_status == "WARN" else "red")
    table.add_row(
        "Baseline",
        f"[{b_style}]{b_status}[/{b_style}]" if not plain else b_status,
        b_detail,
    )

    # 3. Models
    try:
        model_dir, is_bundled = resolve_model_dir()
        runtime = RuntimeModels.load(model_dir)
        is_synth = bool(runtime.card.get("synthetic"))
        if is_synth:
            m_status = "WARN"
            m_detail = "Synthetic demo models loaded (DEMO)"
        else:
            m_status = "OK"
            m_detail = f"Production models loaded ({model_dir.name})"
    except Exception as e:
        m_status = "FAIL"
        m_detail = f"Failed to load: {e}"
        fails.append("Models failed to load")
    m_style = "green" if m_status == "OK" else ("yellow" if m_status == "WARN" else "red")
    m_cell = f"[{m_style}]{m_status}[/{m_style}]" if not plain else m_status
    table.add_row("ML Models", m_cell, m_detail)

    # 4. Probe Stability (12 probes)
    probes_ns: list[float] = []
    for _ in range(12):
        time.sleep(0.1)
        probes_ns.append(float(run_probe().total_ns))
    med_ns = float(np.median(probes_ns))
    p95_ns = float(np.percentile(probes_ns, 95))
    jitter_ratio = (p95_ns / med_ns) if med_ns > 0 else 1.0
    if jitter_ratio <= 1.4:
        s_status = "OK"
        s_detail = f"p95/median = {jitter_ratio:.2f} <= 1.4"
    else:
        s_status = "WARN"
        s_detail = f"Jitter high: p95/median = {jitter_ratio:.2f} > 1.4"
    s_style = "green" if s_status == "OK" else "yellow"
    s_cell = f"[{s_style}]{s_status}[/{s_style}]" if not plain else s_status
    table.add_row("Probe Stability", s_cell, s_detail)

    # 5. Free Disk Space
    usage = shutil.disk_usage(".")
    free_gb = usage.free / (1024**3)
    if free_gb >= 1.0:
        d_status = "OK"
        d_detail = f"{free_gb:.1f} GB free"
    else:
        d_status = "FAIL"
        d_detail = f"Low disk space: {free_gb:.1f} GB < 1.0 GB"
        fails.append("Low disk space")
    d_style = "green" if d_status == "OK" else "red"
    d_cell = f"[{d_style}]{d_status}[/{d_style}]" if not plain else d_status
    table.add_row("Disk Space", d_cell, d_detail)

    # 6. CPU Utilisation under 15%
    psutil.cpu_percent(interval=None)
    cpu_readings: list[float] = []
    for _ in range(5):
        time.sleep(0.3)
        cpu_readings.append(psutil.cpu_percent(interval=None))
    mean_cpu = float(np.mean(cpu_readings))
    if mean_cpu < 15.0:
        c_status = "OK"
        c_detail = f"{mean_cpu:.1f}% (< 15%)"
    else:
        c_status = "WARN"
        c_detail = f"CPU busy: {mean_cpu:.1f}% (>= 15%)"
    c_style = "green" if c_status == "OK" else "yellow"
    c_cell = f"[{c_style}]{c_status}[/{c_style}]" if not plain else c_status
    table.add_row("CPU Idle Check", c_cell, c_detail)

    # 7. Terminal UTF-8 and width >= 100
    enc = sys.stdout.encoding or ""
    cols, _ = shutil.get_terminal_size(fallback=(100, 30))
    is_utf8 = "utf" in enc.lower()
    if is_utf8 and cols >= 100:
        t_status = "OK"
        t_detail = f"{enc.upper()}, {cols} cols"
    elif cols >= 80:
        t_status = "WARN"
        t_detail = f"{enc}, {cols} cols (< 100 cols recommended)"
    else:
        t_status = "WARN"
        t_detail = f"{enc}, {cols} cols (narrow terminal)"
    t_style = "green" if t_status == "OK" else "yellow"
    t_cell = f"[{t_style}]{t_status}[/{t_style}]" if not plain else t_status
    table.add_row("Terminal UI", t_cell, t_detail)

    # 8. Demo App (tkinter)
    try:
        import tkinter

        # Check basic tkinter init
        tk_root = tkinter.Tk()
        tk_root.withdraw()
        tk_root.destroy()
        tk_status = "OK"
        tk_detail = "tkinter functional"
    except Exception as e:
        tk_status = "WARN"
        tk_detail = f"tkinter unavailable: {e}"
    tk_style = "green" if tk_status == "OK" else "yellow"
    tk_cell = f"[{tk_style}]{tk_status}[/{tk_style}]" if not plain else tk_status
    table.add_row("Demo App GUI", tk_cell, tk_detail)

    console.print(table)
    if fails:
        console.print(
            f"[red bold]Not ready for demo:[/red bold] {', '.join(fails)}"
            if not plain
            else f"Not ready for demo: {', '.join(fails)}"
        )
        raise typer.Exit(code=1)
    else:
        msg = f"Ready for demo on {power_state}"
        console.print(f"[green bold]{msg}[/green bold]" if not plain else msg)
