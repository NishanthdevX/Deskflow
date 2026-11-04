"""CLI commands for fault-injection lab operations."""

import subprocess
import sys
import time
from datetime import UTC, datetime

import typer
from rich.console import Console
from rich.table import Table

from deskflow.lab import (
    Intensity,
    Pattern,
    create_scenario,
)
from deskflow.lab.plans import PlanName, get_plan
from deskflow.lab.runner import LabRunner
from deskflow.paths import get_data_dir

app = typer.Typer(help="Manage and execute controlled fault-injection lab sessions.")
console = Console()


@app.command("demo-app")
def demo_app_command(
    memory_mb: int = typer.Option(2048, "--memory-mb"),
    cpu_percent: float = typer.Option(0.0, "--cpu-percent"),
    foreground: bool = typer.Option(False, "--foreground"),
) -> None:
    """Start the visible demo workload used by the remediation demonstration."""
    try:
        import tkinter  # noqa: F401
    except ImportError:
        console.print("[red]tkinter is unavailable; demo app cannot start.[/red]")
        raise typer.Exit(1) from None
    cmd = [
        sys.executable,
        "-m",
        "deskflow.lab.demo_app",
        "--memory-mb",
        str(memory_mb),
        "--cpu-percent",
        str(cpu_percent),
    ]
    if foreground:
        raise typer.Exit(subprocess.call(cmd))
    flags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    flags |= getattr(subprocess, "DETACHED_PROCESS", 0)
    proc = subprocess.Popen(
        cmd,
        creationflags=flags,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    console.print(f"Demo app started with PID {proc.pid}.")


@app.command("list")
def list_command() -> None:
    """List available fault injection scenarios, intensities, and patterns."""
    scenarios_table = Table(title="Available Fault Injection Scenarios", show_header=True)
    scenarios_table.add_column("Scenario", style="bold")
    scenarios_table.add_column("Description")
    scenarios_table.add_column("Resource Bounds")

    scenarios_table.add_row(
        "cpu_pressure",
        "Multi-core CPU busy loops using spawn context workers.",
        "Low: 25% cores, Med: 50% cores, High: 100% cores",
    )
    scenarios_table.add_row(
        "memory_pressure",
        "Sequential 64 MiB page touching up to available RAM target.",
        "Bounded by max(2 GiB, 15% RAM) safety floor",
    )
    scenarios_table.add_row(
        "disk_pressure",
        "Cyclic 4 MiB fsync disk writes to fixed 512 MiB scratch files.",
        "1-4 writer threads, requires >= 5 GiB free disk",
    )
    scenarios_table.add_row(
        "background_load",
        "Spawns multiple idle child processes each holding pinned memory.",
        "10-40 subprocesses scaled to respect memory floor",
    )
    scenarios_table.add_row(
        "idle",
        "Zero-injection healthy reference baseline.",
        "None",
    )
    scenarios_table.add_row(
        "<combo>",
        "Concurrent scenarios combined with '+' (e.g. cpu_pressure+memory_pressure).",
        "Sum of constituent resource bounds",
    )

    console.print(scenarios_table)

    patterns_table = Table(title="Injection Patterns", show_header=True)
    patterns_table.add_column("Pattern", style="bold")
    patterns_table.add_column("Description")

    patterns_table.add_row(
        "constant", "Full targeted intensity maintained for the entire hold phase."
    )
    patterns_table.add_row(
        "ramp", "Linear intensity ramp from near zero to target across the hold phase."
    )
    patterns_table.add_row(
        "burst", "Cyclic alternation between 20 seconds at target and 20 seconds idle."
    )

    console.print(patterns_table)


@app.command("plan")
def plan_command(
    plan: PlanName = typer.Option(
        "standard",
        "--plan",
        help="Plan configuration: 'standard' (20 runs, ~2h) or 'quick' (5 runs, ~22m).",
    ),
) -> None:
    """Print the execution schedule and duration table for a lab plan."""
    runs = get_plan(plan)
    total_time_s = sum(r.total_duration_s for r in runs)
    cooldown_s = 45.0 * max(0, len(runs) - 1)
    grand_total_s = total_time_s + cooldown_s

    table = Table(title=f"Lab Execution Plan: {plan.upper()}", show_header=True)
    table.add_column("#", justify="right")
    table.add_column("Scenario", style="bold")
    table.add_column("Intensity")
    table.add_column("Pattern")
    table.add_column("Pre (s)", justify="right")
    table.add_column("Hold (s)", justify="right")
    table.add_column("Post (s)", justify="right")
    table.add_column("Total (s)", justify="right")

    for i, r in enumerate(runs, start=1):
        table.add_row(
            str(i),
            r.scenario,
            r.intensity,
            r.pattern,
            f"{r.pre_s:.0f}",
            f"{r.hold_s:.0f}",
            f"{r.post_s:.0f}",
            f"{r.total_duration_s:.0f}",
        )

    console.print(table)
    console.print(
        f"Total runs: {len(runs)} | Execution time: {total_time_s / 60:.1f} min | "
        f"Cooldowns: {cooldown_s / 60:.1f} min | Total: {grand_total_s / 60:.1f} min"
    )


@app.command("run")
def run_command(
    scenario: str = typer.Option(
        ...,
        "--scenario",
        "-s",
        help="Scenario: cpu_pressure, memory_pressure, disk_pressure, "
        "background_load, idle, or combo (e.g. cpu_pressure+memory_pressure).",
    ),
    intensity: Intensity = typer.Option(
        "medium",
        "--intensity",
        "-i",
        help="Stress intensity level: low, medium, or high.",
    ),
    pattern: Pattern = typer.Option(
        "constant",
        "--pattern",
        "-p",
        help="Stress pattern: constant, ramp, or burst.",
    ),
    pre: float = typer.Option(
        30.0,
        "--pre",
        help="Pre-injection baseline duration in seconds (default: 30s).",
    ),
    hold: float = typer.Option(
        180.0,
        "--hold",
        help="Active stress injection duration in seconds (default: 180s).",
    ),
    post: float = typer.Option(
        60.0,
        "--post",
        help="Post-injection recovery duration in seconds (default: 60s).",
    ),
    yes: bool = typer.Option(
        False,
        "--yes",
        "-y",
        help="Skip interactive confirmation.",
    ),
    storage: bool = typer.Option(
        False,
        "--storage",
        help="Enable background storage cache scanning.",
    ),
) -> None:
    """Execute a single fault injection lab run with continuous telemetry recording."""
    total_time_s = pre + hold + post
    console.print("CAUTION: This command will generate controlled hardware stress on this machine.")
    console.print(
        f"Target: scenario={scenario}, intensity={intensity}, pattern={pattern}, "
        f"duration={total_time_s:.0f}s."
    )
    console.print(
        "A safety watchdog will monitor RAM (>2GB) and disk (>5GB) and abort if breached."
    )

    if not yes:
        confirmed = typer.confirm("Do you wish to proceed with fault injection?")
        if not confirmed:
            console.print("Operation canceled by user.")
            raise typer.Exit(code=0)

    scen = create_scenario(scenario_str=scenario, intensity=intensity, pattern=pattern)

    def _status(phase: str, elapsed: float, total: float) -> None:
        pct = (elapsed / total) * 100 if total > 0 else 0
        sys.stdout.write(
            f"\rPhase: {phase.upper():<6} | Elapsed: {elapsed:5.1f}s / {total:5.1f}s ({pct:4.1f}%)"
        )
        sys.stdout.flush()

    runner = LabRunner(
        scenario=scen,
        pre_duration_s=pre,
        hold_duration_s=hold,
        post_duration_s=post,
        interval_s=2.0,
        probe_interval_s=4.0,
        status_callback=_status,
        enable_storage=storage,
    )

    console.print(f"Starting lab run (session={runner.session_id})...")
    output_path = runner.run()
    console.print("\n")
    console.print(f"[green]Run complete.[/green] Recorded: {output_path}")
    console.print(f"Sidecar metadata: {output_path.with_suffix('.meta.json')}")


@app.command("session")
def session_command(
    plan: PlanName = typer.Option(
        "standard",
        "--plan",
        help="Plan configuration: 'standard' or 'quick'.",
    ),
    resume: str | None = typer.Option(
        None,
        "--resume",
        help="Optional session ID to resume, skipping already completed runs.",
    ),
    yes: bool = typer.Option(
        False,
        "--yes",
        "-y",
        help="Skip interactive confirmation.",
    ),
    storage: bool = typer.Option(
        False,
        "--storage",
        help="Enable background storage cache scanning.",
    ),
) -> None:
    """Execute a complete multi-run lab plan sequentially with inter-run cooldown."""
    runs = get_plan(plan)
    total_dur_m = (sum(r.total_duration_s for r in runs) + 45.0 * (len(runs) - 1)) / 60.0

    session_id = resume or f"lab-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
    console.print(f"Starting lab session: {session_id} (Plan: {plan.upper()})")
    console.print(f"Total plan runs: {len(runs)} (estimated time: {total_dur_m:.1f} minutes).")

    # Find completed runs in session if resuming
    completed_keys: set[str] = set()
    runs_dir = get_data_dir() / "runs"
    if runs_dir.exists():
        for meta_file in runs_dir.glob("*.meta.json"):
            try:
                import json

                with open(meta_file, encoding="utf-8") as f:
                    meta = json.load(f)
                if meta.get("session_id") == session_id and not meta.get("aborted"):
                    s = meta.get("scenario")
                    i = meta.get("intensity")
                    p = meta.get("pattern")
                    if s and i and p:
                        completed_keys.add(f"{s}:{i}:{p}")
            except Exception:
                pass

    if completed_keys:
        console.print(
            f"Resuming session: {len(completed_keys)} run(s) previously completed will be skipped."
        )

    if not yes:
        confirmed = typer.confirm("Do you wish to begin the lab session?")
        if not confirmed:
            console.print("Session canceled by user.")
            raise typer.Exit(code=0)

    for idx, spec in enumerate(runs, start=1):
        if spec.key in completed_keys:
            console.print(f"[{idx}/{len(runs)}] SKIPPING (already completed): {spec.key}")
            continue

        console.print(
            f"\n[{idx}/{len(runs)}] Running {spec.scenario} ({spec.intensity}, {spec.pattern}) "
            f"for {spec.total_duration_s:.0f}s..."
        )

        scen = create_scenario(
            scenario_str=spec.scenario,
            intensity=spec.intensity,
            pattern=spec.pattern,
        )

        def _status(phase: str, elapsed: float, total: float) -> None:
            pct = (elapsed / total) * 100 if total > 0 else 0
            sys.stdout.write(
                f"\rPhase: {phase.upper():<6} | Elapsed: {elapsed:5.1f}s / {total:5.1f}s "
                f"({pct:4.1f}%)"
            )
            sys.stdout.flush()

        runner = LabRunner(
            scenario=scen,
            pre_duration_s=spec.pre_s,
            hold_duration_s=spec.hold_s,
            post_duration_s=spec.post_s,
            session_id=session_id,
            status_callback=_status,
        )

        out_path = runner.run()
        console.print(f"\nRun finished: {out_path.name}")
        completed_keys.add(spec.key)

        # Cooldown between runs
        if idx < len(runs):
            cooldown_s = spec.cooldown_s
            console.print(f"Cooling down system for {cooldown_s:.0f}s before next run...")
            time.sleep(cooldown_s)

    console.print(f"\n[green]Lab session {session_id} finished successfully.[/green]")
