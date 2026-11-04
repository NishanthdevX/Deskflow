"""Fix CLI command: execute user-confirmed remediation actions and verify impact with re-scan."""

import logging
import time
from datetime import UTC, datetime
from typing import Any

import typer
from rich import box
from rich.console import Console
from rich.table import Table

from deskflow.baseline import get_current_power_state, load_baseline
from deskflow.collector.probe import run_probe
from deskflow.collector.sampler import Sampler
from deskflow.collector.storage import StorageScanner
from deskflow.features.extract import build_row
from deskflow.health.score import compute_health_score
from deskflow.ml.artifacts import load_model_card
from deskflow.paths import get_data_dir
from deskflow.platform.factory import get_adapter
from deskflow.remediation.actions import (
    ActionResult,
    clean_temp,
    close_idle_apps,
    end_busy_process,
    trim_memory,
)
from deskflow.remediation.log import append_fix_log
from deskflow.remediation.recommend import recommend_fixes

logger = logging.getLogger(__name__)
console = Console()


def _gather_state(adapter: Any, baseline: Any, runtime: Any, card: Any) -> dict[str, Any]:
    """Capture current system metrics, probe latency, and predicted health score."""
    import numpy as np

    sampler = Sampler(adapter=adapter, interval_s=1.0)
    data_dir = get_data_dir()

    # Run fresh probe
    probe_res = run_probe(data_dir=data_dir, repeats=3)
    sys_sample, top_procs = sampler.sample()

    from deskflow.collector.models import Record

    rec = Record(
        schema_version=1,
        run_id="fix-measure",
        label="fix",
        sample=sys_sample,
        probe=probe_res,
        top_processes=top_procs,
        storage=None,
    )

    feat_dict = build_row(records_up_to_t=[rec], baseline=baseline, current_record=rec)
    vec = np.array([feat_dict[name] for name in runtime.feature_order], dtype=np.float64)
    preds = runtime.predict(vec)

    primary_reg = card.get("primary_models", {}).get("regressor", "XGBoost")
    slowdown_preds = preds.get("slowdown_ratio", {})
    predicted_ratio = float(slowdown_preds.get(primary_reg, slowdown_preds.get("XGBoost", 1.0)))
    score, label = compute_health_score(predicted_ratio)

    scanner = StorageScanner(adapter=adapter, time_budget_s=2.0)
    storage_report = scanner.scan()

    safe_roots = {loc.path.lower() for loc in adapter.temp_and_cache_dirs() if loc.safe_to_clean}
    temp_bytes = 0
    if storage_report and storage_report.entries:
        for entry in storage_report.entries:
            if entry.kind in ("temp", "cache") and entry.path.lower() in safe_roots:
                temp_bytes += getattr(entry, "cleanable_bytes", entry.bytes)

    return {
        "score": score,
        "label": label,
        "predicted_ratio": predicted_ratio,
        "probe_ms": probe_res.total_ns / 1_000_000.0,
        "available_memory_bytes": sys_sample.mem_available_bytes,
        "temp_bytes": temp_bytes,
        "top_processes": top_procs,
        "storage_report": storage_report,
    }


def fix_command(
    actions: list[str] = typer.Option(
        None,
        "--action",
        "-a",
        help="One or more actions to execute directly (clean_temp, close_idle_apps, trim_memory).",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Simulate actions and print expected impact without making changes.",
    ),
    yes: bool = typer.Option(
        False,
        "--yes",
        "-y",
        help="Bypass interactive selection and confirmation prompts.",
    ),
    no_rescan: bool = typer.Option(
        False,
        "--no-rescan",
        help="Skip post-fix verification scan.",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        help="Allow process termination after graceful close timeout with typed confirmation.",
    ),
    plain: bool = typer.Option(False, "--plain"),
) -> None:
    """Diagnose system, execute confirmed safe remediation actions, and re-scan impact."""
    from deskflow.ui.theme import use_plain_output

    plain = use_plain_output(plain)
    console = Console(no_color=plain)
    adapter = get_adapter()
    power_state = get_current_power_state(adapter)

    try:
        baseline = load_baseline(power_state=power_state)
    except FileNotFoundError as err:
        err_msg = (
            "[red]Error: Baseline required for fix scan. "
            "Run 'deskflow baseline calibrate' first.[/red]"
        )
        console.print(err_msg)
        raise typer.Exit(code=1) from err

    from deskflow.ml.artifacts import resolve_model_dir
    from deskflow.ml.runtime import RuntimeModels

    try:
        model_dir, _is_bundled = resolve_model_dir()
    except FileNotFoundError as err:
        console.print("[red]Error: No trained model set found. Run 'deskflow train' first.[/red]")
        raise typer.Exit(code=1) from err

    card = load_model_card(model_dir)
    runtime = RuntimeModels.load(model_dir)

    console.print("[bold]Diagnosing current system state...[/bold]")
    before = _gather_state(adapter=adapter, baseline=baseline, runtime=runtime, card=card)

    console.print(
        f"Initial Health Score: [bold]{before['score']}/100 ({before['label']})[/bold] | "
        f"Predicted Slowdown: {before['predicted_ratio']:.2f}x | "
        f"Probe: {before['probe_ms']:.1f} ms"
    )

    recs = recommend_fixes(
        top_processes=before["top_processes"],
        storage_report=before["storage_report"],
        issue_tag="healthy",
        is_anomaly=False,
        adapter=adapter,
    )

    # Resolve actions to run
    chosen_actions: list[str] = []

    if actions:
        chosen_actions = list(actions)
    else:
        if not recs:
            console.print("[green]No remediation actions needed. System is healthy.[/green]")
            return

        console.print("\n[bold]Available Fix Actions:[/bold]")
        for idx, r in enumerate(recs, start=1):
            warn_str = f"\n     [yellow]Warning: {r.warning}[/yellow]" if r.warning else ""
            console.print(
                f"  {idx}. [bold]{r.title}[/bold] ({r.action_id}): {r.evidence}{warn_str}"
            )

        if yes:
            chosen_actions = [r.action_id for r in recs]
        else:
            selection = typer.prompt(
                "\nSelect action numbers to execute (comma-separated, default none)",
                default="",
                show_default=False,
            )
            if not selection.strip():
                console.print("No actions selected. Exiting.")
                return

            indices = [int(s.strip()) for s in selection.split(",") if s.strip().isdigit()]
            for i in indices:
                if 1 <= i <= len(recs):
                    chosen_actions.append(recs[i - 1].action_id)

    if not chosen_actions:
        console.print("No valid actions selected. Exiting.")
        return

    # Confirmation
    if not dry_run and not yes:
        action_names = ", ".join(chosen_actions)
        confirmed = typer.confirm(f"Execute action(s): {action_names}?", default=True)
        if not confirmed:
            console.print("Aborted by user.")
            return

    def _confirm_force(app_name: str, pid: int) -> bool:
        console.print(
            f"\n[bold red]WARNING: Graceful close timed out. "
            f"Unsaved work in {app_name} will be lost.[/bold red]"
        )
        typed: str = typer.prompt(
            f"Force terminate {app_name} (PID {pid})? Type 'yes' to kill",
            default="no",
            show_default=False,
        )
        return bool(typed.strip().lower() == "yes")

    # Execute Actions
    results: list[ActionResult] = []
    console.print("\n[bold]Executing actions...[/bold]")

    for act in chosen_actions:
        if act == "clean_temp":
            res = clean_temp(adapter=adapter, dry_run=dry_run)
        elif act == "close_idle_apps":
            res = close_idle_apps(
                before["top_processes"],
                adapter=adapter,
                dry_run=dry_run,
                force=force,
                confirm_force_cb=_confirm_force if force else None,
            )
        elif act == "end_busy_process":
            res = end_busy_process(
                before["top_processes"],
                adapter=adapter,
                dry_run=dry_run,
                force=force,
                confirm_force_cb=_confirm_force if force else None,
            )
        elif act == "trim_memory":
            res = trim_memory(before["top_processes"], adapter=adapter, dry_run=dry_run)
        else:
            res = ActionResult(action=act, success=False, message=f"Unknown action: {act}")

        results.append(res)
        status_color = "green" if res.success else "red"
        console.print(f"[{status_color}]* {act}: {res.message}[/{status_color}]")

    after: dict[str, Any] | None = None

    if not dry_run and not no_rescan:
        console.print("\nWaiting 3 seconds for system state to stabilize...")
        time.sleep(3.0)

        console.print("[bold]Running verification re-scan...[/bold]")
        after = _gather_state(adapter=adapter, baseline=baseline, runtime=runtime, card=card)

        # Before & After Table
        table = Table(
            title="Remediation Verification",
            box=box.ASCII if plain else box.SQUARE,
            show_header=True,
        )
        table.add_column("Metric", style="bold")
        table.add_column("Before", justify="right")
        table.add_column("After", justify="right")
        table.add_column("Change", justify="right")

        score_diff = after["score"] - before["score"]
        s_sign = "+" if score_diff >= 0 else ""
        table.add_row(
            "Health Score",
            f"{before['score']}/100",
            f"{after['score']}/100",
            f"{s_sign}{score_diff}",
        )

        ratio_diff = after["predicted_ratio"] - before["predicted_ratio"]
        r_sign = "+" if ratio_diff >= 0 else ""
        table.add_row(
            "Predicted Slowdown",
            f"{before['predicted_ratio']:.2f}x",
            f"{after['predicted_ratio']:.2f}x",
            f"{r_sign}{ratio_diff:.2f}x",
        )

        probe_diff = after["probe_ms"] - before["probe_ms"]
        p_sign = "+" if probe_diff >= 0 else ""
        table.add_row(
            "Probe Latency",
            f"{before['probe_ms']:.1f} ms",
            f"{after['probe_ms']:.1f} ms",
            f"{p_sign}{probe_diff:.1f} ms",
        )

        mem_before_gb = before["available_memory_bytes"] / (1024**3)
        mem_after_gb = after["available_memory_bytes"] / (1024**3)
        m_diff = mem_after_gb - mem_before_gb
        m_sign = "+" if m_diff >= 0 else ""
        table.add_row(
            "Available Memory",
            f"{mem_before_gb:.2f} GB",
            f"{mem_after_gb:.2f} GB",
            f"{m_sign}{m_diff:.2f} GB",
        )

        temp_before_mb = before["temp_bytes"] / (1024 * 1024)
        temp_after_mb = after["temp_bytes"] / (1024 * 1024)
        t_diff = temp_after_mb - temp_before_mb
        t_sign = "+" if t_diff >= 0 else ""
        table.add_row(
            "Cleanable Temp",
            f"{temp_before_mb:.1f} MB",
            f"{temp_after_mb:.1f} MB",
            f"{t_sign}{t_diff:.1f} MB",
        )

        console.print(table)

        if after["score"] > before["score"]:
            console.print("[green]Health score improved successfully.[/green]")
        elif after["score"] == before["score"]:
            console.print("[yellow]Health score remained unchanged.[/yellow]")
        else:
            console.print("[yellow]Health score did not improve after remediation.[/yellow]")
    elif dry_run:
        console.print("\n[dim]Dry run complete. No modifications were made to the system.[/dim]")

    # Audit Logging (Fix log honesty: dry-run records store estimated_* and null bytes_freed/after)
    if dry_run:
        results_log = [
            {
                "action": r.action,
                "success": r.success,
                "message": r.message,
                "bytes_freed": None,
                "estimated_bytes_freed": r.bytes_freed,
                "estimated_details": r.details,
            }
            for r in results
        ]
        log_record = {
            "ts_utc": datetime.now(UTC).isoformat(),
            "actions": chosen_actions,
            "dry_run": True,
            "results": results_log,
            "before": {
                "score": before["score"],
                "predicted_ratio": before["predicted_ratio"],
                "probe_ms": before["probe_ms"],
                "available_memory_bytes": before["available_memory_bytes"],
                "temp_bytes": before["temp_bytes"],
            },
            "after": None,
        }
    else:
        results_log = [
            {
                "action": r.action,
                "success": r.success,
                "message": r.message,
                "bytes_freed": r.bytes_freed,
                "details": r.details,
            }
            for r in results
        ]
        log_record = {
            "ts_utc": datetime.now(UTC).isoformat(),
            "actions": chosen_actions,
            "dry_run": False,
            "results": results_log,
            "before": {
                "score": before["score"],
                "predicted_ratio": before["predicted_ratio"],
                "probe_ms": before["probe_ms"],
                "available_memory_bytes": before["available_memory_bytes"],
                "temp_bytes": before["temp_bytes"],
            },
            "after": {
                "score": after["score"],
                "predicted_ratio": after["predicted_ratio"],
                "probe_ms": after["probe_ms"],
                "available_memory_bytes": after["available_memory_bytes"],
                "temp_bytes": after["temp_bytes"],
            }
            if after is not None
            else None,
        }

    log_path = append_fix_log(log_record)
    console.print(f"[dim]Fix execution logged to {log_path}[/dim]")
