"""Fix Center CLI: fast inventory, safe remediation execution, and re-scan verification."""

from __future__ import annotations

import contextlib
import json
import logging
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import typer
from rich import box
from rich.console import Console
from rich.table import Table
from rich.text import Text

from deskflow.baseline import get_current_power_state, load_baseline
from deskflow.collector.models import Record
from deskflow.collector.probe import run_probe
from deskflow.collector.sampler import Sampler
from deskflow.collector.storage import StorageScanner
from deskflow.features.extract import build_row
from deskflow.health.score import compute_health_score
from deskflow.ml.artifacts import load_model_card, resolve_model_dir
from deskflow.ml.runtime import RuntimeModels
from deskflow.paths import get_data_dir
from deskflow.platform.base import PlatformAdapter
from deskflow.platform.factory import get_adapter
from deskflow.remediation.actions import (
    ActionResult,
    HeavyTaskGroup,
    clean_app_caches,
    clean_temp,
    close_idle_apps,
    empty_recycle_bin,
    end_heavy_tasks,
    find_heavy_tasks,
    trim_memory,
)
from deskflow.remediation.apps import find_idle_apps
from deskflow.remediation.log import append_fix_log

logger = logging.getLogger(__name__)
app = typer.Typer(help="Diagnose system, inspect fix tools, and execute verified remediations.")
console = Console()


@dataclass(frozen=True)
class FixToolInfo:
    """Metadata and measured potential for a Fix Center remediation tool."""

    number: int
    tool_id: str
    title: str
    risk: str  # "SAFE" or "CAUTION"
    potential_str: str
    potential_bytes: int
    what_it_changes: str
    details: dict[str, Any]


def run_fast_inventory(adapter: PlatformAdapter) -> list[FixToolInfo]:
    """Execute fast inventory in < 5 seconds: 3s storage scan + one process pass."""
    # 1. Fast storage scan (3s budget)
    scanner = StorageScanner(adapter=adapter, time_budget_s=3.0)
    storage_report = scanner.scan()
    safe_roots = {loc.path.lower() for loc in adapter.temp_and_cache_dirs() if loc.safe_to_clean}
    cleanable_temp_bytes = 0
    if storage_report and storage_report.entries:
        for entry in storage_report.entries:
            if entry.kind in ("temp", "cache") and entry.path.lower() in safe_roots:
                cleanable_temp_bytes += getattr(entry, "cleanable_bytes", entry.bytes)

    # 2. Browser & app caches
    app_cache_locs = adapter.app_cache_dirs()
    total_app_cache_bytes = 0
    app_cache_count = 0
    for loc in app_cache_locs:
        p = Path(loc.path)
        if p.exists() and p.is_dir():
            try:
                for f in p.glob("*"):
                    try:
                        if f.is_file() and not f.is_symlink():
                            total_app_cache_bytes += f.stat().st_size
                            app_cache_count += 1
                    except (PermissionError, OSError):
                        pass
            except (PermissionError, OSError):
                pass

    # 3. Recycle Bin
    rb_items, rb_bytes = adapter.query_recycle_bin()

    # 4. Full process pass
    all_procs = adapter.full_process_snapshot()

    # 5. Trim memory potential (non-protected RSS >= 50MB)
    trim_procs = [
        p for p in all_procs if not p.is_protected and p.rss_bytes >= 50 * 1024 * 1024 and p.pid > 4
    ]
    trim_bytes = sum(p.rss_bytes for p in trim_procs)

    # 6. Idle apps
    idle_apps = find_idle_apps(all_procs, adapter=adapter)
    idle_bytes = sum(g.total_rss_bytes for g in idle_apps)

    # 7. Heavy tasks
    heavy_tasks = find_heavy_tasks(adapter, all_processes=all_procs)
    heavy_bytes = sum(g.rss_bytes for g in heavy_tasks)

    tools = [
        FixToolInfo(
            number=1,
            tool_id="clean_temp",
            title="Clean temporary files",
            risk="SAFE",
            potential_str=f"{cleanable_temp_bytes / (1024**2):.1f} MB cleanable",
            potential_bytes=cleanable_temp_bytes,
            what_it_changes="Deletes files older than 24h in temp roots",
            details={"cleanable_bytes": cleanable_temp_bytes},
        ),
        FixToolInfo(
            number=2,
            tool_id="clean_app_caches",
            title="Clean browser & app caches",
            risk="SAFE",
            potential_str=(f"{total_app_cache_bytes / (1024**2):.1f} MB ({app_cache_count} files)"),
            potential_bytes=total_app_cache_bytes,
            what_it_changes="Clears browser caches (pages load slower once)",
            details={
                "cache_bytes": total_app_cache_bytes,
                "files_count": app_cache_count,
            },
        ),
        FixToolInfo(
            number=3,
            tool_id="empty_recycle_bin",
            title="Empty Recycle Bin / Trash",
            risk="CAUTION",
            potential_str=f"{rb_bytes / (1024**2):.1f} MB ({rb_items} items)",
            potential_bytes=rb_bytes,
            what_it_changes="Empties Recycle Bin permanently (irreversible)",
            details={"items": rb_items, "bytes": rb_bytes},
        ),
        FixToolInfo(
            number=4,
            tool_id="trim_memory",
            title="Trim process working sets",
            risk="SAFE",
            potential_str=(
                f"{len(trim_procs)} processes using "
                f"{trim_bytes / (1024**2):.1f} MB (reclaimable amount unknown)"
            ),
            potential_bytes=trim_bytes,
            what_it_changes="Reclaims memory working sets (effect is temporary)",
            details={
                "procs_count": len(trim_procs),
                "rss_bytes": trim_bytes,
                "candidates": trim_procs,
            },
        ),
        FixToolInfo(
            number=5,
            tool_id="close_idle_apps",
            title="Close idle windowed apps",
            risk="CAUTION",
            potential_str=(f"{idle_bytes / (1024**2):.1f} MB across {len(idle_apps)} app group(s)"),
            potential_bytes=idle_bytes,
            what_it_changes="Gracefully closes idle background windowed applications",
            details={"groups": idle_apps, "all_procs": all_procs},
        ),
        FixToolInfo(
            number=6,
            tool_id="end_heavy_tasks",
            title="End heavy background tasks",
            risk="CAUTION",
            potential_str=(
                f"{len(heavy_tasks)} heavy tasks using "
                f"{heavy_bytes / (1024**2):.1f} MB (reclaimable amount unknown)"
            ),
            potential_bytes=heavy_bytes,
            what_it_changes="Terminates top non-protected heavy background processes",
            details={"groups": heavy_tasks, "all_procs": all_procs},
        ),
    ]
    return tools


def _gather_state(adapter: Any, baseline: Any, runtime: Any, card: Any) -> dict[str, Any]:
    """Capture current system metrics, probe latency, and predicted health score."""
    import numpy as np

    sampler = Sampler(adapter=adapter, interval_s=1.0)
    data_dir = get_data_dir()

    probe_res = run_probe(data_dir=data_dir, repeats=3)
    sys_sample, top_procs = sampler.sample()

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


def _render_tools_table(tools: list[FixToolInfo], plain: bool = False) -> Table:
    """Format the Fix Center tools table with dim indicators for empty tools (D22)."""
    from deskflow.ui.components import badge

    table = Table(
        title="[DESKFLOW FIX CENTER - TOOLS INVENTORY]",
        box=box.ASCII if plain else box.ROUNDED,
        show_header=True,
    )
    table.add_column("#", justify="center", style="bold cyan")
    table.add_column("Tool", style="bold")
    table.add_column("Risk", justify="center")
    table.add_column("Measured Potential", justify="right")
    table.add_column("What it changes")

    for t in tools:
        has_work = t.potential_bytes > 0
        risk_cell = badge(t.risk, t.risk, plain=plain)
        pot_disp = t.potential_str if has_work else ("[dim]nothing to do[/dim]" if not plain else "nothing to do")
        if not has_work and not plain:
            table.add_row(
                Text(str(t.number), style="dim"),
                Text(t.title, style="dim"),
                Text(t.risk, style="dim"),
                pot_disp,
                Text(t.what_it_changes, style="dim"),
            )
        else:
            table.add_row(
                str(t.number),
                t.title,
                risk_cell,
                pot_disp,
                t.what_it_changes,
            )
    return table


@app.callback(invoke_without_command=True)
def fix_command(
    actions: list[str] = typer.Option(
        None,
        "--action",
        "-a",
        help="One or more tools to execute directly (e.g. clean_temp, 1, 2).",
    ),
    tool: list[str] = typer.Option(
        None,
        "--tool",
        "-t",
        help="Specify tool ID or number to execute directly.",
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
        help="Execute SAFE tools without interactive prompting (never CAUTION).",
    ),
    include_caution: bool = typer.Option(
        False,
        "--include-caution",
        help="Allow selection of CAUTION tools with --yes or unattended mode.",
    ),
    confirm_caution: str | None = typer.Option(
        None,
        "--confirm-caution",
        help="Comma-separated IDs of CAUTION tools to confirm non-interactively.",
    ),
    list_tools: bool = typer.Option(
        False,
        "--list",
        "-l",
        help="Print tools inventory table without prompting.",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Output inventory or results as JSON.",
    ),
    no_rescan: bool = typer.Option(
        False,
        "--no-rescan",
        help="Skip post-fix verification re-scan.",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        help="Allow process termination after graceful timeout with typed confirmation.",
    ),
    plain: bool = typer.Option(False, "--plain", help="Pure ASCII output."),
) -> None:
    """Diagnose system, execute confirmed safe remediation actions, and verify with re-scan."""
    import typer.models
    from deskflow.ui.theme import use_plain_output

    list_tools = bool(list_tools) if not isinstance(list_tools, typer.models.OptionInfo) else False
    json_output = bool(json_output) if not isinstance(json_output, typer.models.OptionInfo) else False
    dry_run = bool(dry_run) if not isinstance(dry_run, typer.models.OptionInfo) else False
    yes = bool(yes) if not isinstance(yes, typer.models.OptionInfo) else False
    include_caution = bool(include_caution) if not isinstance(include_caution, typer.models.OptionInfo) else False
    no_rescan = bool(no_rescan) if not isinstance(no_rescan, typer.models.OptionInfo) else False
    force = bool(force) if not isinstance(force, typer.models.OptionInfo) else False
    plain = bool(plain) if not isinstance(plain, typer.models.OptionInfo) else False
    actions = list(actions) if (actions is not None and not isinstance(actions, typer.models.OptionInfo)) else None
    tool = list(tool) if (tool is not None and not isinstance(tool, typer.models.OptionInfo)) else None

    plain = use_plain_output(plain)
    adapter = get_adapter()
    power_state = get_current_power_state(adapter)

    # 1. Fast Inventory
    tools = run_fast_inventory(adapter)

    # If --list flag is given, display inventory and exit
    if list_tools:
        if json_output:
            tools_data = [
                {
                    "number": t.number,
                    "tool_id": t.tool_id,
                    "title": t.title,
                    "risk": t.risk,
                    "potential": t.potential_str,
                    "potential_bytes": t.potential_bytes,
                    "what_it_changes": t.what_it_changes,
                }
                for t in tools
            ]
            console.print_json(json.dumps(tools_data))
        else:
            console.print(_render_tools_table(tools, plain=plain))
        return

    # Check baseline & models for re-scan
    baseline = None
    with contextlib.suppress(Exception):
        baseline = load_baseline(power_state=power_state)

    runtime = None
    card = {}
    with contextlib.suppress(Exception):
        model_dir, _ = resolve_model_dir()
        card = load_model_card(model_dir)
        runtime = RuntimeModels.load(model_dir)

    # Gather baseline state
    before: dict[str, Any] | None = None
    if baseline and runtime:
        console.print("[dim]Diagnosing current system state...[/dim]")
        before = _gather_state(adapter=adapter, baseline=baseline, runtime=runtime, card=card)

    # Combine input tools from --tool and --action
    input_specs: list[str] = []
    if actions:
        input_specs.extend(actions)
    if tool:
        input_specs.extend(tool)

    chosen_tools: list[FixToolInfo] = []

    if input_specs:
        tool_by_id = {t.tool_id.lower(): t for t in tools}
        tool_by_num = {str(t.number): t for t in tools}
        for spec in input_specs:
            s_clean = spec.strip().lower()
            if s_clean in ("all", "*"):
                chosen_tools = list(tools)
                break
            elif s_clean == "safe":
                for t in tools:
                    if t.risk == "SAFE" and t not in chosen_tools:
                        chosen_tools.append(t)
            elif s_clean in tool_by_num:
                t = tool_by_num[s_clean]
                if t not in chosen_tools:
                    chosen_tools.append(t)
            elif s_clean in tool_by_id:
                t = tool_by_id[s_clean]
                if t not in chosen_tools:
                    chosen_tools.append(t)
            elif s_clean == "end_busy_process":  # backward compatibility alias
                t = tool_by_id["end_heavy_tasks"]
                if t not in chosen_tools:
                    chosen_tools.append(t)
    elif yes:
        # --yes automatically selects all SAFE tools; NEVER CAUTION unless --include-caution
        for t in tools:
            if t.risk == "SAFE" or include_caution:
                chosen_tools.append(t)
    else:
        # Interactive Fix Center mode
        console.print(_render_tools_table(tools, plain=plain))
        console.print(
            "\n[bold cyan]Select tools to run (e.g. '1,2,4'), 'safe' for SAFE tools, "
            "or 'Q' to quit:[/bold cyan]"
        )
        choice = typer.prompt("Select", default="safe", show_default=True).strip()
        if choice.upper() in ("Q", "QUIT"):
            console.print("Exited Fix Center.")
            return

        tool_by_num = {str(t.number): t for t in tools}
        if choice.lower() == "safe":
            chosen_tools = [t for t in tools if t.risk == "SAFE"]
        elif choice.lower() in ("all", "*"):
            chosen_tools = list(tools)
        else:
            for item in choice.replace(" ", ",").split(","):
                part = item.strip()
                if part in tool_by_num:
                    t = tool_by_num[part]
                    if t not in chosen_tools:
                        chosen_tools.append(t)

    if not chosen_tools:
        console.print("[yellow]No tools selected. Exiting.[/yellow]")
        return

    # Filter out CAUTION tools if --yes was specified without --include-caution
    if yes and not include_caution:
        skipped_caution = [t.title for t in chosen_tools if t.risk == "CAUTION"]
        chosen_tools = [t for t in chosen_tools if t.risk == "SAFE"]
        if skipped_caution:
            console.print(
                f"[yellow]Note: Skipped CAUTION tools under --yes: "
                f"{', '.join(skipped_caution)} (use --include-caution to run).[/yellow]"
            )

    # Parse confirmed caution IDs from --confirm-caution
    confirmed_caution_ids: set[str] = set()
    if confirm_caution:
        for cid in confirm_caution.replace(" ", ",").split(","):
            c_str = cid.strip().lower()
            if c_str:
                confirmed_caution_ids.add(c_str)

    # Validate CAUTION tool confirmation in interactive mode or prompt
    confirmed_tools: list[FixToolInfo] = []
    for t in chosen_tools:
        if t.risk == "SAFE" or dry_run:
            confirmed_tools.append(t)
            continue

        # 1. Non-interactive confirmation via --confirm-caution
        if t.tool_id.lower() in confirmed_caution_ids:
            confirmed_tools.append(t)
            continue

        # 2. Under --yes or unattended mode, CAUTION tools require --confirm-caution
        if yes:
            console.print(
                f"[yellow]Skipped CAUTION tool '{t.tool_id}' under --yes: "
                f"unattended execution requires explicit --confirm-caution {t.tool_id}[/yellow]"
            )
            continue

        # 3. Interactive typed confirmation of the tool id
        console.print(f"\n[bold yellow][CAUTION TOOL: {t.title.upper()}][/bold yellow]")
        if t.tool_id == "empty_recycle_bin":
            rb_items = t.details.get("items", 0)
            rb_bytes = t.details.get("bytes", 0)
            console.print(
                f"Recycle Bin contains {rb_items} item(s) totalling {rb_bytes / (1024**2):.1f} MB."
            )
            console.print("[bold red]This will permanently delete all recycled files.[/bold red]")
            typed = typer.prompt(f"Type '{t.tool_id}' to confirm permanent deletion", default="")
            if typed.strip().lower() in (t.tool_id.lower(), "empty"):
                confirmed_tools.append(t)
            else:
                console.print(f"[dim]Skipped {t.title}.[/dim]")

        elif t.tool_id == "close_idle_apps":
            groups = t.details.get("groups", [])
            if not groups:
                console.print("No idle applications currently eligible to be closed.")
                confirmed_tools.append(t)
            else:
                console.print("The following idle application groups will be gracefully closed:")
                for g in groups:
                    console.print(
                        f"  - {g.name}: {len(g.pids)} process(es), "
                        f"{g.total_rss_bytes / (1024**2):.1f} MB RSS (PIDs: {list(g.pids)})"
                    )
                typed = typer.prompt(
                    f"Type '{t.tool_id}' to confirm closing idle applications", default=""
                )
                if typed.strip().lower() in (t.tool_id.lower(), "close"):
                    confirmed_tools.append(t)
                else:
                    console.print(f"[dim]Skipped {t.title}.[/dim]")

        elif t.tool_id == "end_heavy_tasks":
            heavy_groups: list[HeavyTaskGroup] = t.details.get("groups", [])
            if not heavy_groups:
                console.print("No heavy background tasks currently eligible to be ended.")
                confirmed_tools.append(t)
            else:
                table_heavy = Table(title="Top Heavy Tasks", show_header=True)
                table_heavy.add_column("#", justify="right")
                table_heavy.add_column("Process Name", style="bold")
                table_heavy.add_column("Processes", justify="right")
                table_heavy.add_column("CPU %", justify="right")
                table_heavy.add_column("RSS Memory", justify="right")
                for i, hg in enumerate(heavy_groups[:5], 1):
                    table_heavy.add_row(
                        str(i),
                        hg.name,
                        str(hg.process_count),
                        f"{hg.cpu_percent:.1f}%",
                        f"{hg.rss_bytes / (1024**2):.1f} MB",
                    )
                console.print(table_heavy)
                console.print("Enter numbers of tasks to end (e.g. '1,3'), or 'Q' to skip:")
                picks = typer.prompt("Select tasks", default="Q").strip()
                if picks.upper() in ("Q", "QUIT", ""):
                    console.print(f"[dim]Skipped {t.title}.[/dim]")
                else:
                    selected_names = []
                    for p in picks.replace(" ", ",").split(","):
                        if p.isdigit():
                            idx = int(p) - 1
                            if 0 <= idx < len(heavy_groups[:5]):
                                selected_names.append(heavy_groups[idx].name)
                    if selected_names:
                        t.details["task_names"] = selected_names
                        confirmed_tools.append(t)
                    else:
                        console.print(f"[dim]Skipped {t.title}.[/dim]")

    if not confirmed_tools:
        console.print("[yellow]No confirmed tools to execute.[/yellow]")
        return

    # Execute tools
    console.print(f"\n[bold]{'[Dry-run] Simulating' if dry_run else 'Executing'} actions...[/bold]")
    results: list[ActionResult] = []

    def _confirm_force(app_name: str, pid: int) -> bool:
        console.print(
            f"\n[bold red]WARNING: Graceful close timed out. "
            f"Unsaved work in {app_name} will be lost.[/bold red]"
        )
        typed_kill = typer.prompt(
            f"Force terminate {app_name} (PID {pid})? Type 'yes' to kill",
            default="no",
            show_default=False,
        )
        return bool(typed_kill.strip().lower() == "yes")

    for t in confirmed_tools:
        if t.tool_id == "clean_temp":
            res = clean_temp(adapter=adapter, dry_run=dry_run)
        elif t.tool_id == "clean_app_caches":
            res = clean_app_caches(adapter=adapter, dry_run=dry_run)
        elif t.tool_id == "empty_recycle_bin":
            res = empty_recycle_bin(adapter=adapter, dry_run=dry_run, confirmed=True)
        elif t.tool_id == "trim_memory":
            res = trim_memory(
                t.details.get("candidates", []),
                adapter=adapter,
                dry_run=dry_run,
            )
        elif t.tool_id == "close_idle_apps":
            res = close_idle_apps(
                t.details.get("all_procs", []),
                adapter=adapter,
                dry_run=dry_run,
                force=force,
                confirm_force_cb=_confirm_force if force else None,
            )
        elif t.tool_id == "end_heavy_tasks":
            res = end_heavy_tasks(
                adapter=adapter,
                dry_run=dry_run,
                force=force,
                task_names=t.details.get("task_names"),
                all_processes=t.details.get("all_procs", []),
                confirm_force_cb=_confirm_force if force else None,
            )
        else:
            res = ActionResult(
                action=t.tool_id,
                success=False,
                message=f"Unknown tool: {t.tool_id}",
            )

        results.append(res)
        color = "green" if res.success else "red"
        console.print(f"[{color}]* {t.title}: {res.message}[/{color}]")

    # Post-fix ML re-scan
    after: dict[str, Any] | None = None
    if not dry_run and not no_rescan and baseline and runtime and before:
        console.print("\nWaiting 3 seconds for system state to stabilize...")
        time.sleep(3.0)
        console.print("[bold]Running verification re-scan...[/bold]")
        after = _gather_state(adapter=adapter, baseline=baseline, runtime=runtime, card=card)

        # Before & After Table (docs/DESIGN.md: [ BEFORE / AFTER ])
        table_ver = Table(
            title="[ BEFORE / AFTER: Remediation Verification ]",
            box=box.ASCII if plain else box.ROUNDED,
            show_header=True,
        )
        table_ver.add_column("Metric", style="bold")
        table_ver.add_column("Before", justify="right")
        table_ver.add_column("After", justify="right")
        table_ver.add_column("Delta", justify="right")

        def _fmt_delta(val: float, is_better_when_higher: bool, unit: str = "") -> str:
            if abs(val) < 1e-4:
                return "same"
            sign = "+" if val > 0 else ""
            is_good = (val > 0) if is_better_when_higher else (val < 0)
            word = "better" if is_good else "worse"
            txt = f"{sign}{val}{unit} ({word})"
            if plain:
                return txt
            col = "spring_green2" if is_good else "red1"
            return f"[{col}]{txt}[/{col}]"

        score_diff = after["score"] - before["score"]
        table_ver.add_row(
            "Health Score",
            f"{before['score']}/100",
            f"{after['score']}/100",
            _fmt_delta(score_diff, is_better_when_higher=True),
        )

        ratio_diff = round(after["predicted_ratio"] - before["predicted_ratio"], 2)
        table_ver.add_row(
            "Predicted Slowdown",
            f"{before['predicted_ratio']:.2f}x",
            f"{after['predicted_ratio']:.2f}x",
            _fmt_delta(ratio_diff, is_better_when_higher=False, unit="x"),
        )

        probe_diff = round(after["probe_ms"] - before["probe_ms"], 1)
        table_ver.add_row(
            "Probe Latency",
            f"{before['probe_ms']:.1f} ms",
            f"{after['probe_ms']:.1f} ms",
            _fmt_delta(probe_diff, is_better_when_higher=False, unit=" ms"),
        )

        mem_before_gb = before["available_memory_bytes"] / (1024**3)
        mem_after_gb = after["available_memory_bytes"] / (1024**3)
        m_diff = round(mem_after_gb - mem_before_gb, 2)
        table_ver.add_row(
            "Available memory change (measured)",
            f"{mem_before_gb:.2f} GB",
            f"{mem_after_gb:.2f} GB",
            _fmt_delta(m_diff, is_better_when_higher=True, unit=" GB"),
        )

        temp_before_mb = before["temp_bytes"] / (1024 * 1024)
        temp_after_mb = after["temp_bytes"] / (1024 * 1024)
        t_diff = round(temp_after_mb - temp_before_mb, 1)
        table_ver.add_row(
            "Cleanable Temp",
            f"{temp_before_mb:.1f} MB",
            f"{temp_after_mb:.1f} MB",
            _fmt_delta(t_diff, is_better_when_higher=False, unit=" MB"),
        )

        closed_app_bytes = sum(
            r.bytes_freed
            for r in results
            if r.action in ("close_idle_apps", "end_heavy_tasks") and r.success
        )
        if closed_app_bytes > 0:
            c_mb = closed_app_bytes / (1024**2)
            table_ver.add_row(
                "Memory used by closed apps (at close)",
                f"{c_mb:.1f} MB",
                "-",
                f"-{c_mb:.1f} MB",
            )

        deleted_file_bytes = sum(
            r.bytes_freed
            for r in results
            if r.action in ("clean_temp", "clean_app_caches", "empty_recycle_bin") and r.success
        )
        if deleted_file_bytes > 0:
            f_mb = deleted_file_bytes / (1024**2)
            table_ver.add_row(
                "Bytes deleted (files)",
                "-",
                "-",
                f"+{f_mb:.1f} MB deleted",
            )

        console.print(table_ver)
    elif dry_run:
        console.print("\n[dim]Dry run complete. No modifications were made to the system.[/dim]")

    # Fix Log Persistence with Honesty Guarantees
    utc_str = datetime.now(tz=UTC).isoformat()
    if dry_run:
        results_log = [
            {
                "tool_id": r.action,
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
            "ts_utc": utc_str,
            "dry_run": True,
            "actions": [t.tool_id for t in confirmed_tools],
            "results": results_log,
            "before": {
                "score": before["score"],
                "predicted_ratio": before["predicted_ratio"],
                "probe_ms": before["probe_ms"],
                "available_memory_bytes": before["available_memory_bytes"],
            }
            if before
            else None,
            "after": None,
        }
    else:
        results_log = [
            {
                "tool_id": r.action,
                "action": r.action,
                "success": r.success,
                "message": r.message,
                "bytes_freed": r.bytes_freed,
                "details": r.details,
            }
            for r in results
        ]
        log_record = {
            "ts_utc": utc_str,
            "dry_run": False,
            "actions": [t.tool_id for t in confirmed_tools],
            "results": results_log,
            "before": {
                "score": before["score"],
                "predicted_ratio": before["predicted_ratio"],
                "probe_ms": before["probe_ms"],
                "available_memory_bytes": before["available_memory_bytes"],
            }
            if before
            else None,
            "after": {
                "score": after["score"],
                "predicted_ratio": after["predicted_ratio"],
                "probe_ms": after["probe_ms"],
                "available_memory_bytes": after["available_memory_bytes"],
            }
            if after
            else None,
        }

    log_path = append_fix_log(log_record)
    console.print(f"\n[dim]Audit record saved to {log_path}[/dim]")
    from deskflow.ui.components import render_key_hints

    console.print(render_key_hints([("Enter", "Return to Menu"), ("Q", "Quit")], plain=plain))
