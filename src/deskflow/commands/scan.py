"""Scan CLI command: measured system scan, primary model scoring, and health assessment."""

import contextlib
import json
import logging
import time
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, TextColumn, TimeRemainingColumn
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
from deskflow.collector.models import ProcessSample, Record
from deskflow.collector.probe import run_probe
from deskflow.collector.sampler import Sampler
from deskflow.collector.storage import StorageScanner
from deskflow.exit_codes import (
    EXIT_BLOCKED_OR_DECLINED,
    EXIT_ERROR,
)
from deskflow.features.extract import build_row
from deskflow.features.registry import DEFAULT_HORIZON_S, FEATURE_MAP
from deskflow.health.score import compute_health_score
from deskflow.ml.artifacts import load_model_card
from deskflow.paths import get_data_dir
from deskflow.platform.factory import get_adapter
from deskflow.remediation.apps import find_idle_apps
from deskflow.remediation.recommend import recommend_fixes

logger = logging.getLogger(__name__)
console = Console()


def scan_command(
    seconds: int = typer.Option(
        14,
        "--seconds",
        "-s",
        min=13,
        help="Measuring duration in seconds (default 14, min 13).",
    ),
    output_json: bool = typer.Option(
        False,
        "--json",
        help="Output raw structured JSON schema instead of human-readable tables.",
    ),
    no_storage: bool = typer.Option(
        False,
        "--no-storage",
        help="Skip filesystem storage scan stage.",
    ),
    timing: bool = typer.Option(
        False,
        "--timing",
        help="Print detailed latency breakdown per stage.",
    ),
    replay: Path | None = typer.Option(
        None,
        "--replay",
        help="Replay a previously recorded run JSONL file instead of live measuring.",
    ),
    yes: bool = typer.Option(
        False,
        "--yes",
        "-y",
        help="Automatically confirm any required baseline calibration.",
    ),
    plain: bool = typer.Option(False, "--plain"),
    fail_on: str | None = typer.Option(
        None,
        "--fail-on",
        help="Exit with code 1 if health rating matches or is worse than fair or poor.",
    ),
) -> None:
    """Run an end-to-end device scan, evaluate health score, and recommend fixes."""
    import numpy as np

    adapter = get_adapter()
    total_start = time.perf_counter()
    measuring_s = 0.0
    storage_s = 0.0

    # 1. Resolve Power State and Baseline
    power_state = get_current_power_state(adapter)
    try:
        baseline = load_baseline(power_state=power_state)
    except (FileNotFoundError, CalibrationError):
        baseline = None

    if baseline is None:
        if not output_json:
            console.print(
                f"[yellow]Baseline calibration for power state '{power_state}' not found.[/yellow]"
            )
        if not yes:
            confirmed = typer.confirm(
                f"Calibrate a baseline for {power_state} power now?",
                default=True,
            )
            if not confirmed:
                console.print("[red]Scan aborted: baseline required.[/red]")
                raise typer.Exit(code=EXIT_BLOCKED_OR_DECLINED)
        if not output_json:
            console.print("[cyan]Running baseline calibration...[/cyan]")
        baseline = calibrate_baseline(
            duration_s=60.0,
            interval_s=2.0,
        )
        save_baseline(baseline, power_state=power_state)
        if not output_json:
            console.print("[green]Baseline calibration complete and saved.[/green]")
    elif not output_json:
        # Age check
        age_days = get_baseline_age_days(baseline)
        if age_days > 7.0:
            console.print(
                f"[yellow]Warning: Baseline is {age_days:.1f} days old "
                "(recommend recalibrating with 'deskflow baseline calibrate').[/yellow]"
            )
        # Drift check
        if replay is None:
            is_drifting, drift_ratio, mean_cpu = check_baseline_drift(baseline)
            if is_drifting:
                console.print(
                    f"[yellow]Baseline may be out of date (drift ratio: {drift_ratio:.2f}x "
                    f"at {mean_cpu:.1f}% CPU).[/yellow]"
                )
                if not yes and typer.confirm(
                    f"Recalibrate baseline for {power_state} power now?",
                    default=False,
                ):
                    console.print("[cyan]Running baseline recalibration...[/cyan]")
                    baseline = calibrate_baseline(duration_s=60.0, interval_s=2.0)
                    save_baseline(baseline, power_state=power_state)
                    console.print("[green]Baseline recalibration complete and saved.[/green]")

    # 2. Acquire Telemetry Records
    records: list[Record] = []
    full_procs_snap: list[ProcessSample] = []

    if replay is not None:
        # Replay mode: load existing records from JSONL file
        if not replay.exists():
            console.print(f"[red]Error: Replay file not found: {replay}[/red]")
            raise typer.Exit(code=1)
        with open(replay, encoding="utf-8") as f:
            for line in f:
                line_str = line.strip()
                if line_str:
                    records.append(Record.from_dict(json.loads(line_str)))
        if not records:
            console.print("[red]Error: Replay file contained no records.[/red]")
            raise typer.Exit(code=1)
        # Seed full process snapshot from replay top processes
        full_procs_snap = records[-1].top_processes
    else:
        # Live Measuring
        # Enforce no sklearn / xgboost imports during this block!
        scan_seconds = max(13, seconds)
        sampler = Sampler(adapter=adapter, interval_s=2.0, process_interval_s=10.0)

        t_meas_start = time.perf_counter()
        data_dir = get_data_dir()
        sample_interval = 2.0
        probe_interval = 4.0
        last_probe_mono = -999.0
        full_procs_10_done = False

        # Phase 10: Full process pass at 0s
        with contextlib.suppress(Exception):
            full_procs_snap = adapter.full_process_snapshot()

        if output_json:
            # Silent live loop
            mono_start = time.monotonic()
            while (time.monotonic() - mono_start) < scan_seconds:
                t_loop_start = time.monotonic()
                spent = time.monotonic() - mono_start
                # Phase 10: Full process pass at 10s
                if not full_procs_10_done and spent >= 10.0:
                    with contextlib.suppress(Exception):
                        full_procs_snap = adapter.full_process_snapshot()
                    full_procs_10_done = True

                sys_sample, top_procs = sampler.sample()
                probe_res = None
                if (sys_sample.monotonic_s - last_probe_mono) >= (probe_interval - 0.2):
                    probe_res = run_probe(data_dir=data_dir, repeats=3)
                    last_probe_mono = sys_sample.monotonic_s
                rec = Record(
                    schema_version=1,
                    run_id="scan-live",
                    label="idle",
                    sample=sys_sample,
                    probe=probe_res,
                    top_processes=top_procs,
                    storage=None,
                )
                records.append(rec)
                elapsed = time.monotonic() - t_loop_start
                sleep_s = max(0.05, sample_interval - elapsed)
                time.sleep(sleep_s)
        else:
            with Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TimeRemainingColumn(),
                console=console,
            ) as progress:
                task = progress.add_task(
                    "Measuring system telemetry and probes...", total=scan_seconds
                )
                mono_start = time.monotonic()
                while (time.monotonic() - mono_start) < scan_seconds:
                    t_loop_start = time.monotonic()
                    spent = time.monotonic() - mono_start
                    # Phase 10: Full process pass at 10s
                    if not full_procs_10_done and spent >= 10.0:
                        with contextlib.suppress(Exception):
                            full_procs_snap = adapter.full_process_snapshot()
                        full_procs_10_done = True

                    sys_sample, top_procs = sampler.sample()
                    probe_res = None
                    if (sys_sample.monotonic_s - last_probe_mono) >= (probe_interval - 0.2):
                        probe_res = run_probe(data_dir=data_dir, repeats=3)
                        last_probe_mono = sys_sample.monotonic_s
                    rec = Record(
                        schema_version=1,
                        run_id="scan-live",
                        label="idle",
                        sample=sys_sample,
                        probe=probe_res,
                        top_processes=top_procs,
                        storage=None,
                    )
                    records.append(rec)
                    progress.update(task, completed=min(scan_seconds, spent))
                    loop_spent = time.monotonic() - t_loop_start
                    sleep_s = max(0.05, sample_interval - loop_spent)
                    time.sleep(sleep_s)

        measuring_s = time.perf_counter() - t_meas_start

    # Ensure at least one probe was completed
    probe_records = [r for r in records if r.probe is not None]
    if not probe_records:
        console.print("[red]Error: No valid probe records collected during scan.[/red]")
        raise typer.Exit(code=1)

    cur_rec = probe_records[-1]

    # 3. Storage Scan (budget 3s, skippable)
    storage_report = None
    if not no_storage:
        t_store_start = time.perf_counter()
        scanner = StorageScanner(adapter=adapter, time_budget_s=3.0)
        storage_report = scanner.scan()
        storage_s = time.perf_counter() - t_store_start

    # 4. Load Models and Predict (ML imports occur now after measuring)
    t_model_start = time.perf_counter()
    from deskflow.ml.artifacts import resolve_model_dir
    from deskflow.ml.explain import top_contributors
    from deskflow.ml.runtime import RuntimeModels

    try:
        model_dir, _is_bundled = resolve_model_dir()
    except FileNotFoundError as err:
        console.print("[red]Error: No trained model set found. Run 'deskflow train' first.[/red]")
        raise typer.Exit(code=1) from err

    card = load_model_card(model_dir)
    runtime = RuntimeModels.load(model_dir)
    model_load_s = time.perf_counter() - t_model_start

    # 5. Feature Extraction and Prediction
    t_infer_start = time.perf_counter()
    features_dict = build_row(
        records_up_to_t=records,
        baseline=baseline,
        current_record=cur_rec,
    )
    feature_vector = np.array(
        [features_dict[name] for name in runtime.feature_order],
        dtype=np.float64,
    )

    preds = runtime.predict(feature_vector)

    # Resolve primary models
    primary_models = card.get(
        "primary_models",
        {"regressor": "XGBoost", "classifier": "XGBoost"},
    )
    primary_reg = primary_models.get("regressor", "XGBoost")
    primary_clf = primary_models.get("classifier", "XGBoost")

    # Regression result
    slowdown_preds = preds.get("slowdown_ratio", {})
    predicted_ratio = float(slowdown_preds.get(primary_reg, slowdown_preds.get("XGBoost", 1.0)))

    # Health score
    score, health_label = compute_health_score(predicted_ratio)

    # Classification result
    clf_probs = preds.get("issue_probabilities", {}).get(primary_clf, {})
    if clf_probs:
        best_class = max(clf_probs.keys(), key=lambda c: clf_probs[c])
        best_prob = float(clf_probs[best_class])
        if best_prob >= 0.5:
            issue_tag = best_class
            confidence = best_prob
        else:
            issue_tag = "none"
            confidence = 0.0
    else:
        issue_tag = "healthy"
        confidence = 1.0

    # Anomaly detector result
    anomaly_dict = preds.get("anomaly", {})
    is_anomaly = bool(anomaly_dict.get("is_anomaly", False))
    anomaly_score = float(anomaly_dict.get("score", 0.0))
    is_out_of_range = bool(preds.get("out_of_range", False))

    # Top-3 TreeSHAP contributors with positive contributions above epsilon (1e-4)
    # Using plain language descriptions only
    reasons: list[str] = []
    if runtime.xgb_reg is not None:
        try:
            top_feats = top_contributors(
                feature_vector,
                runtime.xgb_reg,
                feature_names=runtime.feature_order,
                k=len(runtime.feature_order),
            )
            for f_name, shap_val in top_feats:
                if shap_val > 1e-4:
                    spec = FEATURE_MAP.get(f_name)
                    phrase = spec.plain if spec and spec.plain else f_name
                    reasons.append(f"{phrase} (+{shap_val:.2f})")
                    if len(reasons) >= 3:
                        break
        except Exception:
            pass

    if not reasons:
        reasons = ["No significant factors"]

    # Recommendations
    recs = recommend_fixes(
        top_processes=cur_rec.top_processes,
        storage_report=storage_report,
        issue_tag=issue_tag,
        is_anomaly=is_anomaly,
        adapter=adapter,
    )

    # Phase 10: Identify heavy background tasks from full process pass
    from deskflow.remediation.actions import find_heavy_tasks

    try:
        heavy_tasks = find_heavy_tasks(
            adapter, all_processes=full_procs_snap or cur_rec.top_processes
        )
    except Exception:
        heavy_tasks = []

    infer_s = time.perf_counter() - t_infer_start
    total_s = time.perf_counter() - total_start

    # Categories assessment
    # 1. Performance: ML Score
    perf_status = health_label

    # 2. Memory: Headroom and Available %
    tot_ram = cur_rec.sample.mem_total_bytes
    avail_ram = cur_rec.sample.mem_available_bytes
    mem_avail_pct = (avail_ram / tot_ram * 100.0) if tot_ram > 0 else 0.0
    headroom_gb = avail_ram / (1024**3)
    if mem_avail_pct >= 25.0:
        mem_status = "Good"
    elif mem_avail_pct >= 15.0:
        mem_status = "Fair"
    else:
        mem_status = "Poor"

    # 3. Storage: Both total bytes and cleanable bytes (>24h old)
    cleanable_bytes = 0
    total_storage_bytes = 0
    if storage_report and storage_report.entries:
        safe_roots = {
            loc.path.lower() for loc in adapter.temp_and_cache_dirs() if loc.safe_to_clean
        }
        for entry in storage_report.entries:
            if entry.kind in ("temp", "cache") and entry.path.lower() in safe_roots:
                cleanable_bytes += getattr(entry, "cleanable_bytes", entry.bytes)
                total_storage_bytes += entry.bytes

    cleanable_gb = cleanable_bytes / (1024**3)
    total_storage_gb = total_storage_bytes / (1024**3)
    cleanable_mb = cleanable_bytes / (1024 * 1024)
    total_storage_mb = total_storage_bytes / (1024 * 1024)

    if cleanable_gb < 2.0:
        storage_status = "Good"
    elif cleanable_gb < 10.0:
        storage_status = "Fair"
    else:
        storage_status = "Poor"

    # 4. Background Apps: Unified find_idle_apps
    idle_groups = find_idle_apps(cur_rec.top_processes, adapter=adapter)
    idle_count = len(idle_groups)
    idle_rss_gb = sum(g.total_rss_bytes for g in idle_groups) / (1024**3)
    if idle_rss_gb < 1.0:
        apps_status = "Good"
    elif idle_rss_gb < 3.0:
        apps_status = "Fair"
    else:
        apps_status = "Poor"

    # 5. Disk Space Category
    low_disk_vols: list[str] = []
    min_disk_free_pct = 100.0
    try:
        import psutil

        for part in psutil.disk_partitions(all=False):
            try:
                du = psutil.disk_usage(part.mountpoint)
                pct_free = (du.free / du.total * 100.0) if du.total > 0 else 100.0
                gb_free = du.free / (1024**3)
                min_disk_free_pct = min(min_disk_free_pct, pct_free)
                if pct_free < 15.0 or gb_free < 20.0:
                    low_disk_vols.append(
                        f"{part.mountpoint} ({gb_free:.0f} GB free, {100.0 - pct_free:.0f}% full)"
                    )
            except Exception:
                pass
    except Exception:
        pass

    if not low_disk_vols:
        disk_status = "Good"
        disk_desc = "Adequate headroom (>15% free)"
    elif min_disk_free_pct < 10.0:
        disk_status = "Poor"
        disk_desc = f"Critical low space: {', '.join(low_disk_vols)}"
    else:
        disk_status = "Fair"
        disk_desc = f"Low space warning: {', '.join(low_disk_vols)}"

    # 6. JSON Output Mode
    if output_json:
        result_payload = {
            "machine_id": baseline.machine_id,
            "power_state": power_state,
            "scan_duration_s": round(total_s, 2),
            "synthetic_model": bool(card.get("synthetic", False)),
            "health": {
                "score": score,
                "label": health_label,
                "predicted_slowdown_ratio": round(predicted_ratio, 3),
                "horizon_s": DEFAULT_HORIZON_S,
            },
            "issue": {
                "tag": issue_tag,
                "confidence": round(confidence, 3),
                "anomaly": is_anomaly,
                "anomaly_score": round(anomaly_score, 4),
            },
            "reasons": reasons,
            "categories": {
                "performance": {
                    "status": perf_status,
                    "score": score,
                    "is_ml": True,
                },
                "memory": {
                    "status": mem_status,
                    "available_gb": round(headroom_gb, 2),
                    "available_percent": round(mem_avail_pct, 1),
                    "is_ml": False,
                },
                "disk_space": {
                    "status": disk_status,
                    "low_disk_volumes": low_disk_vols,
                    "min_free_percent": round(min_disk_free_pct, 1),
                    "is_ml": False,
                },
                "storage": {
                    "status": storage_status,
                    "cleanable_gb": round(cleanable_gb, 2),
                    "total_gb": round(total_storage_gb, 2),
                    "is_ml": False,
                },
                "background_apps": {
                    "status": apps_status,
                    "idle_count": idle_count,
                    "idle_rss_gb": round(idle_rss_gb, 2),
                    "is_ml": False,
                },
            },
            "recommendations": [
                {
                    "action_id": r.action_id,
                    "title": r.title,
                    "description": r.description,
                    "evidence": r.evidence,
                    "warning": r.warning,
                }
                for r in recs
            ],
            "heavy_tasks": [
                {
                    "name": t.name,
                    "process_count": t.process_count,
                    "cpu_percent": round(t.cpu_percent, 2),
                    "rss_mb": round(t.rss_bytes / (1024 * 1024), 2),
                }
                for t in heavy_tasks
            ],
            "timings": {
                "measuring_s": round(measuring_s, 2),
                "storage_s": round(storage_s, 2),
                "model_load_s": round(model_load_s, 2),
                "inference_s": round(infer_s, 4),
                "total_s": round(total_s, 2),
            },
        }
        print(json.dumps(result_payload, indent=2))
        if fail_on:
            thresh = fail_on.strip().lower()
            lbl = health_label.strip().lower()
            if thresh == "fair" and lbl in ("fair", "poor"):
                raise typer.Exit(code=EXIT_ERROR)
            if thresh == "poor" and lbl == "poor":
                raise typer.Exit(code=EXIT_ERROR)
        return

    # 7. Human-readable CLI Output
    import platform

    from deskflow.commands.fix import run_fast_inventory
    from deskflow.ui.theme import use_plain_output
    from deskflow.ui.widgets import scan_dashboard

    plain = use_plain_output(plain)
    dashboard_console = Console(no_color=plain)
    if is_out_of_range:
        dashboard_console.print(
            "[yellow]Warning: Out-of-range telemetry detected "
            "(features exceed training distribution).[/yellow]"
        )

    category_rows = [
        ("Performance", perf_status, f"{score}/100, ML prediction", float(score)),
        ("Memory", mem_status, f"{headroom_gb:.1f} GB free", mem_avail_pct),
        ("Disk space", disk_status, disk_desc, min_disk_free_pct),
        ("Storage", storage_status, f"{cleanable_mb:.1f} MB cleanable", 0.0),
        ("Background apps", apps_status, f"{idle_count} group(s), {idle_rss_gb:.1f} GB", 0.0),
    ]
    recommendation_rows = [(r.title, r.evidence, r.warning) for r in recs]
    baseline_age = get_baseline_age_str(baseline)

    # Fast inventory tools for the ACTIONS area
    try:
        fix_tools = run_fast_inventory(adapter)
    except Exception:
        fix_tools = None

    # Trend sparklines over last 12 seconds (last 6 records at 2s interval)
    recent_records = records[-6:] if len(records) >= 6 else records
    cpu_vals = [r.sample.cpu_percent_mean for r in recent_records]
    mem_vals = [r.sample.mem_percent for r in recent_records]
    disk_vals = [(r.sample.disk_busy_ms_per_s or 0.0) / 10.0 for r in recent_records]
    swap_vals = [float(r.sample.swap_percent or 0.0) for r in recent_records]

    cur_disk_busy = (cur_rec.sample.disk_busy_ms_per_s or 0.0) / 10.0
    cur_swap = float(cur_rec.sample.swap_percent or 0.0)

    resources_data = [
        ("CPU", cur_rec.sample.cpu_percent_mean / 100.0, cpu_vals),
        ("Memory", cur_rec.sample.mem_percent / 100.0, mem_vals),
        ("Disk Busy", min(1.0, cur_disk_busy / 100.0), disk_vals),
        ("Swap", min(1.0, cur_swap / 100.0), swap_vals),
    ]

    timings_dict = {
        "measuring_s": measuring_s,
        "storage_s": storage_s,
        "model_load_s": model_load_s,
        "inference_s": infer_s,
        "total_s": total_s,
    }
    os_info = f"{platform.system()} {platform.release()}"

    for panel in scan_dashboard(
        baseline_age,
        power_state,
        primary_reg,
        bool(card.get("synthetic")),
        score,
        health_label,
        predicted_ratio,
        issue_tag,
        confidence,
        is_anomaly,
        reasons,
        category_rows,
        recommendation_rows,
        total_s,
        plain,
        anomaly_score=anomaly_score,
        anomaly_threshold=card.get("thresholds", {}).get("anomaly_threshold", 0.5),
        heavy_tasks=heavy_tasks,
        fix_tools=fix_tools,
        resources=resources_data,
        timings=timings_dict,
        os_info=os_info,
    ):
        dashboard_console.print(panel)
    dashboard_console.print(f"Scan completed in {total_s:.1f}s.")
    return

    # Legacy compact output retained below for reference.
    if card.get("synthetic"):
        console.print(
            Panel.fit(
                "DEMO MODEL trained on synthetic data",
                style="bold yellow",
            )
        )

    # Title line
    console.print(
        f"[bold]Deskflow Scan[/bold] - Machine: {baseline.machine_id[:8]}... | Power: {power_state}"
    )

    # Health score
    color_map = {"Good": "green", "Fair": "yellow", "Poor": "red"}
    s_color = color_map.get(health_label, "white")
    console.print(
        f"Health Score: [{s_color}]{score}/100 ({health_label})[/{s_color}] | "
        f"Predicted Slowdown (in 20s): {predicted_ratio:.2f}x baseline"
    )

    # Merged status line:
    # If anomaly is true and tag is healthy: "Unusual pattern detected, cause not identified"
    # If tag is not healthy: print tag and confidence
    # Otherwise: print nothing extra
    # NEVER print "healthy" next to "Anomaly: Yes"
    if is_anomaly and issue_tag.lower() == "healthy":
        console.print("Status: [yellow]Unusual pattern detected, cause not identified[/yellow]")
    elif issue_tag.lower() not in ("healthy", "none"):
        console.print(f"Status: [yellow]{issue_tag}[/yellow] (confidence: {confidence:.0%})")

    # Reasons
    console.print("\n[bold]Top Contributing Factors:[/bold]")
    for r in reasons:
        console.print(f"  - {r}")

    # Category Table
    table_cat = Table(title="\nSystem Categories", show_header=True)
    table_cat.add_column("Category", style="bold", width=18)
    table_cat.add_column("Status", width=10)
    table_cat.add_column("Details")

    table_cat.add_row(
        "Performance",
        f"[{color_map.get(perf_status, 'white')}]{perf_status}[/]",
        f"{score}/100 (ML predicted slowdown {predicted_ratio:.2f}x)",
    )
    table_cat.add_row(
        "Memory",
        f"[{color_map.get(mem_status, 'white')}]{mem_status}[/]",
        f"{headroom_gb:.1f} GB ({mem_avail_pct:.0f}% free)",
    )
    storage_desc = (
        f"{cleanable_mb:.1f} MB cleanable (>24h) / {total_storage_mb:.1f} MB total"
        if not no_storage
        else "Skipped (--no-storage)"
    )
    table_cat.add_row(
        "Storage",
        f"[{color_map.get(storage_status, 'white')}]{storage_status}[/]",
        storage_desc,
    )
    table_cat.add_row(
        "Background apps",
        f"[{color_map.get(apps_status, 'white')}]{apps_status}[/]",
        f"{idle_count} idle app group(s) ({idle_rss_gb:.1f} GB RAM)",
    )
    console.print(table_cat)

    # Recommended Fixes
    if recs:
        console.print("\n[bold]Recommended Fixes:[/bold]")
        for idx, rec_action in enumerate(recs, start=1):
            warn_str = (
                f" [yellow](Warning: {rec_action.warning})[/yellow]" if rec_action.warning else ""
            )
            console.print(
                f"  {idx}. [bold]{rec_action.title}[/bold]: {rec_action.evidence}.{warn_str}"
            )
    else:
        console.print("\n[dim]No fixes needed. System running within normal parameters.[/dim]")

    # Footer
    footer_note = (
        "Only Performance and Issue Tag are ML-derived; other categories are direct measurements."
    )
    console.print(f"\n[dim]Scan completed in {total_s:.1f}s. {footer_note}[/dim]")

    if timing:
        t_str = (
            f"Measuring {measuring_s:.2f}s | Storage {storage_s:.2f}s | "
            f"Model Load {model_load_s:.2f}s | Inference {infer_s:.4f}s | Total {total_s:.2f}s"
        )
        console.print(f"[dim]Timings: {t_str}[/dim]")

    if fail_on:
        thresh = fail_on.strip().lower()
        lbl = health_label.strip().lower()
        if thresh == "fair" and lbl in ("fair", "poor"):
            raise typer.Exit(code=EXIT_ERROR)
        if thresh == "poor" and lbl == "poor":
            raise typer.Exit(code=EXIT_ERROR)
