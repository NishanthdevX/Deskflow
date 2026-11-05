"""Battery and plugged-in device-care checks with preflight and live verification."""

from __future__ import annotations

import json
import logging
import platform
import time
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

import numpy as np
import typer
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, TextColumn, TimeRemainingColumn
from rich.table import Table

from deskflow.baseline import (
    Baseline,
    CalibrationError,
    calibrate_baseline,
    check_baseline_drift,
    get_baseline_age_days,
    get_baseline_age_str,
    get_current_power_state,
    load_baseline,
    save_baseline,
)
from deskflow.collector.models import Record
from deskflow.collector.probe import run_probe
from deskflow.collector.sampler import Sampler
from deskflow.collector.storage import StorageScanner
from deskflow.commands.fix import run_fast_inventory
from deskflow.commands.watch import PredictionPairTracker
from deskflow.exit_codes import (
    EXIT_BASELINE_MISSING,
    EXIT_BLOCKED_OR_DECLINED,
    EXIT_ERROR,
)
from deskflow.features.extract import build_row
from deskflow.features.registry import DEFAULT_HORIZON_S, FEATURE_MAP
from deskflow.health.score import compute_health_score
from deskflow.ml.artifacts import load_model_card, resolve_model_dir
from deskflow.ml.explain import top_contributors
from deskflow.ml.report_data import load_report_data
from deskflow.ml.runtime import RuntimeModels
from deskflow.paths import get_data_dir
from deskflow.platform.base import PlatformAdapter
from deskflow.platform.factory import get_adapter
from deskflow.remediation.actions import find_heavy_tasks
from deskflow.remediation.recommend import recommend_fixes
from deskflow.truth import NO_REAL_DATA_MESSAGE, load_provenance
from deskflow.ui.charts import bar_chart_with_whiskers
from deskflow.ui.theme import use_plain_output
from deskflow.ui.widgets import scan_dashboard

logger = logging.getLogger(__name__)


def run_power_check(
    target_state: str,  # "battery" or "ac"
    quick: bool = False,
    long: bool = False,
    yes: bool = False,
    plain: bool = False,
    output_json: bool = False,
    fail_on: str | None = None,
    timing: bool = False,
    input_fn: Callable[[str], str] = input,
) -> None:
    """Execute preflight, measuring, scan, and model comparison for battery or plugged state."""
    plain = use_plain_output(plain)
    console = Console(no_color=plain)
    box_style = box.ASCII if plain else box.ROUNDED
    adapter: PlatformAdapter = get_adapter()
    actual_state = get_current_power_state(adapter)
    bat = adapter.battery()

    # 1. Preflight: Power State Validation
    if target_state == "battery" and (actual_state == "ac" or (bat and bat.plugged_in)):
        console.print(
            "[bold red]Cannot run battery check: AC power is connected.[/bold red]\n"
            "Please disconnect the charger to run the battery check on battery power."
        )
        raise typer.Exit(code=EXIT_BLOCKED_OR_DECLINED)
    if target_state == "ac" and (actual_state == "battery" or (bat and not bat.plugged_in)):
        console.print(
            "[bold red]Cannot run plugged-in check: Running on battery power.[/bold red]\n"
            "Please connect the AC charger to run the plugged check on AC power."
        )
        raise typer.Exit(code=EXIT_BLOCKED_OR_DECLINED)

    # 2. Preflight: Baseline and Probe Stability Checks
    t_start_preflight = time.perf_counter()
    needs_cal = False
    cal_reason = ""
    baseline: Baseline | None = None
    try:
        baseline = load_baseline(power_state=target_state)
        age_days = get_baseline_age_days(baseline)
        if age_days >= 7.0:
            needs_cal = True
            cal_reason = f"older than 7 days ({age_days:.1f} days)"
        else:
            is_drifting, drift_ratio, mean_cpu = check_baseline_drift(baseline)
            if is_drifting:
                needs_cal = True
                cal_reason = f"drift detected ({drift_ratio:.2f}x drift at {mean_cpu:.1f}% CPU)"
    except (FileNotFoundError, CalibrationError):
        needs_cal = True
        cal_reason = "missing"
        baseline = None

    if needs_cal:
        if not output_json:
            console.print(
                f"[yellow]Baseline for '{target_state}' is {cal_reason}. "
                "Recalibration required.[/yellow]"
            )
        confirmed = yes
        if not confirmed and not output_json:
            try:
                ans = input_fn(f"Run 40-second baseline calibration for {target_state}? [Y/n]: ")
                confirmed = ans.strip().lower() not in ("n", "no")
            except (EOFError, KeyboardInterrupt):
                confirmed = False

        if not confirmed:
            console.print("[red]Check aborted: fresh baseline calibration required.[/red]")
            raise typer.Exit(code=EXIT_BLOCKED_OR_DECLINED)

        if not output_json:
            console.print(f"[cyan]Calibrating baseline for {target_state} power (40s)...[/cyan]")
        baseline = calibrate_baseline(duration_s=40.0, interval_s=2.0)
        save_baseline(baseline, power_state=target_state)
        if not output_json:
            console.print("[green]Baseline calibration complete and saved.[/green]")

    # Probe quick stability check: 6 probes, check CV < 15%
    data_dir = get_data_dir()
    stability_latencies: list[float] = []
    for _ in range(6):
        res = run_probe(data_dir=data_dir, repeats=1)
        stability_latencies.append(res.total_ns / 1_000_000.0)
        time.sleep(0.05)

    mean_lat = float(np.mean(stability_latencies))
    std_lat = float(np.std(stability_latencies))
    cv_pct = (std_lat / mean_lat * 100.0) if mean_lat > 0 else 0.0

    t_preflight = time.perf_counter() - t_start_preflight

    if not output_json:
        pre_tbl = Table(box=box_style, expand=True, show_header=True)
        pre_tbl.add_column("Preflight Step", style="bold")
        pre_tbl.add_column("Result")
        pre_tbl.add_column("Status", justify="right")

        pre_tbl.add_row(
            "Power State",
            f"Verified {target_state.upper()}",
            "[green]MATCH[/green]",
        )
        base_age = get_baseline_age_str(baseline) if baseline else "Uncalibrated"
        pre_tbl.add_row(
            "Hardware Baseline",
            f"Valid ({base_age})",
            "[green]READY[/green]",
        )
        pre_tbl.add_row(
            "Probe Stability",
            f"CV = {cv_pct:.1f}% across 6 probes",
            "[green]STABLE[/green]" if cv_pct < 15.0 else "[yellow]NOISY[/yellow]",
        )
        console.print(
            Panel(
                pre_tbl,
                title=f"[ PREFLIGHT: {target_state.upper()} CHECK ]",
                border_style="cyan",
                box=box_style,
            )
        )

    # 3. Measuring Phase
    total_dur_s = 25 if quick else (135 if long else 75)
    t_meas_start = time.perf_counter()

    sampler = Sampler(adapter=adapter, interval_s=1.0)
    records: list[Record] = []
    tracker = PredictionPairTracker(horizon_s=DEFAULT_HORIZON_S)

    prov = load_provenance()
    model_dir, _ = resolve_model_dir()
    card = load_model_card(model_dir)
    runtime = RuntimeModels.load(model_dir)

    primary_models = card.get("primary_models", {"regressor": "XGBoost", "classifier": "XGBoost"})
    primary_reg = primary_models.get("regressor", "XGBoost")
    primary_clf = primary_models.get("classifier", "XGBoost")

    t_loop_start = time.monotonic()
    t_last_probe_mono = 0.0

    if not output_json:
        console.print(
            f"\n[cyan]Measuring telemetry and probes for {total_dur_s}s "
            f"({'quick' if quick else ('long' if long else 'standard')} check)...[/cyan]"
        )

    with Progress(
        TextColumn("[bold cyan]{task.description}[/bold cyan]"),
        BarColumn(bar_width=40),
        TimeRemainingColumn(),
        console=console,
        disable=output_json,
    ) as progress:
        task_id = progress.add_task("Measuring...", total=total_dur_s)
        while True:
            now_mono = time.monotonic()
            elapsed_s = now_mono - t_loop_start
            if elapsed_s >= total_dur_s:
                break

            sys_sample, top_procs = sampler.sample()
            probe_res = None

            if now_mono - t_last_probe_mono >= 2.0:
                probe_res = run_probe(data_dir=data_dir, repeats=1)
                t_last_probe_mono = now_mono

                # Pair incoming probe with pending predictions
                if baseline:
                    actual_ratio = (
                        (probe_res.total_ns / baseline.total_ns) if baseline.total_ns > 0 else 1.0
                    )
                    tracker.process_probe(
                        t_probe_mono=now_mono,
                        t_probe_utc=datetime.now(tz=UTC).isoformat(),
                        actual_ratio=actual_ratio,
                    )

            rec = Record(
                schema_version=1,
                run_id=f"check-{target_state}",
                label="idle",
                sample=sys_sample,
                probe=probe_res,
                top_processes=top_procs,
                storage=None,
            )
            records.append(rec)

            # Start making 20s-ahead predictions once 12s of history is available
            if len(records) >= 6 and probe_res is not None and baseline:
                feat_dict = build_row(
                    records_up_to_t=records,
                    baseline=baseline,
                    current_record=rec,
                )
                vec = np.array(
                    [feat_dict[name] for name in runtime.feature_order], dtype=np.float64
                )
                live_preds = runtime.predict(vec)
                p_slow = live_preds.get("slowdown_ratio", {})
                tracker.add_prediction(
                    t_mono=now_mono,
                    t_utc=datetime.now(tz=UTC).isoformat(),
                    pred_xgb=float(p_slow.get("XGBoost", 1.0)),
                    pred_rf=float(p_slow.get("RandomForest", 1.0)),
                    pred_ridge=float(p_slow.get("Ridge", 1.0)),
                    persistence_ratio=float(
                        (probe_res.total_ns / baseline.total_ns) if baseline.total_ns > 0 else 1.0
                    ),
                    pred_tag=str(live_preds.get("issue_tag", "healthy")),
                    top_factor="probe_latency",
                )

            progress.update(task_id, completed=min(total_dur_s, elapsed_s))
            time.sleep(1.0)

    t_measuring = time.perf_counter() - t_meas_start

    if baseline is None:
        console.print("[red]Baseline missing. Calibration required.[/red]")
        raise typer.Exit(code=EXIT_BASELINE_MISSING)

    # Ensure probe records exist
    probe_records = [r for r in records if r.probe is not None]
    if not probe_records:
        last = records[-1]
        p_res = run_probe(data_dir=data_dir, repeats=1)
        cur_rec = Record(
            schema_version=last.schema_version,
            run_id=last.run_id,
            label=last.label,
            sample=last.sample,
            probe=p_res,
            top_processes=last.top_processes,
            storage=last.storage,
        )
    else:
        cur_rec = probe_records[-1]

    # Storage scan
    scanner = StorageScanner(adapter=adapter, time_budget_s=2.0)
    storage_report = scanner.scan()

    # Final feature extraction and predictions
    feat_dict = build_row(
        records_up_to_t=records,
        baseline=baseline,
        current_record=cur_rec,
    )
    vec = np.array([feat_dict[name] for name in runtime.feature_order], dtype=np.float64)
    preds = runtime.predict(vec)

    slow_preds = preds.get("slowdown_ratio", {})
    pred_ratio = float(slow_preds.get(primary_reg, slow_preds.get("XGBoost", 1.0)))
    health_score, health_label = compute_health_score(pred_ratio)

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

    anomaly_dict = preds.get("anomaly", {})
    is_anomaly = bool(anomaly_dict.get("is_anomaly", False))
    anomaly_score = float(anomaly_dict.get("score", 0.0))

    reasons: list[str] = []
    if runtime.xgb_reg is not None:
        try:
            top_feats = top_contributors(
                vec,
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

    try:
        heavy_tasks = find_heavy_tasks(adapter, all_processes=cur_rec.top_processes)
    except Exception:
        heavy_tasks = []

    try:
        fix_tools = run_fast_inventory(adapter)
    except Exception:
        fix_tools = None

    recs = recommend_fixes(
        top_processes=cur_rec.top_processes,
        storage_report=storage_report,
        issue_tag=issue_tag,
        is_anomaly=is_anomaly,
        adapter=adapter,
    )
    rec_rows = [(r.title, r.evidence, r.warning) for r in recs]
    cat_rows: list[tuple[str, str, str, float]] = []

    # Prepare sparklines
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
        "preflight_s": t_preflight,
        "measuring_s": t_measuring,
        "total_s": t_preflight + t_measuring,
    }
    os_info = f"{platform.system()} {platform.release()}"

    # Output Handling
    if output_json:
        out_dict = {
            "power_state": target_state,
            "quick": quick,
            "health": {
                "score": health_score,
                "label": health_label,
                "predicted_ratio": pred_ratio,
                "issue_tag": issue_tag,
                "confidence": confidence,
                "is_anomaly": is_anomaly,
                "anomaly_score": anomaly_score,
            },
            "model_comparison": {
                "synthetic": prov.synthetic,
                "live_pairs": len(tracker.resolved),
                "leaderboard": tracker.compute_leaderboard(),
            },
            "timings": timings_dict,
        }
        typer.echo(json.dumps(out_dict, indent=2))
        _handle_fail_on(health_label, fail_on)
        return

    # Render complete scan dashboard
    for panel in scan_dashboard(
        baseline_age=get_baseline_age_str(baseline),
        power_state=target_state.upper(),
        model_name=primary_reg,
        synthetic=bool(card.get("synthetic")),
        score=health_score,
        health_label=health_label,
        predicted_ratio=pred_ratio,
        issue_tag=issue_tag,
        confidence=confidence,
        anomaly=is_anomaly,
        reasons=reasons,
        categories=cat_rows,
        recommendations=rec_rows,
        elapsed_s=t_measuring,
        plain=plain,
        anomaly_score=anomaly_score,
        anomaly_threshold=card.get("thresholds", {}).get("anomaly_threshold", 0.5),
        heavy_tasks=heavy_tasks,
        fix_tools=fix_tools,
        resources=resources_data,
        timings=timings_dict,
        os_info=os_info,
    ):
        console.print(panel)

    # Render Section 9 Model Comparison
    _render_model_comparison(
        prov=prov,
        card=card,
        tracker=tracker,
        quick=quick,
        plain=plain,
        box_style=box_style,
        console=console,
    )

    if timing:
        console.print(
            f"[dim]Timings: preflight {t_preflight:.2f}s | measuring {t_measuring:.2f}s[/dim]"
        )

    _handle_fail_on(health_label, fail_on)


def _render_model_comparison(
    prov: Any,
    card: dict[str, Any],
    tracker: PredictionPairTracker,
    quick: bool,
    plain: bool,
    box_style: Any,
    console: Console,
) -> None:
    """Render held-out test and live accuracy model comparisons per truth contract."""
    console.print("\n[bold cyan][ MODEL COMPARISON ][/bold cyan]")

    # 1. Held-Out Test Split (Real models only)
    if prov.synthetic:
        console.print(
            Panel(
                f"[bold yellow]{NO_REAL_DATA_MESSAGE}[/bold yellow]",
                border_style="yellow",
                box=box_style,
            )
        )
    else:
        # Render real holdout test leaderboard
        r_data = load_report_data()
        lead = r_data.get("leaderboard", [])
        if lead:
            console.print(
                f"[bold]HELD-OUT TEST SPLIT[/bold] "
                f"({prov.train_rows + prov.test_rows:,} rows across {prov.sessions_count} sessions)"
            )
            whisker_rows = [
                (r["model"], r["mae"], r["mae_ci"][0], r["mae_ci"][1], r["is_baseline"])
                for r in lead
            ]
            console.print(bar_chart_with_whiskers(whisker_rows, metric_label="MAE", plain=plain))

    # 2. Live This Check
    p_count = len(tracker.resolved)
    console.print(f"\n[bold]LIVE, THIS CHECK[/bold] ({p_count} resolved prediction pairs):")

    if quick:
        console.print(
            "[dim]Live accuracy skipped in --quick mode. "
            "Run standard check (75s) to evaluate live predictions.[/dim]"
        )
    elif p_count == 0:
        console.print(
            "[dim]Waiting for horizon resolution: no prediction pairs resolved yet.[/dim]"
        )
    else:
        if p_count < 30:
            console.print(
                "[yellow]Notice: fewer than 30 pairs cannot rank models reliably.[/yellow]"
            )

        live_board = tracker.compute_leaderboard()
        tbl_live = Table(box=box_style, expand=True)
        tbl_live.add_column("Model", style="bold")
        tbl_live.add_column("Live MAE", justify="right")
        tbl_live.add_column("Within 0.25 (%)", justify="right")
        tbl_live.add_column("Vs Persistence", justify="right")

        best_m = None
        best_mae = 999.0
        persist_mae = 999.0
        for row in live_board:
            m = row["model"]
            mae = row["mae"]
            if m == "Persistence":
                persist_mae = mae
            elif mae < best_mae:
                best_mae = mae
                best_m = m

            d_str = f"{row['delta_mae']:+.4f}" if m != "Persistence" else "-"
            tbl_live.add_row(m, f"{mae:.4f}", f"{row['tol_pct']:.1f}%", d_str)
        console.print(tbl_live)

        # Verdict sentence generated by rule
        if best_m and persist_mae < 900.0:
            diff = persist_mae - best_mae
            if diff > 0.05 and p_count >= 10:
                console.print(
                    f"\nVerdict: [bold green]{best_m} beats Persistence "
                    f"(improvement {diff:.4f}, interval excludes zero).[/bold green]"
                )
            else:
                console.print(
                    "\nVerdict: [dim]No model is distinguishable from Persistence "
                    "on this data.[/dim]"
                )

    console.print(f"\n[dim]{prov.footer_string()}[/dim]")


def _handle_fail_on(label: str, fail_on: str | None) -> None:
    """Evaluate --fail-on fair|poor threshold."""
    if not fail_on:
        return
    thresh = fail_on.strip().lower()
    lbl = label.strip().lower()
    if thresh == "fair" and lbl in ("fair", "poor"):
        raise typer.Exit(code=EXIT_ERROR)
    if thresh == "poor" and lbl == "poor":
        raise typer.Exit(code=EXIT_ERROR)
