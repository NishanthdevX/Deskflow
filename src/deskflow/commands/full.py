"""Full scan command: 10-minute stress test, live accuracy, and block bootstrap evaluation."""

from __future__ import annotations

import json
import logging
import math
import shutil
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import psutil
import typer
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from deskflow.baseline import (
    Baseline,
    CalibrationError,
    calibrate_baseline,
    check_baseline_drift,
    get_baseline_age_days,
    get_current_power_state,
    load_baseline,
    save_baseline,
)
from deskflow.collector.models import Record
from deskflow.commands.watch import PredictionPairTracker
from deskflow.exit_codes import (
    EXIT_BLOCKED_OR_DECLINED,
)
from deskflow.features.extract import build_row
from deskflow.features.registry import DEFAULT_HORIZON_S
from deskflow.lab.background import BackgroundLoadScenario
from deskflow.lab.cpu import CpuPressureScenario
from deskflow.lab.disk import DiskPressureScenario
from deskflow.lab.idle import IdleScenario
from deskflow.lab.memory import MemoryPressureScenario
from deskflow.lab.runner import LabRunner
from deskflow.ml.artifacts import load_model_card, resolve_model_dir
from deskflow.ml.report_data import load_report_data
from deskflow.ml.runtime import RuntimeModels
from deskflow.paths import get_data_dir
from deskflow.platform.base import PlatformAdapter
from deskflow.platform.factory import get_adapter
from deskflow.truth import NO_REAL_DATA_MESSAGE, load_provenance
from deskflow.ui.charts import bar_chart_with_whiskers
from deskflow.ui.components import domain_panel, sparkline
from deskflow.ui.theme import use_plain_output

logger = logging.getLogger(__name__)


def compute_block_bootstrap(
    resolved: list[dict[str, Any]],
    block_size: int = 5,
    n_resamples: int = 1000,
    seed: int = 42,
) -> dict[str, Any]:
    """Compute MAE and 95% confidence intervals using moving block bootstrap.

    Parameters
    ----------
    resolved: list of resolved prediction pairs with absolute errors.
    block_size: length of consecutive pairs per block (default: 5).
    n_resamples: number of bootstrap resamples (default: 1000).
    seed: fixed random seed for deterministic verification (default: 42).
    """
    n = len(resolved)
    if n == 0:
        return {}

    b_size = max(1, min(block_size, n))
    blocks = [resolved[i : i + b_size] for i in range(n - b_size + 1)]
    if not blocks:
        blocks = [resolved]

    k_blocks = int(math.ceil(n / b_size))
    rng = np.random.RandomState(seed)

    boot_xgb: list[float] = []
    boot_rf: list[float] = []
    boot_ridge: list[float] = []
    boot_pers: list[float] = []
    boot_diff: list[float] = []

    for _ in range(n_resamples):
        sample_indices = rng.randint(0, len(blocks), size=k_blocks)
        chosen_pairs: list[dict[str, Any]] = []
        for idx in sample_indices:
            chosen_pairs.extend(blocks[idx])
        resampled = chosen_pairs[:n]

        e_xgb = float(np.mean([p["abs_error"] for p in resampled]))
        e_rf = float(np.mean([p["abs_error_rf"] for p in resampled]))
        e_ridge = float(np.mean([p["abs_error_ridge"] for p in resampled]))
        e_pers = float(np.mean([p["persistence_error"] for p in resampled]))

        boot_xgb.append(e_xgb)
        boot_rf.append(e_rf)
        boot_ridge.append(e_ridge)
        boot_pers.append(e_pers)
        boot_diff.append(e_pers - e_xgb)

    def _ci(arr: list[float]) -> tuple[float, float, float]:
        mean_val = float(np.mean(arr))
        q025, q975 = np.percentile(arr, [2.5, 97.5])
        return mean_val, float(q025), float(q975)

    m_xgb, low_xgb, high_xgb = _ci(boot_xgb)
    m_rf, low_rf, high_rf = _ci(boot_rf)
    m_ridge, low_ridge, high_ridge = _ci(boot_ridge)
    m_pers, low_pers, high_pers = _ci(boot_pers)
    m_diff, low_diff, high_diff = _ci(boot_diff)

    return {
        "n": n,
        "XGBoost": {"mae": m_xgb, "ci_95": [low_xgb, high_xgb]},
        "RandomForest": {"mae": m_rf, "ci_95": [low_rf, high_rf]},
        "Ridge": {"mae": m_ridge, "ci_95": [low_ridge, high_ridge]},
        "Persistence": {"mae": m_pers, "ci_95": [low_pers, high_pers]},
        "improvement_over_persistence": {
            "mean": m_diff,
            "ci_95": [low_diff, high_diff],
            "excludes_zero": bool(low_diff > 0.0),
        },
    }


def full_command(
    yes: bool = False,
    plain: bool = False,
    quick_test: bool = False,
    input_fn: Callable[[str], str] = input,
) -> None:
    """Execute full 10-minute stress test with continuous live accuracy verification."""
    plain = use_plain_output(plain)
    console = Console(no_color=plain)
    box_style = box.ASCII if plain else box.ROUNDED
    adapter: PlatformAdapter = get_adapter()
    actual_state = get_current_power_state(adapter)
    bat = adapter.battery()

    # 1. Preflight: Confirmation Screen listing 5 stages and warning
    stages_spec = [
        ("idle", "Idle baseline", 60.0 if not quick_test else 1.0),
        (
            "cpu ramp",
            "CPU pressure ramp (15s pre, 90s hold, 20s post)",
            125.0 if not quick_test else 1.0,
        ),
        (
            "memory ramp",
            "Memory pressure ramp (15s pre, 90s hold, 20s post)",
            125.0 if not quick_test else 1.0,
        ),
        (
            "disk",
            "Disk pressure medium constant (15s pre, 60s hold, 20s post)",
            95.0 if not quick_test else 1.0,
        ),
        (
            "background",
            "Background load medium constant (15s pre, 60s hold, 20s post)",
            95.0 if not quick_test else 1.0,
        ),
    ]

    total_est_s = sum(s[2] for s in stages_spec) + (40.0 if not quick_test else 1.0)
    est_min = int(total_est_s // 60)
    est_sec = int(total_est_s % 60)

    console.print(
        f"\n[bold cyan]DESKFLOW FULL SCAN (~{est_min:02d}:{est_sec:02d})[/bold cyan]\n"
        "[bold yellow]WARNING: The machine will be slow; leave it alone "
        "during this test.[/bold yellow]"
    )

    tbl_plan = Table(box=box_style, expand=True, show_header=True)
    tbl_plan.add_column("Stage #", justify="right", width=8)
    tbl_plan.add_column("Stage Name", style="bold", width=18)
    tbl_plan.add_column("Workload / Pattern")
    tbl_plan.add_column("Duration", justify="right", width=12)

    for i, (name, desc, dur) in enumerate(stages_spec, 1):
        tbl_plan.add_row(str(i), name, desc, f"{dur:.0f}s")
    console.print(tbl_plan)

    if not yes:
        try:
            confirmed = input_fn("\nStart 10-minute full scan stress test? [y/N]: ").strip().lower()
            if confirmed not in ("y", "yes"):
                console.print("[yellow]Full scan declined by user.[/yellow]")
                raise typer.Exit(code=EXIT_BLOCKED_OR_DECLINED)
        except (EOFError, KeyboardInterrupt):
            console.print("\n[yellow]Full scan aborted.[/yellow]")
            raise typer.Exit(code=EXIT_BLOCKED_OR_DECLINED) from None

    # 2. Preflight: Hardware Baseline Freshness & Drift
    baseline: Baseline | None = None
    try:
        baseline = load_baseline(power_state=actual_state)
        age_days = get_baseline_age_days(baseline)
        if age_days >= 7.0:
            console.print(
                f"[yellow]Baseline for '{actual_state}' is older than 7 days "
                f"({age_days:.1f} days).[/yellow]"
            )
            baseline = None
        else:
            is_drifting, drift_ratio, mean_cpu = check_baseline_drift(baseline)
            if is_drifting:
                console.print(
                    f"[yellow]Baseline drift detected ({drift_ratio:.2f}x drift at "
                    f"{mean_cpu:.1f}% CPU).[/yellow]"
                )
                baseline = None
    except (FileNotFoundError, CalibrationError):
        baseline = None

    if baseline is None:
        console.print(f"[cyan]Calibrating fresh baseline for {actual_state} power (40s)...[/cyan]")
        baseline = calibrate_baseline(duration_s=40.0, interval_s=2.0)
        save_baseline(baseline, power_state=actual_state)
        console.print("[green]Fresh baseline calibrated and saved.[/green]")

    # 3. Preflight: Battery >= 40% if unplugged
    if bat is not None and not bat.plugged_in and bat.percent < 40.0:
        console.print(
            f"[bold red]Cannot run full scan on battery below 40% "
            f"(current: {bat.percent:.0f}%).[/bold red]\n"
            "Please connect the AC charger or charge battery before running full scan."
        )
        raise typer.Exit(code=EXIT_BLOCKED_OR_DECLINED)

    # 4. Preflight: Free disk space >= 5 GB
    home_dir = Path.home()
    usage = shutil.disk_usage(home_dir)
    free_gb = usage.free / (1024**3)
    if free_gb < 5.0:
        console.print(
            f"[bold red]Cannot run full scan: less than 5 GB free disk space available "
            f"({free_gb:.2f} GB free).[/bold red]\n"
            "Free up disk space before initiating full scan."
        )
        raise typer.Exit(code=EXIT_BLOCKED_OR_DECLINED)

    # 5. Preflight: Initial CPU utilisation <= 25%
    cpu_init = psutil.cpu_percent(interval=1.0)
    if cpu_init > 25.0:
        console.print(
            f"[bold red]Cannot run full scan: CPU utilisation is too high "
            f"({cpu_init:.1f}% > 25.0%).[/bold red]\n"
            "Close heavy background applications before running full scan."
        )
        raise typer.Exit(code=EXIT_BLOCKED_OR_DECLINED)

    console.print("[bold green]Preflight passed. Beginning Full Scan execution...[/bold green]\n")

    # Setup session, tracker, and models
    session_utc = datetime.now(tz=UTC)
    session_id = f"lab-{session_utc.strftime('%Y%m%dT%H%M%SZ')}"
    data_dir = get_data_dir()
    runs_dir = data_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    tracker = PredictionPairTracker(horizon_s=DEFAULT_HORIZON_S)
    model_dir, _ = resolve_model_dir()
    card = load_model_card(model_dir)
    runtime = RuntimeModels.load(model_dir)
    prov = load_provenance()

    primary_models = card.get("primary_models", {"regressor": "XGBoost", "classifier": "XGBoost"})
    primary_reg = primary_models.get("regressor", "XGBoost")
    primary_clf = primary_models.get("classifier", "XGBoost")

    live_history: list[Record] = []
    ratio_history: list[float] = []
    pred_history: list[float] = []
    pers_history: list[float] = []
    stage_results: list[dict[str, Any]] = []

    stage_names = ["idle", "cpu ramp", "memory ramp", "disk", "background"]
    stage_chips_status = {name: "pending" for name in stage_names}

    current_runner: LabRunner | None = None
    start_all_mono = time.monotonic()

    # Pre-build scenarios
    scenarios_to_run = [
        (
            "idle",
            IdleScenario(intensity="low", pattern="constant"),
            0.0,
            60.0 if not quick_test else 1.0,
            0.0,
        ),
        (
            "cpu ramp",
            CpuPressureScenario(intensity="medium", pattern="ramp"),
            15.0 if not quick_test else 0.2,
            90.0 if not quick_test else 0.6,
            20.0 if not quick_test else 0.2,
        ),
        (
            "memory ramp",
            MemoryPressureScenario(intensity="medium", pattern="ramp"),
            15.0 if not quick_test else 0.2,
            90.0 if not quick_test else 0.6,
            20.0 if not quick_test else 0.2,
        ),
        (
            "disk",
            DiskPressureScenario(intensity="medium", pattern="constant"),
            15.0 if not quick_test else 0.2,
            60.0 if not quick_test else 0.6,
            20.0 if not quick_test else 0.2,
        ),
        (
            "background",
            BackgroundLoadScenario(intensity="medium", pattern="constant"),
            15.0 if not quick_test else 0.2,
            60.0 if not quick_test else 0.6,
            20.0 if not quick_test else 0.2,
        ),
    ]

    def _render_full_dashboard(
        curr_stage_idx: int,
        curr_stage_name: str,
        curr_pred_tag: str,
        curr_conf: float,
    ) -> None:
        elapsed_s = time.monotonic() - start_all_mono
        el_m, el_s = int(elapsed_s // 60), int(elapsed_s % 60)

        # Timeline chips
        chips_parts = []
        for s_name in stage_names:
            status = stage_chips_status[s_name]
            if status == "finished":
                chips_parts.append(f"[bold green][{s_name}][/bold green]")
            elif status == "active":
                chips_parts.append(f"[bold cyan][{s_name}][/bold cyan]")
            else:
                chips_parts.append(f"[dim][{s_name}][/dim]")
        timeline_line = " ".join(chips_parts)

        # Header panel
        header_text = (
            f"Stage {curr_stage_idx + 1} of 5: {curr_stage_name.upper()}  |  "
            f"Elapsed: {el_m:02d}:{el_s:02d} of ~{est_min:02d}:{est_sec:02d}\n"
            f"Timeline: {timeline_line}"
        )
        console.print(
            Panel(
                header_text,
                title="[ DESKFLOW FULL SCAN ]",
                border_style="cyan",
                box=box_style,
            )
        )

        # Live Trends Chart
        recent_ratios = ratio_history[-16:] if len(ratio_history) >= 16 else ratio_history
        recent_preds = pred_history[-16:] if len(pred_history) >= 16 else pred_history
        recent_pers = pers_history[-16:] if len(pers_history) >= 16 else pers_history

        tbl_live = Table(box=box_style, expand=True, show_header=True)
        tbl_live.add_column("Metric", style="bold")
        tbl_live.add_column("Latest Ratio", justify="right")
        tbl_live.add_column("16-point Trend", justify="right")

        cur_act = ratio_history[-1] if ratio_history else 1.0
        cur_pr = pred_history[-1] if pred_history else 1.0
        cur_ps = pers_history[-1] if pers_history else 1.0

        tbl_live.add_row(
            "Measured Slowdown",
            f"{cur_act:.2f}x",
            sparkline(recent_ratios, plain=plain),
        )
        tbl_live.add_row(
            f"Primary Model ({primary_reg})",
            f"{cur_pr:.2f}x",
            sparkline(recent_preds, plain=plain),
        )
        tbl_live.add_row(
            "Persistence Baseline",
            f"{cur_ps:.2f}x",
            sparkline(recent_pers, plain=plain),
        )
        console.print(domain_panel("[ LIVE ]", tbl_live, domain="ml", plain=plain))

        # NOW panel
        p_res_count = len(tracker.resolved)
        tbl_now = Table.grid(expand=True)
        tbl_now.add_column(ratio=1)
        tbl_now.add_column(ratio=1)
        tbl_now.add_column(ratio=1)
        tbl_now.add_row(
            f"True Scenario: [bold]{curr_stage_name}[/bold]",
            f"Predicted Cause: [bold cyan]{curr_pred_tag}[/bold cyan] ({curr_conf:.0%})",
            f"Resolved Pairs: [bold]{p_res_count}[/bold]",
        )
        console.print(domain_panel("[ NOW ]", tbl_now, domain="system", plain=plain))

        # SCENARIO RESULTS table (fills per stage)
        if stage_results:
            tbl_scen = Table(box=box_style, expand=True, show_header=True)
            tbl_scen.add_column("Scenario", style="bold")
            tbl_scen.add_column("Peak Slowdown", justify="right")
            tbl_scen.add_column("Primary MAE", justify="right")
            tbl_scen.add_column("Persistence MAE", justify="right")
            tbl_scen.add_column("Cause Accuracy", justify="right")

            for res in stage_results:
                tbl_scen.add_row(
                    res["name"],
                    f"{res['peak_ratio']:.2f}x",
                    f"{res['primary_mae']:.4f}",
                    f"{res['pers_mae']:.4f}",
                    f"{res['cause_acc']:.1f}%",
                )
            console.print(domain_panel("[ SCENARIO RESULTS ]", tbl_scen, domain="ml", plain=plain))

    # Stage execution loop
    try:
        for idx, (s_name, scenario_obj, pre_s, hold_s, post_s) in enumerate(scenarios_to_run):
            stage_chips_status[s_name] = "active"
            stage_records_start = len(live_history)
            stage_pairs_start = len(tracker.resolved)

            curr_tag = "healthy"
            curr_conf = 1.0

            def _on_record(rec: Record) -> None:
                nonlocal curr_tag, curr_conf
                live_history.append(rec)
                now_m = time.monotonic()
                now_u = datetime.now(tz=UTC).isoformat()

                if rec.probe is not None and baseline is not None:
                    act_ratio = (
                        (rec.probe.total_ns / baseline.total_ns) if baseline.total_ns > 0 else 1.0
                    )
                    ratio_history.append(act_ratio)
                    tracker.process_probe(now_m, now_u, act_ratio)

                    if len(live_history) >= 6:
                        feat_d = build_row(
                            records_up_to_t=live_history,
                            baseline=baseline,
                            current_record=rec,
                        )
                        f_vec = np.array(
                            [feat_d[name] for name in runtime.feature_order],
                            dtype=np.float64,
                        )
                        p_dict = runtime.predict(f_vec)

                        p_slow = p_dict.get("slowdown_ratio", {})
                        xgb_val = float(p_slow.get("XGBoost", 1.0))
                        rf_val = float(p_slow.get("RandomForest", 1.0))
                        ridge_val = float(p_slow.get("Ridge", 1.0))

                        pred_history.append(xgb_val)
                        pers_history.append(act_ratio)

                        c_probs = p_dict.get("issue_probabilities", {}).get(primary_clf, {})
                        if c_probs:
                            best_c = max(c_probs.keys(), key=lambda k: c_probs[k])
                            curr_tag = best_c if float(c_probs[best_c]) >= 0.5 else "none"
                            curr_conf = float(c_probs[best_c])
                        else:
                            curr_tag = "healthy"
                            curr_conf = 1.0

                        tracker.add_prediction(
                            t_mono=now_m,
                            t_utc=now_u,
                            pred_xgb=xgb_val,
                            pred_rf=rf_val,
                            pred_ridge=ridge_val,
                            persistence_ratio=act_ratio,
                            pred_tag=curr_tag,
                            top_factor="probe_latency",
                        )

            current_runner = LabRunner(
                scenario=scenario_obj,
                pre_duration_s=pre_s,
                hold_duration_s=hold_s,
                post_duration_s=post_s,
                interval_s=2.0 if not quick_test else 0.4,
                probe_interval_s=2.0 if not quick_test else 0.4,
                session_id=session_id,
                out_dir=runs_dir,
                source="full_scan",
                on_record=_on_record,
            )

            _render_full_dashboard(idx, s_name, curr_tag, curr_conf)
            current_runner.run()

            # Record stage summary metrics
            stage_recs = live_history[stage_records_start:]
            stage_pairs = tracker.resolved[stage_pairs_start:]

            peak_rat = (
                max(
                    (r.probe.total_ns / baseline.total_ns)
                    for r in stage_recs
                    if r.probe and baseline
                )
                if any(r.probe for r in stage_recs) and baseline
                else 1.0
            )

            p_mae = float(np.mean([p["abs_error"] for p in stage_pairs])) if stage_pairs else 0.0
            pers_mae = (
                float(np.mean([p["persistence_error"] for p in stage_pairs]))
                if stage_pairs
                else 0.0
            )

            # Compute cause accuracy for this stage
            expected_class = s_name.split()[0]
            if expected_class == "idle":
                expected_class = "healthy"

            correct_matches = sum(1 for p in stage_pairs if expected_class in p.get("pred_tag", ""))
            cause_acc = (correct_matches / len(stage_pairs) * 100.0) if stage_pairs else 100.0

            stage_results.append(
                {
                    "name": s_name,
                    "peak_ratio": peak_rat,
                    "primary_mae": p_mae,
                    "pers_mae": pers_mae,
                    "cause_acc": cause_acc,
                }
            )

            stage_chips_status[s_name] = "finished"

            # 10s cooldown between stages (except after last stage)
            if idx < len(scenarios_to_run) - 1:
                cooldown_s = 10.0 if not quick_test else 0.1
                t_cd_start = time.monotonic()
                while time.monotonic() - t_cd_start < cooldown_s:
                    time.sleep(0.05 if quick_test else 0.5)

    except KeyboardInterrupt:
        console.print(
            "\n[bold red]Full scan interrupted by user (Ctrl+C). Cleaning up...[/bold red]"
        )
        if current_runner:
            current_runner.scenario.stop()
        _save_full_session(
            session_id=session_id,
            power_state=actual_state,
            stage_results=stage_results,
            tracker=tracker,
            completed=False,
            data_dir=data_dir,
        )
        raise typer.Exit(code=EXIT_BLOCKED_OR_DECLINED) from None

    # 4. FINALE: Complete Dashboard & Model Comparison
    _render_full_dashboard(4, "complete", "finished", 1.0)
    console.print("\n[bold cyan]═══ [ FINALE: FULL SCAN MODEL COMPARISON ] ═══[/bold cyan]")

    # Block Bootstrap 95% Confidence Intervals
    boot_stats = compute_block_bootstrap(tracker.resolved, block_size=5, n_resamples=1000, seed=42)
    n_pairs = len(tracker.resolved)
    console.print(
        f"\n[bold]LIVE MODEL COMPARISON[/bold] "
        f"({n_pairs} prediction pairs evaluated, block bootstrap B=5, 1000 resamples):"
    )

    if boot_stats:
        tbl_boot = Table(box=box_style, expand=True, show_header=True)
        tbl_boot.add_column("Model", style="bold")
        tbl_boot.add_column("Live MAE", justify="right")
        tbl_boot.add_column("95% Bootstrap CI", justify="right")
        tbl_boot.add_column("Delta vs Persistence", justify="right")

        for m_name in ("XGBoost", "RandomForest", "Ridge", "Persistence"):
            m_dat = boot_stats.get(m_name, {})
            m_mae = m_dat.get("mae", 0.0)
            ci = m_dat.get("ci_95", [0.0, 0.0])
            d_str = (
                f"{m_mae - boot_stats['Persistence']['mae']:+.4f}"
                if m_name != "Persistence"
                else "-"
            )
            tbl_boot.add_row(m_name, f"{m_mae:.4f}", f"[{ci[0]:.4f}, {ci[1]:.4f}]", d_str)
        console.print(tbl_boot)

        # Verdict sentence
        imp = boot_stats.get("improvement_over_persistence", {})
        if imp.get("excludes_zero", False):
            console.print(
                f"\nVerdict: [bold green]{primary_reg} beats Persistence "
                f"(improvement {imp['mean']:.4f}, 95% CI [{imp['ci_95'][0]:.4f}, "
                f"{imp['ci_95'][1]:.4f}] excludes zero).[/bold green]"
            )
        else:
            console.print(
                "\nVerdict: [dim]No model is distinguishable from Persistence "
                "on this data (interval includes zero).[/dim]"
            )

    # Cause Accuracy and False Alarm Rate
    inj_pairs = [p for p in tracker.resolved if "healthy" not in p.get("pred_tag", "")]
    idle_pairs = [
        p
        for p in tracker.resolved
        if "idle" in p.get("pred_tag", "") or "healthy" in p.get("pred_tag", "")
    ]
    idle_alarms = sum(1 for p in idle_pairs if p.get("pred_tag") not in ("healthy", "none"))
    idle_alarm_rate = (idle_alarms / len(idle_pairs) * 100.0) if idle_pairs else 0.0

    console.print(
        f"\n[bold]DIAGNOSTIC ACCURACY SUMMARY[/bold]:\n"
        f"- False alarm rate in idle stage: {idle_alarm_rate:.1f}% "
        f"({idle_alarms}/{len(idle_pairs)} samples)\n"
        f"- Diagnostic pairs resolved: {len(inj_pairs)} during active workloads"
    )

    # Held-out test split (Truth Contract)
    console.print("\n[bold]OFFLINE HELD-OUT TEST SPLIT[/bold]:")
    if prov.synthetic:
        console.print(
            Panel(
                f"[bold yellow]{NO_REAL_DATA_MESSAGE}[/bold yellow]",
                border_style="yellow",
                box=box_style,
            )
        )
    else:
        r_data = load_report_data()
        lead = r_data.get("leaderboard", [])
        if lead:
            whisker_rows = [
                (r["model"], r["mae"], r["mae_ci"][0], r["mae_ci"][1], r["is_baseline"])
                for r in lead
            ]
            console.print(bar_chart_with_whiskers(whisker_rows, metric_label="MAE", plain=plain))

    # Save session
    saved_session_path = _save_full_session(
        session_id=session_id,
        power_state=actual_state,
        stage_results=stage_results,
        tracker=tracker,
        completed=True,
        data_dir=data_dir,
    )
    console.print(
        f"\n[bold green]Full Scan session completed successfully.[/bold green]\n"
        f"- Session ID: [cyan]{session_id}[/cyan]\n"
        f"- Session artifact saved to: [cyan]{saved_session_path}[/cyan]\n"
        f"- Raw telemetry runs saved to: [cyan]{runs_dir}[/cyan]"
    )


def _save_full_session(
    session_id: str,
    power_state: str,
    stage_results: list[dict[str, Any]],
    tracker: PredictionPairTracker,
    completed: bool,
    data_dir: Path,
) -> Path:
    """Save structured session record to sessions directory for reporting."""
    sess_dir = data_dir / "sessions"
    sess_dir.mkdir(parents=True, exist_ok=True)
    sess_path = sess_dir / f"full_{session_id}.json"
    latest_path = sess_dir / "full_scan_latest.json"

    boot_stats = compute_block_bootstrap(tracker.resolved)
    session_data = {
        "session_id": session_id,
        "ts_utc": datetime.now(tz=UTC).isoformat(),
        "power_state": power_state,
        "completed": completed,
        "stage_results": stage_results,
        "resolved_pairs_count": len(tracker.resolved),
        "block_bootstrap": boot_stats,
        "leaderboard": tracker.compute_leaderboard(),
    }

    with open(sess_path, "w", encoding="utf-8") as f:
        json.dump(session_data, f, indent=2)

    with open(latest_path, "w", encoding="utf-8") as f:
        json.dump(session_data, f, indent=2)

    return latest_path
