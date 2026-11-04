"""Live accuracy verification command: deskflow watch."""

import json
import logging
import shutil
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import typer
from rich import box
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table

from deskflow.baseline import (
    Baseline,
    CalibrationError,
    get_current_power_state,
    load_baseline,
)
from deskflow.collector.models import ProbeResult, Record, SystemSample
from deskflow.collector.probe import run_probe
from deskflow.collector.sampler import Sampler
from deskflow.features.extract import build_row
from deskflow.features.registry import FEATURE_MAP, FEATURE_NAMES
from deskflow.ml.artifacts import load_model_card, resolve_model_dir
from deskflow.ml.runtime import RuntimeModels
from deskflow.paths import get_data_dir
from deskflow.platform.factory import get_adapter

logger = logging.getLogger(__name__)
app = typer.Typer(help="Live accuracy verification against future 20s horizons.")


class PredictionPairTracker:
    """Buffers predictions made at time t and pairs them with actual probes at t + 20s."""

    def __init__(self, horizon_s: float = 20.0, tolerance_s: float = 2.5) -> None:
        self.horizon_s = horizon_s
        self.tolerance_s = tolerance_s
        self.pending: list[dict[str, Any]] = []
        self.resolved: list[dict[str, Any]] = []

    def add_prediction(
        self,
        t_mono: float,
        t_utc: str,
        pred_xgb: float,
        pred_rf: float,
        pred_ridge: float,
        persistence_ratio: float,
        pred_tag: str,
        top_factor: str,
    ) -> None:
        """Register a pending prediction for horizon t + horizon_s."""
        self.pending.append(
            {
                "t_made": t_mono,
                "t_made_utc": t_utc,
                "t_due": t_mono + self.horizon_s,
                "pred_xgb": pred_xgb,
                "pred_rf": pred_rf,
                "pred_ridge": pred_ridge,
                "pred_persistence": persistence_ratio,
                "pred_tag": pred_tag,
                "top_factor": top_factor,
            }
        )

    def process_probe(
        self,
        t_probe_mono: float,
        t_probe_utc: str,
        actual_ratio: float,
    ) -> list[dict[str, Any]]:
        """Pair incoming probe with pending predictions whose horizon has arrived."""
        newly_resolved: list[dict[str, Any]] = []
        remaining: list[dict[str, Any]] = []

        for p in self.pending:
            delta = t_probe_mono - p["t_due"]
            # Within tolerance window or overdue
            if abs(delta) <= self.tolerance_s or delta > 0:
                abs_err_xgb = abs(p["pred_xgb"] - actual_ratio)
                abs_err_rf = abs(p["pred_rf"] - actual_ratio)
                abs_err_ridge = abs(p["pred_ridge"] - actual_ratio)
                abs_err_pers = abs(p["pred_persistence"] - actual_ratio)

                pair = {
                    "time_utc": t_probe_utc,
                    "t_made_utc": p["t_made_utc"],
                    "pred_xgb": p["pred_xgb"],
                    "pred_rf": p["pred_rf"],
                    "pred_ridge": p["pred_ridge"],
                    "pred_persistence": p["pred_persistence"],
                    "actual_ratio": actual_ratio,
                    "abs_error": abs_err_xgb,
                    "abs_error_rf": abs_err_rf,
                    "abs_error_ridge": abs_err_ridge,
                    "persistence_error": abs_err_pers,
                    "within_tol": abs_err_xgb <= 0.25,
                    "pred_tag": p["pred_tag"],
                    "top_factor": p["top_factor"],
                }
                self.resolved.append(pair)
                newly_resolved.append(pair)
            else:
                remaining.append(p)

        self.pending = remaining
        return newly_resolved

    def compute_leaderboard(self) -> list[dict[str, Any]]:
        """Compute live MAE and metrics for all models vs Persistence baseline."""
        if not self.resolved:
            return []

        n = len(self.resolved)
        errs_xgb = [r["abs_error"] for r in self.resolved]
        errs_rf = [r["abs_error_rf"] for r in self.resolved]
        errs_ridge = [r["abs_error_ridge"] for r in self.resolved]
        errs_pers = [r["persistence_error"] for r in self.resolved]

        mae_xgb = float(np.mean(errs_xgb))
        mae_rf = float(np.mean(errs_rf))
        mae_ridge = float(np.mean(errs_ridge))
        mae_pers = float(np.mean(errs_pers))

        tol_xgb = sum(1 for e in errs_xgb if e <= 0.25) / n * 100.0
        tol_rf = sum(1 for e in errs_rf if e <= 0.25) / n * 100.0
        tol_ridge = sum(1 for e in errs_ridge if e <= 0.25) / n * 100.0
        tol_pers = sum(1 for e in errs_pers if e <= 0.25) / n * 100.0

        return [
            {
                "model": "XGBoost",
                "n": n,
                "mae": mae_xgb,
                "tol_pct": tol_xgb,
                "delta_mae": mae_xgb - mae_pers,
            },
            {
                "model": "RandomForest",
                "n": n,
                "mae": mae_rf,
                "tol_pct": tol_rf,
                "delta_mae": mae_rf - mae_pers,
            },
            {
                "model": "Ridge",
                "n": n,
                "mae": mae_ridge,
                "tol_pct": tol_ridge,
                "delta_mae": mae_ridge - mae_pers,
            },
            {
                "model": "Persistence Baseline",
                "n": n,
                "mae": mae_pers,
                "tol_pct": tol_pers,
                "delta_mae": 0.0,
            },
        ]


def render_watch_layout(
    elapsed_s: float,
    total_duration_s: float,
    power_state: str,
    model_name: str,
    is_synthetic: bool,
    tracker: PredictionPairTracker,
    current_mono: float,
    plain: bool = False,
) -> Panel:
    """Build the dense, magenta-bordered terminal dashboard for watch."""
    box_style = box.ASCII if plain else box.ROUNDED
    table_box = box.ASCII if plain else box.SQUARE

    status_table = Table.grid(expand=True)
    status_table.add_column(justify="left")
    status_table.add_column(justify="center")
    status_table.add_column(justify="center")
    status_table.add_column(justify="center")
    status_table.add_column(justify="right")

    time_str = f"ELAPSED: {int(elapsed_s)}s"
    pwr_str = f"POWER: {power_state.upper()}"
    demo_badge = " [DEMO]" if is_synthetic else ""
    mdl_str = f"MODEL: {model_name[:8]}{demo_badge}"
    hrz_str = "H: 20s"
    res_str = f"PAIRS: {len(tracker.resolved)}"

    status_table.add_row(
        f"[bold]{time_str}[/bold]",
        pwr_str,
        f"[magenta]{mdl_str}[/magenta]",
        hrz_str,
        f"[bold green]{res_str}[/bold green]",
    )

    # 1. Pending predictions table
    tbl_pending = Table(
        title="[PENDING PREDICTIONS (COUNTDOWN TO t + 20s)]",
        box=table_box,
        expand=True,
    )
    tbl_pending.add_column("Time Made", style="cyan", overflow="fold")
    tbl_pending.add_column("Countdown", justify="right", overflow="fold")
    tbl_pending.add_column("Predicted Ratio", justify="right", overflow="fold")
    tbl_pending.add_column("Predicted Tag", overflow="fold")
    tbl_pending.add_column("Top Contributing Factor", overflow="fold")

    term_lines = shutil.get_terminal_size(fallback=(100, 30)).lines
    max_rows = 6
    if term_lines < 26:
        max_rows = 2
    elif term_lines < 32:
        max_rows = 4

    def _local_time(iso_str: str) -> str:
        try:
            dt = datetime.fromisoformat(iso_str)
            return dt.astimezone().strftime("%H:%M:%S")
        except Exception:
            return iso_str[11:19]

    for p in tracker.pending[-max_rows:]:
        rem = max(0.0, p["t_due"] - current_mono)
        tbl_pending.add_row(
            _local_time(p["t_made_utc"]),
            f"{rem:4.1f}s left",
            f"{p['pred_xgb']:.2f}x",
            p["pred_tag"],
            p["top_factor"],
        )
    if not tracker.pending:
        tbl_pending.add_row("-", "Warming up...", "-", "-", "-")

    # 2. Resolved predictions table
    tbl_resolved = Table(
        title="[RESOLVED PREDICTIONS (PAIRED AT t + 20s)]",
        box=table_box,
        expand=True,
    )
    tbl_resolved.add_column("Time", style="cyan", overflow="fold")
    tbl_resolved.add_column("Predicted", justify="right", overflow="fold")
    tbl_resolved.add_column("Actual", justify="right", overflow="fold")
    tbl_resolved.add_column("Abs Error", justify="right", overflow="fold")
    tbl_resolved.add_column("Within Tol", justify="center", overflow="fold")
    tbl_resolved.add_column("Persistence Error", justify="right", overflow="fold")

    for r in tracker.resolved[-max_rows:]:
        tol_color = "green" if r["within_tol"] else "yellow"
        label = "YES" if r["within_tol"] else "NO"
        tol_text = f"[{tol_color}]{label}[/{tol_color}]"
        tbl_resolved.add_row(
            _local_time(r["time_utc"]),
            f"{r['pred_xgb']:.2f}x",
            f"{r['actual_ratio']:.2f}x",
            f"{r['abs_error']:.3f}",
            tol_text,
            f"{r['persistence_error']:.3f}",
        )
    if not tracker.resolved:
        tbl_resolved.add_row("-", "-", "-", "-", "Waiting for horizon...", "-")

    # 3. Leaderboard table
    tbl_board = Table(title="[LIVE MODEL LEADERBOARD]", box=table_box, expand=True)
    tbl_board.add_column("Model", style="bold", overflow="fold")
    tbl_board.add_column("Resolved N", justify="right", overflow="fold")
    tbl_board.add_column("Live MAE", justify="right", overflow="fold")
    tbl_board.add_column("Within 0.25 (%)", justify="right", overflow="fold")
    tbl_board.add_column("Vs Persistence", justify="right", overflow="fold")

    leaderboard = tracker.compute_leaderboard()
    for row in leaderboard:
        delta = row["delta_mae"]
        if row["model"] == "Persistence Baseline":
            delta_str = "-"
        else:
            d_color = "green" if delta <= 0 else "red"
            delta_str = f"[{d_color}]{delta:+.4f}[/{d_color}]"

        mae_val = row["mae"]
        if mae_val > 100.0 or np.isnan(mae_val) or np.isinf(mae_val):
            mae_str = "[red]invalid[/red]" if not plain else "invalid"
        else:
            mae_str = f"{mae_val:.4f}"

        tbl_board.add_row(
            row["model"],
            str(row["n"]),
            mae_str,
            f"{row['tol_pct']:.1f}%",
            delta_str,
        )
    if not leaderboard:
        tbl_board.add_row("-", "0", "-", "-", "-")

    main_grid = Table.grid(expand=True)
    main_grid.add_row(status_table)
    main_grid.add_row("")
    main_grid.add_row(tbl_pending)
    main_grid.add_row("")
    main_grid.add_row(tbl_resolved)
    main_grid.add_row("")
    main_grid.add_row(tbl_board)

    border_color = "magenta" if not plain else "white"
    return Panel(
        main_grid,
        title="[DESKFLOW LIVE ACCURACY WATCH - HORIZON 20s]",
        border_style=border_color,
        box=box_style,
    )


@app.callback(invoke_without_command=True)
def watch_command(
    duration: int = typer.Option(90, "--duration", help="Watch session duration in seconds."),
    plain: bool = typer.Option(False, "--plain", help="Render pure ASCII without colors."),
    replay: Path | None = typer.Option(
        None, "--replay", help="Replay a previously recorded run JSONL file."
    ),
    fast: bool = typer.Option(
        False, "--fast", help="Run at maximum speed without wall-clock sleeps (for replay/tests)."
    ),
) -> None:
    """Live accuracy verification: prove that predictions at t match measured slowdown at t+20s."""
    console = Console(no_color=plain)
    adapter = get_adapter()
    power_state = get_current_power_state(adapter)

    # 1. Resolve baseline
    try:
        baseline = load_baseline(power_state=power_state)
    except (FileNotFoundError, CalibrationError):
        try:
            baseline = load_baseline()
        except (FileNotFoundError, CalibrationError):
            baseline = Baseline(
                machine_id="default",
                created_utc=datetime.now(UTC).isoformat(),
                platform=adapter.name,
                cpu_ns=15_000_000.0,
                mem_ns=2_000_000.0,
                disk_ns=3_000_000.0,
                total_ns=20_000_000.0,
                p10_total_ns=15_000_000.0,
                p90_total_ns=25_000_000.0,
                sample_count=30,
                deskflow_version="0.1.0",
                power_state=power_state,
            )

    # 2. Resolve model
    try:
        model_dir, is_bundled = resolve_model_dir()
    except FileNotFoundError:
        console.print("[red]Error: No trained or bundled model set found.[/red]")
        raise typer.Exit(code=1) from None

    card = load_model_card(model_dir)
    runtime = RuntimeModels.load(model_dir)
    is_synth = bool(card.get("synthetic", False)) or is_bundled
    model_name = card.get("model_id", "Default")

    tracker = PredictionPairTracker(horizon_s=20.0, tolerance_s=3.0)
    records_buffer: list[Record] = []
    start_mono = time.monotonic()
    t_last_probe_mono = start_mono

    watch_dir = get_data_dir() / "watch"
    watch_dir.mkdir(parents=True, exist_ok=True)
    utc_timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_session_file = watch_dir / f"watch-{utc_timestamp}.json"

    console.print(
        Panel(
            f"Starting Live Accuracy Watch ({duration}s session)...\n"
            "Warming up telemetry buffer (12s)...",
            style="magenta",
            box=box.ASCII if plain else box.ROUNDED,
        )
    )

    baseline_median_ms = baseline.total_ns / 1_000_000.0 if baseline.total_ns > 0 else 1.0

    if replay and replay.exists():
        # Replay mode
        with open(replay, encoding="utf-8") as f:
            lines = [json.loads(line) for line in f if line.strip()]

        with Live(
            render_watch_layout(
                0.0,
                duration,
                power_state,
                model_name,
                is_synth,
                tracker,
                start_mono,
                plain=plain,
            ),
            console=console,
            refresh_per_second=2,
            vertical_overflow="crop",
        ) as live:
            for entry in lines:
                raw_sample = entry.get("sample", {})
                raw_probe = entry.get("probe")
                probe_res = ProbeResult.from_dict(raw_probe) if raw_probe else None
                sample_obj = SystemSample(**raw_sample)
                rec = Record(
                    schema_version=1,
                    run_id="watch-replay",
                    label="watch",
                    sample=sample_obj,
                    probe=probe_res,
                    top_processes=[],
                    storage=None,
                )
                records_buffer.append(rec)
                cur_mono = rec.sample.monotonic_s

                if probe_res is not None:
                    actual_ratio = (
                        (probe_res.total_ns / 1_000_000.0) / baseline_median_ms
                        if baseline_median_ms > 0
                        else 1.0
                    )
                    tracker.process_probe(cur_mono, rec.sample.ts_utc, actual_ratio)

                    # If we have at least 12s history, make a prediction
                    history_s = (
                        records_buffer[-1].sample.monotonic_s - records_buffer[0].sample.monotonic_s
                    )
                    if len(records_buffer) >= 2 and history_s >= 12.0:
                        feat_dict = build_row(records_buffer, baseline=baseline)
                        X_row = np.array([[feat_dict[k] for k in FEATURE_NAMES]])
                        pred_dict = runtime.predict(X_row)
                        slow_preds = pred_dict["slowdown_ratio"]

                        pred_xgb = float(
                            np.clip(slow_preds.get("XGBoost", actual_ratio), 0.3, 10.0)
                        )
                        pred_rf = float(
                            np.clip(slow_preds.get("RandomForest", actual_ratio), 0.3, 10.0)
                        )
                        pred_ridge = float(
                            np.clip(slow_preds.get("Ridge", actual_ratio), 0.3, 10.0)
                        )
                        persistence_ratio = float(np.clip(actual_ratio, 0.3, 10.0))

                        out_flag = pred_dict.get("out_of_range", False)
                        top_fac = (
                            "out-of-range telemetry"
                            if out_flag
                            else "system metrics within normal range"
                        )
                        if not out_flag:
                            for k in FEATURE_NAMES:
                                if feat_dict[k] > 0.05 and k in FEATURE_MAP:
                                    top_fac = FEATURE_MAP[k].plain
                                    break

                        pred_tag = "healthy"
                        if (
                            "predicted_issue" in pred_dict
                            and "XGBoost" in pred_dict["predicted_issue"]
                        ):
                            pred_tag = pred_dict["predicted_issue"]["XGBoost"]

                        tracker.add_prediction(
                            t_mono=cur_mono,
                            t_utc=rec.sample.ts_utc,
                            pred_xgb=pred_xgb,
                            pred_rf=pred_rf,
                            pred_ridge=pred_ridge,
                            persistence_ratio=persistence_ratio,
                            pred_tag=pred_tag,
                            top_factor=top_fac,
                        )

                elapsed = cur_mono - records_buffer[0].sample.monotonic_s
                live.update(
                    render_watch_layout(
                        elapsed,
                        duration,
                        power_state,
                        model_name,
                        is_synth,
                        tracker,
                        cur_mono,
                        plain=plain,
                    )
                )
                if not fast:
                    time.sleep(0.1)

    else:
        # Live execution mode
        sampler = Sampler(adapter=adapter)
        with Live(
            render_watch_layout(
                0.0,
                duration,
                power_state,
                model_name,
                is_synth,
                tracker,
                start_mono,
                plain=plain,
            ),
            console=console,
            refresh_per_second=2,
            vertical_overflow="crop",
        ) as live:
            try:
                while (time.monotonic() - start_mono) < duration:
                    now_mono = time.monotonic()
                    elapsed = now_mono - start_mono

                    # Collect sample every 2 seconds
                    sys_sample, top_procs = sampler.sample()
                    probe_res = None

                    # Probe every 4 seconds
                    if (now_mono - t_last_probe_mono) >= 4.0 or not records_buffer:
                        t_last_probe_mono = now_mono
                        probe_res = run_probe()

                    rec = Record(
                        schema_version=1,
                        run_id="watch-live",
                        label="watch",
                        sample=sys_sample,
                        probe=probe_res,
                        top_processes=top_procs,
                        storage=None,
                    )
                    records_buffer.append(rec)

                    # Drop records older than 45s
                    while (
                        records_buffer and (now_mono - records_buffer[0].sample.monotonic_s) > 45.0
                    ):
                        records_buffer.pop(0)

                    if probe_res is not None:
                        actual_ratio = (
                            (probe_res.total_ns / 1_000_000.0) / baseline_median_ms
                            if baseline_median_ms > 0
                            else 1.0
                        )
                        tracker.process_probe(now_mono, sys_sample.ts_utc, actual_ratio)

                        # If warm-up (12s) is reached, generate next prediction
                        history_s = (
                            records_buffer[-1].sample.monotonic_s
                            - records_buffer[0].sample.monotonic_s
                        )
                        if len(records_buffer) >= 2 and history_s >= 12.0:
                            feat_dict = build_row(records_buffer, baseline=baseline)
                            X_row = np.array([[feat_dict[k] for k in FEATURE_NAMES]])
                            runtime_preds = runtime.predict(X_row)
                            slow_dict = runtime_preds.get("slowdown_ratio", {})
                            pred_xgb = float(slow_dict.get("XGBoost", actual_ratio))
                            pred_rf = float(slow_dict.get("RandomForest", actual_ratio))
                            pred_ridge = float(slow_dict.get("Ridge", actual_ratio))

                            top_fac = "system metrics within normal range"
                            for k in FEATURE_NAMES:
                                if feat_dict[k] > 0.05 and k in FEATURE_MAP:
                                    top_fac = FEATURE_MAP[k].plain
                                    break

                            tracker.add_prediction(
                                t_mono=now_mono,
                                t_utc=sys_sample.ts_utc,
                                pred_xgb=pred_xgb,
                                pred_rf=pred_rf,
                                pred_ridge=pred_ridge,
                                persistence_ratio=actual_ratio,
                                pred_tag="healthy",
                                top_factor=top_fac,
                            )

                    live.update(
                        render_watch_layout(
                            elapsed,
                            duration,
                            power_state,
                            model_name,
                            is_synth,
                            tracker,
                            now_mono,
                            plain=plain,
                        )
                    )
                    time.sleep(2.0)
            except KeyboardInterrupt:
                pass

    # Print final summary
    final_board = tracker.compute_leaderboard()
    console.print("\n[bold]Final Live Accuracy Leaderboard:[/bold]")
    table_fin = Table(box=box.ASCII if plain else box.ROUNDED)
    table_fin.add_column("Model", style="bold")
    table_fin.add_column("Resolved Pairs", justify="right")
    table_fin.add_column("Live MAE", justify="right")
    table_fin.add_column("Within 0.25 (%)", justify="right")
    table_fin.add_column("Vs Persistence", justify="right")

    for row in final_board:
        delta = row["delta_mae"]
        if row["mae"] > 100.0:
            mae_str = "invalid"
            delta_str = "invalid"
        else:
            mae_str = f"{row['mae']:.4f}"
            delta_str = "-" if row["model"] == "Persistence Baseline" else f"{delta:+.4f}"
        table_fin.add_row(
            row["model"],
            str(row["n"]),
            mae_str,
            f"{row['tol_pct']:.1f}%",
            delta_str,
        )
    console.print(table_fin)

    # Save session
    session_data = {
        "session_utc": utc_timestamp,
        "duration_s": duration,
        "power_state": power_state,
        "model_id": model_name,
        "is_synthetic": is_synth,
        "leaderboard": final_board,
        "resolved_pairs": tracker.resolved,
    }
    with open(out_session_file, "w", encoding="utf-8") as f:
        json.dump(session_data, f, indent=2)
    console.print(f"[dim]Watch session log saved to {out_session_file}[/dim]")
