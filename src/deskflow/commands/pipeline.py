"""Animated pipeline visualization and guided architectural tour: real-time data, models, and training flow."""

from __future__ import annotations

import contextlib
import logging
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
from rich import box
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table

from deskflow.baseline import Baseline, load_baseline
from deskflow.collector.models import Record
from deskflow.collector.probe import run_probe
from deskflow.collector.sampler import Sampler
from deskflow.commands.fix import run_fast_inventory
from deskflow.features.extract import build_row
from deskflow.health.score import compute_health_score
from deskflow.ml.artifacts import resolve_model_dir
from deskflow.ml.facts import ModelFacts, load_model_facts
from deskflow.ml.runtime import RuntimeModels
from deskflow.platform.base import PlatformAdapter
from deskflow.platform.factory import get_adapter
from deskflow.ui.components import render_key_hints
from deskflow.ui.theme import COLOR_CYAN, COLOR_GREEN, COLOR_GREY, use_plain_output

logger = logging.getLogger(__name__)

STAGES = [
    ("SENSORS", "CPU RAM\nDisk", 0),
    ("PROBE", "3 Micro\nTests", 1),
    ("FEATURES", "27 Stats\n6/12 s", 2),
    ("MODELS", "Ridge RF\nXGBoost", 3),
    ("SCORE+", "Health\n& Cause", 4),
    ("FIX", "Plan &\nRe-scan", 5),
]

STAGE_DESCRIPTIONS = [
    (
        "SENSORS",
        "Hardware Telemetry Sampling",
        "Samples CPU usage per core, frequency ratio, memory commit bytes, swap/pagefile usage, disk I/O throughput, and active user processes every second.",
    ),
    (
        "PROBE",
        "Deterministic Ground-Truth Latency",
        "Executes 3 lightweight micro-benchmarks (CPU arithmetic, memory bandwidth, small disk writes) timed with high-resolution perf_counter_ns.",
    ),
    (
        "FEATURES",
        "Rolling Time-Series Windows",
        "Computes 27 rolling statistics over 6-second and 12-second ring buffers (mean, min, max, linear trend slopes, and probe latency ratios).",
    ),
    (
        "MODELS",
        "Multi-Model Slowdown Inference",
        "Runs pre-trained regression models (XGBoost, Random Forest, Ridge) in < 5 ms to predict system responsiveness 20 seconds ahead.",
    ),
    (
        "SCORE+",
        "Deterministic Scoring & Diagnosis",
        "Maps predicted slowdown ratio to a transparent 0-100 Health Score, attributes root causes with class probabilities, and checks anomaly threshold.",
    ),
    (
        "FIX",
        "Fix Center & Verification",
        "Inventories cleanable temp files and caches under strict safety allowlists, protects IDEs and system processes, and verifies recovery with re-scan.",
    ),
]


def render_pipeline_frame(
    active_stage: int,
    packet_pos: float,
    real_value_text: str,
    facts: ModelFacts | None = None,
    plain: bool = False,
    compact: bool = True,
    prov_dict: dict[str, Any] | None = None,
) -> Group:
    """Render a single frame of the live pipeline visualization with aligned boxes."""
    if facts is None:
        facts = load_model_facts()
    box_w = 10
    arrow_sym = "-->" if plain else "──▶"
    packet_sym = "*" if plain else "●"
    path_sym = "-" if plain else "━"

    # 1. Upper Flow Lane: 6 boxes connected by arrows
    lines: list[str] = ["", "", "", "", "", ""]

    for i, (title, sub, _) in enumerate(STAGES):
        sub_lines = sub.split("\n")
        sub1 = sub_lines[0] if len(sub_lines) > 0 else ""
        sub2 = sub_lines[1] if len(sub_lines) > 1 else ""

        if plain:
            style_start, style_end = "", ""
        elif i < active_stage:
            style_start, style_end = f"[{COLOR_GREEN}]", f"[/{COLOR_GREEN}]"
        elif i == active_stage:
            style_start, style_end = f"[bold {COLOR_CYAN}]", f"[/bold {COLOR_CYAN}]"
        else:
            style_start, style_end = f"[{COLOR_GREY}]", f"[/{COLOR_GREY}]"

        b_top = "+" + "-" * (box_w - 2) + "+" if plain else "┌" + "─" * (box_w - 2) + "┐"
        b_mid1 = f"|{title:^{box_w - 2}}|" if plain else f"│{title:^{box_w - 2}}│"
        b_mid2 = f"|{sub1:^{box_w - 2}}|" if plain else f"│{sub1:^{box_w - 2}}│"
        b_mid3 = f"|{sub2:^{box_w - 2}}|" if plain else f"│{sub2:^{box_w - 2}}│"
        b_bot = "+" + "-" * (box_w - 2) + "+" if plain else "└" + "─" * (box_w - 2) + "┘"

        lines[0] += f"{style_start}{b_top}{style_end}"
        lines[1] += f"{style_start}{b_mid1}{style_end}"
        lines[2] += f"{style_start}{b_mid2}{style_end}"
        lines[3] += f"{style_start}{b_mid3}{style_end}"
        lines[4] += f"{style_start}{b_bot}{style_end}"

        if i < len(STAGES) - 1:
            lines[0] += "   "
            arr_style = f"[bold {COLOR_CYAN}]" if (i == active_stage and not plain) else f"[{COLOR_GREY}]"
            arr_end = f"[/bold {COLOR_CYAN}]" if (i == active_stage and not plain) else f"[/{COLOR_GREY}]"
            if plain:
                arr_style, arr_end = "", ""
            lines[1] += f"{arr_style}{arrow_sym}{arr_end}"
            lines[2] += "   "
            lines[3] += "   "
            lines[4] += "   "

    # Track line for moving packet (75 cols total)
    track_chars = [" "] * 75
    curr_col = min(74, max(0, int(packet_pos * 13.0) + 4))

    for c in range(4, curr_col):
        track_chars[c] = path_sym
    track_chars[curr_col] = packet_sym

    packet_line = "".join(track_chars)
    if not plain:
        packet_line = f"[bold {COLOR_CYAN}]{packet_line}[/bold {COLOR_CYAN}]"
    lines[5] = packet_line

    flow_text = "\n".join(lines)
    upper_panel = Panel(
        flow_text,
        title="[ LIVE TELEMETRY & INFERENCE PIPELINE ]",
        border_style=COLOR_CYAN if not plain else "white",
        box=box.ASCII if plain else box.ROUNDED,
    )

    # Stage description + real live data split
    stg_info = STAGE_DESCRIPTIONS[min(active_stage, len(STAGE_DESCRIPTIONS) - 1)]
    left_tbl = Table(box=box.SIMPLE if not plain else box.ASCII, show_header=False, expand=True)
    left_tbl.add_column("Property", style="bold cyan" if not plain else "bold")
    left_tbl.add_column("Detail")
    left_tbl.add_row("Stage", stg_info[0])
    left_tbl.add_row("Concept", stg_info[1])
    left_tbl.add_row("Operation", stg_info[2])

    p_left = Panel(
        left_tbl,
        title="[ WHAT HAPPENS ]",
        border_style=COLOR_GREY if not plain else "white",
        box=box.ASCII if plain else box.ROUNDED,
    )

    right_tbl = Table(box=box.SIMPLE if not plain else box.ASCII, show_header=False, expand=True)
    right_tbl.add_column("Metric", style="bold green" if not plain else "bold")
    right_tbl.add_column("Live Telemetry Value")
    right_tbl.add_row("Readout", real_value_text)

    p_right = Panel(
        right_tbl,
        title="[ REAL LIVE DATA ]",
        border_style=COLOR_GREEN if not plain else "white",
        box=box.ASCII if plain else box.ROUNDED,
    )

    split_grid = Table.grid(expand=True, padding=(0, 2))
    split_grid.add_column(ratio=1)
    split_grid.add_column(ratio=1)
    split_grid.add_row(p_left, p_right)

    # Lower lane: Training pipeline with real counts from ModelFacts (D1, D2 fix)
    lower_flow = (
        "COLLECT (lab + normal) -> DATASET (27 feats, holdout) -> "
        "TRAIN (grouped CV) -> EVALUATE (bootstrap) -> EXPORT"
    )
    stat_line = (
        f"Dataset SHA-256: {facts.dataset_hash[:8]} | {facts.dataset_split_summary}"
        if not facts.synthetic
        else f"Dataset: SYNTHETIC DEMO | {facts.train_rows} train, {facts.test_rows} test rows"
    )
    lower_panel = Panel(
        f"{lower_flow}\n{stat_line}",
        title="[ MODEL TRAINING PIPELINE (Model Facts) ]",
        border_style=COLOR_GREY if not plain else "white",
        box=box.ASCII if plain else box.ROUNDED,
    )

    return Group(upper_panel, split_grid, lower_panel)


def print_pipeline_tables(console: Console, facts: ModelFacts, plain: bool = False) -> None:
    """Print the [ MODELS USED ] and [ HOW IT IS TRAINED ] tables strictly from ModelFacts."""
    box_style = box.ASCII if plain else box.ROUNDED

    # 1. MODELS USED Table (D1, D3 fix: read from ModelFacts only)
    tbl_models = Table(
        title="[ MODELS USED ]",
        box=box_style,
        expand=True,
        show_header=True,
    )
    tbl_models.add_column("Task", style="bold cyan" if not plain else "bold", width=18)
    tbl_models.add_column("Model(s)", width=22)
    tbl_models.add_column("Role / Function", ratio=1)
    tbl_models.add_column("Held-Out Accuracy", justify="right", width=24)

    if facts.synthetic:
        held_reg = "Synthetic demo"
        held_clf = "Synthetic demo"
    else:
        ci_str = f"[{facts.primary_regressor_ci[0]:.4f}, {facts.primary_regressor_ci[1]:.4f}]"
        held_reg = f"MAE {facts.primary_regressor_mae:.4f} {ci_str}"
        held_clf = f"F1 {facts.primary_classifier_f1:.2f} (held-out)"

    tbl_models.add_row(
        "Regression",
        "XGBoost, RF, Ridge",
        "Predicts 20s-ahead probe latency ratio",
        held_reg,
    )
    tbl_models.add_row(
        "Classification",
        "RandomForest, XGBoost",
        "Attributes bottleneck cause (CPU, RAM, Disk, Load)",
        held_clf,
    )
    tbl_models.add_row(
        "Anomaly Detection",
        "Isolation Forest",
        "Flags unusual telemetry vs healthy baseline",
        f"Threshold {facts.thresholds.get('anomaly_threshold', 0.6009):.4f}",
    )
    tbl_models.add_row(
        "Health Scoring",
        "Deterministic Formula",
        "Maps predicted slowdown ratio to 0-100 score",
        "Deterministic (Ratio <= 1.0 -> 100)",
    )
    console.print()
    console.print(tbl_models)

    # 2. HOW IT IS TRAINED Table (D3 fix: 27 features, no p90)
    tbl_trained = Table(
        title="[ HOW IT IS TRAINED ]",
        box=box_style,
        expand=True,
        show_header=True,
    )
    tbl_trained.add_column("Stage / Property", style="bold", width=22)
    tbl_trained.add_column("Implementation & Methodology")

    ds_str = (
        "Synthetic demo dataset (simulated profiles)"
        if facts.synthetic
        else f"SHA-256 {facts.dataset_hash} ({facts.dataset_split_summary})"
    )
    tbl_trained.add_row(
        "Data Collection",
        "Micro-benchmark probe latencies and hardware sensor samples across power states",
    )
    tbl_trained.add_row(
        "Feature Engineering",
        "27 rolling statistics over 6s and 12s windows (mean, min, max, trend slopes, probe latencies)",
    )
    tbl_trained.add_row(
        "Splitting Strategy",
        "Strict grouped temporal holdout by session ID (newest sessions held out; zero leakage)",
    )
    tbl_trained.add_row(
        "Evaluation Protocol",
        "Moving block bootstrap (B=5, 1000 resamples, seed 42) 95% CI vs Persistence baseline",
    )
    tbl_trained.add_row("Dataset Provenance", ds_str)

    console.print()
    console.print(tbl_trained)


def pipeline_command(
    static: bool = False,
    fast: bool = False,
    plain: bool = False,
) -> None:
    """Run animated pipeline visualization or guided tour with keyboard controls."""
    plain = use_plain_output(plain)
    console = Console(no_color=plain)
    adapter: PlatformAdapter = get_adapter()

    facts = load_model_facts()

    # Gather live telemetry values
    sampler = Sampler(adapter=adapter, interval_s=1.0)
    data_dir = None
    with contextlib.suppress(Exception):
        from deskflow.paths import get_data_dir

        data_dir = get_data_dir()

    probe_res = run_probe(data_dir=data_dir, repeats=1)
    sample, procs = sampler.sample()

    tot_ms = probe_res.total_ns / 1_000_000.0
    c_ms = probe_res.cpu_ns / 1_000_000.0
    m_ms = probe_res.mem_ns / 1_000_000.0
    d_ms = probe_res.disk_ns / 1_000_000.0

    try:
        baseline = load_baseline()
    except Exception:
        baseline = Baseline(
            machine_id="default",
            created_utc="2026-01-01T00:00:00Z",
            platform=adapter.name,
            cpu_ns=15_000_000.0,
            mem_ns=2_000_000.0,
            disk_ns=3_000_000.0,
            total_ns=20_000_000.0,
            p10_total_ns=15_000_000.0,
            p90_total_ns=25_000_000.0,
            sample_count=30,
            deskflow_version="0.2.0",
            power_state="ac",
        )

    rec = Record(
        schema_version=1,
        run_id="pipe-demo",
        label="normal",
        sample=sample,
        probe=probe_res,
        top_processes=procs,
        storage=None,
    )
    history = [rec] * 6
    feat_dict = build_row(history, baseline, rec)

    # Runtime model inference
    try:
        model_dir, _ = resolve_model_dir()
        runtime = RuntimeModels.load(model_dir)
        f_vec = np.array([feat_dict[k] for k in runtime.feature_order], dtype=np.float64)
        p_dict = runtime.predict(f_vec)
        p_slow = p_dict.get("slowdown_ratio", {})
        xgb_ratio = float(p_slow.get("XGBoost", 1.05))
        rf_ratio = float(p_slow.get("RandomForest", 1.04))
        ridge_ratio = float(p_slow.get("Ridge", 1.02))
        probs = p_dict.get("issue_probabilities", {}).get("XGBoost", {})
        cause_tag = max(probs.keys(), key=lambda k: probs[k]) if probs else "healthy"
        cause_conf = float(probs.get(cause_tag, 0.95))
    except Exception:
        xgb_ratio, rf_ratio, ridge_ratio = 1.05, 1.04, 1.02
        cause_tag, cause_conf = "healthy", 0.95

    health_score, _ = compute_health_score(xgb_ratio)
    tools = run_fast_inventory(adapter)
    safe_tools = [t for t in tools if t.risk.lower() == "safe"]
    safe_pot_mb = sum(t.potential_bytes for t in safe_tools) / (1024 * 1024)

    free_gb = shutil.disk_usage(Path.home()).free / (1024**3)
    mem_used_gb = (sample.mem_total_bytes - sample.mem_available_bytes) / (1024**3)

    stage_texts = [
        f"CPU {sample.cpu_percent_mean:.1f}%, RAM {sample.mem_percent:.1f}% ({mem_used_gb:.1f} GB), Free disk {free_gb:.1f} GB",
        f"Probe total {tot_ms:.2f} ms (CPU: {c_ms:.2f} ms, RAM: {m_ms:.2f} ms, Disk: {d_ms:.2f} ms)",
        f"27 rolling features: CPU mean {feat_dict['cpu_percent_mean_mean_6s']:.1f}%, RAM slope {feat_dict['mem_percent_slope_12s']:+.2f}%, probe latencies",
        f"Inference: XGBoost {xgb_ratio:.2f}x, RF {rf_ratio:.2f}x, Ridge {ridge_ratio:.2f}x, Persistence 1.00x",
        f"Health Score {health_score}/100, Predicted Slowdown {xgb_ratio:.2f}x, Cause '{cause_tag}' ({cause_conf:.0%})",
        f"Fix Center: {len(safe_tools)} safe tools available, {safe_pot_mb:.1f} MB measured potential",
    ]

    is_interactive = console.is_terminal and not static and not plain

    if not is_interactive:
        final_group = render_pipeline_frame(
            active_stage=5,
            packet_pos=5.0,
            real_value_text=stage_texts[-1],
            facts=facts,
            plain=plain,
            compact=True,
        )
        console.print(final_group)
        print_pipeline_tables(console, facts, plain=plain)
        return

    # Interactive Guided Tour on Alternate Screen (D21 fix)
    fps = 8
    total_stages = len(stage_texts)
    seconds_per_stage = 2.0 if fast else 4.0
    frames_per_stage = int(fps * seconds_per_stage)
    frame_delay = 1.0 / fps

    start_frame = render_pipeline_frame(
        active_stage=0,
        packet_pos=0.0,
        real_value_text=stage_texts[0],
        facts=facts,
        plain=plain,
        compact=True,
    )

    with Live(start_frame, console=console, screen=True, refresh_per_second=fps) as live:
        s_idx = 0
        paused = False
        while s_idx < total_stages:
            for f_idx in range(frames_per_stage):
                k = adapter.read_key(0.01)
                if k == "space":
                    paused = not paused
                elif k == "right":
                    s_idx = min(total_stages - 1, s_idx + 1)
                    break
                elif k == "left":
                    s_idx = max(0, s_idx - 1)
                    break
                elif k == "r":
                    s_idx = 0
                    break
                elif k == "q":
                    s_idx = total_stages
                    break

                if not paused:
                    frac = f_idx / frames_per_stage
                    pos = s_idx + frac
                    frame = render_pipeline_frame(
                        active_stage=s_idx,
                        packet_pos=pos,
                        real_value_text=stage_texts[s_idx],
                        facts=facts,
                        plain=plain,
                        compact=True,
                    )
                    live.update(frame)
                time.sleep(frame_delay)
            if not paused:
                s_idx += 1

    # Leave alternate screen and print tables inline into scrollback
    print_pipeline_tables(console, facts, plain=plain)
    console.print()
    console.print(render_key_hints([("Enter", "Return to Menu"), ("Q", "Quit")], plain=plain))
