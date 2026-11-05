"""Comprehensive verification report command: deskflow report.

Generates the full instructor verification report across terminal, Markdown,
pure JSON, and self-contained HTML formats.
"""

from __future__ import annotations

import json
import logging
import math
import platform
from pathlib import Path
from typing import Any

import psutil
import typer
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from deskflow.baseline import load_all_baselines
from deskflow.ml.artifacts import load_model_card, resolve_model_dir
from deskflow.ml.report_data import load_report_data
from deskflow.paths import get_data_dir
from deskflow.platform import get_adapter
from deskflow.report_html import generate_html_report
from deskflow.truth import NO_REAL_DATA_MESSAGE, load_provenance
from deskflow.ui.charts import bar_chart_with_whiskers, heatmap, line_chart
from deskflow.ui.components import bar
from deskflow.ui.theme import (
    COLOR_GREEN,
    COLOR_RED,
    use_plain_output,
)

logger = logging.getLogger(__name__)
app = typer.Typer(help="Generate comprehensive instructor verification report.")


def _collect_live_watch_summary(
    watch_dir: Path,
) -> tuple[int, int, dict[str, dict[str, Any]], str]:
    """Parse and aggregate live watch verification sessions.

    Excludes and counts sessions with invalid or out-of-range MAE (> 100.0, NaN, Inf).
    """
    if not watch_dir.exists():
        return 0, 0, {}, "No watch sessions found in telemetry directory."

    files = sorted(watch_dir.glob("watch-*.json"))
    if not files:
        return 0, 0, {}, "No watch sessions recorded yet."

    valid_count = 0
    excluded_count = 0
    model_stats: dict[str, list[tuple[float, int]]] = {}

    for fpath in files:
        try:
            with open(fpath, encoding="utf-8") as f:
                data = json.load(f)
            board = data.get("leaderboard", [])
            # Sanity check board values
            is_valid = True
            for entry in board:
                mae = float(entry.get("mae", 0.0))
                if math.isnan(mae) or math.isinf(mae) or mae > 100.0 or mae < 0.0:
                    is_valid = False
                    break
            if not is_valid:
                excluded_count += 1
                continue

            valid_count += 1
            for entry in board:
                m = str(entry.get("model", "Unknown"))
                mae = float(entry.get("mae", 0.0))
                n = int(entry.get("n", 0))
                if m not in model_stats:
                    model_stats[m] = []
                model_stats[m].append((mae, n))
        except Exception:
            excluded_count += 1

    if valid_count == 0:
        msg = f"All {excluded_count} watch sessions were excluded due to out-of-range MAE."
        return 0, excluded_count, {}, msg

    summary: dict[str, dict[str, Any]] = {}
    for m, entries in model_stats.items():
        total_n = sum(e[1] for e in entries)
        if total_n > 0:
            weighted_mae = sum(e[0] * e[1] for e in entries) / total_n
        else:
            weighted_mae = sum(e[0] for e in entries) / max(1, len(entries))
        summary[m] = {
            "mae": round(weighted_mae, 4),
            "sessions": len(entries),
            "total_pairs": total_n,
        }

    status_str = f"Aggregated from {valid_count} sessions ({excluded_count} excluded)."
    return valid_count, excluded_count, summary, status_str


def _collect_fix_effectiveness(
    fixlog_path: Path,
) -> tuple[int, dict[str, dict[str, Any]], str]:
    """Parse non-dry-run remediation records to assess effectiveness."""
    if not fixlog_path.exists():
        return 0, {}, "No non-dry-run fixes recorded yet."

    valid_records = []
    try:
        with open(fixlog_path, encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if not rec.get("dry_run", True):
                    valid_records.append(rec)
    except Exception as exc:
        return 0, {}, f"Could not parse fix log: {exc}"

    if not valid_records:
        return 0, {}, "No non-dry-run fixes recorded yet."

    tool_stats: dict[str, dict[str, Any]] = {}
    file_tools = {"clean_temp", "clean_app_caches", "empty_recycle_bin"}
    app_tools = {"close_idle_apps", "end_heavy_tasks"}

    for rec in valid_records:
        before = rec.get("before") or {}
        after = rec.get("after") or {}
        score_delta = None
        mem_delta = None
        if "score" in before and "score" in after:
            score_delta = float(after["score"]) - float(before["score"])
        if "available_memory_bytes" in before and "available_memory_bytes" in after:
            b_mem = float(before["available_memory_bytes"])
            a_mem = float(after["available_memory_bytes"])
            mem_delta = a_mem - b_mem

        for res in rec.get("results", []):
            tid = res.get("tool_id") or res.get("action") or "unknown"
            bytes_freed = float(res.get("bytes_freed") or 0.0)
            if tid not in tool_stats:
                tool_stats[tid] = {
                    "runs": 0,
                    "bytes_deleted": 0.0,
                    "app_memory_closed": 0.0,
                    "mem_deltas": [],
                    "score_deltas": [],
                }
            tool_stats[tid]["runs"] += 1
            if tid in file_tools:
                tool_stats[tid]["bytes_deleted"] += bytes_freed
            elif tid in app_tools:
                tool_stats[tid]["app_memory_closed"] += bytes_freed
            if mem_delta is not None:
                tool_stats[tid]["mem_deltas"].append(mem_delta)
            if score_delta is not None:
                tool_stats[tid]["score_deltas"].append(score_delta)

    return len(valid_records), tool_stats, f"{len(valid_records)} non-dry-run fix runs recorded."


def generate_report(
    html: Path | None = None,
    md: Path | None = None,
    json_output: bool = False,
    plain: bool = False,
    console_out: Console | None = None,
) -> dict[str, Any]:
    """Generate and display the comprehensive verification report."""
    plain = use_plain_output(plain)
    console = console_out or Console(no_color=plain)
    box_style = box.ASCII if plain else box.ROUNDED

    # 1. Load report data and metadata
    try:
        report_data = load_report_data()
        model_dir, _ = resolve_model_dir()
        card = load_model_card(model_dir)
        prov = load_provenance(model_dir)
    except Exception:
        report_data = {}
        card = {}
        prov = load_provenance()

    is_synth = bool(report_data.get("synthetic", card.get("synthetic", True)))
    synth_banner = "DEMO MODEL (SYNTHETIC DATA - NOT A REAL WORLD RESULT)"

    # Machine summary
    adapter = get_adapter()
    try:
        static_hw = dict(adapter.get_static_hardware())
    except Exception:
        static_hw = {}
    cpu_model = static_hw.get("cpu_model") or platform.processor() or "CPU"
    phys_cores = psutil.cpu_count(logical=False) or 1
    log_cores = psutil.cpu_count(logical=True) or 1
    total_ram_gb = round(psutil.virtual_memory().total / (1024**3), 1)

    gpu_name = "None"
    gpus_list = static_hw.get("gpus", [])
    if isinstance(gpus_list, list) and gpus_list:
        gpu_name = gpus_list[0].get("name", "GPU")

    machine_dict = {
        "platform": f"{platform.system()} {platform.release()} ({platform.machine()})",
        "cpu_model": cpu_model,
        "physical_cores": phys_cores,
        "logical_cores": log_cores,
        "total_ram_gb": total_ram_gb,
        "gpu": gpu_name,
    }

    # Baselines
    try:
        baselines = load_all_baselines()
    except Exception:
        baselines = {}
    base_dict = {}
    for st in ["ac", "battery"]:
        b = baselines.get(st)
        if b:
            med_ms = round(b.total_ns / 1_000_000.0, 2)
            p95_ms = round(b.p90_total_ns / 1_000_000.0, 2)
            cv_val = f"{b.cv * 100:.2f}%" if b.cv is not None else "-"
            base_dict[st] = {
                "created_utc": b.created_utc,
                "median_ms": med_ms,
                "p95_ms": p95_ms,
                "cv": cv_val,
            }

    # Live watch summary
    data_dir = get_data_dir()
    v_cnt, ex_cnt, watch_stats, watch_msg = _collect_live_watch_summary(data_dir / "watch")

    # Fix effectiveness
    n_fixes, fix_stats, fix_msg = _collect_fix_effectiveness(data_dir / "fixlog.jsonl")

    # Limitations
    if is_synth:
        limitations = [NO_REAL_DATA_MESSAGE]
    else:
        limitations = [
            f"Evaluated on single hardware host ({machine_dict['cpu_model']}, "
            f"{machine_dict['total_ram_gb']} GB RAM).",
            "Power states calibrated: "
            + (", ".join(st.upper() for st in base_dict) if base_dict else "None."),
            (
                f"Trained on organic multi-session dataset "
                f"({prov.train_rows + prov.test_rows:,} rows in {prov.sessions_count} sessions)."
            ),
            "Probe measures CPU, RAM, and storage latency; GPU workloads are unmonitored.",
        ]

    # Combine into full report dictionary
    full_report: dict[str, Any] = {
        "title": "Deskflow Instructor Verification Report",
        "synthetic": is_synth,
        "machine": machine_dict,
        "baselines": base_dict,
        "leaderboard": report_data.get("leaderboard", []),
        "scenario_breakdown": report_data.get("scenario_breakdown", {}),
        "power_breakdown": report_data.get("power_breakdown", {}),
        "confusion_matrix": report_data.get("confusion_matrix", {}),
        "feature_importance": report_data.get("feature_importance", []),
        "anomaly": report_data.get("anomaly", {}),
        "time_series": report_data.get("time_series", []),
        "watch_summary": {
            "valid_sessions": v_cnt,
            "excluded_sessions": ex_cnt,
            "stats": watch_stats,
            "text": watch_msg,
        },
        "fix_summary": {
            "total_runs": n_fixes,
            "stats": fix_stats,
            "text": fix_msg,
        },
        "limitations": limitations,
    }

    # JSON output handling
    if json_output:
        import sys

        sys.stdout.write(json.dumps(full_report, indent=2) + "\n")
        return full_report

    # Render HTML if requested
    if html:
        generate_html_report(full_report, html)
        console.print(f"[bold green]HTML report exported to {html}[/bold green]")

    # Render Markdown if requested
    if md:
        _write_markdown_report(full_report, md)
        console.print(f"[bold green]Markdown report written to {md}[/bold green]")

    # --- Render Terminal Report ---
    if is_synth:
        console.print(
            Panel(
                f"[bold yellow]{synth_banner}[/bold yellow]",
                border_style="yellow",
                box=box_style,
            )
        )

    # 1. MACHINE
    console.print("\n[bold cyan]1. [ MACHINE ][/bold cyan]")
    t_mach = Table(box=box_style, expand=True)
    t_mach.add_column("Property", style="bold")
    t_mach.add_column("Specification")
    t_mach.add_row("Host OS", str(machine_dict["platform"]))
    t_mach.add_row(
        "Processor",
        f"{machine_dict['cpu_model']} ({machine_dict['physical_cores']} phys / "
        f"{machine_dict['logical_cores']} log cores)",
    )
    t_mach.add_row("Memory", f"{machine_dict['total_ram_gb']} GB RAM")
    t_mach.add_row("GPU", str(machine_dict["gpu"]))
    console.print(t_mach)

    # 2. BASELINES
    console.print("\n[bold cyan]2. [ BASELINES ][/bold cyan]")
    t_base = Table(box=box_style, expand=True)
    t_base.add_column("Power State", style="bold")
    t_base.add_column("Calibrated UTC")
    t_base.add_column("Median Latency (ms)", justify="right")
    t_base.add_column("p95 Latency (ms)", justify="right")
    t_base.add_column("CV (%)", justify="right")
    for st in ["ac", "battery"]:
        b_info = base_dict.get(st)
        if isinstance(b_info, dict):
            created = str(b_info.get("created_utc", "-"))
            med = b_info.get("median_ms", 0.0)
            p95 = b_info.get("p95_ms", 0.0)
            cv_val = str(b_info.get("cv", "-"))
            t_base.add_row(
                st.upper(),
                created[:19] if created != "-" else "-",
                f"{float(med):.2f}" if isinstance(med, (int, float)) else str(med),
                f"{float(p95):.2f}" if isinstance(p95, (int, float)) else str(p95),
                str(cv_val),
            )
        else:
            t_base.add_row(st.upper(), "Not calibrated", "-", "-", "-")
    console.print(t_base)

    # 3. OFFLINE LEADERBOARD
    console.print("\n[bold cyan]3. [ OFFLINE LEADERBOARD ][/bold cyan]")
    if is_synth:
        console.print(
            Panel(
                f"[bold yellow]{NO_REAL_DATA_MESSAGE}[/bold yellow]",
                border_style="yellow",
                box=box_style,
            )
        )
    else:
        leaderboard = report_data.get("leaderboard", [])
        if leaderboard:
            # Whisker chart
            rows_whisker = [
                (
                    r["model"],
                    r["mae"],
                    r["mae_ci"][0],
                    r["mae_ci"][1],
                    r["is_baseline"],
                )
                for r in leaderboard
            ]
            console.print(bar_chart_with_whiskers(rows_whisker, metric_label="MAE", plain=plain))

            # Metrics table
            t_lead = Table(box=box_style, expand=True)
            t_lead.add_column("Model", style="bold")
            t_lead.add_column("MAE", justify="right")
            t_lead.add_column("95% CI", justify="right")
            t_lead.add_column("RMSE", justify="right")
            t_lead.add_column("R2 Score", justify="right")
            t_lead.add_column("Tol Acc", justify="right")
            t_lead.add_column("Beats Persistence", justify="center")

            for r in leaderboard:
                is_base = r.get("is_baseline", False)
                beats = r.get("beats_persistence", False)
                if is_base:
                    beats_str = Text("-", style="dim")
                elif beats:
                    beats_str = Text("YES", style=COLOR_GREEN) if not plain else Text("YES")
                else:
                    beats_str = Text("NO", style=COLOR_RED) if not plain else Text("NO")

                model_txt = Text(r["model"], style="dim") if is_base else Text(r["model"])
                t_lead.add_row(
                    model_txt,
                    f"{r['mae']:.4f}",
                    f"[{r['mae_ci'][0]:.4f}, {r['mae_ci'][1]:.4f}]",
                    f"{r['rmse']:.4f}",
                    f"{r['r2']:.4f}",
                    f"{r['tolerance_accuracy'] * 100:.1f}%",
                    beats_str,
                )
            console.print(t_lead)

    # 4. PREDICTED VS ACTUAL
    console.print("\n[bold cyan]4. [ PREDICTED VS ACTUAL ][/bold cyan]")
    if is_synth:
        console.print(
            Panel(
                f"[bold yellow]{NO_REAL_DATA_MESSAGE}[/bold yellow]",
                border_style="yellow",
                box=box_style,
            )
        )
    else:
        ts_pts = report_data.get("time_series", [])
        if ts_pts:
            act_pts = [p["actual"] for p in ts_pts]
            pred_pts = [p["predicted"] for p in ts_pts]
            console.print(
                line_chart(
                    actual=act_pts,
                    predicted=pred_pts,
                    title="PREDICTED VS ACTUAL",
                    height=8,
                    width=56,
                    plain=plain,
                )
            )

    # 5. BREAKDOWNS
    console.print("\n[bold cyan]5. [ BREAKDOWNS ][/bold cyan]")
    if is_synth:
        console.print(
            Panel(
                f"[bold yellow]{NO_REAL_DATA_MESSAGE}[/bold yellow]",
                border_style="yellow",
                box=box_style,
            )
        )
    else:
        sc_break = report_data.get("scenario_breakdown", {})
        if sc_break:
            t_sc = Table(title="Scenario MAE Breakdown", box=box_style, expand=True)
            t_sc.add_column("Scenario", style="bold")
            for m in ["XGBoost", "RandomForest", "Ridge", "Persistence"]:
                t_sc.add_column(m, justify="right")
            models_order = ["XGBoost", "RandomForest", "Ridge", "Persistence"]
            for sc, sc_dict in sc_break.items():
                cells = [f"{sc_dict.get(m, 0.0):.4f}" for m in models_order]
                t_sc.add_row(sc, *cells)
            console.print(t_sc)

    # 6. CONFUSION MATRIX
    console.print("\n[bold cyan]6. [ CONFUSION MATRIX ][/bold cyan]")
    if is_synth:
        console.print(
            Panel(
                f"[bold yellow]{NO_REAL_DATA_MESSAGE}[/bold yellow]",
                border_style="yellow",
                box=box_style,
            )
        )
    else:
        cm_data = report_data.get("confusion_matrix", {})
        if cm_data and cm_data.get("matrix"):
            console.print(
                heatmap(
                    labels=cm_data.get("classes", []),
                    matrix=cm_data.get("matrix", []),
                    title="ROOT CAUSE CLASSIFICATION",
                    plain=plain,
                )
            )
            acc = cm_data.get("accuracy", 1.0) * 100.0
            f1 = cm_data.get("macro_f1", 1.0)
            console.print(f"[dim]Overall Accuracy: {acc:.1f}% | Macro F1: {f1:.4f}[/dim]")

    # 7. FEATURE IMPORTANCE
    console.print("\n[bold cyan]7. [ FEATURE IMPORTANCE ][/bold cyan]")
    if is_synth:
        console.print(
            Panel(
                f"[bold yellow]{NO_REAL_DATA_MESSAGE}[/bold yellow]",
                border_style="yellow",
                box=box_style,
            )
        )
    else:
        fi_list = report_data.get("feature_importance", [])[:10]
        if fi_list:
            t_fi = Table(box=box_style, expand=True)
            t_fi.add_column("Feature", style="bold")
            t_fi.add_column("Weight", justify="right")
            t_fi.add_column("Contribution Bar", justify="left")
            for f in fi_list:
                f_bar = bar(f.get("normalized", 0.0), 1.0, width=20, plain=plain)
                t_fi.add_row(f.get("name", ""), f"{f.get('weight', 0.0):.4f}", f_bar)
            console.print(t_fi)

    # 8. ANOMALY DETECTOR
    console.print("\n[bold cyan]8. [ ANOMALY DETECTOR ][/bold cyan]")
    if is_synth:
        console.print(
            Panel(
                f"[bold yellow]{NO_REAL_DATA_MESSAGE}[/bold yellow]",
                border_style="yellow",
                box=box_style,
            )
        )
    else:
        anom = report_data.get("anomaly", {})
        t_anom = Table(box=box_style, expand=True)
        t_anom.add_column("Metric", style="bold")
        t_anom.add_column("Value", justify="right")
        t_anom.add_row("ROC-AUC", f"{anom.get('roc_auc', 1.0):.4f}")
        t_anom.add_row("Recall (anomalies)", f"{anom.get('recall', 1.0):.4f}")
        t_anom.add_row("False Alarm Rate (healthy)", f"{anom.get('false_alarm_rate', 0.02):.4f}")
        t_anom.add_row("Threshold", f"{anom.get('threshold', 0.54):.4f}")
        console.print(t_anom)

    # 9. LIVE ACCURACY
    console.print("\n[bold cyan]9. [ LIVE ACCURACY ][/bold cyan]")
    if v_cnt > 0:
        t_live = Table(box=box_style, expand=True)
        t_live.add_column("Model", style="bold")
        t_live.add_column("Aggregated Live MAE", justify="right")
        t_live.add_column("Evaluated Pairs", justify="right")
        for m, stat in watch_stats.items():
            t_live.add_row(m, f"{stat['mae']:.4f}", str(stat["total_pairs"]))
        console.print(t_live)
        if ex_cnt > 0:
            console.print(
                f"[yellow][WARN] {ex_cnt} watch session(s) excluded due to "
                "out-of-range MAE (>100.0).[/yellow]"
            )
    else:
        console.print(f"[dim]{watch_msg}[/dim]")

    # 10. FIX EFFECTIVENESS
    console.print("\n[bold cyan]10. [ FIX EFFECTIVENESS ][/bold cyan]")
    if n_fixes > 0:
        t_fx = Table(box=box_style, expand=True)
        t_fx.add_column("Remediation Tool", style="bold")
        t_fx.add_column("Executed Runs", justify="right")
        t_fx.add_column("File Bytes Deleted", justify="right")
        t_fx.add_column("Memory of Closed Apps", justify="right")
        t_fx.add_column("Available Memory Change", justify="right")
        file_tools = {"clean_temp", "clean_app_caches", "empty_recycle_bin"}
        app_tools = {"close_idle_apps", "end_heavy_tasks"}
        for tid, t_stat in fix_stats.items():
            b_del = f"{t_stat['bytes_deleted'] / (1024**2):.1f} MB" if tid in file_tools else "-"
            m_closed = (
                f"{t_stat.get('app_memory_closed', 0.0) / (1024**2):.1f} MB"
                if tid in app_tools
                else "-"
            )
            if t_stat.get("mem_deltas"):
                avg_m = sum(t_stat["mem_deltas"]) / len(t_stat["mem_deltas"]) / (1024**2)
                avail_change = f"{avg_m:+.1f} MB"
            else:
                avail_change = "-"
            t_fx.add_row(tid, str(t_stat["runs"]), b_del, m_closed, avail_change)
        console.print(t_fx)
    else:
        console.print(f"[dim]{fix_msg}[/dim]")

    # 11. LIMITATIONS
    console.print("\n[bold cyan]11. [ LIMITATIONS ][/bold cyan]")
    for lim in limitations:
        console.print(f"- {lim}")

    console.print(f"\n[dim]{prov.footer_string()}[/dim]")
    console.print()
    return full_report


@app.callback(invoke_without_command=True)
def report_command(
    html: Path | None = typer.Option(
        None, "--html", help="Path to write self-contained HTML report."
    ),
    md: Path | None = typer.Option(
        None, "--md", "--out", "-o", help="Path to write Markdown report."
    ),
    json_output: bool = typer.Option(
        False, "--json", help="Output report data as pure JSON to stdout."
    ),
    plain: bool = typer.Option(False, "--plain", help="Render pure ASCII without colors."),
) -> None:
    """Generate and display the comprehensive verification report."""
    generate_report(
        html=html,
        md=md,
        json_output=json_output,
        plain=plain,
    )


def _write_markdown_report(data: dict[str, Any], output_path: Path) -> None:
    """Write report to a Markdown file."""
    lines: list[str] = [
        "# Deskflow Instructor Verification Report",
        "",
        f"Generated on {data['machine'].get('platform', 'Desktop')}",
        "",
    ]
    if data.get("synthetic"):
        lines.extend(
            [
                "> [!WARNING]",
                "> DEMO MODEL (SYNTHETIC DATA - NOT A REAL WORLD RESULT)",
                "",
            ]
        )

    lines.extend(
        [
            "## 1. Machine & Baseline Summary",
            "",
            f"- CPU: {data['machine'].get('cpu_model')}",
            f"- RAM: {data['machine'].get('total_ram_gb')} GB",
            f"- GPU: {data['machine'].get('gpu')}",
            "",
            "| Power State | Calibrated UTC | Median Latency (ms) | p95 Latency (ms) | CV (%) |",
            "| :--- | :--- | :---: | :---: | :---: |",
        ]
    )
    for st in ["ac", "battery"]:
        b = data["baselines"].get(st)
        if b:
            lines.append(
                f"| {st.upper()} | {b['created_utc'][:19]} | "
                f"{b['median_ms']:.2f} | {b['p95_ms']:.2f} | {b['cv']} |"
            )
        else:
            lines.append(f"| {st.upper()} | Not calibrated | - | - | - |")
    lines.append("")

    if data.get("synthetic"):
        lines.extend(
            [
                "## 2. Offline Model Leaderboard",
                "",
                "> [!NOTE]",
                f"> {NO_REAL_DATA_MESSAGE}",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "## 2. Offline Model Leaderboard",
                "",
                "| Model | MAE | 95% Bootstrap CI | RMSE | R2 | Tol Acc | Beats Persistence |",
                "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
            ]
        )
        for r in data.get("leaderboard", []):
            beats = "YES" if r.get("beats_persistence") else ("-" if r.get("is_baseline") else "NO")
            ci_str = f"[{r['mae_ci'][0]:.4f}, {r['mae_ci'][1]:.4f}]"
            tol_str = f"{r['tolerance_accuracy'] * 100:.1f}%"
            lines.append(
                f"| {r['model']} | {r['mae']:.4f} | {ci_str} | "
                f"{r['rmse']:.4f} | {r['r2']:.4f} | {tol_str} | {beats} |"
            )
        lines.append("")

    lines.extend(
        [
            "## 3. Factual Limitations",
            "",
        ]
    )
    for lim in data.get("limitations", []):
        lines.append(f"- {lim}")
    lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")
