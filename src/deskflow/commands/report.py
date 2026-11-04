"""Instructor summary command: deskflow report."""

import json
import logging
import platform
from pathlib import Path

import typer
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from deskflow.baseline import load_all_baselines
from deskflow.ml.artifacts import load_model_card, resolve_model_dir
from deskflow.paths import get_data_dir

logger = logging.getLogger(__name__)
app = typer.Typer(help="Generate comprehensive instructor verification report.")


@app.callback(invoke_without_command=True)
def report_command(
    out: Path | None = typer.Option(
        Path("report.md"), "--out", "-o", help="Output path for markdown report file."
    ),
    plain: bool = typer.Option(False, "--plain", help="Render pure ASCII without colors."),
) -> None:
    """Generate and display the instructor verification report.

    Combines baselines, offline metrics, and live accuracy.
    """
    console = Console(no_color=plain)
    box_style = box.ASCII if plain else box.ROUNDED

    lines_md: list[str] = [
        "# Deskflow Instructor Verification Report",
        "",
        f"Generated on {platform.system()} ({platform.machine()})",
        "",
    ]

    # 1. Model resolution and synthetic check
    try:
        model_dir, is_bundled = resolve_model_dir()
        card = load_model_card(model_dir)
        is_synth = bool(card.get("synthetic", False)) or is_bundled
    except FileNotFoundError:
        card = {}
        is_synth = True

    if is_synth:
        banner_text = "DEMO MODEL (SYNTHETIC DATA - NOT A REAL WORLD RESULT)"
        console.print(
            Panel(
                f"[bold yellow]{banner_text}[/bold yellow]",
                border_style="yellow",
                box=box_style,
            )
        )
        lines_md.extend([f"> [!WARNING]\n> {banner_text}\n", ""])

    # 2. Machine & Baseline Summary
    try:
        baselines = load_all_baselines()
    except FileNotFoundError:
        baselines = {}
    console.print("[bold]1. Machine & Baseline Summary[/bold]")
    table_base = Table(box=box_style)
    table_base.add_column("Power State", style="bold")
    table_base.add_column("Calibrated (UTC)")
    table_base.add_column("Median Latency (ms)", justify="right")
    table_base.add_column("p95 Latency (ms)", justify="right")
    table_base.add_column("CV (%)", justify="right")

    lines_md.extend(
        [
            "## 1. Machine & Baseline Summary",
            "",
            "| Power State | Calibrated (UTC) | Median Latency (ms) | p95 Latency (ms) | CV (%) |",
            "| :--- | :--- | :---: | :---: | :---: |",
        ]
    )

    for state in ["ac", "battery"]:
        b = baselines.get(state)
        if b:
            cv_val = f"{b.cv * 100:.2f}%" if b.cv is not None else "-"
            med_ms = b.total_ns / 1_000_000.0
            p90_ms = b.p90_total_ns / 1_000_000.0
            table_base.add_row(
                state.upper(),
                b.created_utc[:19] if b.created_utc else "-",
                f"{med_ms:.2f}",
                f"{p90_ms:.2f}",
                cv_val,
            )
            lines_md.append(
                f"| {state.upper()} | {b.created_utc[:19] if b.created_utc else '-'} | "
                f"{med_ms:.2f} | {p90_ms:.2f} | {cv_val} |"
            )
        else:
            table_base.add_row(state.upper(), "Not calibrated", "-", "-", "-")
            lines_md.append(f"| {state.upper()} | Not calibrated | - | - | - |")

    console.print(table_base)
    lines_md.append("")

    # 3. Offline Hold-Out Test Leaderboard
    console.print("\n[bold]2. Offline Hold-Out Model Performance[/bold]")
    test_metrics = card.get("test_metrics", {})
    reg_metrics = test_metrics.get("regression", {})

    table_reg = Table(box=box_style)
    table_reg.add_column("Model", style="bold")
    table_reg.add_column("Test MAE", justify="right")
    table_reg.add_column("RMSE", justify="right")
    table_reg.add_column("R2 Score", justify="right")
    table_reg.add_column("Tol Acc (<=0.25)", justify="right")
    table_reg.add_column("Beats Persistence", justify="center")

    lines_md.extend(
        [
            "## 2. Offline Hold-Out Model Performance",
            "",
            "| Model | Test MAE | RMSE | R2 Score | Tol Acc (<=0.25) | Beats Persistence |",
            "| :--- | :---: | :---: | :---: | :---: | :---: |",
        ]
    )

    pers_mae = reg_metrics.get("Persistence", {}).get("mae", 999.0)

    for m_name in ["XGBoost", "RandomForest", "Ridge", "Persistence"]:
        m_data = reg_metrics.get(m_name)
        if m_data:
            mae_val = m_data.get("mae", 0.0)
            rmse_val = m_data.get("rmse", 0.0)
            r2_val = m_data.get("r2", 0.0)
            tol_val = f"{m_data.get('tolerance_accuracy', 0.0) * 100:.1f}%"
            beats = "YES" if mae_val < pers_mae else ("-" if m_name == "Persistence" else "NO")
            beats_colored = f"[green]{beats}[/green]" if beats == "YES" else beats

            table_reg.add_row(
                m_name,
                f"{mae_val:.4f}",
                f"{rmse_val:.4f}",
                f"{r2_val:.4f}",
                tol_val,
                beats_colored,
            )
            lines_md.append(
                f"| {m_name} | {mae_val:.4f} | {rmse_val:.4f} | "
                f"{r2_val:.4f} | {tol_val} | {beats} |"
            )

    console.print(table_reg)
    lines_md.append("")

    # 4. Latest Live Accuracy Watch Verification
    watch_dir = get_data_dir() / "watch"
    watch_sessions = sorted(watch_dir.glob("watch-*.json")) if watch_dir.exists() else []

    console.print("\n[bold]3. Latest Live Accuracy Verification (deskflow watch)[/bold]")
    lines_md.extend(
        [
            "## 3. Latest Live Accuracy Verification (deskflow watch)",
            "",
        ]
    )

    if watch_sessions:
        latest_watch_file = watch_sessions[-1]
        try:
            with open(latest_watch_file, encoding="utf-8") as f:
                watch_data = json.load(f)
            leaderboard = watch_data.get("leaderboard", [])

            table_watch = Table(
                title=f"Session {latest_watch_file.stem} ({watch_data.get('duration_s', 0)}s)",
                box=box_style,
            )
            table_watch.add_column("Model", style="bold")
            table_watch.add_column("Resolved Pairs", justify="right")
            table_watch.add_column("Live MAE", justify="right")
            table_watch.add_column("Within 0.25 (%)", justify="right")
            table_watch.add_column("Vs Persistence", justify="right")

            p_state = watch_data.get("power_state", "unknown")
            dur = watch_data.get("duration_s", 0)
            lines_md.extend(
                [
                    f"**Session:** `{latest_watch_file.name}` ({dur}s on {p_state})",
                    "",
                    "| Model | Resolved Pairs | Live MAE | Within 0.25 (%) | Vs Persistence |",
                    "| :--- | :---: | :---: | :---: | :---: |",
                ]
            )

            for row in leaderboard:
                delta = row.get("delta_mae", 0.0)
                delta_str = "-" if row["model"] == "Persistence Baseline" else f"{delta:+.4f}"
                table_watch.add_row(
                    row["model"],
                    str(row["n"]),
                    f"{row['mae']:.4f}",
                    f"{row['tol_pct']:.1f}%",
                    delta_str,
                )
                lines_md.append(
                    f"| {row['model']} | {row['n']} | {row['mae']:.4f} | "
                    f"{row['tol_pct']:.1f}% | {delta_str} |"
                )

            console.print(table_watch)
        except Exception as e:
            console.print(f"[yellow]Could not parse {latest_watch_file}: {e}[/yellow]")
            lines_md.append(f"Error parsing watch session: {e}")
    else:
        msg = (
            "No live watch verification sessions found. "
            "Run 'deskflow watch' to verify live accuracy."
        )
        console.print(f"[dim]{msg}[/dim]")
        lines_md.append(f"*{msg}*")

    lines_md.append("")

    # Write output markdown report
    if out:
        out.write_text("\n".join(lines_md), encoding="utf-8")
        console.print(f"\n[green]Report written to {out}[/green]")
