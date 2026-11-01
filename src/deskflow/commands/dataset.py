"""CLI commands for building, inspecting, validating, and merging ML datasets."""

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from deskflow.features.builder import (
    build_dataset,
    describe_dataset,
    merge_datasets,
    validate_dataset,
)

app = typer.Typer(help="Build, validate, describe, and merge feature datasets.")
console = Console()


@app.command("build")
def build_command(
    runs: list[Path] = typer.Option(
        ...,
        "--runs",
        "-r",
        help="One or more directories or JSONL files containing run records.",
    ),
    baseline: Path | None = typer.Option(
        None,
        "--baseline",
        "-b",
        help="Optional custom path to baseline.json.",
    ),
    horizon: float = typer.Option(
        30.0,
        "--horizon",
        "-h",
        help="Prediction target horizon in seconds (default: 30s).",
    ),
    out: Path = typer.Option(
        Path("dataset.parquet"),
        "--out",
        "-o",
        help="Output parquet file path (default: dataset.parquet).",
    ),
    keep_transition: bool = typer.Option(
        False,
        "--keep-transition",
        help="Retain injection warm-up and cool-down transition rows.",
    ),
) -> None:
    """Build a labeled parquet feature dataset from raw run records."""
    console.print(f"Building dataset from {len(runs)} path(s) at horizon={horizon:.0f}s...")

    try:
        parquet_path, manifest_path = build_dataset(
            run_paths=runs,
            baseline_path=baseline,
            horizon_s=horizon,
            out_path=out,
            keep_transition=keep_transition,
        )
    except Exception as e:
        console.print(f"[red]Build failed:[/red] {e}")
        raise typer.Exit(code=1) from e

    with open(manifest_path, encoding="utf-8") as f:
        meta = json.load(f)

    console.print("[green]Dataset built successfully.[/green]")
    console.print(f"Parquet file:  {parquet_path}")
    console.print(f"Manifest file: {manifest_path}")
    console.print(
        f"Rows: {meta['row_count']} (Train: {meta['train_rows']}, Test: {meta['test_rows']}) | "
        f"Strategy: {meta['split_strategy']}"
    )


@app.command("describe")
def describe_command(
    file: Path = typer.Argument(
        ...,
        help="Path to parquet dataset file to describe.",
    ),
) -> None:
    """Print statistical summaries, label distributions, and split sizes of a dataset."""
    if not file.exists():
        console.print(f"[red]Error:[/red] File not found: {file}")
        raise typer.Exit(code=1)

    try:
        info = describe_dataset(file)
    except Exception as e:
        console.print(f"[red]Error reading dataset:[/red] {e}")
        raise typer.Exit(code=1) from e

    console.print(f"Dataset Summary: {file.name}")
    console.print(f"Total Rows: {info['total_rows']}")
    console.print(
        f"Splits: Train {info['train_rows']} rows, Test {info['test_rows']} rows "
        f"({info['test_share'] * 100:.1f}%)"
    )
    console.print(f"Time Span: {info['time_span']['start']} to {info['time_span']['end']}")
    console.print(f"Sessions ({info['session_count']}): {', '.join(info['sessions'])}")

    # Distribution Table
    table = Table(title="Row Distribution by Label (issue_now)", show_header=True)
    table.add_column("Issue Label", style="bold")
    table.add_column("Rows", justify="right")
    table.add_column("Share", justify="right")

    for k, v in info["rows_per_issue"].items():
        share = (v / info["total_rows"]) * 100 if info["total_rows"] > 0 else 0
        table.add_row(str(k), str(v), f"{share:.1f}%")
    console.print(table)

    # Scenarios Table
    scen_table = Table(title="Row Distribution by Scenario", show_header=True)
    scen_table.add_column("Scenario", style="bold")
    scen_table.add_column("Rows", justify="right")
    for k, v in info["rows_per_scenario"].items():
        scen_table.add_row(str(k), str(v))
    console.print(scen_table)

    # Target Statistics
    target_stats = info["target_total_ratio"]
    console.print(
        f"Target Total Ratio: min={target_stats['min']:.2f}, median={target_stats['median']:.2f}, "
        f"p95={target_stats['p95']:.2f}, max={target_stats['max']:.2f}"
    )
    console.print(f"Share of target_slow (ratio >= 1.5): {info['target_slow_share'] * 100:.1f}%")

    if info["missing_features"]:
        console.print("[yellow]Features with missing values:[/yellow]")
        for f, frac in info["missing_features"].items():
            console.print(f"  {f}: {frac * 100:.1f}% missing")
    else:
        console.print("Features with missing values: None")


@app.command("validate")
def validate_command(
    file: Path = typer.Argument(
        ...,
        help="Path to parquet dataset file to validate.",
    ),
) -> None:
    """Validate dataset columns against registry, check target completeness, and test uniqueness."""
    if not file.exists():
        console.print(f"[red]Error:[/red] File not found: {file}")
        raise typer.Exit(code=1)

    errors = validate_dataset(file)
    if errors:
        console.print(f"[red]Dataset validation failed ({len(errors)} error(s)):[/red]")
        for err in errors:
            console.print(f"  - {err}")
        raise typer.Exit(code=1)

    console.print(
        "[green]Dataset validation passed.[/green] "
        "All registry columns, targets, and constraints verified."
    )


@app.command("merge")
def merge_command(
    files: list[Path] = typer.Argument(
        ...,
        help="Two or more parquet dataset files to merge.",
    ),
    out: Path = typer.Option(
        ...,
        "--out",
        "-o",
        help="Output merged parquet file path.",
    ),
) -> None:
    """Merge datasets from different machines or sessions into a single unified dataset."""
    console.print(f"Merging {len(files)} dataset files into {out}...")

    try:
        out_parquet, out_manifest = merge_datasets(dataset_files=files, out_path=out)
    except Exception as e:
        console.print(f"[red]Merge failed:[/red] {e}")
        raise typer.Exit(code=1) from e

    with open(out_manifest, encoding="utf-8") as f:
        meta = json.load(f)

    console.print("[green]Merge complete.[/green]")
    console.print(f"Merged Parquet:  {out_parquet}")
    console.print(f"Merged Manifest: {out_manifest}")
    console.print(
        f"Total rows: {meta['row_count']} (Train: {meta['train_rows']}, Test: {meta['test_rows']})"
    )
