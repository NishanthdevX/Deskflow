"""Implementation of the collect CLI command."""

import time
from pathlib import Path

import typer
from rich.console import Console

from deskflow.collector import (
    SCHEMA_VERSION,
    ProbeResult,
    Record,
    RunRecorder,
    Sampler,
    StorageScanner,
    run_probe,
)
from deskflow.platform import get_adapter

app = typer.Typer(help="Collect continuous telemetry and probe data into a JSONL run file.")


@app.callback(invoke_without_command=True)
def collect_command(
    duration: float = typer.Option(
        600.0,
        "--duration",
        help="Total collection duration in seconds.",
    ),
    interval: float = typer.Option(
        2.0,
        "--interval",
        help="Telemetry sampling interval in seconds.",
    ),
    probe_interval: float = typer.Option(
        4.0,
        "--probe-interval",
        help="Performance probe interval in seconds.",
    ),
    process_interval: float = typer.Option(
        10.0,
        "--process-interval",
        help="Process table sampling interval in seconds.",
    ),
    storage: bool = typer.Option(
        False,
        "--storage",
        help="Enable background storage cache scanning.",
    ),
    label: str = typer.Option(
        "unlabeled",
        "--label",
        help="Dataset label tag for this collection run.",
    ),
    out: Path | None = typer.Option(
        None,
        "--out",
        help="Optional custom output path for JSONL file.",
    ),
) -> None:
    """Collect continuous telemetry and probe records into a JSONL run file."""
    console = Console()

    # Validate intervals
    if probe_interval < interval:
        console.print(
            f"[red]Error:[/red] --probe-interval ({probe_interval}) must be >= "
            f"--interval ({interval})."
        )
        raise typer.Exit(code=1)

    ratio = probe_interval / interval
    if abs(ratio - round(ratio)) > 1e-6:
        console.print(
            f"[yellow]Warning:[/yellow] --probe-interval ({probe_interval}) is not an exact "
            f"multiple of --interval ({interval}). Probes will run at the nearest tick."
        )

    adapter = get_adapter()
    sampler = Sampler(
        adapter=adapter,
        interval_s=interval,
        process_interval_s=process_interval,
    )
    storage_scanner: StorageScanner | None = None
    if storage:
        storage_scanner = StorageScanner(adapter=adapter)
        storage_scanner.start_background_scan()

    recorder = RunRecorder(out_path=out)
    console.print(
        f"Starting collection (duration: {duration}s, interval: {interval}s, "
        f"probe: every {probe_interval}s, label: '{label}')."
    )
    console.print(f"Output target: {recorder.path}")

    start_mono = time.monotonic()
    next_sample_mono = start_mono
    next_probe_mono = start_mono
    last_status_mono = start_mono
    last_probe_str = "n/a"

    try:
        while True:
            now_mono = time.monotonic()
            elapsed = now_mono - start_mono
            if elapsed >= duration:
                break

            # 1. Take telemetry sample
            sample, processes = sampler.sample()

            # 2. Check if probe is due (pausing storage scanner during execution)
            probe: ProbeResult | None = None
            if now_mono >= next_probe_mono:
                if storage_scanner:
                    with storage_scanner.pause_during_probe():
                        probe = run_probe()
                else:
                    probe = run_probe()
                last_probe_str = f"{probe.total_ns / 1_000_000:.2f} ms"
                # Advance next probe target
                while next_probe_mono <= now_mono:
                    next_probe_mono += probe_interval

            # 3. Storage snapshot from background scan (if enabled)
            storage_report = storage_scanner.get_cached_report() if storage_scanner else None

            # 4. Form and persist record
            record = Record(
                schema_version=SCHEMA_VERSION,
                run_id=recorder.run_id,
                label=label,
                sample=sample,
                probe=probe,
                top_processes=processes[:10],
                storage=storage_report,
            )
            recorder.write(record)

            # 5. Periodic status output every 10 seconds
            if now_mono - last_status_mono >= 10.0:
                ts_str = time.strftime("%H:%M:%S")
                console.print(
                    f"[{ts_str}] samples: {recorder.records_written}, last probe: {last_probe_str}"
                )
                last_status_mono = now_mono

            # 6. Sleep with drift compensation
            next_sample_mono += interval
            sleep_time = next_sample_mono - time.monotonic()
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                # If falling behind, reset schedule to current monotonic
                next_sample_mono = time.monotonic()

    except KeyboardInterrupt:
        console.print("\nCollection stopped by user.")
    finally:
        if storage_scanner:
            storage_scanner.stop()
        recorder.close()
        recorder.close()
        console.print(f"Finished. Wrote {recorder.records_written} records to {recorder.path}")
