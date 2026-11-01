"""Machine identity and baseline calibration for performance normalization."""

import json
import statistics
import time
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import platformdirs

from deskflow import __version__
from deskflow.collector.models import ProbeResult
from deskflow.collector.probe import run_probe
from deskflow.platform import get_adapter


class CalibrationError(Exception):
    """Raised when baseline calibration fails or is rejected."""


def get_default_config_dir() -> Path:
    """Return user configuration directory for deskflow."""
    return Path(platformdirs.user_config_dir("deskflow", ensure_exists=True))


def get_or_create_machine_id(config_dir: Path | None = None) -> str:
    """Retrieve existing random machine UUID or generate and persist a new one.

    Never derives machine identity from hostname, username, or hardware properties.
    """
    cdir = config_dir or get_default_config_dir()
    machine_path = cdir / "machine.json"
    if machine_path.exists():
        try:
            with open(machine_path, encoding="utf-8") as f:
                data = json.load(f)
                mid = data.get("machine_id")
                if isinstance(mid, str) and mid:
                    return mid
        except (json.JSONDecodeError, OSError):
            pass

    new_id = str(uuid.uuid4())
    cdir.mkdir(parents=True, exist_ok=True)
    temp_path = machine_path.with_suffix(".tmp")
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "machine_id": new_id,
                "created_utc": datetime.now(UTC).isoformat(),
            },
            f,
            indent=2,
        )
    temp_path.replace(machine_path)
    return new_id


@dataclass(frozen=True)
class Baseline:
    """System baseline latency medians and bounds recorded during idle state."""

    machine_id: str
    created_utc: str
    platform: str
    cpu_ns: float
    mem_ns: float
    disk_ns: float
    total_ns: float
    p10_total_ns: float
    p90_total_ns: float
    sample_count: int
    deskflow_version: str

    def to_dict(self) -> dict[str, Any]:
        """Convert baseline record to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Baseline":
        """Instantiate baseline from dictionary."""
        return cls(
            machine_id=str(data["machine_id"]),
            created_utc=str(data["created_utc"]),
            platform=str(data["platform"]),
            cpu_ns=float(data["cpu_ns"]),
            mem_ns=float(data["mem_ns"]),
            disk_ns=float(data["disk_ns"]),
            total_ns=float(data["total_ns"]),
            p10_total_ns=float(data["p10_total_ns"]),
            p90_total_ns=float(data["p90_total_ns"]),
            sample_count=int(data["sample_count"]),
            deskflow_version=str(data["deskflow_version"]),
        )


def save_baseline(
    baseline: Baseline,
    path: Path | None = None,
    config_dir: Path | None = None,
) -> Path:
    """Save baseline dataclass to a JSON file atomically."""
    target_path = path or ((config_dir or get_default_config_dir()) / "baseline.json")
    target_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = target_path.with_suffix(".tmp")
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(baseline.to_dict(), f, indent=2)
    temp_path.replace(target_path)
    return target_path


def load_baseline(
    path: Path | None = None,
    config_dir: Path | None = None,
) -> Baseline:
    """Load baseline from file. Raises FileNotFoundError if missing."""
    target_path = path or ((config_dir or get_default_config_dir()) / "baseline.json")
    if not target_path.exists():
        raise FileNotFoundError(f"Baseline file not found at {target_path}")
    with open(target_path, encoding="utf-8") as f:
        data = json.load(f)
    return Baseline.from_dict(data)


def calibrate_baseline(
    duration_s: float = 120.0,
    interval_s: float = 2.0,
    warmup_samples: int = 3,
    probe_fn: Callable[[], ProbeResult] = run_probe,
    config_dir: Path | None = None,
    out_path: Path | None = None,
    progress_callback: Callable[[int, int, float], None] | None = None,
) -> Baseline:
    """Run baseline calibration loop on idle system.

    Discards the first `warmup_samples` probes to avoid cache/warmup effects.
    Computes medians and p10/p90 percentiles. Refuses to save if coefficient of
    variation of total_ns exceeds 0.5.
    """
    total_planned = int(max(duration_s // interval_s, warmup_samples + 3))
    probes: list[ProbeResult] = []

    start_mono = time.monotonic()
    current_probe_idx = 0

    while len(probes) < total_planned:
        target_time = start_mono + (current_probe_idx * interval_s)
        sleep_dur = target_time - time.monotonic()
        if sleep_dur > 0:
            time.sleep(sleep_dur)

        probe = probe_fn()
        probes.append(probe)
        current_probe_idx += 1

        if progress_callback is not None:
            progress_callback(current_probe_idx, total_planned, probe.total_ns / 1_000_000.0)

    # Discard warmup probes
    usable = probes[warmup_samples:]
    if len(usable) < 2:
        raise CalibrationError(
            f"Insufficient usable samples ({len(usable)}) for baseline calibration."
        )

    totals = [p.total_ns for p in usable]
    mean_total = statistics.mean(totals)
    std_total = statistics.stdev(totals) if len(totals) > 1 else 0.0
    cv = std_total / mean_total if mean_total > 0 else 0.0

    if cv > 0.5:
        raise CalibrationError(
            f"Coefficient of variation of probe latency ({cv:.3f}) exceeded 0.5 threshold. "
            "Machine was not sufficiently idle during calibration."
        )

    cpu_list = [p.cpu_ns for p in usable]
    mem_list = [p.mem_ns for p in usable]
    disk_list = [p.disk_ns for p in usable]

    machine_id = get_or_create_machine_id(config_dir=config_dir)
    adapter = get_adapter()

    baseline = Baseline(
        machine_id=machine_id,
        created_utc=datetime.now(UTC).isoformat(),
        platform=adapter.name,
        cpu_ns=float(np.median(cpu_list)),
        mem_ns=float(np.median(mem_list)),
        disk_ns=float(np.median(disk_list)),
        total_ns=float(np.median(totals)),
        p10_total_ns=float(np.percentile(totals, 10)),
        p90_total_ns=float(np.percentile(totals, 90)),
        sample_count=len(usable),
        deskflow_version=__version__,
    )

    save_baseline(baseline, path=out_path, config_dir=config_dir)
    return baseline
