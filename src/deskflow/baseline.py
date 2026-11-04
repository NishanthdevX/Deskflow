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

from deskflow import __version__
from deskflow.collector.models import ProbeResult
from deskflow.collector.probe import (
    CPU_PROBE_ITERATIONS,
    DISK_PROBE_BYTES,
    MEM_PROBE_BYTES,
    MEM_PROBE_STRIDE,
    PROBE_REPEATS,
    PROBE_WARMUP_MS,
    run_probe,
)
from deskflow.paths import get_config_dir
from deskflow.platform import get_adapter
from deskflow.platform.base import PlatformAdapter


class CalibrationError(Exception):
    """Raised when baseline calibration fails or is rejected."""


def get_default_config_dir() -> Path:
    """Return user configuration directory for deskflow."""
    return get_config_dir()


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
    cv: float = 0.0
    power_state: str = "ac"
    probe_repeats: int = PROBE_REPEATS
    cpu_iterations: int = CPU_PROBE_ITERATIONS
    mem_bytes: int = MEM_PROBE_BYTES
    mem_stride: int = MEM_PROBE_STRIDE
    disk_bytes: int = DISK_PROBE_BYTES
    probe_warmup_ms: int = PROBE_WARMUP_MS
    p50_total_ns: float = 0.0
    spread_fastest_half: float = 0.0
    probe_fingerprint: str = ""
    mean_cpu_percent: float = 0.0

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
            cv=float(data.get("cv", 0.0)),
            power_state=str(data.get("power_state", "ac")),
            probe_repeats=int(data.get("probe_repeats", PROBE_REPEATS)),
            cpu_iterations=int(data.get("cpu_iterations", CPU_PROBE_ITERATIONS)),
            mem_bytes=int(data.get("mem_bytes", MEM_PROBE_BYTES)),
            mem_stride=int(data.get("mem_stride", MEM_PROBE_STRIDE)),
            disk_bytes=int(data.get("disk_bytes", DISK_PROBE_BYTES)),
            probe_warmup_ms=int(data.get("probe_warmup_ms", 0)),
            p50_total_ns=float(data.get("p50_total_ns", data.get("total_ns", 0.0))),
            spread_fastest_half=float(data.get("spread_fastest_half", data.get("cv", 0.0))),
            probe_fingerprint=str(data.get("probe_fingerprint", get_probe_fingerprint())),
            mean_cpu_percent=float(data.get("mean_cpu_percent", 0.0)),
        )


def get_probe_fingerprint() -> str:
    """Return canonical parameter fingerprint string for the probe workload."""
    return (
        f"r{PROBE_REPEATS}_c{CPU_PROBE_ITERATIONS}_m{MEM_PROBE_BYTES}_"
        f"s{MEM_PROBE_STRIDE}_d{DISK_PROBE_BYTES}_w{PROBE_WARMUP_MS}"
    )


def get_baseline_age_days(baseline: Baseline) -> float:
    """Return age of baseline in fractional days."""
    try:
        dt = datetime.fromisoformat(baseline.created_utc)
        return max(0.0, (datetime.now(UTC) - dt).total_seconds() / 86400.0)
    except Exception:
        return 0.0


def get_baseline_age_str(baseline: Baseline) -> str:
    """Format baseline age as a human-readable string (e.g. 2h ago, 3d ago)."""
    try:
        dt = datetime.fromisoformat(baseline.created_utc)
        delta_s = max(0.0, (datetime.now(UTC) - dt).total_seconds())
        if delta_s < 3600:
            return f"{int(delta_s // 60)}m ago"
        elif delta_s < 86400:
            return f"{int(delta_s // 3600)}h ago"
        else:
            days = int(delta_s // 86400)
            return f"{days}d ago"
    except Exception:
        return "unknown"


def check_baseline_drift(
    baseline: Baseline,
    probe_fn: Callable[[], ProbeResult] = run_probe,
    num_probes: int = 3,
) -> tuple[bool, float, float]:
    """Compare initial probes with baseline. Returns (is_drifting, ratio, mean_cpu)."""
    import psutil

    psutil.cpu_percent(interval=None)
    probes = []
    cpu_readings = []
    for _ in range(num_probes):
        time.sleep(0.2)
        probes.append(probe_fn())
        cpu_readings.append(psutil.cpu_percent(interval=None))
    median_ns = statistics.median([p.total_ns for p in probes])
    mean_cpu = statistics.mean(cpu_readings) if cpu_readings else 0.0
    ratio = (median_ns / baseline.total_ns) if baseline.total_ns > 0 else 1.0
    is_drifting = bool(mean_cpu < 15.0 and (ratio < 0.75 or ratio > 1.3))
    return is_drifting, ratio, mean_cpu


def get_current_power_state(adapter: PlatformAdapter | None = None) -> str:
    """Determine current device power state ('ac' or 'battery')."""
    ad = adapter or get_adapter()
    bat = ad.battery()
    if bat is None or bat.plugged_in:
        return "ac"
    return "battery"


def save_baseline(
    baseline: Baseline,
    power_state: str | None = None,
    path: Path | None = None,
    config_dir: Path | None = None,
) -> Path:
    """Save baseline dataclass merged by power state to JSON file atomically."""
    target_path = path or ((config_dir or get_default_config_dir()) / "baseline.json")
    target_path.parent.mkdir(parents=True, exist_ok=True)
    state = power_state or baseline.power_state

    current_data: dict[str, Any] = {"ac": None, "battery": None}
    if target_path.exists():
        try:
            with open(target_path, encoding="utf-8") as f:
                loaded = json.load(f)
                if isinstance(loaded, dict):
                    if "ac" in loaded or "battery" in loaded:
                        current_data["ac"] = loaded.get("ac")
                        current_data["battery"] = loaded.get("battery")
                    elif "machine_id" in loaded:
                        current_data["ac"] = loaded
        except Exception:
            pass

    current_data[state] = baseline.to_dict()

    temp_path = target_path.with_suffix(".tmp")
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(current_data, f, indent=2)
    temp_path.replace(target_path)
    return target_path


def load_all_baselines(
    path: Path | None = None,
    config_dir: Path | None = None,
) -> dict[str, Baseline | None]:
    """Load both power-state baselines from baseline.json."""
    target_path = path or ((config_dir or get_default_config_dir()) / "baseline.json")
    if not target_path.exists():
        raise FileNotFoundError(f"Baseline file not found at {target_path}")

    with open(target_path, encoding="utf-8") as f:
        data = json.load(f)

    result: dict[str, Baseline | None] = {"ac": None, "battery": None}
    if isinstance(data, dict):
        if "ac" in data or "battery" in data:
            if data.get("ac"):
                result["ac"] = Baseline.from_dict(data["ac"])
            if data.get("battery"):
                result["battery"] = Baseline.from_dict(data["battery"])
        elif "machine_id" in data:
            result["ac"] = Baseline.from_dict(data)
    return result


def load_baseline(
    power_state: str | None = None,
    path: Path | None = None,
    config_dir: Path | None = None,
) -> Baseline:
    """Load baseline for a specific power state ('ac' or 'battery')."""
    target_path = path or ((config_dir or get_default_config_dir()) / "baseline.json")
    if not target_path.exists():
        raise FileNotFoundError(f"Baseline file not found at {target_path}")

    state = power_state or get_current_power_state()
    all_baselines = load_all_baselines(path=path, config_dir=config_dir)

    target_baseline = all_baselines.get(state)
    if target_baseline is None and power_state is None:
        # Fallback to 'ac' or first available baseline when power_state not explicitly requested
        target_baseline = all_baselines.get("ac") or next(
            (b for b in all_baselines.values() if b is not None), None
        )
    if target_baseline is None:
        raise FileNotFoundError(
            f"Baseline for power state '{state}' not found in {target_path}. "
            f"Run 'deskflow baseline calibrate' while on {state} power."
        )
    if target_baseline.probe_warmup_ms != PROBE_WARMUP_MS:
        raise CalibrationError(
            "Baseline probe settings do not match this Deskflow version. "
            "Run 'deskflow baseline calibrate' again."
        )
    return target_baseline


def calibrate_baseline(
    duration_s: float = 90.0,
    interval_s: float = 2.0,
    warmup_samples: int = 3,
    probe_fn: Callable[[], ProbeResult] = run_probe,
    config_dir: Path | None = None,
    out_path: Path | None = None,
    progress_callback: Callable[[int, int, float], None] | None = None,
    min_duration_s: float = 0.0,
) -> Baseline:
    """Run baseline calibration loop on idle system.

    Discards the first `warmup_samples` probes to avoid cache/warmup effects.
    Computes medians of the fastest 50 percent of probe totals for robust baseline
    estimation. Refuses if mean CPU utilisation exceeds 15% or spread of fastest
    half exceeds 0.25.
    """
    if min_duration_s > 0.0 and duration_s < min_duration_s:
        raise CalibrationError(
            f"Calibration duration {duration_s:.0f}s is below minimum "
            f"required {min_duration_s:.0f}s."
        )

    import psutil

    psutil.cpu_percent(interval=None)

    total_planned = int(max(duration_s // interval_s, warmup_samples + 3))
    probes: list[ProbeResult] = []
    cpu_samples: list[float] = []

    start_mono = time.monotonic()
    current_probe_idx = 0

    while len(probes) < total_planned:
        target_time = start_mono + (current_probe_idx * interval_s)
        sleep_dur = target_time - time.monotonic()
        if sleep_dur > 0:
            time.sleep(sleep_dur)

        probe = probe_fn()
        probes.append(probe)
        cpu_samples.append(psutil.cpu_percent(interval=None))
        current_probe_idx += 1

        if progress_callback is not None:
            progress_callback(current_probe_idx, total_planned, probe.total_ns / 1_000_000.0)

    # Discard warmup probes
    usable = probes[warmup_samples:]
    usable_cpu = cpu_samples[warmup_samples:]
    if len(usable) < 2:
        raise CalibrationError(
            f"Insufficient usable samples ({len(usable)}) for baseline calibration."
        )

    mean_cpu = statistics.mean(usable_cpu) if usable_cpu else 0.0
    if mean_cpu > 15.0:
        raise CalibrationError(
            f"Mean CPU utilisation was {mean_cpu:.1f}%, exceeding 15% threshold. "
            "System must be idle during calibration."
        )

    # Baseline statistic: median of the fastest 50 percent of probe totals
    sorted_by_total = sorted(usable, key=lambda p: p.total_ns)
    half_count = max(1, len(sorted_by_total) // 2)
    fastest_half = sorted_by_total[:half_count]

    fastest_totals = [p.total_ns for p in fastest_half]
    all_totals = [p.total_ns for p in usable]

    import numpy as np

    stat_total = float(np.median(fastest_totals))
    stat_cpu = float(np.median([p.cpu_ns for p in fastest_half]))
    stat_mem = float(np.median([p.mem_ns for p in fastest_half]))
    stat_disk = float(np.median([p.disk_ns for p in fastest_half]))

    p10_half = float(np.percentile(fastest_totals, 10))
    p90_half = float(np.percentile(fastest_totals, 90))
    spread_fastest_half = (p90_half - p10_half) / stat_total if stat_total > 0 else 0.0

    p10_total_all = float(np.percentile(all_totals, 10))
    p50_total_all = float(np.median(all_totals))
    p90_total_all = float(np.percentile(all_totals, 90))
    std_total = statistics.stdev(all_totals) if len(all_totals) > 1 else 0.0
    mean_total = statistics.mean(all_totals) if all_totals else 1.0
    cv = std_total / mean_total

    if cv > 0.5:
        raise CalibrationError(
            f"Coefficient of variation of probe latency ({cv:.3f}) exceeded 0.5 threshold. "
            "Machine was not sufficiently idle during calibration."
        )

    if spread_fastest_half > 0.25:
        raise CalibrationError(
            f"Spread of the fastest half ({spread_fastest_half:.3f}) exceeded 0.25 threshold. "
            "Machine performance was not sufficiently stable."
        )

    machine_id = get_or_create_machine_id(config_dir=config_dir)
    adapter = get_adapter()
    power_state = get_current_power_state(adapter)

    baseline = Baseline(
        machine_id=machine_id,
        created_utc=datetime.now(UTC).isoformat(),
        platform=adapter.name,
        cpu_ns=stat_cpu,
        mem_ns=stat_mem,
        disk_ns=stat_disk,
        total_ns=stat_total,
        p10_total_ns=p10_total_all,
        p90_total_ns=p90_total_all,
        sample_count=len(usable),
        deskflow_version=__version__,
        cv=float(cv),
        power_state=power_state,
        probe_repeats=PROBE_REPEATS,
        cpu_iterations=CPU_PROBE_ITERATIONS,
        mem_bytes=MEM_PROBE_BYTES,
        mem_stride=MEM_PROBE_STRIDE,
        disk_bytes=DISK_PROBE_BYTES,
        probe_warmup_ms=PROBE_WARMUP_MS,
        p50_total_ns=p50_total_all,
        spread_fastest_half=float(spread_fastest_half),
        probe_fingerprint=get_probe_fingerprint(),
        mean_cpu_percent=float(mean_cpu),
    )

    save_baseline(baseline, power_state=power_state, path=out_path, config_dir=config_dir)
    return baseline
