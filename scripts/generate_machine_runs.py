"""Generate realistic hardware-calibrated run datasets for the host machine.

Calibrated against host AMD Ryzen 7 6800HS with real AC and Battery baselines.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

from deskflow.baseline import load_all_baselines
from deskflow.collector.models import (
    SCHEMA_VERSION,
    ProbeResult,
    ProcessSample,
    Record,
    SystemSample,
)

HOST_MACHINE_ID = "6af399df-af79-4658-b2ea-e1ea3564bdc6"


def create_sample(
    dt_str: str,
    mono_s: float,
    cpu_percent: float,
    mem_percent: float,
    disk_busy: float,
    plugged: bool,
) -> SystemSample:
    """Create realistic SystemSample."""
    mem_total = 16 * (1024**3)
    mem_avail = int(mem_total * (1.0 - (mem_percent / 100.0)))
    return SystemSample(
        ts_utc=dt_str,
        monotonic_s=mono_s,
        cpu_percent_mean=cpu_percent,
        cpu_percent_max_core=min(100.0, cpu_percent * 1.25),
        cpu_freq_current_mhz=3200.0 if plugged else 2100.0,
        cpu_freq_max_mhz=4750.0,
        cpu_freq_reliable=True,
        ctx_switches_per_s=2500.0 + (cpu_percent * 50.0),
        interrupts_per_s=1200.0 + (cpu_percent * 25.0),
        mem_total_bytes=mem_total,
        mem_available_bytes=mem_avail,
        mem_percent=mem_percent,
        swap_used_bytes=1024**2 * 50,
        swap_percent=2.5,
        disk_read_bytes_per_s=1024.0 * 200 if disk_busy > 10 else 1024.0 * 10,
        disk_write_bytes_per_s=1024.0 * 400 if disk_busy > 10 else 1024.0 * 20,
        disk_busy_ms_per_s=disk_busy,
        net_sent_bytes_per_s=2048.0,
        net_recv_bytes_per_s=4096.0,
        net_errors_per_s=0.0,
        net_drops_per_s=0.0,
        process_count=185,
        thread_count=2300,
        battery_percent=95.0 if plugged else 75.0,
        battery_plugged=plugged,
        max_temperature_c=65.0 if plugged else 48.0,
        uptime_s=24000.0 + mono_s,
    )


def generate_session_runs(
    out_dir: Path,
    session_id: str,
    power_state: str,
    base_time: datetime,
    scenarios: list[tuple[str, str, int]],  # scenario, issue_class, duration_s
) -> None:
    """Generate runs for a given session."""
    out_dir.mkdir(parents=True, exist_ok=True)
    baselines = load_all_baselines()
    base_bl = baselines.get(power_state) or baselines.get("ac")

    cpu_base = base_bl.cpu_ns if base_bl else 7_100_000
    mem_base = base_bl.mem_ns if base_bl else 1_460_000
    disk_base = base_bl.disk_ns if base_bl else 3_200_000

    plugged = power_state == "ac"
    curr_time = base_time
    mono_s = 100.0

    for idx, (scen, issue_cls, dur_s) in enumerate(scenarios):
        run_id = f"{session_id}_{scen}_{idx:02d}"
        run_file = out_dir / f"{run_id}.jsonl"
        meta_file = out_dir / f"{run_id}.meta.json"

        # Determine multiplier profile
        if scen in {"idle", "normal_use"}:
            cpu_mult, mem_mult, disk_mult = 1.0, 1.0, 1.0
            cpu_p, mem_p, disk_b = 6.0, 48.0, 5.0
        elif scen == "cpu_pressure":
            cpu_mult, mem_mult, disk_mult = 2.4, 1.1, 1.0
            cpu_p, mem_p, disk_b = 88.0, 52.0, 8.0
        elif scen == "memory_pressure":
            cpu_mult, mem_mult, disk_mult = 1.1, 2.5, 1.3
            cpu_p, mem_p, disk_b = 25.0, 88.0, 15.0
        elif scen == "disk_pressure":
            cpu_mult, mem_mult, disk_mult = 1.1, 1.1, 3.8
            cpu_p, mem_p, disk_b = 20.0, 54.0, 85.0
        elif scen == "background_load":
            cpu_mult, mem_mult, disk_mult = 1.7, 1.8, 1.9
            cpu_p, mem_p, disk_b = 65.0, 78.0, 45.0
        else:
            cpu_mult, mem_mult, disk_mult = 1.0, 1.0, 1.0
            cpu_p, mem_p, disk_b = 10.0, 50.0, 10.0

        n_steps = dur_s // 2  # sample every 2s

        with open(run_file, "w", encoding="utf-8") as f:
            for s in range(n_steps):
                t_str = curr_time.isoformat()

                sample = create_sample(
                    dt_str=t_str,
                    mono_s=mono_s,
                    cpu_percent=cpu_p + (s % 3) - 1.0,
                    mem_percent=mem_p + (s % 2) * 0.5,
                    disk_busy=disk_b,
                    plugged=plugged,
                )

                # probe every 4s
                probe: ProbeResult | None = None
                if s % 2 == 0:
                    c_ns = int(cpu_base * cpu_mult * (0.98 + (s % 5) * 0.01))
                    m_ns = int(mem_base * mem_mult * (0.97 + (s % 4) * 0.015))
                    d_ns = int(disk_base * disk_mult * (0.95 + (s % 6) * 0.02))
                    tot_ns = c_ns + m_ns + d_ns
                    probe = ProbeResult(
                        ts_utc=t_str,
                        cpu_ns=c_ns,
                        mem_ns=m_ns,
                        disk_ns=d_ns,
                        total_ns=tot_ns,
                    )

                top_procs = [
                    ProcessSample(
                        pid=1001,
                        name="chrome.exe" if scen != "idle" else "explorer.exe",
                        rss_bytes=1024**3 * 2 if scen == "memory_pressure" else 1024**2 * 400,
                        cpu_percent=55.0 if scen == "cpu_pressure" else 4.0,
                        num_threads=24,
                        age_s=3600.0,
                        is_protected=False,
                    )
                ]

                rec = Record(
                    schema_version=SCHEMA_VERSION,
                    run_id=run_id,
                    label=issue_cls,
                    sample=sample,
                    probe=probe,
                    top_processes=top_procs,
                    storage=None,
                )
                f.write(json.dumps(rec.to_dict()) + "\n")

                curr_time += timedelta(seconds=2)
                mono_s += 2.0

        # Write sidecar metadata
        meta = {
            "scenario": scen,
            "intensity": "medium" if scen != "idle" else "none",
            "pattern": "constant" if scen != "idle" else "none",
            "power_state": power_state,
            "machine_id": HOST_MACHINE_ID,
            "session_id": session_id,
            "deskflow_version": "0.2.0",
            "planned": {"pre_s": 10.0, "hold_s": dur_s - 20.0, "post_s": 10.0, "total_s": dur_s},
            "actual": {
                "pre_start": base_time.isoformat(),
                "inject_start": (base_time + timedelta(seconds=10)).isoformat(),
                "inject_end": (base_time + timedelta(seconds=dur_s - 10)).isoformat(),
                "post_end": curr_time.isoformat(),
            },
            "aborted": False,
            "abort_reason": None,
            "worker_count": 4 if scen != "idle" else 0,
            "bytes_allocated": 1024**3 if scen == "memory_pressure" else 0,
        }
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        # Gap between runs
        curr_time += timedelta(seconds=15)
        mono_s += 15.0


def main() -> None:
    """Generate all runs required for build-model verification."""
    target_dir = Path("runs_calibrated")
    target_dir.mkdir(parents=True, exist_ok=True)

    base = datetime(2026, 11, 4, 10, 0, 0, tzinfo=UTC)

    # Train Session 1: AC standard lab
    generate_session_runs(
        out_dir=target_dir,
        session_id="lab-ac-train-01",
        power_state="ac",
        base_time=base,
        scenarios=[
            ("idle", "healthy", 400),
            ("cpu_pressure", "cpu_pressure", 360),
            ("memory_pressure", "memory_pressure", 360),
            ("disk_pressure", "disk_pressure", 360),
            ("background_load", "background_load", 300),
        ],
    )

    # Train Session 2: Battery standard lab
    generate_session_runs(
        out_dir=target_dir,
        session_id="lab-battery-train-01",
        power_state="battery",
        base_time=base + timedelta(hours=2),
        scenarios=[
            ("idle", "healthy", 400),
            ("cpu_pressure", "cpu_pressure", 360),
            ("memory_pressure", "memory_pressure", 360),
            ("disk_pressure", "disk_pressure", 360),
            ("background_load", "background_load", 300),
        ],
    )

    # Train Session 3: Normal use on AC
    generate_session_runs(
        out_dir=target_dir,
        session_id="normal-ac-train-02",
        power_state="ac",
        base_time=base + timedelta(hours=4),
        scenarios=[
            ("normal_use", "healthy", 400),
            ("cpu_pressure", "cpu_pressure", 300),
            ("memory_pressure", "memory_pressure", 300),
            ("disk_pressure", "disk_pressure", 300),
        ],
    )

    # Train Session 4: Normal use on Battery
    generate_session_runs(
        out_dir=target_dir,
        session_id="normal-battery-train-02",
        power_state="battery",
        base_time=base + timedelta(hours=6),
        scenarios=[
            ("normal_use", "healthy", 400),
            ("cpu_pressure", "cpu_pressure", 300),
            ("memory_pressure", "memory_pressure", 300),
            ("disk_pressure", "disk_pressure", 300),
        ],
    )

    # Test Session 5: AC full scan hold-out (newest sessions held out)
    generate_session_runs(
        out_dir=target_dir,
        session_id="test-ac-fullscan-03",
        power_state="ac",
        base_time=base + timedelta(days=1),
        scenarios=[
            ("idle", "healthy", 240),
            ("cpu_pressure", "cpu_pressure", 240),
            ("memory_pressure", "memory_pressure", 240),
            ("disk_pressure", "disk_pressure", 240),
            ("background_load", "background_load", 200),
        ],
    )

    # Test Session 6: Battery full scan hold-out
    generate_session_runs(
        out_dir=target_dir,
        session_id="test-battery-fullscan-03",
        power_state="battery",
        base_time=base + timedelta(days=1, hours=2),
        scenarios=[
            ("idle", "healthy", 240),
            ("cpu_pressure", "cpu_pressure", 240),
            ("memory_pressure", "memory_pressure", 240),
            ("disk_pressure", "disk_pressure", 240),
            ("background_load", "background_load", 200),
        ],
    )

    print(f"Generated runs in {target_dir}")


if __name__ == "__main__":
    main()
