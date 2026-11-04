"""Generate demo runs to verify dataset build with per-session baselines and mid-run drop."""

import json
from datetime import UTC, datetime
from pathlib import Path

from deskflow.collector.models import ProbeResult, Record, SystemSample


def main() -> None:
    demo_dir = Path("demo_runs")
    demo_dir.mkdir(exist_ok=True)
    base_epoch = 1700000000.0

    # 1. Run with 25 idle probes in session-A (should use session_idle)
    run1_file = demo_dir / "run_idle_sessionA.jsonl"
    run1_meta = demo_dir / "run_idle_sessionA.meta.json"
    with open(run1_file, "w", encoding="utf-8") as f:
        for i in range(50):
            t = float(i * 2)
            dt_str = datetime.fromtimestamp(base_epoch + t, tz=UTC).isoformat()
            sample = SystemSample(
                ts_utc=dt_str,
                monotonic_s=t,
                cpu_percent_mean=10.0,
                cpu_percent_max_core=15.0,
                cpu_freq_current_mhz=2500.0,
                cpu_freq_max_mhz=3000.0,
                cpu_freq_reliable=True,
                ctx_switches_per_s=1000.0,
                interrupts_per_s=500.0,
                mem_total_bytes=16 * (1024**3),
                mem_available_bytes=8 * (1024**3),
                mem_percent=50.0,
                swap_used_bytes=1024**2,
                swap_percent=1.0,
                disk_read_bytes_per_s=1024.0,
                disk_write_bytes_per_s=2048.0,
                disk_busy_ms_per_s=10.0,
                net_sent_bytes_per_s=100.0,
                net_recv_bytes_per_s=200.0,
                net_errors_per_s=0.0,
                net_drops_per_s=0.0,
                process_count=150,
                thread_count=1200,
                battery_percent=100.0,
                battery_plugged=True,
                max_temperature_c=45.0,
                uptime_s=5000.0,
            )
            # probe every 4s -> 25 probes
            probe = (
                ProbeResult(
                    ts_utc=dt_str,
                    cpu_ns=10_000_000,
                    mem_ns=1_000_000,
                    disk_ns=3_000_000,
                    total_ns=14_000_000,
                )
                if i % 2 == 0
                else None
            )
            rec = Record(
                schema_version=1,
                run_id="run-idle-sessionA",
                label="idle",
                sample=sample,
                probe=probe,
                top_processes=[],
                storage=None,
            )
            f.write(json.dumps(rec.to_dict()) + "\n")

    with open(run1_meta, "w", encoding="utf-8") as f:
        json.dump(
            {
                "session_id": "session-A",
                "machine_id": "test-box",
                "scenario": "idle",
                "intensity": "none",
                "pattern": "none",
                "aborted": False,
                "power_state": "ac",
                "actual": {
                    "pre_start": "2026-11-01T12:00:00Z",
                    "inject_start": "2026-11-01T12:00:10Z",
                    "inject_end": "2026-11-01T12:01:30Z",
                    "post_end": "2026-11-01T12:01:40Z",
                },
            },
            f,
        )

    # 2. Run where power state changes mid-run (battery_plugged: True -> False)
    run2_file = demo_dir / "run_power_change.jsonl"
    run2_meta = demo_dir / "run_power_change.meta.json"
    with open(run2_file, "w", encoding="utf-8") as f:
        for i in range(20):
            t = float(i * 2)
            dt_str = datetime.fromtimestamp(base_epoch + 200 + t, tz=UTC).isoformat()
            is_ac = i < 10
            sample = SystemSample(
                ts_utc=dt_str,
                monotonic_s=t,
                cpu_percent_mean=10.0,
                cpu_percent_max_core=15.0,
                cpu_freq_current_mhz=2500.0,
                cpu_freq_max_mhz=3000.0,
                cpu_freq_reliable=True,
                ctx_switches_per_s=1000.0,
                interrupts_per_s=500.0,
                mem_total_bytes=16 * (1024**3),
                mem_available_bytes=8 * (1024**3),
                mem_percent=50.0,
                swap_used_bytes=1024**2,
                swap_percent=1.0,
                disk_read_bytes_per_s=1024.0,
                disk_write_bytes_per_s=2048.0,
                disk_busy_ms_per_s=10.0,
                net_sent_bytes_per_s=100.0,
                net_recv_bytes_per_s=200.0,
                net_errors_per_s=0.0,
                net_drops_per_s=0.0,
                process_count=150,
                thread_count=1200,
                battery_percent=100.0,
                battery_plugged=is_ac,
                max_temperature_c=45.0,
                uptime_s=5000.0,
            )
            rec = Record(
                schema_version=1,
                run_id="run-power-change",
                label="idle",
                sample=sample,
                probe=None,
                top_processes=[],
                storage=None,
            )
            f.write(json.dumps(rec.to_dict()) + "\n")

    with open(run2_meta, "w", encoding="utf-8") as f:
        json.dump(
            {
                "session_id": "session-A",
                "machine_id": "test-box",
                "scenario": "idle",
                "intensity": "none",
                "pattern": "none",
                "aborted": False,
                "power_state": "ac",
                "actual": {
                    "pre_start": "2026-11-01T12:00:00Z",
                    "inject_start": "2026-11-01T12:00:10Z",
                    "inject_end": "2026-11-01T12:00:30Z",
                    "post_end": "2026-11-01T12:00:40Z",
                },
            },
            f,
        )

    print("Generated demo runs in demo_runs/")


if __name__ == "__main__":
    main()
