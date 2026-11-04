"""Generate a realistic synthetic dataset in v2 schema for Phase 3 ML development."""

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from deskflow import __version__
from deskflow.features.builder import METADATA_COLUMNS, TARGET_COLUMNS
from deskflow.features.registry import (
    DEFAULT_HORIZON_S,
    FEATURE_NAMES,
    FEATURE_SET_VERSION,
    SLOW_THRESHOLD,
)


def _compute_sha256(file_path: Path) -> str:
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def generate_synthetic_dataset(
    out_path: Path = Path("synthetic.parquet"),
    n_rows: int = 3000,
    seed: int = 42,
) -> tuple[Path, Path]:
    """Generate ~3000 synthetic rows across 14 runs on 2 machines in Feature Set v2 schema."""
    rng = np.random.RandomState(seed)

    # Ensure every scenario has ample representation in both train and test splits
    # 5 base classes + combo = 6 distinct classes
    base_scenarios = [
        ("idle", "healthy", "none", "none"),
        ("cpu_pressure", "cpu_pressure", "medium", "constant"),
        ("memory_pressure", "memory_pressure", "medium", "constant"),
        ("disk_pressure", "disk_pressure", "medium", "constant"),
        ("background_load", "background_load", "medium", "constant"),
        ("combo", "combo", "high", "constant"),
    ]

    machines = ["mach-desktop-intel", "mach-laptop-ryzen"]
    runs = []

    # 3 repetitions per scenario: 2 train, 1 test = 18 runs total
    for rep in range(3):
        split = "test" if rep == 2 else "train"
        for idx, scen_info in enumerate(base_scenarios):
            run_idx = rep * len(base_scenarios) + idx
            m = machines[run_idx % len(machines)]
            runs.append(
                {
                    "run_id": f"synthetic-run-{run_idx:03d}",
                    "machine_id": m,
                    "scenario": scen_info[0],
                    "issue_now": scen_info[1],
                    "intensity": scen_info[2],
                    "pattern": scen_info[3],
                    "split": split,
                    "power_state": "battery" if "laptop" in m and run_idx % 2 == 0 else "ac",
                }
            )

    rows_per_run = n_rows // len(runs)
    all_rows: list[dict[str, Any]] = []

    base_time = 1700000000.0

    for run in runs:
        issue = run["issue_now"]

        for step in range(rows_per_run):
            t_offset = step * 4.0
            ts_str = datetime.fromtimestamp(base_time + t_offset, tz=UTC).isoformat()
            row: dict[str, Any] = {}

            # Generate scenario-specific characteristic feature distributions
            if issue == "healthy":
                cpu_mean = rng.uniform(5.0, 25.0)
                cpu_max = cpu_mean + rng.uniform(2.0, 10.0)
                cpu_slope = rng.normal(0.0, 0.1)
                freq_ratio_mean = rng.uniform(0.95, 1.0)
                freq_ratio_min = rng.uniform(0.90, 0.98)
                mem_pct = rng.uniform(30.0, 50.0)
                mem_slope = rng.normal(0.0, 0.2)
                mem_avail = rng.uniform(8.0, 12.0) * (1024**3)
                swap_pct = rng.uniform(0.0, 5.0)
                disk_r = rng.uniform(0.1, 5.0)
                disk_w = rng.uniform(0.1, 5.0)
                ctx = rng.uniform(500.0, 2000.0)
                procs = float(rng.randint(120, 180))
                threads = float(rng.randint(1200, 1800))
                top1_rss = rng.uniform(0.05, 0.15)
                top1_cpu = rng.uniform(2.0, 10.0)
                busy_procs = float(rng.randint(0, 1))
                idle_rss = rng.uniform(1.0, 3.0)
                idle_count = float(rng.randint(2, 6))

                p_cpu = rng.normal(1.0, 0.05)
                p_mem = rng.normal(1.0, 0.05)
                p_disk = rng.normal(1.0, 0.05)
                p_tot = rng.normal(1.0, 0.05)
                p_tot_12s = rng.normal(1.0, 0.04)
                slow_idle = 0.0

                tgt_tot = float(np.clip(rng.normal(1.02, 0.06), 0.8, 1.3))
                tgt_cpu = float(np.clip(rng.normal(1.01, 0.06), 0.8, 1.3))
                tgt_mem = float(np.clip(rng.normal(1.01, 0.06), 0.8, 1.3))
                tgt_disk = float(np.clip(rng.normal(1.01, 0.06), 0.8, 1.3))

            elif issue == "cpu_pressure":
                cpu_mean = rng.uniform(75.0, 98.0)
                cpu_max = rng.uniform(90.0, 100.0)
                cpu_slope = rng.normal(0.5, 0.3)
                freq_ratio_mean = rng.uniform(0.70, 0.90)
                freq_ratio_min = rng.uniform(0.60, 0.85)
                mem_pct = rng.uniform(40.0, 60.0)
                mem_slope = rng.normal(0.0, 0.3)
                mem_avail = rng.uniform(6.0, 9.0) * (1024**3)
                swap_pct = rng.uniform(0.0, 8.0)
                disk_r = rng.uniform(0.5, 10.0)
                disk_w = rng.uniform(0.5, 10.0)
                ctx = rng.uniform(8000.0, 25000.0)
                procs = float(rng.randint(150, 220))
                threads = float(rng.randint(1800, 2800))
                top1_rss = rng.uniform(0.08, 0.20)
                top1_cpu = rng.uniform(60.0, 95.0)
                busy_procs = float(rng.randint(2, 6))
                idle_rss = rng.uniform(1.0, 3.0)
                idle_count = float(rng.randint(1, 4))

                p_cpu = rng.normal(2.6, 0.3)
                p_mem = rng.normal(1.2, 0.1)
                p_disk = rng.normal(1.1, 0.1)
                p_tot = rng.normal(2.2, 0.25)
                p_tot_12s = rng.normal(2.1, 0.2)
                slow_idle = 0.0

                tgt_tot = float(np.clip(p_tot + rng.normal(0.2, 0.1), 1.6, 4.0))
                tgt_cpu = float(np.clip(p_cpu + rng.normal(0.2, 0.1), 1.8, 4.5))
                tgt_mem = float(np.clip(p_mem, 1.0, 2.0))
                tgt_disk = float(np.clip(p_disk, 1.0, 2.0))

            elif issue == "memory_pressure":
                cpu_mean = rng.uniform(20.0, 45.0)
                cpu_max = cpu_mean + rng.uniform(10.0, 20.0)
                cpu_slope = rng.normal(0.0, 0.2)
                freq_ratio_mean = rng.uniform(0.90, 0.98)
                freq_ratio_min = rng.uniform(0.85, 0.95)
                mem_pct = rng.uniform(88.0, 98.0)
                mem_slope = rng.uniform(1.0, 5.0)
                mem_avail = rng.uniform(0.2, 1.2) * (1024**3)
                swap_pct = rng.uniform(45.0, 85.0)
                disk_r = rng.uniform(20.0, 80.0)  # Paging
                disk_w = rng.uniform(20.0, 80.0)
                ctx = rng.uniform(4000.0, 10000.0)
                procs = float(rng.randint(140, 200))
                threads = float(rng.randint(1400, 2200))
                top1_rss = rng.uniform(0.40, 0.70)
                top1_cpu = rng.uniform(10.0, 30.0)
                busy_procs = float(rng.randint(1, 3))
                idle_rss = rng.uniform(0.2, 1.0)
                idle_count = float(rng.randint(1, 3))

                p_cpu = rng.normal(1.3, 0.1)
                p_mem = rng.normal(2.8, 0.3)
                p_disk = rng.normal(2.0, 0.3)
                p_tot = rng.normal(2.3, 0.25)
                p_tot_12s = rng.normal(2.2, 0.2)
                slow_idle = 0.0

                tgt_tot = float(np.clip(p_tot + rng.normal(0.2, 0.1), 1.6, 4.0))
                tgt_cpu = float(np.clip(p_cpu, 1.0, 2.0))
                tgt_mem = float(np.clip(p_mem + rng.normal(0.2, 0.1), 2.0, 4.5))
                tgt_disk = float(np.clip(p_disk + rng.normal(0.1, 0.1), 1.5, 3.5))

            elif issue == "disk_pressure":
                cpu_mean = rng.uniform(15.0, 35.0)
                cpu_max = cpu_mean + rng.uniform(5.0, 15.0)
                cpu_slope = rng.normal(0.0, 0.1)
                freq_ratio_mean = rng.uniform(0.92, 0.99)
                freq_ratio_min = rng.uniform(0.88, 0.95)
                mem_pct = rng.uniform(40.0, 65.0)
                mem_slope = rng.normal(0.0, 0.2)
                mem_avail = rng.uniform(5.0, 9.0) * (1024**3)
                swap_pct = rng.uniform(0.0, 5.0)
                disk_r = rng.uniform(120.0, 350.0)
                disk_w = rng.uniform(80.0, 250.0)
                ctx = rng.uniform(3000.0, 8000.0)
                procs = float(rng.randint(130, 190))
                threads = float(rng.randint(1300, 2000))
                top1_rss = rng.uniform(0.10, 0.25)
                top1_cpu = rng.uniform(8.0, 20.0)
                busy_procs = float(rng.randint(1, 2))
                idle_rss = rng.uniform(1.0, 2.5)
                idle_count = float(rng.randint(2, 5))

                p_cpu = rng.normal(1.1, 0.08)
                p_mem = rng.normal(1.1, 0.08)
                p_disk = rng.normal(3.8, 0.4)
                p_tot = rng.normal(2.5, 0.3)
                p_tot_12s = rng.normal(2.4, 0.25)
                slow_idle = 0.0

                tgt_tot = float(np.clip(p_tot + rng.normal(0.15, 0.1), 1.7, 4.5))
                tgt_cpu = float(np.clip(p_cpu, 1.0, 1.8))
                tgt_mem = float(np.clip(p_mem, 1.0, 1.8))
                tgt_disk = float(np.clip(p_disk + rng.normal(0.2, 0.15), 2.5, 5.5))

            elif issue == "background_load":
                cpu_mean = rng.uniform(40.0, 65.0)
                cpu_max = rng.uniform(65.0, 85.0)
                cpu_slope = rng.normal(0.0, 0.2)
                freq_ratio_mean = rng.uniform(0.85, 0.95)
                freq_ratio_min = rng.uniform(0.80, 0.90)
                mem_pct = rng.uniform(55.0, 75.0)
                mem_slope = rng.normal(0.0, 0.3)
                mem_avail = rng.uniform(4.0, 7.0) * (1024**3)
                swap_pct = rng.uniform(5.0, 15.0)
                disk_r = rng.uniform(10.0, 40.0)
                disk_w = rng.uniform(15.0, 50.0)
                ctx = rng.uniform(5000.0, 12000.0)
                procs = float(rng.randint(180, 260))
                threads = float(rng.randint(2000, 3200))
                top1_rss = rng.uniform(0.15, 0.30)
                top1_cpu = rng.uniform(25.0, 45.0)
                busy_procs = float(rng.randint(3, 7))
                idle_rss = rng.uniform(3.0, 6.0)
                idle_count = float(rng.randint(6, 14))

                p_cpu = rng.normal(1.6, 0.15)
                p_mem = rng.normal(1.4, 0.12)
                p_disk = rng.normal(1.5, 0.15)
                p_tot = rng.normal(1.55, 0.12)
                p_tot_12s = rng.normal(1.52, 0.10)
                slow_idle = 0.0

                tgt_tot = float(np.clip(p_tot + rng.normal(0.1, 0.08), 1.3, 2.5))
                tgt_cpu = float(np.clip(p_cpu, 1.2, 2.2))
                tgt_mem = float(np.clip(p_mem, 1.1, 2.0))
                tgt_disk = float(np.clip(p_disk, 1.1, 2.2))

            else:  # combo
                cpu_mean = rng.uniform(80.0, 98.0)
                cpu_max = 99.0
                cpu_slope = rng.normal(0.3, 0.2)
                freq_ratio_mean = rng.uniform(0.65, 0.85)
                freq_ratio_min = rng.uniform(0.55, 0.75)
                mem_pct = rng.uniform(90.0, 99.0)
                mem_slope = rng.uniform(2.0, 6.0)
                mem_avail = rng.uniform(0.1, 0.8) * (1024**3)
                swap_pct = rng.uniform(50.0, 90.0)
                disk_r = rng.uniform(50.0, 150.0)
                disk_w = rng.uniform(50.0, 150.0)
                ctx = rng.uniform(15000.0, 30000.0)
                procs = float(rng.randint(180, 260))
                threads = float(rng.randint(2200, 3500))
                top1_rss = rng.uniform(0.40, 0.65)
                top1_cpu = rng.uniform(70.0, 95.0)
                busy_procs = float(rng.randint(4, 8))
                idle_rss = rng.uniform(1.0, 3.0)
                idle_count = float(rng.randint(2, 5))

                p_cpu = rng.normal(3.2, 0.35)
                p_mem = rng.normal(3.0, 0.35)
                p_disk = rng.normal(2.5, 0.3)
                p_tot = rng.normal(3.0, 0.3)
                p_tot_12s = rng.normal(2.9, 0.25)
                slow_idle = 0.0

                tgt_tot = float(np.clip(p_tot + rng.normal(0.2, 0.1), 2.2, 5.0))
                tgt_cpu = float(np.clip(p_cpu, 2.0, 5.0))
                tgt_mem = float(np.clip(p_mem, 2.0, 5.0))
                tgt_disk = float(np.clip(p_disk, 1.8, 4.5))

            # Assemble canonical features (27 features)
            row["cpu_percent_mean_mean_6s"] = cpu_mean
            row["cpu_percent_mean_mean_12s"] = cpu_mean
            row["cpu_percent_mean_max_12s"] = cpu_max
            row["cpu_percent_mean_slope_12s"] = cpu_slope
            row["cpu_freq_ratio_mean_12s"] = freq_ratio_mean
            row["cpu_freq_ratio_min_12s"] = freq_ratio_min
            row["mem_percent_mean_12s"] = mem_pct
            row["mem_percent_slope_12s"] = mem_slope
            row["mem_available_bytes"] = mem_avail
            row["swap_percent"] = swap_pct
            row["disk_read_mb_s_mean_12s"] = disk_r
            row["disk_write_mb_s_mean_12s"] = disk_w
            row["ctx_switches_per_s_mean_12s"] = ctx
            row["process_count"] = procs
            row["thread_count"] = threads
            row["battery_plugged"] = 0.0 if run["power_state"] == "battery" else 1.0
            row["top1_rss_share"] = top1_rss
            row["top1_cpu_percent"] = top1_cpu
            row["busy_process_count"] = busy_procs
            row["idle_app_rss_gb"] = idle_rss
            row["idle_app_count"] = idle_count
            row["probe_cpu_ratio"] = p_cpu
            row["probe_mem_ratio"] = p_mem
            row["probe_disk_ratio"] = p_disk
            row["probe_total_ratio"] = p_tot
            row["probe_total_ratio_mean_12s"] = p_tot_12s
            row["slow_while_idle"] = slow_idle

            # Target values
            row["target_total_ratio"] = tgt_tot
            row["target_cpu_ratio"] = tgt_cpu
            row["target_mem_ratio"] = tgt_mem
            row["target_disk_ratio"] = tgt_disk
            row["target_slow"] = int(tgt_tot >= SLOW_THRESHOLD)

            # Metadata
            row["issue_now"] = issue
            row["baseline_state"] = run["power_state"]
            row["baseline_source"] = "synthetic_profile"
            row["run_id"] = run["run_id"]
            row["session_id"] = f"session-{run['run_id']}"
            row["machine_id"] = run["machine_id"]
            row["scenario"] = run["scenario"]
            row["intensity"] = run["intensity"]
            row["pattern"] = run["pattern"]
            row["ts_utc"] = ts_str
            row["horizon_s"] = DEFAULT_HORIZON_S
            row["baseline_total_ms"] = 25.0
            row["split"] = run["split"]

            all_rows.append(row)

    df = pd.DataFrame(all_rows)

    # Order columns canonically
    canonical_columns = list(FEATURE_NAMES) + TARGET_COLUMNS + METADATA_COLUMNS + ["split"]
    df = df[canonical_columns]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False, engine="pyarrow")

    # Generate manifest
    manifest_path = out_path.with_suffix(".manifest.json")
    sha256_hex = _compute_sha256(out_path)

    manifest = {
        "synthetic": True,
        "feature_set_version": FEATURE_SET_VERSION,
        "deskflow_version": __version__,
        "horizon_s": DEFAULT_HORIZON_S,
        "row_count": len(df),
        "train_rows": int((df["split"] == "train").sum()),
        "test_rows": int((df["split"] == "test").sum()),
        "split_strategy": "group_run_id",
        "source_run_ids": sorted(list(df["run_id"].unique())),
        "build_utc": datetime.now(UTC).isoformat(),
        "sha256": sha256_hex,
    }

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return out_path, manifest_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic dataset for deskflow.")
    parser.add_argument(
        "--out", type=Path, default=Path("synthetic.parquet"), help="Output parquet path"
    )
    parser.add_argument("--rows", type=int, default=3000, help="Approximate number of rows")
    args = parser.parse_args()

    p, m = generate_synthetic_dataset(out_path=args.out, n_rows=args.rows)
    print(f"Generated synthetic dataset: {p} ({p.stat().st_size / 1024:.1f} KB)")
    print(f"Generated manifest: {m}")
