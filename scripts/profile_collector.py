"""Profile individual stages of a collection tick and run cProfile on collector."""

import cProfile
import io
import pstats
import tempfile
import time
from pathlib import Path

import psutil

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


def profile_stages(num_ticks: int = 30) -> dict[str, list[float]]:
    """Time each stage of a collection tick separately over num_ticks."""
    adapter = get_adapter()
    sampler = Sampler(adapter=adapter, interval_s=1.0, process_interval_s=10.0)
    storage_scanner = StorageScanner(adapter=adapter)

    with tempfile.TemporaryDirectory() as tmpdir:
        run_file = Path(tmpdir) / "profile_run.jsonl"
        with RunRecorder(out_path=run_file) as recorder:
            stage_timings: dict[str, list[float]] = {
                "system_counters": [],
                "process_enumeration": [],
                "thread_counting": [],
                "storage_report": [],
                "probe": [],
                "json_write": [],
                "total_tick": [],
            }

            # Prime sampler
            sampler.sample()
            time.sleep(0.5)

            for _tick in range(num_ticks):
                t_tick_start = time.perf_counter()

                # 1. System counters via sampler.sample()
                t0 = time.perf_counter()
                sample, processes = sampler.sample()
                t1 = time.perf_counter()
                stage_timings["system_counters"].append((t1 - t0) * 1000.0)

                # 2. Process enumeration alone (psutil process_iter)
                t_proc_start = time.perf_counter()
                attrs = ["pid", "name", "memory_info", "cpu_percent"]
                for proc in psutil.process_iter(attrs=attrs):
                    _ = proc.info
                t_proc_end = time.perf_counter()
                stage_timings["process_enumeration"].append((t_proc_end - t_proc_start) * 1000.0)

                # 3. Thread counting via adapter GetPerformanceInfo
                t_th_start = time.perf_counter()
                _ = adapter.thread_count()
                t_th_end = time.perf_counter()
                stage_timings["thread_counting"].append((t_th_end - t_th_start) * 1000.0)

                # 4. Storage report retrieval
                t_st_start = time.perf_counter()
                storage = storage_scanner.get_cached_report()
                t_st_end = time.perf_counter()
                stage_timings["storage_report"].append((t_st_end - t_st_start) * 1000.0)

                # 5. Probe
                t_pr_start = time.perf_counter()
                probe: ProbeResult | None = run_probe()
                t_pr_end = time.perf_counter()
                stage_timings["probe"].append((t_pr_end - t_pr_start) * 1000.0)

                # 6. JSON write
                t_wr_start = time.perf_counter()
                record = Record(
                    schema_version=SCHEMA_VERSION,
                    run_id=recorder.run_id,
                    label="profiling",
                    sample=sample,
                    probe=probe,
                    top_processes=processes[:10],
                    storage=storage,
                )
                recorder.write(record)
                t_wr_end = time.perf_counter()
                stage_timings["json_write"].append((t_wr_end - t_wr_start) * 1000.0)

                t_tick_end = time.perf_counter()
                stage_timings["total_tick"].append((t_tick_end - t_tick_start) * 1000.0)

                time.sleep(0.05)

    return stage_timings


def run_cprofile(duration_s: float = 30.0) -> str:
    """Run cProfile on a continuous collect session for duration_s."""
    adapter = get_adapter()
    sampler = Sampler(adapter=adapter, interval_s=2.0, process_interval_s=10.0)

    pr = cProfile.Profile()
    with tempfile.TemporaryDirectory() as tmpdir:
        run_file = Path(tmpdir) / "cprofile_run.jsonl"
        with RunRecorder(out_path=run_file) as recorder:
            pr.enable()
            start = time.monotonic()
            while time.monotonic() - start < duration_s:
                sample, processes = sampler.sample()
                probe = run_probe()
                record = Record(
                    schema_version=SCHEMA_VERSION,
                    run_id=recorder.run_id,
                    label="cprofile",
                    sample=sample,
                    probe=probe,
                    top_processes=processes[:10],
                    storage=None,
                )
                recorder.write(record)
                time.sleep(2.0)
            pr.disable()

    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
    ps.print_stats(35)
    return s.getvalue()


def main() -> None:
    print("=" * 60)
    print("STAGE-BY-STAGE COLLECTOR PROFILE (30 ticks)")
    print("=" * 60)
    timings = profile_stages(num_ticks=30)
    print(f"{'Stage':<25} {'Mean (ms)':<15} {'Max (ms)':<15}")
    print("-" * 55)
    for stage, times in timings.items():
        mean_ms = sum(times) / len(times)
        max_ms = max(times)
        print(f"{stage:<25} {mean_ms:<15.2f} {max_ms:<15.2f}")

    print("\n" + "=" * 60)
    print("CPROFILE SUMMARY (30s collect run)")
    print("=" * 60)
    cprof_output = run_cprofile(duration_s=30.0)
    print(cprof_output)


if __name__ == "__main__":
    main()
