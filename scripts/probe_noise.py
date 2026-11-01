"""Measure idle probe noise across three 120s runs and evaluate stability."""

import time
from typing import Any

import numpy as np

from deskflow.collector.probe import run_probe


def run_probe_session(duration_s: float = 120.0, interval_s: float = 2.0) -> list[float]:
    """Execute probes every interval_s seconds for duration_s and return total latencies in ms."""
    latencies_ms: list[float] = []
    start_mono = time.monotonic()
    next_probe_mono = start_mono

    while True:
        now_mono = time.monotonic()
        if now_mono - start_mono >= duration_s:
            break

        if now_mono >= next_probe_mono:
            res = run_probe()
            latencies_ms.append(res.total_ns / 1_000_000.0)
            while next_probe_mono <= now_mono:
                next_probe_mono += interval_s

        time.sleep(0.05)

    return latencies_ms


def analyze_run(latencies: list[float]) -> dict[str, Any]:
    """Calculate statistical noise metrics for a sequence of probe latencies."""
    arr = np.array(latencies, dtype=float)
    median = float(np.median(arr))
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
    cv = std / mean if mean > 0 else 0.0
    p95 = float(np.percentile(arr, 95))
    ratio_p95_median = p95 / median if median > 0 else 0.0
    threshold = 1.5 * median
    spikes = int(np.sum(arr >= threshold))

    return {
        "count": len(arr),
        "median_ms": median,
        "mean_ms": mean,
        "std_ms": std,
        "cv": cv,
        "p95_ms": p95,
        "p95_div_median": ratio_p95_median,
        "spikes_above_1_5x": spikes,
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Probe noise benchmark")
    parser.add_argument(
        "--duration",
        type=float,
        default=120.0,
        help="Duration per run in seconds (default: 120.0)",
    )
    args = parser.parse_args()

    dur = args.duration
    print("=" * 65)
    print(f"PROBE NOISE BENCHMARK (3 runs of {dur:.0f}s idle probing @ 2s interval)")
    print("Please keep the machine idle and close heavy background applications.")
    print("=" * 65)

    all_latencies: list[float] = []
    run_results: list[dict[str, Any]] = []

    for run_num in range(1, 4):
        print(f"\nStarting Run {run_num}/3 ({dur:.0f}s)...")
        latencies = run_probe_session(duration_s=dur, interval_s=2.0)
        metrics = analyze_run(latencies)
        run_results.append(metrics)
        all_latencies.extend(latencies)
        print(
            f"Run {run_num} complete: {metrics['count']} probes | "
            f"Median: {metrics['median_ms']:.2f} ms | "
            f"CV: {metrics['cv']:.3f} | "
            f"p95/Median: {metrics['p95_div_median']:.2f} | "
            f">1.5x Median: {metrics['spikes_above_1_5x']}"
        )

    overall = analyze_run(all_latencies)

    print("\n" + "=" * 65)
    print("PROBE NOISE BENCHMARK RESULTS")
    print("=" * 65)
    hdr = (
        f"{'Run':<8} {'Probes':<8} {'Median (ms)':<14} {'CV':<8} "
        f"{'p95/Median':<12} {'>1.5x Med':<10}"
    )
    print(hdr)
    print("-" * 65)
    for idx, r in enumerate(run_results, 1):
        print(
            f"Run {idx:<6} {r['count']:<8} {r['median_ms']:<14.2f} {r['cv']:<10.3f} "
            f"{r['p95_div_median']:<12.2f} {r['spikes_above_1_5x']:<10}"
        )
    print("-" * 65)
    print(
        f"{'OVERALL':<10} {overall['count']:<8} {overall['median_ms']:<14.2f} "
        f"{overall['cv']:<10.3f} {overall['p95_div_median']:<12.2f} "
        f"{overall['spikes_above_1_5x']:<10}"
    )
    print("=" * 65)

    print("\nDecision Rule Check:")
    if overall["p95_div_median"] > 1.4:
        print(
            f"Verdict: p95/median ({overall['p95_div_median']:.2f}) > 1.4. "
            "REPEAT PROBES REQUIRED (CPU and memory repeat 3x min)."
        )
    else:
        print(
            f"Verdict: p95/median ({overall['p95_div_median']:.2f}) <= 1.4. "
            "Single repeat is sufficiently quiet. No multi-repeat workload needed."
        )


if __name__ == "__main__":
    main()
