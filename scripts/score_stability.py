"""Script to test health score and slowdown ratio stability across 5 back-to-back scans."""

import statistics
import time

import numpy as np

from deskflow.baseline import get_current_power_state, load_baseline
from deskflow.collector.models import Record
from deskflow.collector.probe import run_probe
from deskflow.collector.sampler import Sampler
from deskflow.features.extract import build_row
from deskflow.health.score import compute_health_score
from deskflow.ml.artifacts import get_latest_model_id, load_model_card
from deskflow.ml.runtime import RuntimeModels
from deskflow.paths import get_data_dir
from deskflow.platform.factory import get_adapter


def run_one_scan(adapter: object, scan_duration_s: float = 14.0) -> tuple[list[Record], Record]:
    """Execute a single scan telemetry and probe measuring window."""
    sampler = Sampler(adapter=adapter, interval_s=2.0, process_interval_s=10.0)  # type: ignore[arg-type]
    data_dir = get_data_dir()
    sample_interval = 2.0
    probe_interval = 4.0
    last_probe_mono = -999.0
    records: list[Record] = []

    mono_start = time.monotonic()
    while (time.monotonic() - mono_start) < scan_duration_s:
        t_loop_start = time.monotonic()
        sys_sample, top_procs = sampler.sample()
        probe_res = None
        if (sys_sample.monotonic_s - last_probe_mono) >= (probe_interval - 0.2):
            probe_res = run_probe(data_dir=data_dir, repeats=3)
            last_probe_mono = sys_sample.monotonic_s
        rec = Record(
            schema_version=1,
            run_id="stability-test",
            label="idle",
            sample=sys_sample,
            probe=probe_res,
            top_processes=top_procs,
            storage=None,
        )
        records.append(rec)
        loop_spent = time.monotonic() - t_loop_start
        sleep_s = max(0.05, sample_interval - loop_spent)
        time.sleep(sleep_s)

    probe_records = [r for r in records if r.probe is not None]
    if not probe_records:
        raise RuntimeError("No probe collected during scan")
    return records, probe_records[-1]


def measure_stability(iterations: int = 5) -> None:
    """Run iterations scans back-to-back and report score/ratio stability."""
    adapter = get_adapter()
    power_state = get_current_power_state(adapter)
    baseline = load_baseline(power_state=power_state)

    latest_id = get_latest_model_id()
    if latest_id is None:
        raise RuntimeError("No trained model found. Run 'deskflow train' first.")

    model_dir = get_data_dir() / "models" / latest_id
    card = load_model_card(model_dir)
    runtime = RuntimeModels.load(model_dir)

    primary_models = card.get(
        "primary_models",
        {"regressor": "XGBoost", "classifier": "XGBoost"},
    )
    primary_reg = primary_models.get("regressor", "XGBoost")

    print("-" * 65)
    print(f"DESKFLOW SCORE STABILITY EVALUATION ({iterations} scans back-to-back)")
    print(f"Machine: {baseline.machine_id[:12]} | Power state: {power_state}")
    print(f"Model ID: {latest_id} (synthetic: {card.get('synthetic', False)})")
    print("-" * 65)
    print(f"{'Run':<6} {'Score':<10} {'Slowdown Ratio':<18} {'Probe Latency':<16}")
    print("-" * 65)

    scores: list[int] = []
    ratios: list[float] = []

    for idx in range(1, iterations + 1):
        print(f"Running scan {idx}/{iterations} (14s)...", end="\r", flush=True)
        records, cur_rec = run_one_scan(adapter=adapter, scan_duration_s=14.0)

        features_dict = build_row(
            records_up_to_t=records,
            baseline=baseline,
            current_record=cur_rec,
        )
        vec = np.array([features_dict[name] for name in runtime.feature_order], dtype=np.float64)
        preds = runtime.predict(vec)

        slowdown_preds = preds.get("slowdown_ratio", {})
        predicted_ratio = float(slowdown_preds.get(primary_reg, slowdown_preds.get("XGBoost", 1.0)))
        score, label = compute_health_score(predicted_ratio)
        probe_ms = (cur_rec.probe.total_ns / 1_000_000.0) if cur_rec.probe else 0.0

        scores.append(score)
        ratios.append(predicted_ratio)

        print(f"{idx:<6} {score:<10} {predicted_ratio:<18.3f} {probe_ms:<16.1f}")

    print("-" * 65)
    score_mean = statistics.mean(scores)
    score_stdev = statistics.stdev(scores) if len(scores) > 1 else 0.0
    ratio_mean = statistics.mean(ratios)
    ratio_stdev = statistics.stdev(ratios) if len(ratios) > 1 else 0.0

    print(f"Score Mean:            {score_mean:.1f}")
    print(f"Score Std Dev:         {score_stdev:.2f} points (target <= 5.0 with real model)")
    print(f"Predicted Ratio Mean:  {ratio_mean:.3f}x")
    print(f"Predicted Ratio Stdev: {ratio_stdev:.4f}x")
    print("-" * 65)


if __name__ == "__main__":
    measure_stability(5)
