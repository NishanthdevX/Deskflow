"""Reproduction and fix demonstration for Phase 9 Ridge 18M bug."""

import numpy as np

from deskflow.features.registry import FEATURE_NAMES
from deskflow.ml.artifacts import resolve_model_dir
from deskflow.ml.runtime import RuntimeModels


def main() -> None:
    model_dir, _ = resolve_model_dir()
    runtime = RuntimeModels.load(model_dir)

    print("=== RIDGE 18,087,065 BUG REPRODUCTION AND FIX ===")
    coef = np.asarray(runtime.ridge_params.get("coef", []), dtype=np.float64)
    scaler_mean = np.asarray(runtime.ridge_params.get("scaler_mean", []), dtype=np.float64)
    scaler_scale = np.asarray(runtime.ridge_params.get("scaler_scale", []), dtype=np.float64)
    intercept = float(runtime.ridge_params.get("intercept", 1.0))

    # Identify offending unscaled feature: mem_available_bytes
    mem_idx = FEATURE_NAMES.index("mem_available_bytes")
    feature_name = FEATURE_NAMES[mem_idx]
    raw_mem_bytes = 8_250_000_000.0  # Typical 8.25 GB unstandardized memory reading

    print(f"Offending feature: {feature_name} (index {mem_idx})")
    print(f"  Raw live value: {raw_mem_bytes:.2e} bytes (~8.25 GB)")
    print(f"  StandardScaler mean: {scaler_mean[mem_idx]:.4e}")
    print(f"  StandardScaler scale: {scaler_scale[mem_idx]:.4e}")
    print(f"  Ridge coefficient: {coef[mem_idx]:.6e}")

    # Bug reproduction (direct dot product in watch.py without StandardScaler):
    # In earlier versions: np.dot(X_row[0], coef) + intercept
    raw_dot = raw_mem_bytes * coef[mem_idx] + intercept
    print("\n[BUG OCCURRENCE IN LIVE WATCH]")
    print(f"  Unscaled dot product contribution: {raw_dot:,.2f}")

    # Fixed calculation through RuntimeModels.predict:
    row = np.zeros((1, len(FEATURE_NAMES)), dtype=np.float64)
    row[0, :] = scaler_mean  # Default other features to mean
    row[0, mem_idx] = raw_mem_bytes

    pred_res = runtime.predict(row)
    print("\n[PHASE 9 GUARDED INFERENCE (runtime.predict)]")
    print(f"  Primary model (XGBoost) prediction: {pred_res['slowdown_ratio']['XGBoost']:.4f}")
    print(f"  Ridge model prediction: {pred_res['slowdown_ratio']['Ridge']:.4f}")
    print("  Model ratio bounds: [0.3000, 10.0000]")
    print(f"  Out of range flag: {pred_res.get('out_of_range')}")
    print(f"  Features clipped count: {pred_res.get('clipped_count')}")


if __name__ == "__main__":
    main()
