"""Dataset building, manifest generation, validation, and multi-machine merging."""

import hashlib
import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from deskflow import __version__
from deskflow.baseline import Baseline, load_baseline
from deskflow.collector.recorder import load_run
from deskflow.features.extract import extract_run_features
from deskflow.features.registry import (
    FEATURE_NAMES,
    FEATURE_SET_VERSION,
)
from deskflow.features.splits import assign_splits

logger = logging.getLogger(__name__)

TARGET_COLUMNS = [
    "target_total_ratio",
    "target_cpu_ratio",
    "target_mem_ratio",
    "target_disk_ratio",
    "target_slow",
]

METADATA_COLUMNS = [
    "issue_now",
    "run_id",
    "session_id",
    "machine_id",
    "scenario",
    "intensity",
    "pattern",
    "ts_utc",
    "horizon_s",
    "baseline_total_ms",
]


class DatasetValidationError(Exception):
    """Raised when a dataset fails schema or integrity validation."""


def _compute_sha256(file_path: Path) -> str:
    """Compute hex SHA-256 checksum of a file."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def build_dataset(
    run_paths: list[Path],
    baseline: Baseline | None = None,
    baseline_path: Path | None = None,
    horizon_s: float = 30.0,
    out_path: Path | None = None,
    keep_transition: bool = False,
) -> tuple[Path, Path]:
    """Build a parquet dataset from JSONL run files and sidecars with manifest."""
    if baseline is None:
        baseline = load_baseline(path=baseline_path)

    # Collect all jsonl files
    jsonl_files: list[Path] = []
    for p in run_paths:
        if p.is_file() and p.suffix == ".jsonl":
            jsonl_files.append(p)
        elif p.is_dir():
            jsonl_files.extend(sorted(p.glob("*.jsonl")))

    if not jsonl_files:
        raise ValueError("No .jsonl run files found in specified paths.")

    all_rows: list[dict[str, Any]] = []
    processed_run_ids: list[str] = []

    for jf in jsonl_files:
        try:
            records = load_run(jf)
        except Exception as e:
            logger.warning("Failed to load %s: %s", jf, e)
            continue

        if not records:
            continue

        meta_path = jf.with_suffix(".meta.json")
        meta: dict[str, Any] | None = None
        if meta_path.exists():
            try:
                with open(meta_path, encoding="utf-8") as f:
                    meta = json.load(f)
            except Exception:
                pass

        if meta and meta.get("aborted"):
            logger.warning("Skipping aborted run %s (%s)", jf.name, meta.get("abort_reason"))
            continue

        run_rows = extract_run_features(
            records=records,
            baseline=baseline,
            meta=meta,
            horizon_s=horizon_s,
            keep_transition=keep_transition,
        )

        if run_rows:
            all_rows.extend(run_rows)
            processed_run_ids.append(records[0].run_id)

    if not all_rows:
        raise ValueError("No valid rows could be extracted from input runs.")

    df = pd.DataFrame(all_rows)

    # Deterministic sorting
    df = df.sort_values(by=["run_id", "ts_utc"], ascending=[True, True]).reset_index(drop=True)

    # Assign splits
    df, strategy = assign_splits(df)

    # Order columns canonically: features, targets, metadata, split
    canonical_columns = list(FEATURE_NAMES) + TARGET_COLUMNS + METADATA_COLUMNS + ["split"]
    df = df[canonical_columns]

    target_parquet = out_path or Path("dataset.parquet")
    target_parquet.parent.mkdir(parents=True, exist_ok=True)

    # Save to parquet
    df.to_parquet(target_parquet, index=False, engine="pyarrow")

    # Write sibling manifest
    manifest_path = target_parquet.with_suffix(".manifest.json")
    sha256_hex = _compute_sha256(target_parquet)

    manifest_dict = {
        "feature_set_version": FEATURE_SET_VERSION,
        "deskflow_version": __version__,
        "horizon_s": horizon_s,
        "row_count": len(df),
        "train_rows": int((df["split"] == "train").sum()),
        "test_rows": int((df["split"] == "test").sum()),
        "split_strategy": strategy,
        "source_run_ids": sorted(list(set(processed_run_ids))),
        "baseline_used": {
            "machine_id": baseline.machine_id,
            "created_utc": baseline.created_utc,
            "total_ns": baseline.total_ns,
        },
        "build_utc": datetime.now(UTC).isoformat(),
        "sha256": sha256_hex,
    }

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_dict, f, indent=2)

    return target_parquet, manifest_path


def describe_dataset(parquet_path: Path) -> dict[str, Any]:
    """Calculate summary statistics and health metrics for a dataset."""
    df = pd.read_parquet(parquet_path)

    total_rows = len(df)
    train_rows = int((df["split"] == "train").sum()) if "split" in df.columns else 0
    test_rows = int((df["split"] == "test").sum()) if "split" in df.columns else 0

    rows_per_issue = df["issue_now"].value_counts().to_dict() if "issue_now" in df.columns else {}
    rows_per_scenario = df["scenario"].value_counts().to_dict() if "scenario" in df.columns else {}
    rows_per_intensity = (
        df["intensity"].value_counts().to_dict() if "intensity" in df.columns else {}
    )
    rows_per_pattern = df["pattern"].value_counts().to_dict() if "pattern" in df.columns else {}

    time_min = str(df["ts_utc"].min()) if "ts_utc" in df.columns and not df.empty else "n/a"
    time_max = str(df["ts_utc"].max()) if "ts_utc" in df.columns and not df.empty else "n/a"

    sessions = df["session_id"].unique().tolist() if "session_id" in df.columns else []

    missing_fractions: dict[str, float] = {}
    for col in FEATURE_NAMES:
        if col in df.columns:
            m_frac = float(df[col].isna().mean())
            if m_frac > 0.0:
                missing_fractions[col] = m_frac

    targets_arr = (
        np.array(df["target_total_ratio"].dropna(), dtype=float)
        if "target_total_ratio" in df.columns
        else np.array([], dtype=float)
    )
    t_min = float(np.min(targets_arr)) if len(targets_arr) > 0 else 0.0
    t_med = float(np.median(targets_arr)) if len(targets_arr) > 0 else 0.0
    t_p95 = float(np.percentile(targets_arr, 95)) if len(targets_arr) > 0 else 0.0
    t_max = float(np.max(targets_arr)) if len(targets_arr) > 0 else 0.0

    slow_share = (
        float(df["target_slow"].mean()) if "target_slow" in df.columns and not df.empty else 0.0
    )

    return {
        "total_rows": total_rows,
        "train_rows": train_rows,
        "test_rows": test_rows,
        "test_share": (test_rows / total_rows) if total_rows > 0 else 0.0,
        "sessions": sessions,
        "session_count": len(sessions),
        "time_span": {"start": time_min, "end": time_max},
        "rows_per_issue": rows_per_issue,
        "rows_per_scenario": rows_per_scenario,
        "rows_per_intensity": rows_per_intensity,
        "rows_per_pattern": rows_per_pattern,
        "missing_features": missing_fractions,
        "target_total_ratio": {
            "min": t_min,
            "median": t_med,
            "p95": t_p95,
            "max": t_max,
        },
        "target_slow_share": slow_share,
    }


def validate_dataset(parquet_path: Path) -> list[str]:
    """Validate schema compliance, column presence, target completeness, and duplicate freedom."""
    errors: list[str] = []
    try:
        df = pd.read_parquet(parquet_path)
    except Exception as e:
        return [f"Failed to read parquet file: {e}"]

    # 1. Check feature columns
    for f in FEATURE_NAMES:
        if f not in df.columns:
            errors.append(f"Missing required feature column: '{f}'")

    # 2. Check target columns
    for t in TARGET_COLUMNS:
        if t not in df.columns:
            errors.append(f"Missing target column: '{t}'")
        elif df[t].isna().any():
            nan_count = int(df[t].isna().sum())
            errors.append(f"Target column '{t}' contains {nan_count} NaN values.")

    # 3. Check metadata and split columns
    for m in METADATA_COLUMNS + ["split"]:
        if m not in df.columns:
            errors.append(f"Missing metadata column: '{m}'")

    # 4. Check duplicate (run_id, ts_utc)
    if "run_id" in df.columns and "ts_utc" in df.columns:
        dups = df.duplicated(subset=["run_id", "ts_utc"]).sum()
        if dups > 0:
            errors.append(f"Found {dups} duplicate (run_id, ts_utc) rows.")

    return errors


def merge_datasets(dataset_files: list[Path], out_path: Path) -> tuple[Path, Path]:
    """Merge datasets from multiple machines or sessions, re-verifying version compatibility."""
    if not dataset_files:
        raise ValueError("No dataset files specified for merge.")

    dfs: list[pd.DataFrame] = []
    horizons: set[float] = set()

    for df_path in dataset_files:
        df = pd.read_parquet(df_path)
        if "horizon_s" in df.columns:
            horizons.add(float(df["horizon_s"].iloc[0]))
        dfs.append(df)

    if len(horizons) > 1:
        raise ValueError(f"Cannot merge datasets with differing horizons: {horizons}")

    merged_df = pd.concat(dfs, ignore_index=True)

    # Deduplicate identical rows if any
    merged_df = merged_df.drop_duplicates(subset=["run_id", "ts_utc"]).reset_index(drop=True)

    # Re-apply splits cleanly across all merged sessions
    merged_df, strategy = assign_splits(merged_df)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged_df.to_parquet(out_path, index=False, engine="pyarrow")

    manifest_path = out_path.with_suffix(".manifest.json")
    sha256_hex = _compute_sha256(out_path)

    manifest_dict = {
        "feature_set_version": FEATURE_SET_VERSION,
        "deskflow_version": __version__,
        "horizon_s": next(iter(horizons)) if horizons else 30.0,
        "row_count": len(merged_df),
        "train_rows": int((merged_df["split"] == "train").sum()),
        "test_rows": int((merged_df["split"] == "test").sum()),
        "split_strategy": strategy,
        "source_run_ids": sorted(list(merged_df["run_id"].unique())),
        "build_utc": datetime.now(UTC).isoformat(),
        "sha256": sha256_hex,
        "source_datasets": [str(p) for p in dataset_files],
    }

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_dict, f, indent=2)

    return out_path, manifest_path
