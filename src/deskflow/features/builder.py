"""Dataset building, manifest generation, validation, and multi-machine merging."""

import hashlib
import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass

from deskflow import __version__
from deskflow.baseline import Baseline, load_all_baselines, load_baseline
from deskflow.collector.recorder import load_run
from deskflow.features.extract import extract_run_features
from deskflow.features.registry import (
    DEFAULT_HORIZON_S,
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
    "baseline_state",
    "baseline_source",
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


@dataclass(frozen=True)
class RunBuildReport:
    """Audit record for a single run processed during dataset build."""

    run_id: str
    label: str
    duration_s: float
    rows_kept: int
    rows_dropped: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "label": self.label,
            "duration_s": round(self.duration_s, 2),
            "rows_kept": self.rows_kept,
            "rows_dropped": self.rows_dropped,
        }


class DatasetBuildResult:
    """Result of building a dataset, preserving 2-tuple unpacking compatibility."""

    def __init__(
        self,
        parquet_path: Path,
        manifest_path: Path,
        reports: list[RunBuildReport],
    ) -> None:
        self.parquet_path = parquet_path
        self.manifest_path = manifest_path
        self.reports = reports

    def __iter__(self) -> Any:
        return iter((self.parquet_path, self.manifest_path))

    def __getitem__(self, idx: int) -> Path:
        return (self.parquet_path, self.manifest_path)[idx]

    def __len__(self) -> int:
        return 2

    def __repr__(self) -> str:
        return (
            f"DatasetBuildResult(parquet_path={self.parquet_path!r}, "
            f"manifest_path={self.manifest_path!r}, reports={self.reports!r})"
        )


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
    baseline: Baseline | dict[str, Baseline | None] | None = None,
    baseline_path: Path | None = None,
    horizon_s: float = DEFAULT_HORIZON_S,
    out_path: Path | None = None,
    keep_transition: bool = False,
) -> DatasetBuildResult:
    """Build a parquet dataset from JSONL run files and sidecars with manifest."""
    if isinstance(baseline, (Path, str)):
        baseline_path = Path(baseline)
        baseline = None

    if baseline is None:
        try:
            baseline = load_all_baselines(path=baseline_path)
        except Exception:
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
    run_reports: list[RunBuildReport] = []

    # First pass: load runs, check mid-run power consistency, and gather session idle probes
    loaded_runs: list[tuple[Path, list[Any], dict[str, Any] | None]] = []
    session_idle_probes: dict[tuple[str, str], list[Any]] = {}

    for jf in jsonl_files:
        try:
            records = load_run(jf)
        except Exception as e:
            logger.warning("Failed to load %s: %s", jf, e)
            continue

        if not records:
            continue

        meta_path = jf.with_suffix(".meta.json")
        meta = None
        if meta_path.exists():
            try:
                with open(meta_path, encoding="utf-8") as f:
                    meta = json.load(f)
            except Exception:
                pass

        loaded_runs.append((jf, records, meta))

        # Check for idle/prephase probes to form per-session baselines
        run_label = records[0].label
        session_id = meta.get("session_id") if meta else f"session-{jf.parent.name}"
        for r in records:
            if r.probe is not None:
                p_state = getattr(r.sample, "power_state", None) or (
                    "ac" if r.sample.battery_plugged else "battery"
                )
                key = (session_id, p_state)
                if run_label == "idle" or (meta and meta.get("scenario") == "idle"):
                    session_idle_probes.setdefault(key, []).append(r.probe)

    # Compute per-session baseline if at least 20 idle probes exist
    import numpy as np

    session_baselines: dict[tuple[str, str], Baseline] = {}
    for key, p_list in session_idle_probes.items():
        if len(p_list) >= 20:
            sorted_probes = sorted(p_list, key=lambda p: p.total_ns)
            half = sorted_probes[: max(1, len(sorted_probes) // 2)]
            fastest_totals = [p.total_ns for p in half]
            sess_base = Baseline(
                machine_id=f"session-{key[0]}",
                created_utc=datetime.now(UTC).isoformat(),
                platform="unknown",
                cpu_ns=float(np.median([p.cpu_ns for p in half])),
                mem_ns=float(np.median([p.mem_ns for p in half])),
                disk_ns=float(np.median([p.disk_ns for p in half])),
                total_ns=float(np.median(fastest_totals)),
                p10_total_ns=float(np.percentile([p.total_ns for p in p_list], 10)),
                p90_total_ns=float(np.percentile([p.total_ns for p in p_list], 90)),
                sample_count=len(p_list),
                deskflow_version=__version__,
                power_state=key[1],
            )
            session_baselines[key] = sess_base

    # Second pass: feature extraction
    for jf, records, meta in loaded_runs:
        run_id = records[0].run_id
        label = records[0].label
        duration_s = (
            records[-1].sample.monotonic_s - records[0].sample.monotonic_s
            if len(records) > 1
            else 0.0
        )
        drop_stats: dict[str, int] = {}

        # 1. Drop if aborted
        if meta and meta.get("aborted"):
            drop_stats["aborted run"] = len(records)
            run_reports.append(
                RunBuildReport(
                    run_id=run_id,
                    label=label,
                    duration_s=duration_s,
                    rows_kept=0,
                    rows_dropped=drop_stats,
                )
            )
            logger.warning("Skipping aborted run %s (%s)", jf.name, meta.get("abort_reason"))
            continue

        # 2. Drop if power state changed mid-run
        p_states = {
            getattr(r.sample, "power_state", None)
            or ("ac" if r.sample.battery_plugged else "battery")
            for r in records
        }
        if len(p_states) > 1:
            drop_stats["power state changed mid-run"] = len(records)
            run_reports.append(
                RunBuildReport(
                    run_id=run_id,
                    label=label,
                    duration_s=duration_s,
                    rows_kept=0,
                    rows_dropped=drop_stats,
                )
            )
            logger.warning("Skipping run %s: power state changed mid-run (%s)", jf.name, p_states)
            continue

        # Resolve active baseline and baseline_source for this run
        run_p_state = str(next(iter(p_states)))
        session_id = (
            str(meta["session_id"])
            if (meta and meta.get("session_id"))
            else f"session-{jf.parent.name}"
        )
        sess_key: tuple[str, str] = (session_id, run_p_state)

        active_b: Baseline | dict[str, Baseline | None]
        if sess_key in session_baselines:
            active_b = session_baselines[sess_key]
            b_source = "session_idle"
        else:
            active_b = baseline
            b_source = "stored_profile"
            if len(session_idle_probes.get(sess_key, [])) > 0:
                logger.warning(
                    "Session %s has fewer than 20 idle probes (%d); "
                    "falling back to stored baseline",
                    session_id,
                    len(session_idle_probes.get(sess_key, [])),
                )

        run_rows = extract_run_features(
            records=records,
            baseline=active_b,
            meta=meta,
            horizon_s=horizon_s,
            keep_transition=keep_transition,
            drop_stats=drop_stats,
            baseline_source=b_source,
        )

        rows_kept = len(run_rows)
        run_reports.append(
            RunBuildReport(
                run_id=run_id,
                label=label,
                duration_s=duration_s,
                rows_kept=rows_kept,
                rows_dropped=drop_stats,
            )
        )

        if run_rows:
            all_rows.extend(run_rows)
            processed_run_ids.append(run_id)

    if not all_rows:
        raise ValueError("No valid rows could be extracted from input runs.")

    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError(
            "Optional research dependency 'pandas' is required to build datasets. "
            "Install with: pip install 'deskflow[train]' or uv pip install -e '.[train]'"
        ) from exc

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

    b_summary = (
        baseline.get("ac") or next((b for b in baseline.values() if b is not None), None)
        if isinstance(baseline, dict)
        else baseline
    )

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
            "machine_id": b_summary.machine_id if b_summary else "unknown",
            "created_utc": b_summary.created_utc if b_summary else "",
            "total_ns": b_summary.total_ns if b_summary else 0,
        },
        "build_utc": datetime.now(UTC).isoformat(),
        "sha256": sha256_hex,
        "run_reports": [r.to_dict() for r in run_reports],
    }

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_dict, f, indent=2)

    return DatasetBuildResult(
        parquet_path=target_parquet,
        manifest_path=manifest_path,
        reports=run_reports,
    )


def describe_dataset(parquet_path: Path) -> dict[str, Any]:
    """Calculate summary statistics and health metrics for a dataset."""
    import numpy as np
    import pandas as pd

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
    import pandas as pd

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

    import pandas as pd

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
