# ADR 0002: Add pyarrow dependency for Parquet dataset storage

Date: 2026-11-01
Status: Accepted

## Context

Phase 2 requires building, validating, merging, and distributing tabular datasets of extracted features and regression/classification targets. Datasets contain dozens of rolling-window time series features across thousands of observations. Storing these datasets in uncompressed CSV or JSON is inefficient in storage size, lacks typed schema guarantees (e.g. distinguishing float32, int64, boolean, and string columns losslessly), and is slow to read and write.

## Decision

We add `pyarrow` as a runtime dependency. The dataset building and loading pipelines use the Apache Parquet format (`.parquet`) via `pyarrow` and `pandas`.

## Consequences

- Datasets are stored compactly with columnar compression (snappy by default).
- Precise column data types (including nullable floats and timestamps) are preserved losslessly.
- Building, validating, and merging large multi-session datasets remains fast and memory-efficient.
- An additional binary package dependency is added to the environment; wheels are available for Windows and macOS on Python 3.11+.

