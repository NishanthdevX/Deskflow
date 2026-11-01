# 3. Session-Level Dataset Splitting

Date: 2026-11-01

## Status

Accepted

## Context

Deskflow predicts device responsiveness and latency spikes 30 seconds ahead. Features are computed across multi-scale rolling temporal windows (10s, 30s, and 60s). Standard random train/test splitting (such as `train_test_split` uniformly across rows) introduces severe data leakage: neighbouring rows from the same continuous run share up to 98% of their rolling historical window. Random row splitting causes models to evaluate on near-identical temporal sequences seen during training, yielding unrealistically inflated validation performance.

## Decision

We enforce deterministic **session-level splitting** as the primary strategy in `deskflow.features.splits`:

1. All runs within a contiguous collection period share a `session_id` (`lab-<UTC>` for lab sessions, `day-<UTC>` for field collections).
2. Sessions are chronologically ordered by start timestamp.
3. Whole sessions are assigned to the `test` split starting from the newest sessions until at least 20% of dataset rows belong to `test`. The remainder form the `train` split.
4. If only a single session exists, a fallback **run-level splitting** strategy is employed: newest complete runs are allocated to `test` until reaching the 20% quota, and an explicit warning is printed and logged indicating that results will be optimistic.
5. Under no circumstances may rows from the same run or session be partitioned across both `train` and `test`.

## Consequences

- Completely eliminates feature leakage across overlapping rolling time windows.
- Tests genuine generalization to future sessions and temporal shifts.
- Single-session fallback clearly alerts developers of potential evaluation optimism.

