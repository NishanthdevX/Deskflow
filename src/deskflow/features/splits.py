"""Session-level and run-level dataset splitting without temporal leakage."""

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)


def assign_splits(df: "pd.DataFrame", test_ratio: float = 0.20) -> tuple["pd.DataFrame", str]:
    """Assign train/test splits at the session level (fallback to run level if only 1 session).

    Whole sessions/runs are assigned to test starting from the newest until test_ratio is met.
    A test run never has rows in train, preventing overlapping window leakage.

    Returns:
        (DataFrame with 'split' column containing 'train' or 'test', strategy_used)
    """
    if df.empty:
        df["split"] = "train"
        return df, "empty"

    df = df.copy()
    total_rows = len(df)
    min_test_rows = int(total_rows * test_ratio)

    # Check sessions
    sessions = df["session_id"].unique().tolist()
    if len(sessions) >= 2:
        strategy = "session"
        # Determine session start times
        session_starts = (
            df.groupby("session_id")["ts_utc"].min().sort_values(ascending=True).index.tolist()
        )
        # Allocate newest sessions to test
        test_sessions: set[str] = set()
        test_row_count = 0

        for s in reversed(session_starts):
            s_rows = int((df["session_id"] == s).sum())
            test_sessions.add(s)
            test_row_count += s_rows
            if test_row_count >= min_test_rows:
                break

        # If all sessions got assigned to test, keep at least one train
        if len(test_sessions) == len(session_starts) and len(session_starts) > 1:
            test_sessions.remove(session_starts[0])

        df["split"] = df["session_id"].apply(lambda s: "test" if s in test_sessions else "train")
        return df, strategy

    # Fallback to run-level splitting
    strategy = "run"
    logger.warning(
        "Only 1 session detected. Falling back to run-level splitting. "
        "Results will be optimistic due to shared machine state across runs."
    )

    run_starts = df.groupby("run_id")["ts_utc"].min().sort_values(ascending=True).index.tolist()

    test_runs: set[str] = set()
    test_row_count = 0

    for r in reversed(run_starts):
        r_rows = int((df["run_id"] == r).sum())
        test_runs.add(r)
        test_row_count += r_rows
        if test_row_count >= min_test_rows:
            break

    if len(test_runs) == len(run_starts) and len(run_starts) > 1:
        test_runs.remove(run_starts[0])

    df["split"] = df["run_id"].apply(lambda r: "test" if r in test_runs else "train")
    return df, strategy
