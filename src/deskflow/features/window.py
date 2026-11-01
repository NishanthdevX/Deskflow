"""Window math utilities and statistical aggregations for time series features."""

import math
import statistics


def compute_mean(values: list[float]) -> float:
    """Compute arithmetic mean of values."""
    if not values:
        return float("nan")
    clean = [v for v in values if not math.isnan(v)]
    if not clean:
        return float("nan")
    return float(statistics.mean(clean))


def compute_std(values: list[float]) -> float:
    """Compute sample standard deviation of values (0.0 if len <= 1)."""
    if not values:
        return float("nan")
    clean = [v for v in values if not math.isnan(v)]
    if len(clean) <= 1:
        return 0.0
    return float(statistics.stdev(clean))


def compute_max(values: list[float]) -> float:
    """Compute maximum value."""
    if not values:
        return float("nan")
    clean = [v for v in values if not math.isnan(v)]
    if not clean:
        return float("nan")
    return float(max(clean))


def compute_min(values: list[float]) -> float:
    """Compute minimum value."""
    if not values:
        return float("nan")
    clean = [v for v in values if not math.isnan(v)]
    if not clean:
        return float("nan")
    return float(min(clean))


def compute_drop(values: list[float]) -> float:
    """Compute drop: maximum value minus last value."""
    if not values:
        return float("nan")
    clean = [v for v in values if not math.isnan(v)]
    if not clean:
        return float("nan")
    return float(max(clean) - clean[-1])


def compute_slope(timestamps: list[float], values: list[float]) -> float:
    """Compute least-squares slope (change per second) against timestamps.

    Returns 0.0 if fewer than 2 valid points or timestamp variance is 0.
    """
    if len(timestamps) != len(values) or len(timestamps) < 2:
        return 0.0

    valid_pts = [(t, v) for t, v in zip(timestamps, values, strict=False) if not math.isnan(v)]
    if len(valid_pts) < 2:
        return 0.0

    ts = [p[0] for p in valid_pts]
    ys = [p[1] for p in valid_pts]

    mean_t = statistics.mean(ts)
    mean_y = statistics.mean(ys)

    numerator = sum((t - mean_t) * (y - mean_y) for t, y in zip(ts, ys, strict=False))
    denominator = sum((t - mean_t) ** 2 for t in ts)

    if denominator <= 1e-12:
        return 0.0

    return float(numerator / denominator)

