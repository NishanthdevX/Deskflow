"""Health-score subpackage.

Derives a 0-100 health score from predicted probe slowdown relative to
this machine's own healthy baseline.
"""

from deskflow.health.score import (
    DEGRADED_SLOWDOWN_RATIO,
    HEALTHY_SLOWDOWN_RATIO,
    LABEL_FAIR_THRESHOLD,
    LABEL_GOOD_THRESHOLD,
    MAX_SLOWDOWN_RATIO,
    MIN_SLOWDOWN_RATIO,
    compute_health_score,
)

__all__ = [
    "DEGRADED_SLOWDOWN_RATIO",
    "HEALTHY_SLOWDOWN_RATIO",
    "LABEL_FAIR_THRESHOLD",
    "LABEL_GOOD_THRESHOLD",
    "MAX_SLOWDOWN_RATIO",
    "MIN_SLOWDOWN_RATIO",
    "compute_health_score",
]
