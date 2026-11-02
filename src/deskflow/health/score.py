"""Health score calculation derived from predicted performance slowdown ratio.

Constants and thresholds are strictly documented:
- Predicted slowdown ratio is clipped to [0.5, 5.0].
- Score is 100 when ratio <= 1.0 (at or faster than baseline).
- Score falls linearly to 0 at ratio 3.0 (3x slower than baseline).
- Labels:
    - Good: score >= 80
    - Fair: 50 <= score <= 79
    - Poor: score < 50
"""

from typing import Final

# Documented constants
MIN_SLOWDOWN_RATIO: Final[float] = 0.5
MAX_SLOWDOWN_RATIO: Final[float] = 5.0
HEALTHY_SLOWDOWN_RATIO: Final[float] = 1.0
DEGRADED_SLOWDOWN_RATIO: Final[float] = 3.0

LABEL_GOOD_THRESHOLD: Final[int] = 80
LABEL_FAIR_THRESHOLD: Final[int] = 50


def compute_health_score(predicted_ratio: float) -> tuple[int, str]:
    """Compute 0-100 health score and qualitative label from predicted slowdown ratio.

    Args:
        predicted_ratio: The predicted total probe latency ratio relative to baseline.

    Returns:
        tuple of (integer_score, label_string) where label is 'Good', 'Fair', or 'Poor'.
    """
    clipped = min(max(float(predicted_ratio), MIN_SLOWDOWN_RATIO), MAX_SLOWDOWN_RATIO)

    if clipped <= HEALTHY_SLOWDOWN_RATIO:
        raw_score = 100.0
    elif clipped >= DEGRADED_SLOWDOWN_RATIO:
        raw_score = 0.0
    else:
        # Linear degradation from 100.0 at ratio 1.0 to 0.0 at ratio 3.0
        range_span = DEGRADED_SLOWDOWN_RATIO - HEALTHY_SLOWDOWN_RATIO
        raw_score = 100.0 * (DEGRADED_SLOWDOWN_RATIO - clipped) / range_span

    score = int(round(raw_score))
    score = max(0, min(100, score))

    if score >= LABEL_GOOD_THRESHOLD:
        label = "Good"
    elif score >= LABEL_FAIR_THRESHOLD:
        label = "Fair"
    else:
        label = "Poor"

    return score, label
