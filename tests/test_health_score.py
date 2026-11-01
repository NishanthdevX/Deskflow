"""Unit tests for health score computation, bounds, and monotonicity."""

from deskflow.health.score import compute_health_score


def test_health_score_bounds_and_labels() -> None:
    # Under or equal to baseline
    score, label = compute_health_score(0.2)  # Below min (clips to 0.5)
    assert score == 100
    assert label == "Good"

    score, label = compute_health_score(1.0)
    assert score == 100
    assert label == "Good"

    # Moderate degradation
    score, label = compute_health_score(1.4)  # 100 - 50 * 0.4 = 80
    assert score == 80
    assert label == "Good"

    score, label = compute_health_score(1.5)  # 100 - 50 * 0.5 = 75
    assert score == 75
    assert label == "Fair"

    score, label = compute_health_score(2.0)  # 100 - 50 * 1.0 = 50
    assert score == 50
    assert label == "Fair"

    score, label = compute_health_score(2.1)  # 100 - 50 * 1.1 = 45
    assert score == 45
    assert label == "Poor"

    # Severe degradation
    score, label = compute_health_score(3.0)
    assert score == 0
    assert label == "Poor"

    score, label = compute_health_score(6.0)  # Above max (clips to 5.0)
    assert score == 0
    assert label == "Poor"


def test_health_score_monotonicity() -> None:
    """Assert score is monotonically non-increasing as slowdown ratio increases."""
    ratios = [0.1 * i for i in range(5, 55)]
    scores = [compute_health_score(r)[0] for r in ratios]

    for i in range(len(scores) - 1):
        assert scores[i] >= scores[i + 1], (
            f"Monotonicity violated at {ratios[i]} -> {ratios[i + 1]}"
        )
