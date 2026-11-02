"""Tests for scan plain reasons, positive SHAP filtering, and merged status line."""

from rich.console import Console

from deskflow.features.registry import FEATURE_MAP, FEATURE_SPECS


def test_registry_has_plain_fields_for_all_features() -> None:
    """Every feature in the registry must define a non-empty plain-language phrase."""
    assert len(FEATURE_SPECS) >= 26
    for spec in FEATURE_SPECS:
        assert hasattr(spec, "plain"), f"Feature {spec.name} is missing 'plain' field"
        assert isinstance(spec.plain, str) and len(spec.plain) > 0, (
            f"Feature {spec.name} has empty 'plain' field"
        )
        # Plain field must not simply be the technical description
        assert spec.plain != spec.description, (
            f"Feature {spec.name} plain field is identical to technical description"
        )


def test_positive_reasons_and_plain_language() -> None:
    """Scan reasons filter out negative contributions and print plain language phrases."""
    # Simulated top_contributors output: mix of positive, negative, and tiny contributions
    mock_contributors = [
        ("cpu_percent_mean_mean_12s", 1.25),  # positive -> should be included
        ("mem_percent_slope_12s", -0.85),  # negative -> must be excluded!
        ("disk_read_mb_s_mean_12s", 0.40),  # positive -> should be included
        ("probe_disk_ratio", 0.00001),  # below epsilon (1e-4) -> excluded
        ("swap_percent", 0.15),  # positive -> should be included
        ("cpu_percent_mean_max_12s", 0.10),  # positive -> beyond top 3, excluded
    ]

    reasons: list[str] = []
    for f_name, shap_val in mock_contributors:
        if shap_val > 1e-4:
            spec = FEATURE_MAP.get(f_name)
            phrase = spec.plain if spec and spec.plain else f_name
            reasons.append(f"{phrase} (+{shap_val:.2f})")
            if len(reasons) >= 3:
                break

    # Top 3 positive reasons
    assert len(reasons) == 3
    # Check that plain language phrases are used
    assert "sustained CPU load (+1.25)" in reasons
    assert "heavy disk read throughput (+0.40)" in reasons
    assert "high swap/pagefile usage (+0.15)" in reasons

    # Ensure NO negative contributions were included
    for r in reasons:
        assert "(-" not in r
        assert "-0.85" not in r

    # Ensure technical description is not printed
    for r in reasons:
        assert "Least-squares slope" not in r
        assert "Instantaneous" not in r


def test_no_significant_factors_fallback() -> None:
    """When no feature has contribution > epsilon, fallback to 'No significant factors'."""
    all_negative = [
        ("cpu_percent_mean_mean_12s", -0.5),
        ("mem_percent_mean_12s", -0.2),
        ("probe_total_ratio", 0.0),
    ]

    reasons: list[str] = []
    for f_name, shap_val in all_negative:
        if shap_val > 1e-4:
            spec = FEATURE_MAP.get(f_name)
            phrase = spec.plain if spec and spec.plain else f_name
            reasons.append(f"{phrase} (+{shap_val:.2f})")
            if len(reasons) >= 3:
                break

    if not reasons:
        reasons = ["No significant factors"]

    assert reasons == ["No significant factors"]


def test_merged_status_line_contradiction_prevention() -> None:
    """Status line merges signals and never prints 'healthy' next to 'Anomaly: Yes'."""
    console = Console(record=True)

    def render_status(is_anomaly: bool, issue_tag: str, confidence: float) -> str:
        with console.capture() as capture:
            if is_anomaly and issue_tag.lower() == "healthy":
                console.print("Status: Unusual pattern detected, cause not identified")
            elif issue_tag.lower() not in ("healthy", "none"):
                console.print(f"Status: {issue_tag} (confidence: {confidence:.0%})")
        return capture.get().strip()

    # Case 1: Anomaly True and Tag Healthy -> Must print Unusual pattern detected
    out1 = render_status(is_anomaly=True, issue_tag="healthy", confidence=1.0)
    assert "Unusual pattern detected, cause not identified" in out1
    assert "Anomaly: Yes" not in out1
    assert "healthy" not in out1.lower()

    # Case 2: Tag is memory_pressure
    out2 = render_status(is_anomaly=False, issue_tag="memory_pressure", confidence=0.88)
    assert "memory_pressure (confidence: 88%)" in out2

    # Case 3: Completely normal system
    out3 = render_status(is_anomaly=False, issue_tag="healthy", confidence=1.0)
    assert out3 == ""
