"""Hard quality gates for model training, release export, and production promotion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

# -----------------------------------------------------------------------------
# Named Constants and Quality Gate Thresholds
# -----------------------------------------------------------------------------

MIN_TRAIN_ROWS: int = 1000
"""Minimum number of rows required in the training split."""

MIN_TEST_ROWS: int = 300
"""Minimum number of rows required in the held-out test split."""

MIN_TEST_SESSIONS: int = 2
"""Minimum number of distinct held-out test sessions."""

MIN_CLASS_ROWS: int = 100
"""Minimum training rows required for every supported issue class."""

REQUIRE_BOTH_POWER_STATES: bool = True
"""Whether both AC and battery power states must be present when both baselines exist."""

ALLOW_NAN_OR_OUT_OF_RANGE: bool = False
"""Disallow any NaN, Inf, or out-of-range (>20.0 or <0.0) predictions."""

VALIDATE_PASSES: bool = True
"""Dataset structural and distribution validation must pass with zero critical errors."""

MIN_VALID_PRED_RATIO: float = 0.0
"""Minimum physically plausible slowdown ratio prediction."""

MAX_VALID_PRED_RATIO: float = 20.0
"""Maximum physically plausible slowdown ratio prediction before clipping."""

SUPPORTED_ISSUE_CLASSES: list[str] = [
    "healthy",
    "cpu_pressure",
    "memory_pressure",
    "disk_pressure",
]
"""Core issue classes required for multi-class fault classification."""


@dataclass
class GateResult:
    """Evaluation result for an individual quality gate."""

    name: str
    required: str
    measured: str
    passed: bool
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert result to dictionary representation."""
        return {
            "name": self.name,
            "required": self.required,
            "measured": self.measured,
            "passed": self.passed,
            "reason": self.reason,
        }


def evaluate_dataset_gates(
    train_rows: int,
    test_rows: int,
    test_sessions_count: int,
    class_counts: dict[str, int],
    power_states_train: list[str],
    has_both_baselines: bool,
    validation_errors: list[str],
) -> list[GateResult]:
    """Evaluate structural and statistical dataset gates prior to model training."""
    results: list[GateResult] = []

    # 1. Train Rows Gate
    pass_train = train_rows >= MIN_TRAIN_ROWS
    results.append(
        GateResult(
            name="Train Rows",
            required=f">={MIN_TRAIN_ROWS:,}",
            measured=f"{train_rows:,}",
            passed=pass_train,
            reason=""
            if pass_train
            else f"Insufficient train rows ({train_rows} < {MIN_TRAIN_ROWS})",
        )
    )

    # 2. Test Rows Gate
    pass_test = test_rows >= MIN_TEST_ROWS
    results.append(
        GateResult(
            name="Test Rows",
            required=f">={MIN_TEST_ROWS:,}",
            measured=f"{test_rows:,}",
            passed=pass_test,
            reason="" if pass_test else f"Insufficient test rows ({test_rows} < {MIN_TEST_ROWS})",
        )
    )

    # 3. Test Sessions Gate
    pass_sessions = test_sessions_count >= MIN_TEST_SESSIONS
    results.append(
        GateResult(
            name="Test Sessions",
            required=f">={MIN_TEST_SESSIONS}",
            measured=f"{test_sessions_count}",
            passed=pass_sessions,
            reason=""
            if pass_sessions
            else f"Insufficient test sessions ({test_sessions_count} < {MIN_TEST_SESSIONS})",
        )
    )

    # 4. Class Balance Gate (at least MIN_CLASS_ROWS per class)
    failing_classes: list[str] = []
    class_details: list[str] = []
    for cls in SUPPORTED_ISSUE_CLASSES:
        cnt = class_counts.get(cls, 0)
        class_details.append(f"{cls}: {cnt}")
        if cnt < MIN_CLASS_ROWS:
            failing_classes.append(f"{cls} ({cnt} < {MIN_CLASS_ROWS})")

    pass_classes = len(failing_classes) == 0
    results.append(
        GateResult(
            name="Class Balance",
            required=f">={MIN_CLASS_ROWS} each",
            measured=", ".join(class_details),
            passed=pass_classes,
            reason="" if pass_classes else f"Classes below minimum: {', '.join(failing_classes)}",
        )
    )

    # 5. Power States Gate
    if has_both_baselines and REQUIRE_BOTH_POWER_STATES:
        has_ac = "ac" in power_states_train
        has_bat = "battery" in power_states_train
        pass_power = has_ac and has_bat
        measured_states = ", ".join(sorted(power_states_train)) or "none"
        results.append(
            GateResult(
                name="Power States",
                required="both ac and battery",
                measured=measured_states,
                passed=pass_power,
                reason=""
                if pass_power
                else f"Missing power states in train split (measured: {measured_states})",
            )
        )
    else:
        results.append(
            GateResult(
                name="Power States",
                required="active baseline state",
                measured=", ".join(sorted(power_states_train)) or "single",
                passed=True,
                reason="",
            )
        )

    # 6. Validation Errors Gate
    pass_val = len(validation_errors) == 0
    results.append(
        GateResult(
            name="Dataset Validation",
            required="0 validation errors",
            measured=f"{len(validation_errors)} error(s)",
            passed=pass_val,
            reason="" if pass_val else "; ".join(validation_errors[:3]),
        )
    )

    return results


def evaluate_prediction_gates(
    test_predictions: dict[str, np.ndarray],
) -> list[GateResult]:
    """Evaluate inference numerical integrity gates on test split predictions."""
    results: list[GateResult] = []

    failing_models: list[str] = []
    max_observed: float = 0.0

    for model_name, preds in test_predictions.items():
        if len(preds) == 0:
            continue

        if np.isnan(preds).any():
            failing_models.append(f"{model_name} (contains NaN)")
        elif np.isinf(preds).any():
            failing_models.append(f"{model_name} (contains Inf)")
        elif np.any(preds < MIN_VALID_PRED_RATIO):
            min_v = float(np.min(preds))
            failing_models.append(f"{model_name} (min {min_v:.2f} < {MIN_VALID_PRED_RATIO})")
        elif np.any(preds > MAX_VALID_PRED_RATIO):
            max_v = float(np.max(preds))
            failing_models.append(f"{model_name} (max {max_v:.2f} > {MAX_VALID_PRED_RATIO})")

        m_max = float(np.max(preds))
        max_observed = max(max_observed, m_max)

    pass_preds = len(failing_models) == 0
    results.append(
        GateResult(
            name="Prediction Integrity",
            required=f"no NaN, range [{MIN_VALID_PRED_RATIO:.1f}, {MAX_VALID_PRED_RATIO:.1f}]",
            measured=f"all finite, max {max_observed:.2f}"
            if pass_preds
            else "; ".join(failing_models),
            passed=pass_preds,
            reason="" if pass_preds else f"Out of range predictions: {'; '.join(failing_models)}",
        )
    )

    return results
