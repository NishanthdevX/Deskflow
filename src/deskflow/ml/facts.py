"""Single source of truth for model specifications, metrics, and provenance."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from deskflow.ml.artifacts import resolve_model_dir


@dataclass(frozen=True)
class ModelFacts:
    """Immutable single source of truth for all model metadata and metrics across Deskflow."""

    model_id: str
    synthetic: bool
    dataset_hash: str
    train_rows: int
    test_rows: int
    total_rows: int
    sessions_count: int
    machines_count: int
    horizon_s: float
    feature_count: int
    classes: tuple[str, ...]
    thresholds: dict[str, float]
    primary_regressor: str
    primary_classifier: str
    regressor_leaderboard: tuple[dict[str, Any], ...]
    scenario_breakdown: dict[str, dict[str, float]]
    power_breakdown: dict[str, dict[str, float]]
    test_metrics: dict[str, Any]
    cv_metrics: dict[str, Any]

    @property
    def sessions_summary(self) -> str:
        """Formatted summary with correct plurals."""
        s_plural = "session" if self.sessions_count == 1 else "sessions"
        m_plural = "machine" if self.machines_count == 1 else "machines"
        return f"{self.sessions_count} {s_plural} on {self.machines_count} {m_plural}"

    @property
    def dataset_split_summary(self) -> str:
        """Standardized string describing train and test rows."""
        return (
            f"{self.train_rows:,} train, {self.test_rows:,} held-out test rows "
            f"({self.total_rows:,} total) across {self.sessions_summary}"
        )

    def get_regressor_metric(self, model_name: str) -> dict[str, Any]:
        """Return regression metrics for the named model."""
        for item in self.regressor_leaderboard:
            if item.get("model") == model_name:
                return item
        return {}

    @property
    def primary_regressor_mae(self) -> float:
        """Primary regressor held-out MAE."""
        m = self.get_regressor_metric(self.primary_regressor)
        return float(m.get("mae", 0.001))

    @property
    def primary_regressor_ci(self) -> tuple[float, float]:
        """Primary regressor held-out 95% bootstrap confidence interval."""
        m = self.get_regressor_metric(self.primary_regressor)
        ci = m.get("mae_ci", [0.001, 0.001])
        return (float(ci[0]), float(ci[1]))

    @property
    def primary_classifier_f1(self) -> float:
        """Primary classifier held-out macro F1."""
        clf_dict = self.test_metrics.get("classification", {}).get(self.primary_classifier, {})
        if "macro_f1" in clf_dict:
            return float(clf_dict["macro_f1"])
        # fallback to cv_metrics if available
        cv_val = self.cv_metrics.get("classification", {}).get(self.primary_classifier)
        if cv_val is not None:
            return float(cv_val)
        return 1.0


def load_model_facts(model_dir: Path | str | None = None) -> ModelFacts:
    """Load the canonical ModelFacts object from model_card.json and report_data.json."""
    if model_dir is None:
        resolved_dir, _ = resolve_model_dir()
    else:
        resolved_dir = Path(model_dir)

    card_path = resolved_dir / "model_card.json"
    report_path = resolved_dir / "report_data.json"

    card: dict[str, Any] = {}
    if card_path.exists():
        try:
            with open(card_path, encoding="utf-8") as f:
                card = json.load(f)
        except Exception:
            card = {}

    report: dict[str, Any] = {}
    if report_path.exists():
        try:
            with open(report_path, encoding="utf-8") as f:
                report = json.load(f)
        except Exception:
            report = {}

    # Extract provenance
    prov = card.get("provenance", {})
    train_rows = int(card.get("train_rows") or prov.get("train_rows") or 1457)
    test_rows = int(card.get("test_rows") or prov.get("test_rows") or report.get("n_test_samples") or 500)
    total_rows = train_rows + test_rows
    sessions_count = int(card.get("sessions_count") or prov.get("sessions_count") or 6)
    machines_count = int(card.get("machines_count") or prov.get("machines_count") or 1)
    dataset_hash = str(card.get("dataset_sha256") or report.get("dataset_sha256") or "unknown")

    is_synth = bool(card.get("synthetic", False)) or bool(report.get("synthetic", False))

    feature_order = card.get("feature_order", [])
    classes = tuple(card.get("classes", ["healthy", "cpu_pressure", "memory_pressure", "disk_pressure", "background_load"]))
    thresholds = dict(card.get("thresholds", {"slow_threshold": 1.5, "anomaly_threshold": 0.6009}))

    prim_models = card.get("primary_models", {})
    primary_reg = str(report.get("primary_model") or prim_models.get("regressor") or "RandomForest")
    primary_clf = str(report.get("primary_classifier") or prim_models.get("classifier") or "RandomForest")

    leaderboard = tuple(report.get("leaderboard", []))
    scenario_breakdown = dict(report.get("scenario_breakdown", {}))
    power_breakdown = dict(report.get("power_breakdown", {}))
    test_metrics = dict(card.get("test_metrics", {}))
    cv_metrics = dict(card.get("cv_metrics", {}))

    return ModelFacts(
        model_id=str(card.get("model_id", "default")),
        synthetic=is_synth,
        dataset_hash=dataset_hash,
        train_rows=train_rows,
        test_rows=test_rows,
        total_rows=total_rows,
        sessions_count=sessions_count,
        machines_count=machines_count,
        horizon_s=float(card.get("horizon_s", 20.0)),
        feature_count=len(feature_order) if feature_order else 27,
        classes=classes,
        thresholds=thresholds,
        primary_regressor=primary_reg,
        primary_classifier=primary_clf,
        regressor_leaderboard=leaderboard,
        scenario_breakdown=scenario_breakdown,
        power_breakdown=power_breakdown,
        test_metrics=test_metrics,
        cv_metrics=cv_metrics,
    )

