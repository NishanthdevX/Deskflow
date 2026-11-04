"""Tests for Phase 8 live accuracy watch, horizon pairing, and instructor reporting."""

import re
from pathlib import Path

from rich.console import Console
from typer.testing import CliRunner

from deskflow.commands.report import app as report_app
from deskflow.commands.watch import (
    PredictionPairTracker,
    render_watch_layout,
)
from deskflow.commands.watch import (
    app as watch_app,
)

runner = CliRunner()


def test_watch_horizon_pairing_buffer() -> None:
    """Verify zero lookahead and exact horizon pairing at t + 20s."""
    tracker = PredictionPairTracker(horizon_s=20.0, tolerance_s=2.5)

    # 1. Prediction made at t = 100.0s, due at t = 120.0s
    tracker.add_prediction(
        t_mono=100.0,
        t_utc="2026-11-04T06:00:00Z",
        pred_xgb=1.50,
        pred_rf=1.45,
        pred_ridge=1.40,
        persistence_ratio=1.10,
        pred_tag="cpu_pressure",
        top_factor="CPU busy",
    )
    assert len(tracker.pending) == 1
    assert len(tracker.resolved) == 0

    # 2. Probe at t = 105.0s (too early, 15s remaining)
    res_early = tracker.process_probe(
        t_probe_mono=105.0,
        t_probe_utc="2026-11-04T06:00:05Z",
        actual_ratio=1.20,
    )
    assert len(res_early) == 0
    assert len(tracker.pending) == 1
    assert len(tracker.resolved) == 0

    # 3. Probe at t = 120.0s (exact horizon match)
    res_match = tracker.process_probe(
        t_probe_mono=120.0,
        t_probe_utc="2026-11-04T06:00:20Z",
        actual_ratio=1.60,
    )
    assert len(res_match) == 1
    assert len(tracker.pending) == 0
    assert len(tracker.resolved) == 1

    pair = tracker.resolved[0]
    assert pair["pred_xgb"] == 1.50
    assert pair["actual_ratio"] == 1.60
    assert abs(pair["abs_error"] - 0.10) < 1e-6
    assert pair["within_tol"] is True
    assert abs(pair["persistence_error"] - 0.50) < 1e-6

    # 4. Leaderboard metrics
    leaderboard = tracker.compute_leaderboard()
    assert len(leaderboard) == 4
    xgb_row = next(r for r in leaderboard if r["model"] == "XGBoost")
    assert xgb_row["n"] == 1
    assert abs(xgb_row["mae"] - 0.10) < 1e-6
    assert xgb_row["tol_pct"] == 100.0
    assert abs(xgb_row["delta_mae"] - (-0.40)) < 1e-6


def test_watch_rendering_at_widths() -> None:
    """Verify watch dashboard renders at widths 80, 100, 120 without errors."""
    tracker = PredictionPairTracker()
    tracker.add_prediction(
        t_mono=100.0,
        t_utc="2026-11-04T06:00:00Z",
        pred_xgb=1.25,
        pred_rf=1.20,
        pred_ridge=1.15,
        persistence_ratio=1.0,
        pred_tag="healthy",
        top_factor="metrics normal",
    )
    tracker.process_probe(120.0, "2026-11-04T06:00:20Z", 1.30)

    for width in (80, 100, 120):
        console = Console(record=True, width=width)
        panel = render_watch_layout(
            elapsed_s=30.0,
            total_duration_s=90.0,
            power_state="battery",
            model_name="XGBoost",
            is_synthetic=True,
            tracker=tracker,
            current_mono=125.0,
            plain=False,
        )
        console.print(panel)
        text = console.export_text()
        assert "LIVE ACCURACY WATCH" in text
        assert "PAIRS: 1" in text
        # Verify no emojis
        assert not re.search(r"[\U00010000-\U0010ffff]", text)


def test_watch_plain_rendering_is_ascii() -> None:
    """Verify --plain output of watch dashboard is strictly ASCII."""
    tracker = PredictionPairTracker()
    tracker.add_prediction(
        100.0,
        "2026-11-04T06:00:00Z",
        1.25,
        1.20,
        1.15,
        1.0,
        "healthy",
        "metrics normal",
    )
    tracker.process_probe(120.0, "2026-11-04T06:00:20Z", 1.30)

    for width in (80, 100, 120):
        console = Console(record=True, width=width, no_color=True)
        panel = render_watch_layout(
            elapsed_s=30.0,
            total_duration_s=90.0,
            power_state="battery",
            model_name="XGBoost",
            is_synthetic=True,
            tracker=tracker,
            current_mono=125.0,
            plain=True,
        )
        console.print(panel)
        text = console.export_text()
        assert text.isascii()


def test_watch_replay_fast() -> None:
    """Test watch replay with sample run fixture in fast mode."""
    fixture_path = Path(__file__).resolve().parent / "fixtures" / "sample_run.jsonl"
    assert fixture_path.exists()

    result = runner.invoke(
        watch_app,
        ["--replay", str(fixture_path), "--fast", "--plain"],
    )
    assert result.exit_code == 0
    assert "Final Live Accuracy Leaderboard:" in result.output
    assert "XGBoost" in result.output


def test_report_command(tmp_path: Path) -> None:
    """Test deskflow report output and markdown report generation."""
    out_md = tmp_path / "test_report.md"
    result = runner.invoke(
        report_app,
        ["--out", str(out_md), "--plain"],
    )
    assert result.exit_code == 0
    assert "1. Machine & Baseline Summary" in result.output
    assert "2. Offline Hold-Out Model Performance" in result.output
    assert out_md.exists()
    content = out_md.read_text(encoding="utf-8")
    assert "# Deskflow Instructor Verification Report" in content
