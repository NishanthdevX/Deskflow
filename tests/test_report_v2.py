"""Unit tests for Phase 12: Report v2, terminal charts, HTML export, and 5-step check."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from typer.testing import CliRunner

from deskflow.commands.check import check_command
from deskflow.commands.report import _collect_live_watch_summary
from deskflow.commands.report import app as report_app
from deskflow.report_html import generate_html_report

runner = CliRunner()


def test_report_all_formats(tmp_path: Path) -> None:
    """Test deskflow report execution across terminal, markdown, HTML, and pure JSON."""
    html_out = tmp_path / "test_report.html"
    md_out = tmp_path / "test_report.md"

    # Terminal & Markdown & HTML
    res = runner.invoke(
        report_app,
        ["--html", str(html_out), "--md", str(md_out), "--plain"],
    )
    assert res.exit_code == 0
    assert "1. [ MACHINE ]" in res.output
    assert "2. [ BASELINES ]" in res.output
    assert "3. [ OFFLINE LEADERBOARD ]" in res.output
    assert "4. [ PREDICTED VS ACTUAL ]" in res.output
    assert "5. [ BREAKDOWNS ]" in res.output
    assert "6. [ CONFUSION MATRIX ]" in res.output
    assert "7. [ FEATURE IMPORTANCE ]" in res.output
    assert "8. [ ANOMALY DETECTOR ]" in res.output
    assert "9. [ LIVE ACCURACY ]" in res.output
    assert "10. [ FIX EFFECTIVENESS ]" in res.output
    assert "11. [ LIMITATIONS ]" in res.output

    assert html_out.exists()
    assert md_out.exists()
    assert html_out.stat().st_size > 0
    assert md_out.stat().st_size > 0

    # Pure JSON
    res_json = runner.invoke(report_app, ["--json"])
    assert res_json.exit_code == 0
    data = json.loads(res_json.output)
    assert "leaderboard" in data
    assert "confusion_matrix" in data
    assert "limitations" in data


def test_synthetic_banner_present(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify synthetic model banner is rendered on terminal, Markdown, and HTML."""
    html_out = tmp_path / "banner_test.html"
    md_out = tmp_path / "banner_test.md"

    import deskflow.commands.report as report_cmd
    from deskflow.truth import ModelProvenance

    orig_load_report_data = report_cmd.load_report_data
    orig_load_card = report_cmd.load_model_card
    orig_load_provenance = report_cmd.load_provenance

    monkeypatch.setattr(
        report_cmd,
        "load_report_data",
        lambda p=None: {**orig_load_report_data(p), "synthetic": True},
    )
    monkeypatch.setattr(
        report_cmd,
        "load_model_card",
        lambda p: {**orig_load_card(p), "synthetic": True},
    )

    def _synthetic_provenance(model_dir: object = None) -> ModelProvenance:
        prov = orig_load_provenance(model_dir)  # type: ignore[arg-type]
        return ModelProvenance(
            synthetic=True,
            dataset_hash=prov.dataset_hash,
            train_rows=prov.train_rows,
            test_rows=prov.test_rows,
            sessions_count=prov.sessions_count,
            machines_count=prov.machines_count,
            power_states=prov.power_states,
            created_utc=prov.created_utc,
            git_commit=prov.git_commit,
        )

    monkeypatch.setattr(report_cmd, "load_provenance", _synthetic_provenance)

    res = runner.invoke(
        report_app,
        ["--html", str(html_out), "--md", str(md_out), "--plain"],
    )
    assert res.exit_code == 0
    expected_banner = "DEMO MODEL (SYNTHETIC DATA - NOT A REAL WORLD RESULT)"

    # Terminal
    assert expected_banner in res.output

    # Markdown
    md_content = md_out.read_text(encoding="utf-8")
    assert expected_banner in md_content

    # HTML
    html_content = html_out.read_text(encoding="utf-8")
    assert expected_banner in html_content


def test_invalid_watch_sessions_excluded(tmp_path: Path) -> None:
    """Verify out-of-range or NaN watch sessions are excluded and counted."""
    watch_dir = tmp_path / "watch"
    watch_dir.mkdir()

    # Valid session
    valid_session = {
        "leaderboard": [
            {"model": "XGBoost", "mae": 0.22, "n": 20},
            {"model": "Persistence", "mae": 0.25, "n": 20},
        ]
    }
    (watch_dir / "watch-valid.json").write_text(json.dumps(valid_session), encoding="utf-8")

    # Invalid session with massive MAE > 100
    invalid_session = {
        "leaderboard": [
            {"model": "Ridge", "mae": 18234500.0, "n": 20},
        ]
    }
    (watch_dir / "watch-invalid-high.json").write_text(
        json.dumps(invalid_session), encoding="utf-8"
    )

    # Invalid session with negative MAE
    invalid_neg = {
        "leaderboard": [
            {"model": "XGBoost", "mae": -0.5, "n": 10},
        ]
    }
    (watch_dir / "watch-invalid-neg.json").write_text(json.dumps(invalid_neg), encoding="utf-8")

    v_cnt, ex_cnt, stats, msg = _collect_live_watch_summary(watch_dir)
    assert v_cnt == 1
    assert ex_cnt == 2
    assert "XGBoost" in stats
    assert stats["XGBoost"]["mae"] == 0.22


def test_html_self_contained_and_under_500kb(tmp_path: Path) -> None:
    """Verify HTML export contains no external URLs, has print styles, and stays < 500 KB."""
    html_file = tmp_path / "standalone.html"
    sample_data = {
        "synthetic": False,
        "machine": {"platform": "Windows 11", "cpu_model": "Test CPU", "total_ram_gb": 16},
        "baselines": {},
        "leaderboard": [
            {
                "model": "XGBoost",
                "mae": 0.075,
                "mae_ci": [0.071, 0.079],
                "rmse": 0.095,
                "r2": 0.984,
                "tolerance_accuracy": 0.99,
                "is_baseline": False,
                "beats_persistence": True,
            },
            {
                "model": "Persistence",
                "mae": 0.158,
                "mae_ci": [0.151, 0.164],
                "rmse": 0.187,
                "r2": 0.938,
                "tolerance_accuracy": 0.81,
                "is_baseline": True,
                "beats_persistence": False,
            },
        ],
        "scenario_breakdown": {"idle": {"XGBoost": 0.04, "Persistence": 0.06}},
        "power_breakdown": {"ac": {"XGBoost": 0.07, "Persistence": 0.15}},
        "confusion_matrix": {"classes": ["healthy", "cpu"], "matrix": [[10, 0], [0, 10]]},
        "feature_importance": [{"name": "probe_ratio", "weight": 0.95, "normalized": 1.0}],
        "anomaly": {"roc_auc": 1.0, "recall": 1.0, "false_alarm_rate": 0.02, "threshold": 0.54},
        "time_series": [{"actual": 1.0, "predicted": 1.02, "persistence": 1.0}],
        "watch_summary": {"valid_sessions": 1, "excluded_sessions": 0, "text": "Aggregated"},
        "fix_summary": {"total_runs": 0, "text": "None"},
        "limitations": ["Test limitation"],
    }

    generate_html_report(sample_data, html_file)
    assert html_file.exists()

    size_bytes = html_file.stat().st_size
    assert size_bytes < 500 * 1024  # Strict 500 KB limit
    assert size_bytes < 60 * 1024  # In practice under 60 KB

    content = html_file.read_text(encoding="utf-8")
    assert "@media print" in content
    assert "<svg" in content
    assert "How to read this report:" in content

    # Check for external URLs (excluding XML namespaces like xmlns="http://www.w3.org/2000/svg")
    raw_urls = re.findall(r'https?://[^\s"\'<>]+', content)
    external_urls = [u for u in raw_urls if "w3.org" not in u]
    assert external_urls == [], f"Found disallowed external assets: {external_urls}"


def test_golden_renders_width_responsiveness() -> None:
    """Verify report rendering at terminal widths 80, 100, and 120."""
    for width in (80, 100, 120):
        for plain in (True, False):
            # Invoke report with simulated terminal width
            res = runner.invoke(
                report_app,
                ["--plain"] if plain else [],
                env={"COLUMNS": str(width)},
            )
            assert res.exit_code == 0
            assert len(res.output) > 200


def test_check_flow_guided(tmp_path: Path) -> None:
    """Verify guided 5-step check command executes with simulated user inputs."""
    prompts_seen = []

    def fake_input(prompt: str) -> str:
        prompts_seen.append(prompt)
        return ""  # Simulates pressing Enter or default choice

    html_path = tmp_path / "check_report.html"
    check_command(
        plain=True,
        yes=False,
        skip_watch=True,  # Skip 60s wait in unit test
        no_input=False,
        html=html_path,
        scan_seconds=1,
        input_fn=fake_input,
    )

    assert any("Health Scan" in p for p in prompts_seen)
    assert any("Fix Center" in p for p in prompts_seen)
    assert html_path.exists()
