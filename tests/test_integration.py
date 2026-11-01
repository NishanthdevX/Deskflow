"""Integration tests touching real system APIs.

These tests are marked @pytest.mark.integration and run on both Windows and macOS in CI.
"""

from pathlib import Path

import pytest
from typer.testing import CliRunner

from deskflow.cli import app
from deskflow.collector import Sampler, load_run, run_probe
from deskflow.platform import get_adapter

runner = CliRunner()


@pytest.mark.integration
def test_real_sampler_returns_plausible_values() -> None:
    """The sampler should sample real hardware metrics without raising exceptions."""
    adapter = get_adapter()
    sampler = Sampler(adapter=adapter)

    sample, processes = sampler.sample()

    assert sample.mem_total_bytes > 0
    assert sample.mem_available_bytes > 0
    assert 0.0 <= sample.mem_percent <= 100.0
    assert sample.process_count > 0
    assert len(processes) > 0

    # Ensure processes are sorted descending by RSS
    for i in range(len(processes) - 1):
        assert processes[i].rss_bytes >= processes[i + 1].rss_bytes


@pytest.mark.integration
def test_real_probe_duration_under_500ms(tmp_path: Path) -> None:
    """The real probe should run and complete well under 500ms on real hardware."""
    probe = run_probe(data_dir=tmp_path)
    assert probe.cpu_ns > 0
    assert probe.mem_ns > 0
    assert probe.disk_ns > 0
    assert probe.total_ns > 0
    assert probe.total_ns < 500_000_000


@pytest.mark.integration
def test_real_collect_command_writes_records(tmp_path: Path) -> None:
    """`deskflow collect` must write at least 5 valid records in a 6-second run."""
    out_file = tmp_path / "integration_run.jsonl"
    result = runner.invoke(
        app,
        [
            "collect",
            "--duration",
            "7",
            "--interval",
            "1",
            "--probe-interval",
            "2",
            "--label",
            "integration_test",
            "--out",
            str(out_file),
        ],
    )
    assert result.exit_code == 0
    assert out_file.exists()

    records = load_run(out_file)
    assert len(records) >= 5
    for r in records:
        assert r.label == "integration_test"
        assert r.schema_version == 1
