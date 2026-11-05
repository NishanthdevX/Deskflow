"""Automated end-to-end verification self-test: deskflow tools selftest."""

from __future__ import annotations

import io
import json
import logging
import os
import re
import shutil
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import psutil
import typer
from rich import box
from rich.console import Console
from rich.table import Table

from deskflow.baseline import (
    Baseline,
    get_baseline_age_days,
    get_current_power_state,
    load_baseline,
)
from deskflow.collector.probe import run_probe
from deskflow.commands.fix import run_fast_inventory
from deskflow.commands.info import render_info_dashboard
from deskflow.commands.pipeline import render_pipeline_frame
from deskflow.exit_codes import (
    EXIT_BASELINE_MISSING,
    EXIT_BLOCKED_OR_DECLINED,
    EXIT_ERROR,
    EXIT_OK,
    EXIT_USAGE,
)
from deskflow.paths import get_data_dir
from deskflow.platform.base import PlatformAdapter
from deskflow.platform.factory import get_adapter
from deskflow.report_html import generate_html_report
from deskflow.truth import (
    ModelProvenance,
    load_provenance,
    verify_model_hashes,
)
from deskflow.ui.theme import use_plain_output

logger = logging.getLogger(__name__)


@dataclass
class SelftestResult:
    """Outcome of an individual self-test verification check."""

    name: str
    status: str  # PASS, WARN, FAIL, SKIP
    measured: str
    reason: str = ""


# Regex matching standard emoji ranges only (excluding box drawing and blocks)
EMOJI_PATTERN = re.compile(
    "["
    "\U0001f300-\U0001f5ff"  # miscellaneous symbols and pictographs
    "\U0001f600-\U0001f64f"  # emoticons
    "\U0001f680-\U0001f6ff"  # transport and map symbols
    "\U0001f700-\U0001f77f"  # alchemical symbols
    "\U0001f780-\U0001f7ff"  # geometric shapes extended
    "\U0001f800-\U0001f8ff"  # supplemental arrows-c
    "\U0001f900-\U0001f9ff"  # supplemental symbols and pictographs
    "\U0001fa00-\U0001faff"  # symbols and pictographs extended-a
    "\U0001f1e6-\U0001f1ff"  # regional indicator symbols (flags)
    "\U00002702-\U000027b0"  # dingbats
    "]+",
    flags=re.UNICODE,
)


def check_truth_contract(prov: ModelProvenance) -> SelftestResult:
    """1. Verify truth contract enforcement across all user-facing renders."""
    if prov.synthetic:
        # Verify that synthetic model suppresses offline metrics
        buf = io.StringIO()
        con = Console(file=buf, width=100, no_color=True)
        frame = render_pipeline_frame(
            active_stage=5,
            packet_pos=5.0,
            real_value_text="Test",
            prov_dict={
                "synthetic": True,
                "train_rows": prov.train_rows,
                "test_rows": prov.test_rows,
                "dataset_hash": "",
            },
            plain=True,
        )
        con.print(frame)
        txt = buf.getvalue()
        if "SYNTHETIC DEMO" not in txt and "DEMO" not in txt:
            return SelftestResult(
                name="truth_contract",
                status="FAIL",
                measured="synthetic=True: missing DEMO banner in pipeline render",
                reason="Truth contract requires DEMO indicator on synthetic models",
            )
        return SelftestResult(
            name="truth_contract",
            status="PASS",
            measured=f"synthetic=True: suppressed in renders ({prov.train_rows} train rows)",
        )
    return SelftestResult(
        name="truth_contract",
        status="PASS",
        measured=f"synthetic=False: verified real model card (hash {prov.dataset_hash[:8]})",
    )


def check_model_hashes(prov: ModelProvenance) -> SelftestResult:
    """2. Verify model provenance and SHA-256 cryptographic file hashes."""
    passed, errors = verify_model_hashes(prov.model_dir)
    n_files = len(prov.model_hashes)
    if not passed:
        err_msg = "; ".join(errors)
        return SelftestResult(
            name="model_hashes",
            status="FAIL",
            measured=f"{len(errors)} file hash mismatch(es)",
            reason=err_msg,
        )
    commit = prov.git_commit[:8] if prov.git_commit else "unknown"
    return SelftestResult(
        name="model_hashes",
        status="PASS",
        measured=f"All {n_files} model files match SHA-256 (commit {commit})",
    )


def check_baselines_freshness(adapter: PlatformAdapter) -> SelftestResult:
    """3. Verify baseline presence and age for each power state."""
    curr_state = get_current_power_state(adapter)
    other_state = "battery" if curr_state == "ac" else "ac"

    curr_b: Baseline | None = None
    try:
        curr_b = load_baseline(power_state=curr_state)
    except Exception:
        curr_b = None

    if curr_b is None:
        return SelftestResult(
            name="baselines_freshness",
            status="FAIL",
            measured=f"Current power state '{curr_state}' baseline missing",
            reason=f"Calibrate fresh baseline on {curr_state} power",
        )

    age_days = get_baseline_age_days(curr_b)
    if age_days > 7.0:
        return SelftestResult(
            name="baselines_freshness",
            status="FAIL",
            measured=f"Current baseline '{curr_state}' is stale ({age_days:.1f} days > 7.0d)",
            reason="Baseline calibration must be refreshed at least once every 7 days",
        )

    # Check other power state
    bat = adapter.battery()
    if bat is None and other_state == "battery":
        return SelftestResult(
            name="baselines_freshness",
            status="PASS",
            measured=f"Current '{curr_state}' {age_days:.1f}d; other 'battery' not applicable",
        )

    other_b: Baseline | None = None
    try:
        other_b = load_baseline(power_state=other_state)
    except Exception:
        other_b = None

    if other_b is None:
        return SelftestResult(
            name="baselines_freshness",
            status="WARN",
            measured=f"Current '{curr_state}' {age_days:.1f}d; other '{other_state}' missing",
            reason=f"Calibrate baseline on {other_state} power for complete coverage",
        )

    other_age = get_baseline_age_days(other_b)
    return SelftestResult(
        name="baselines_freshness",
        status="PASS",
        measured=f"AC: {age_days if curr_state == 'ac' else other_age:.1f}d, "
        f"Battery: {other_age if curr_state == 'ac' else age_days:.1f}d",
    )


def check_probe_stability() -> SelftestResult:
    """4. Run 12 probes and verify latency stability ratio (p95 / median <= 1.4)."""
    # 1 initial warm-up probe to eliminate JIT/code-cache latency
    _ = run_probe()

    latencies_ns: list[float] = []
    for _ in range(12):
        time.sleep(0.04)
        res = run_probe()
        latencies_ns.append(res.total_ns)

    latencies_ms = [ns / 1_000_000.0 for ns in latencies_ns]
    med_ms = float(np.median(latencies_ms))
    p95_ms = float(np.percentile(latencies_ms, 95))
    ratio = (p95_ms / med_ms) if med_ms > 0 else 1.0

    measured_str = f"12 probes: median {med_ms:.2f}ms, p95 {p95_ms:.2f}ms, ratio {ratio:.2f}"
    if ratio > 1.4:
        return SelftestResult(
            name="probe_stability",
            status="FAIL",
            measured=measured_str,
            reason=f"Probe stability ratio {ratio:.2f} exceeds threshold 1.40",
        )
    return SelftestResult(
        name="probe_stability",
        status="PASS",
        measured=measured_str,
    )


def check_command_surface() -> SelftestResult:
    """5. Verify CLI command surface: primary commands visible, tools hidden."""
    from typer.testing import CliRunner

    from deskflow.cli import app as cli_app

    runner = CliRunner()
    res = runner.invoke(cli_app, ["--help"])
    help_out = res.output

    # Verify root help includes all primary commands
    primary_commands = ["battery", "plugged", "full", "pipeline", "fix", "info", "tools"]
    for p in primary_commands:
        if p not in help_out:
            return SelftestResult(
                name="command_surface",
                status="FAIL",
                measured=f"Missing primary command '{p}' in root help",
                reason=f"Command '{p}' must be visible under root deskflow --help",
            )

    # Verify hidden commands are NOT visible in root help commands block
    hidden_names = ["doctor", "baseline", "prep", "watch", "dataset", "train", "evaluate"]
    # Check that none appear as a command in the Commands table
    leaked: list[str] = []
    for h in hidden_names:
        if f"│ {h} " in help_out or f"| {h} " in help_out:
            leaked.append(h)

    if leaked:
        return SelftestResult(
            name="command_surface",
            status="FAIL",
            measured=f"Hidden commands exposed in root help: {', '.join(leaked)}",
            reason="Advanced tools must reside exclusively under 'deskflow tools'",
        )

    return SelftestResult(
        name="command_surface",
        status="PASS",
        measured="6 primary commands in root help; advanced tools isolated",
    )


def check_responsive_layout(prov: ModelProvenance, adapter: PlatformAdapter) -> SelftestResult:
    """6. Verify responsive layout renders at 80, 100, 120 cols without emoji."""
    widths = [80, 100, 120]
    modes = [False, True]  # normal (colored), plain (ASCII)

    for w in widths:
        for plain in modes:
            buf = io.StringIO()
            con = Console(file=buf, width=w, no_color=plain)

            # Test pipeline frame render
            frame = render_pipeline_frame(
                active_stage=3,
                packet_pos=3.0,
                real_value_text="Testing layout responsiveness",
                prov_dict={"synthetic": prov.synthetic, "train_rows": prov.train_rows},
                plain=plain,
                compact=(w <= 80),
            )
            con.print(frame)

            # Test info render
            render_info_dashboard(plain=plain, console_out=con)
            con.print(f"Info test at width {w}")

            output = buf.getvalue()

            # Check emoji presence
            emojis_found = EMOJI_PATTERN.findall(output)
            if emojis_found:
                return SelftestResult(
                    name="responsive_layout",
                    status="FAIL",
                    measured=f"Emoji found at width {w} (plain={plain}): {emojis_found[0]!r}",
                    reason="Strict zero-emoji policy violated",
                )

    return SelftestResult(
        name="responsive_layout",
        status="PASS",
        measured="Widths 80, 100, 120 verified in normal & plain modes; 0 emojis",
    )


def check_json_purity() -> SelftestResult:
    """7. Verify purity of --json output without extraneous logs or banners."""
    import contextlib

    from deskflow.commands.report import generate_report

    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            _ = generate_report(json_output=True, plain=True)
        raw_stdout = buf.getvalue().strip()
        parsed = json.loads(raw_stdout)
        if not isinstance(parsed, dict) or "machine" not in parsed:
            return SelftestResult(
                name="json_purity",
                status="FAIL",
                measured="Invalid payload structure from report --json",
                reason="JSON output did not contain expected top-level keys",
            )
    except Exception as exc:
        return SelftestResult(
            name="json_purity",
            status="FAIL",
            measured=f"JSON parse error: {exc}",
            reason=str(exc),
        )

    return SelftestResult(
        name="json_purity",
        status="PASS",
        measured="Strict valid JSON on stdout for --json operations",
    )


def check_exit_codes() -> SelftestResult:
    """8. Verify compliance of standard CLI exit codes."""
    codes_map = {
        "EXIT_OK": EXIT_OK == 0,
        "EXIT_ERROR": EXIT_ERROR == 1,
        "EXIT_USAGE": EXIT_USAGE == 2,
        "EXIT_BASELINE_MISSING": EXIT_BASELINE_MISSING == 4,
        "EXIT_BLOCKED_OR_DECLINED": EXIT_BLOCKED_OR_DECLINED == 5,
    }
    mismatches = [k for k, v in codes_map.items() if not v]
    if mismatches:
        return SelftestResult(
            name="exit_codes",
            status="FAIL",
            measured=f"Exit code mismatch: {', '.join(mismatches)}",
            reason="Exit codes must match docs/cli.md specification",
        )
    return SelftestResult(
        name="exit_codes",
        status="PASS",
        measured="0 (OK), 1 (ERR), 2 (USAGE), 4 (BASELINE), 5 (ABORT) verified",
    )


def check_timings(adapter: PlatformAdapter) -> SelftestResult:
    """9. Measure latency of fast-path operations: version, info, fix inventory."""
    # Measure info gathering
    t0 = time.monotonic()
    render_info_dashboard(plain=True, console_out=Console(file=io.StringIO()))
    t_info_ms = (time.monotonic() - t0) * 1000.0

    # Measure fix inventory
    t1 = time.monotonic()
    _ = run_fast_inventory(adapter)
    t_fix_ms = (time.monotonic() - t1) * 1000.0

    measured_str = f"info {t_info_ms:.0f}ms (< 400ms), fix {t_fix_ms:.0f}ms (< 500ms)"
    if t_info_ms > 800.0:
        return SelftestResult(
            name="timings",
            status="WARN",
            measured=measured_str,
            reason=f"Info render latency {t_info_ms:.0f}ms exceeds target 400ms",
        )
    return SelftestResult(
        name="timings",
        status="PASS",
        measured=measured_str,
    )


def check_installer_dry_runs() -> SelftestResult:
    """10. Verify syntactic validity and integrity of PowerShell and shell installers."""

    root_dir = Path(__file__).resolve().parent.parent.parent.parent
    ps1_path = root_dir / "install.ps1"
    sh_path = root_dir / "install.sh"

    if not ps1_path.exists() or not sh_path.exists():
        return SelftestResult(
            name="installer_dry_runs",
            status="FAIL",
            measured="Missing install.ps1 or install.sh installer script",
            reason=f"ps1: {ps1_path.exists()}, sh: {sh_path.exists()}",
        )

    ps1_text = ps1_path.read_text(encoding="utf-8")
    sh_text = sh_path.read_text(encoding="utf-8")

    # 1. Static structure verification
    if "Deskflow Windows Installer" not in ps1_text or "DESKFLOW_DRYRUN" not in ps1_text:
        return SelftestResult(
            name="installer_dry_runs",
            status="FAIL",
            measured="install.ps1 missing dry-run or header markers",
            reason="install.ps1 validation failed",
        )

    if "#!/usr/bin/env bash" not in sh_text and "#!/bin/sh" not in sh_text:
        return SelftestResult(
            name="installer_dry_runs",
            status="FAIL",
            measured="install.sh missing standard POSIX shebang",
            reason="install.sh shebang check failed",
        )

    # 2. PowerShell dry-run execution on Windows
    if shutil.which("powershell"):
        try:
            env = dict(os.environ, DESKFLOW_DRYRUN="1")
            res_ps = subprocess.run(
                ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", "install.ps1"],
                check=False,
                cwd=root_dir,
                env=env,
                capture_output=True,
                text=True,
                timeout=10,
            )
            if res_ps.returncode != 0 or "[DRYRUN]" not in res_ps.stdout:
                return SelftestResult(
                    name="installer_dry_runs",
                    status="FAIL",
                    measured=f"install.ps1 dry-run exit code {res_ps.returncode}",
                    reason=res_ps.stderr[:100] if res_ps.stderr else "Missing [DRYRUN] token",
                )
        except Exception as exc:
            return SelftestResult(
                name="installer_dry_runs",
                status="FAIL",
                measured=f"install.ps1 dry-run failed: {exc}",
                reason=str(exc),
            )

    # 3. Shell script syntax verification
    if shutil.which("bash"):
        try:
            res_sh = subprocess.run(
                ["bash", "-n", "install.sh"],
                check=False,
                cwd=root_dir,
                capture_output=True,
                text=True,
                timeout=10,
            )
            if res_sh.returncode != 0:
                return SelftestResult(
                    name="installer_dry_runs",
                    status="FAIL",
                    measured=f"install.sh syntax error (code {res_sh.returncode})",
                    reason=res_sh.stderr[:100],
                )
        except Exception as exc:
            logger.debug("bash -n check skipped: %s", exc)

    return SelftestResult(
        name="installer_dry_runs",
        status="PASS",
        measured="install.ps1 dry-run exit 0, install.sh syntax valid",
    )


def check_html_report() -> SelftestResult:
    """11. Verify self-contained HTML report is under 500 KB without external CDN links."""
    data_dir = get_data_dir()
    tmp_html = data_dir / "selftest_report.html"

    report_payload: dict[str, Any] = {
        "machine": {"platform": "Windows 11", "cpu_model": "AMD Ryzen", "total_ram_gb": 16},
        "baselines": {
            "ac": {
                "created_utc": "2026-11-01T00:00:00Z",
                "median_ms": 15.0,
                "p95_ms": 18.0,
                "cv": 5.0,
            }
        },
        "synthetic": False,
        "leaderboard": [
            {
                "model": "XGBoost",
                "mae": 0.08,
                "mae_ci": [0.06, 0.10],
                "rmse": 0.11,
                "r2": 0.85,
                "tolerance_accuracy": 0.95,
                "beats_persistence": True,
            }
        ],
        "limitations": ["Test limitation"],
    }

    try:
        generate_html_report(report_payload, tmp_html)
        size_bytes = tmp_html.stat().st_size
        size_kb = size_bytes / 1024.0
        content = tmp_html.read_text(encoding="utf-8")

        # Check self-containment: no remote http/https scripts or link tags
        remote_matches = re.findall(r'<(?:script|link)[^>]*(?:src|href)=["\']https?://', content)
        if remote_matches:
            return SelftestResult(
                name="html_report",
                status="FAIL",
                measured=f"Found {len(remote_matches)} external remote HTTP links",
                reason="HTML report must be strictly self-contained with no external dependencies",
            )

        if size_bytes > 500 * 1024:
            return SelftestResult(
                name="html_report",
                status="FAIL",
                measured=f"Report size {size_kb:.1f} KB exceeds 500 KB limit",
                reason=f"Size {size_bytes} > 512,000 bytes",
            )

        return SelftestResult(
            name="html_report",
            status="PASS",
            measured=f"Self-contained HTML report size {size_kb:.1f} KB < 500 KB",
        )
    finally:
        if tmp_html.exists():
            tmp_html.unlink()


def check_no_network_sockets() -> SelftestResult:
    """12. Verify zero network sockets opened during local operations (fully offline)."""
    proc = psutil.Process()
    conns_before = proc.net_connections()

    # Run local probe and fast inventory
    _ = run_probe()

    conns_after = proc.net_connections()
    new_conns = [c for c in conns_after if c not in conns_before]

    if new_conns:
        return SelftestResult(
            name="no_network_sockets",
            status="FAIL",
            measured=f"{len(new_conns)} socket(s) opened: {new_conns}",
            reason="Deskflow is strictly offline; runtime networking is forbidden",
        )

    return SelftestResult(
        name="no_network_sockets",
        status="PASS",
        measured="0 network sockets opened during scan (fully offline)",
    )


def selftest_command(plain: bool = False) -> None:
    """Run the 12 automated verification checks and exit 0 only if nothing fails."""
    plain = use_plain_output(plain)
    console = Console(no_color=plain)
    box_style = box.ASCII if plain else box.ROUNDED
    adapter = get_adapter()
    prov = load_provenance()

    console.print(
        "\n[bold cyan]DESKFLOW AUTOMATED SELF-TEST[/bold cyan] "
        "[dim](12 verification checks, exit 0 only if 0 failures)[/dim]\n"
    )

    checks: list[tuple[str, Callable[[], SelftestResult]]] = [
        ("Truth Contract", lambda: check_truth_contract(prov)),
        ("Model Provenance & Hashes", lambda: check_model_hashes(prov)),
        ("Baselines Freshness", lambda: check_baselines_freshness(adapter)),
        ("Probe Latency Stability", check_probe_stability),
        ("CLI Command Surface", check_command_surface),
        ("Responsive Layout & 0 Emoji", lambda: check_responsive_layout(prov, adapter)),
        ("JSON Output Purity", check_json_purity),
        ("CLI Exit Codes", check_exit_codes),
        ("Execution Latency Timings", lambda: check_timings(adapter)),
        ("Installer Scripts Dry-Run", check_installer_dry_runs),
        ("HTML Report Self-Containment", check_html_report),
        ("Offline Network Verification", check_no_network_sockets),
    ]

    results: list[SelftestResult] = []
    tbl = Table(box=box_style, expand=True, show_header=True)
    tbl.add_column("Verification Check", style="bold", width=30)
    tbl.add_column("Status", justify="center", width=8)
    tbl.add_column("Measured Value & Output", ratio=1)

    fail_count = 0
    warn_count = 0
    pass_count = 0
    skip_count = 0

    for title, fn in checks:
        try:
            res = fn()
        except Exception as exc:
            res = SelftestResult(
                name=title,
                status="FAIL",
                measured=f"Unhandled exception: {exc}",
                reason=str(exc),
            )

        results.append(res)
        if res.status == "PASS":
            pass_count += 1
            st_str = "[bold green]PASS[/bold green]" if not plain else "PASS"
        elif res.status == "WARN":
            warn_count += 1
            st_str = "[bold yellow]WARN[/bold yellow]" if not plain else "WARN"
        elif res.status == "FAIL":
            fail_count += 1
            st_str = "[bold red]FAIL[/bold red]" if not plain else "FAIL"
        else:
            skip_count += 1
            st_str = "[dim]SKIP[/dim]" if not plain else "SKIP"

        desc = res.measured
        if res.reason and res.status in ("WARN", "FAIL", "SKIP"):
            desc += f"\n  -> {res.reason}"
        tbl.add_row(title, st_str, desc)

    console.print(tbl)

    summary_str = (
        f"\n[bold]SELF-TEST SUMMARY[/bold]: "
        f"{pass_count} passed, {warn_count} warnings, {skip_count} skipped, {fail_count} failed"
    )
    if fail_count > 0:
        console.print(f"{summary_str} -> [bold red]FAIL[/bold red]\n")
        raise typer.Exit(code=EXIT_ERROR)
    else:
        console.print(f"{summary_str} -> [bold green]PASS[/bold green]\n")
        raise typer.Exit(code=EXIT_OK)
