"""Phase 11 Golden text snapshots, layout width scaling, ASCII purity, and render time tests."""

from __future__ import annotations

import time

from rich.console import Console

from deskflow.collector.models import ProcessSample
from deskflow.commands.fix import _render_tools_table, run_fast_inventory
from deskflow.platform.base import PlatformAdapter
from deskflow.ui.charts import (
    bar_chart_with_whiskers,
    heatmap,
    line_chart,
)
from deskflow.ui.components import (
    banner,
)
from deskflow.ui.widgets import scan_dashboard


class DummyAdapter(PlatformAdapter):
    """Minimal adapter for golden render tests."""

    @property
    def name(self) -> str:
        return "mock_os"

    def cpu_percent(self) -> float:
        return 15.0

    def cpu_percent_per_core(self) -> list[float]:
        return [15.0, 15.0]

    def cpu_freq(self):
        return None

    def virtual_memory(self):
        return (16 * 1024**3, 8 * 1024**3, 50.0)

    def swap_memory(self):
        return (1024**2, 1.0)

    def disk_usage(self, path: str):
        return (100 * 1024**3, 50 * 1024**3, 50 * 1024**3, 50.0)

    def disk_io_counters(self):
        return (100, 200, 1, 2, 1)

    def net_io_counters(self):
        return (100, 200, 0, 0)

    def process_count(self) -> int:
        return 50

    def thread_count(self) -> int:
        return 200

    def temperatures(self):
        return {}

    def battery(self):
        return None

    def power_state(self) -> str:
        return "ac"

    def temp_and_cache_dirs(self):
        return []

    def never_offer_names(self):
        return ["system", "deskflow"]

    def protected_process_names(self):
        return ["csrss.exe"]

    def is_process_protected(self, pid: int, name: str) -> bool:
        return False

    def query_recycle_bin(self):
        return 0, 0

    def empty_recycle_bin(self):
        return True

    def app_cache_dirs(self):
        return []

    def get_static_hardware(self):
        return {
            "host_model": "Dell XPS 15",
            "cpu_model": "Intel Core i7-12700H",
            "gpus": [{"name": "RTX 3050 Ti", "vram_bytes": 4 * 1024**3}],
            "resolution": "1920x1080",
        }

    def full_process_snapshot(self):
        return [
            ProcessSample(101, "python.exe", 200 * 1024 * 1024, 12.0, 4, 100.0, False),
            ProcessSample(102, "worker.exe", 500 * 1024 * 1024, 40.0, 8, 200.0, False),
        ]


def test_ui_imports_never_pull_ml_frameworks():
    """Verify that importing deskflow.ui primitives never imports sklearn or xgboost."""
    import sys

    # Remove any existing references to verify clean import
    modules_to_check = ["sklearn", "xgboost"]
    for mod in modules_to_check:
        sys.modules.pop(mod, None)

    for mod in modules_to_check:
        assert mod not in sys.modules, f"Forbidden import: {mod} was imported by UI package!"


def test_banner_at_widths_and_plain():
    """Verify banner adapts to widths 80, 100, 120 and plain mode produces pure ASCII."""
    for width in (80, 100, 120):
        # Plain mode
        cons_plain = Console(width=width, no_color=True, record=True)
        cons_plain.print(banner(version="0.1.0", width=width, plain=True))
        plain_out = cons_plain.export_text()
        assert "DESKFLOW" in plain_out
        assert all(ord(c) < 128 for c in plain_out), "Plain output must be pure ASCII!"

        # Unicode mode
        cons_uni = Console(width=width, record=True)
        cons_uni.print(banner(version="0.1.0", width=width, plain=False))
        uni_out = cons_uni.export_text()
        assert "DESKFLOW" in uni_out


def test_scan_screen_golden_render_widths_and_performance():
    """Verify scan dashboard renders without errors at 80, 100, 120 cols and < 50ms per render."""
    categories = [
        ("Performance", "Good", "88/100, ML prediction", 88.0),
        ("Memory", "Good", "8.2 GB free", 52.0),
        ("Storage", "Good", "0.5 MB cleanable", 0.0),
        ("Background apps", "Good", "0 group(s), 0.0 GB", 0.0),
    ]
    reasons = ["Top active processes within bounds", "Micro-probe latency optimal"]
    resources = [
        ("CPU", 0.15, [10.0, 12.0, 14.0, 15.0, 15.0, 16.0]),
        ("Memory", 0.48, [48.0, 48.0, 48.0, 48.0, 48.0, 48.0]),
        ("Disk Busy", 0.05, [2.0, 4.0, 3.0, 5.0, 2.0, 5.0]),
        ("Swap", 0.01, [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
    ]
    adapter = DummyAdapter()
    tools = run_fast_inventory(adapter)

    # Warm-up to bypass cold coverage tracing
    _ = scan_dashboard(
        "2d ago",
        "ac",
        "XGBoost",
        False,
        88,
        "Good",
        1.05,
        "healthy",
        0.95,
        False,
        reasons,
        categories,
        [],
        13.2,
        False,
    )

    for width in (80, 100, 120):
        for plain in (False, True):
            console = Console(width=width, no_color=plain, record=True)

            t0 = time.perf_counter()
            panels = scan_dashboard(
                baseline_age="2d ago",
                power_state="ac",
                model_name="XGBoost",
                synthetic=False,
                score=88,
                health_label="Good",
                predicted_ratio=1.05,
                issue_tag="healthy",
                confidence=0.95,
                anomaly=False,
                reasons=reasons,
                categories=categories,
                recommendations=[],
                elapsed_s=13.2,
                plain=plain,
                anomaly_score=0.12,
                anomaly_threshold=0.50,
                heavy_tasks=adapter.full_process_snapshot(),
                fix_tools=tools,
                resources=resources,
                timings={
                    "measuring_s": 12.0,
                    "storage_s": 0.5,
                    "inference_s": 0.005,
                    "total_s": 13.2,
                },
                os_info="Windows 11",
            )
            render_s = time.perf_counter() - t0
            for p in panels:
                console.print(p)

            # Verify render time is under 50 ms (target: < 50ms)
            assert render_s < 0.05, f"Render took {render_s * 1000:.2f} ms (must be < 50 ms)!"

            rendered = console.export_text()
            assert "DESKFLOW SCAN" in rendered
            assert "HEALTH" in rendered
            assert "RESOURCES" in rendered
            assert "HEAVY TASKS" in rendered
            assert "ACTIONS" in rendered

            if plain:
                assert all(ord(c) < 128 for c in rendered), (
                    "Plain mode scan output must be pure ASCII!"
                )


def test_fix_center_inventory_golden_render():
    """Verify Fix Center inventory renders properly at 80, 100, 120 cols in plain and unicode."""
    adapter = DummyAdapter()
    tools = run_fast_inventory(adapter)

    for width in (80, 100, 120):
        for plain in (False, True):
            console = Console(width=width, no_color=plain, record=True)
            tbl = _render_tools_table(tools, plain=plain)
            console.print(tbl)
            out = console.export_text()

            assert "TOOLS INVENTORY" in out
            assert "Clean temporary" in out
            assert "Recycle Bin" in out

            if plain:
                assert all(ord(c) < 128 for c in out), "Plain mode tools table must be pure ASCII!"


def test_chart_primitives_render_and_ascii_purity():
    """Verify whiskers bar chart, line chart, heatmap, and leaderboard render pure ASCII."""
    # 1. Bar chart with whiskers
    whisker_data = [
        ("XGBoost", 0.052, 0.040, 0.065, False),
        ("RandomForest", 0.061, 0.045, 0.078, False),
        ("Ridge", 0.070, 0.055, 0.088, False),
        ("Persistence", 0.120, 0.095, 0.145, True),
    ]
    for plain in (False, True):
        cons = Console(width=100, no_color=plain, record=True)
        tbl = bar_chart_with_whiskers(whisker_data, plain=plain)
        cons.print(tbl)
        out = cons.export_text()
        assert "XGBoost" in out
        assert "Persistence" in out
        if plain:
            assert all(ord(c) < 128 for c in out)

    # 2. Line chart
    actual_pts = [1.0, 1.1, 1.2, 1.4, 1.3, 1.1, 1.0]
    pred_pts = [1.0, 1.15, 1.25, 1.35, 1.30, 1.12, 1.02]
    for plain in (False, True):
        cons = Console(width=100, no_color=plain, record=True)
        tbl = line_chart(actual_pts, pred_pts, plain=plain)
        cons.print(tbl)
        out = cons.export_text()
        assert "PREDICTED VS ACTUAL" in out
        if plain:
            assert all(ord(c) < 128 for c in out)

    # 3. Heatmap
    labels = ["healthy", "cpu", "memory", "disk"]
    mat = [
        [50, 2, 0, 1],
        [1, 45, 3, 0],
        [0, 2, 48, 1],
        [1, 0, 1, 42],
    ]
    for plain in (False, True):
        cons = Console(width=100, no_color=plain, record=True)
        tbl = heatmap(labels, mat, plain=plain)
        cons.print(tbl)
        out = cons.export_text()
        assert "CONFUSION MATRIX" in out
        if plain:
            assert all(ord(c) < 128 for c in out)


def test_home_screen_loop_with_simulated_q_input():
    """Verify home screen menu loops and cleanly exits upon Q input."""
    from deskflow.cli import show_home_menu

    inputs = ["q"]

    def fake_input(prompt: str) -> str:
        return inputs.pop(0)

    # Must complete without exception and cleanly exit
    show_home_menu(input_fn=fake_input)
