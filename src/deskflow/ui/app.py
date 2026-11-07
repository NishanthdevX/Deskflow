"""Screen base class, router, and interactive navigation architecture."""

from __future__ import annotations

import sys
from abc import ABC, abstractmethod

from rich.console import Console, RenderableType
from rich.markdown import Markdown
from rich.panel import Panel

from deskflow.platform.factory import get_adapter
from deskflow.ui.theme import COLOR_CYAN

GLOSSARY_TEXT = """
# Deskflow Architecture Glossary

- **Micro-Benchmark Probe**: Lightweight, deterministic CPU arithmetic, memory copy, and small disk writes timed via `time.perf_counter_ns()`. Measured latency serves as the ground-truth definition of system responsiveness.
- **Baseline**: Calibrated probe latency captured when the machine is completely healthy and idle. Baselines are maintained independently for AC and Battery power states.
- **Slowdown Ratio**: Current or predicted probe total latency divided by baseline latency. 1.0x is optimal; 1.5x is noticeable sluggishness; 2.0x is severe resource contention.
- **Held-Out Test Data**: Strict session-based split (testing on strictly later recorded sessions) with zero time overlap and zero run_id overlap, ensuring no future data leakage.
- **Persistence Baseline**: The standard time-series baseline predicting that latency 20 seconds from now will equal current latency. Machine learning models must certify that they beat Persistence.
- **Fix Center**: Reclaimable disk and memory remediation tools governed by safety allowlists and process protection.
"""


class Screen(ABC):
    """Base class for full-terminal interactive screens."""

    @abstractmethod
    def render(self, width: int, height: int, plain: bool = False) -> RenderableType:
        """Render the complete screen for given terminal dimensions."""

    def handle_key(self, key: str) -> str | None:
        """Handle a key token. Return action string or None to keep active."""
        return None


def show_glossary(console: Console, plain: bool = False) -> None:
    """Display the Deskflow glossary overlay and wait for a keypress."""
    panel = Panel(
        Markdown(GLOSSARY_TEXT),
        title="[ DESKFLOW GLOSSARY & ML ARCHITECTURE ]",
        border_style=COLOR_CYAN if not plain else "white",
        expand=True,
    )
    console.print(panel)
    console.print("\n[dim]Press any key to return...[/dim]")
    adapter = get_adapter()
    if sys.stdin.isatty():
        while True:
            k = adapter.read_key(0.1)
            if k is not None:
                break
    else:
        try:
            input()
        except Exception:
            pass

