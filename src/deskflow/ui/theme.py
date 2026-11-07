"""Terminal capability detection, semantic palette, named Theme, and color blending."""

from __future__ import annotations

import contextlib
import os
import sys

from rich.style import Style
from rich.theme import Theme

# Semantic palette from docs/DESIGN.md
COLOR_CYAN = "bright_cyan"  # #00E5FF identity, ML predictions, benchmarks
COLOR_GREEN = "spring_green2"  # #00E676 good, passed, safe
COLOR_YELLOW = "yellow1"  # #FFEA00 fair, warnings, recommendations, caution
COLOR_RED = "red1"  # #FF1744 poor, errors, failed
COLOR_MAGENTA = "magenta2"  # #D500F9 live accuracy, model tags
COLOR_WHITE = "bold white"  # primary values
COLOR_GREY = "grey50"  # #64748B secondary text, units, inactive borders

# Domain border styles
DOMAIN_BORDER_STYLES: dict[str, str] = {
    "ml": COLOR_CYAN,
    "health": COLOR_GREEN,
    "actions": COLOR_YELLOW,
    "accuracy": COLOR_MAGENTA,
    "system": COLOR_GREY,
}

STATUS_STYLES: dict[str, str] = {
    "Good": COLOR_GREEN,
    "Fair": COLOR_YELLOW,
    "Poor": COLOR_RED,
    "Safe": COLOR_GREEN,
    "Caution": COLOR_YELLOW,
    "Pass": COLOR_GREEN,
    "OK": COLOR_GREEN,
    "Warn": COLOR_YELLOW,
    "Fail": COLOR_RED,
}

# Official Deskflow Rich Theme with named styles (Phase 18 Section C)
DESKFLOW_THEME = Theme(
    {
        "df.title": Style(color="bright_cyan", bold=True),
        "df.good": Style(color="spring_green2", bold=True),
        "df.warn": Style(color="yellow1", bold=True),
        "df.bad": Style(color="red1", bold=True),
        "df.accent": Style(color="bright_cyan", bold=True),
        "df.muted": Style(color="grey50"),
        "df.value": Style(color="white", bold=True),
        "df.border.ml": Style(color="bright_cyan"),
        "df.border.health": Style(color="spring_green2"),
        "df.border.action": Style(color="yellow1"),
        "df.border.live": Style(color="magenta2"),
        "df.border.system": Style(color="grey50"),
    }
)


def configure_stdout() -> None:
    """Configure Windows stdout for UTF-8 output when the stream supports it."""
    if hasattr(sys.stdout, "reconfigure"):
        with contextlib.suppress(Exception):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")


configure_utf8_stdout = configure_stdout


def use_plain_output(requested: bool = False) -> bool:
    """Return whether output must use portable ASCII-only formatting."""
    encoding = (getattr(sys.stdout, "encoding", None) or "").lower()
    term = os.environ.get("TERM", "")
    return requested or bool(os.environ.get("NO_COLOR")) or term == "dumb" or "utf" not in encoding


def blend_health_color(score: float, plain: bool = False) -> str:
    """Blend color from Red (0) -> Amber (50) -> Green (100)."""
    if plain:
        return "none"
    norm = max(0.0, min(100.0, score)) / 100.0
    if norm < 0.5:
        # Red to Yellow
        return COLOR_RED if norm < 0.25 else COLOR_YELLOW
    return COLOR_YELLOW if norm < 0.75 else COLOR_GREEN


def blend_load_color(ratio: float, plain: bool = False) -> str:
    """Blend color from Green (low load <= 1.0) -> Yellow (1.5) -> Red (>= 2.5)."""
    if plain:
        return "none"
    if ratio <= 1.2:
        return COLOR_GREEN
    if ratio <= 1.8:
        return COLOR_YELLOW
    return COLOR_RED
