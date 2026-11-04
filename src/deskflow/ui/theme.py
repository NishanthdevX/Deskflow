"""Terminal capability detection, semantic palette, and domain border themes."""

from __future__ import annotations

import contextlib
import os
import sys

# Semantic palette from docs/DESIGN.md
COLOR_CYAN = "bright_cyan"  # #00E5FF identity, ML predictions, benchmarks
COLOR_GREEN = "spring_green2"  # #00E676 good, passed, safe
COLOR_YELLOW = "yellow1"  # #FFEA00 fair, warnings, recommendations, caution
COLOR_RED = "red1"  # #FF1744 poor, errors, failed
COLOR_MAGENTA = "magenta2"  # #D500F9 live accuracy, model tags
COLOR_WHITE = "bold white"  # primary values
COLOR_GREY = "grey50"  # #64748B secondary text, units, inactive borders

# Border colour marks domain: cyan = ML, green = health, yellow = actions,
# magenta = live accuracy, grey = system facts
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


def configure_stdout() -> None:
    """Configure Windows stdout for UTF-8 output when the stream supports it."""
    if hasattr(sys.stdout, "reconfigure"):
        with contextlib.suppress(Exception):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def use_plain_output(requested: bool = False) -> bool:
    """Return whether output must use portable ASCII-only formatting."""
    encoding = (getattr(sys.stdout, "encoding", None) or "").lower()
    term = os.environ.get("TERM", "")
    return requested or bool(os.environ.get("NO_COLOR")) or term == "dumb" or "utf" not in encoding
