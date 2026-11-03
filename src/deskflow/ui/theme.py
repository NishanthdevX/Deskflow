"""Terminal capability detection and shared Deskflow presentation settings."""

import os
import sys


def configure_stdout() -> None:
    """Configure Windows stdout for UTF-8 output when the stream supports it."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def use_plain_output(requested: bool = False) -> bool:
    """Return whether output must use portable ASCII-only formatting."""
    encoding = (getattr(sys.stdout, "encoding", None) or "").lower()
    term = os.environ.get("TERM", "")
    return requested or bool(os.environ.get("NO_COLOR")) or term == "dumb" or "utf" not in encoding
