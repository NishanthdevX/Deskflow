"""Regenerate plain-text screen captures in docs/screens/ using real commands."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
SCREENS_DIR = ROOT_DIR / "docs" / "screens"

ANSI_REGEX = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]")


def run_command_plain(args: list[str]) -> str:
    """Run CLI command and return stripped text output."""
    cmd = [sys.executable, "-m", "deskflow"] + args
    res = subprocess.run(
        cmd,
        cwd=ROOT_DIR,
        capture_output=True,
        text=True,
        check=False,
    )
    raw = res.stdout
    if not raw.strip() and res.stderr.strip():
        raw = res.stderr
    # Strip ANSI sequences if any remain
    cleaned = ANSI_REGEX.sub("", raw)
    return cleaned.replace("\r\n", "\n")


def update_screens() -> None:
    """Capture real CLI commands and write to docs/screens/."""
    SCREENS_DIR.mkdir(parents=True, exist_ok=True)

    commands: dict[str, list[str]] = {
        "home.txt": ["--help"],
        "info.txt": ["info", "--plain"],
        "fix_list.txt": ["fix", "--list", "--plain"],
        "pipeline.txt": ["pipeline", "--plain"],
        "scan.txt": ["scan", "--plain", "--no-storage"],
        "report.txt": ["report", "--plain"],
    }

    for filename, args in commands.items():
        dest = SCREENS_DIR / filename
        print(f"Generating {dest.name} via: deskflow {' '.join(args)}...")
        out = run_command_plain(args)
        dest.write_text(out, encoding="utf-8")
        print(f"  Wrote {len(out.splitlines())} lines to {dest.name}")

    print("All plain screens updated successfully.")


if __name__ == "__main__":
    update_screens()
