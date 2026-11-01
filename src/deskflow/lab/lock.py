"""Process tracking and leftover cleanup for the fault-injection lab."""

import contextlib
import json
import logging
import os
import shutil
from pathlib import Path

import platformdirs
import psutil

logger = logging.getLogger(__name__)


def get_scratch_dir() -> Path:
    """Return user data directory for lab scratch files."""
    data_dir = Path(platformdirs.user_data_dir("deskflow", ensure_exists=True))
    scratch_dir = data_dir / "lab-scratch"
    scratch_dir.mkdir(parents=True, exist_ok=True)
    return scratch_dir


def cleanup_leftovers(scratch_dir: Path | None = None) -> None:
    """Clean up leftover files and terminate leftover child processes from crashed runs."""
    sdir = scratch_dir or get_scratch_dir()
    lock_file = sdir / "lab_children.lock"

    if lock_file.exists():
        try:
            with open(lock_file, encoding="utf-8") as f:
                pids: list[int] = json.load(f)
            current_pid = os.getpid()
            for pid in pids:
                if pid != current_pid and psutil.pid_exists(pid):
                    try:
                        p = psutil.Process(pid)
                        # Check that this process looks like python/deskflow before killing
                        p.terminate()
                        try:
                            p.wait(timeout=1.0)
                        except psutil.TimeoutExpired:
                            p.kill()
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
        except Exception as e:
            logger.warning("Error reading lab lock file %s: %s", lock_file, e)

        with contextlib.suppress(OSError):
            lock_file.unlink(missing_ok=True)

    # Clean up scratch files
    for item in sdir.glob("*"):
        if item.name == "lab_children.lock":
            continue
        try:
            if item.is_file() or item.is_symlink():
                item.unlink(missing_ok=True)
            elif item.is_dir():
                shutil.rmtree(item, ignore_errors=True)
        except OSError as e:
            logger.warning("Error removing scratch file %s: %s", item, e)


def register_child_pids(pids: list[int], scratch_dir: Path | None = None) -> None:
    """Record child PIDs in the scratch lock file."""
    sdir = scratch_dir or get_scratch_dir()
    lock_file = sdir / "lab_children.lock"
    try:
        with open(lock_file, "w", encoding="utf-8") as f:
            json.dump(pids, f)
    except OSError as e:
        logger.warning("Failed to record child PIDs in %s: %s", lock_file, e)


def unregister_children(scratch_dir: Path | None = None) -> None:
    """Remove the scratch lock file when child processes terminate cleanly."""
    sdir = scratch_dir or get_scratch_dir()
    lock_file = sdir / "lab_children.lock"
    with contextlib.suppress(OSError):
        lock_file.unlink(missing_ok=True)
