"""macOS platform adapter implementation."""

import logging
import os
import subprocess
import tempfile
import time
from pathlib import Path

import psutil

from deskflow.platform.base import BatteryInfo, CacheLocation, CpuFreq, PlatformAdapter

logger = logging.getLogger(__name__)

_MACOS_PROTECTED_PROCESSES = frozenset(
    {
        "kernel_task",
        "launchd",
        "windowserver",
        "loginwindow",
        "finder",
        "dock",
        "systemuiserver",
        "coreaudiod",
        "mds",
        "mds_stores",
        "distnoted",
        "cfprefsd",
    }
)

_MACOS_NEVER_OFFER = (
    frozenset(
        {
            # Shells and terminals
            "terminal",
            "iterm2",
            "iterm",
            "alacritty",
            "kitty",
            "wezterm",
            "bash",
            "zsh",
            "sh",
            # IDEs and editors
            "code",
            "cursor",
            "antigravity",
            "pycharm",
            "intellij idea",
            "idea",
            "xcode",
            "sublime text",
            "textmate",
            # File manager and shell UI
            "finder",
            "dock",
            "systemuiserver",
            "windowserver",
            "loginwindow",
            # Security & background
            "syspolicyd",
            "securityd",
            # Presentation and conferencing apps
            "keynote",
            "powerpoint",
            "microsoft powerpoint",
            "zoom",
            "zoom.us",
            "teams",
            "microsoft teams",
            "obs",
            "slack",
            "webex",
            "discord",
        }
    )
    | _MACOS_PROTECTED_PROCESSES
)


class MacOSAdapter(PlatformAdapter):
    """Platform adapter for macOS 12+."""

    @property
    def name(self) -> str:
        """Return platform name."""
        return "macos"

    def cpu_freq(self) -> CpuFreq | None:
        """Sample CPU frequency via psutil.

        On Apple Silicon and modern macOS, psutil often returns None or 0.
        When reported, if current equals max, it is marked reliable=False.
        """
        try:
            freq = psutil.cpu_freq()
            if freq is None or freq.current <= 0:
                return None
            max_mhz = freq.max if freq.max > 0 else freq.current
            reliable = freq.current != max_mhz
            return CpuFreq(current_mhz=freq.current, max_mhz=max_mhz, reliable=reliable)
        except Exception:
            return None

    def temperatures(self) -> dict[str, float]:
        """Hardware temperatures on macOS.

        psutil.sensors_temperatures() is typically unsupported on macOS without external tools.
        Returns empty dict when unavailable.
        """
        if not hasattr(psutil, "sensors_temperatures"):
            return {}
        try:
            readings = psutil.sensors_temperatures()
            if not readings:
                return {}
            result: dict[str, float] = {}
            for name, entries in readings.items():
                for entry in entries:
                    sensor_name = f"{name}_{entry.label}" if entry.label else name
                    result[sensor_name] = entry.current
            return result
        except Exception:
            return {}

    def battery(self) -> BatteryInfo | None:
        """Battery status via psutil."""
        if not hasattr(psutil, "sensors_battery"):
            return None
        try:
            bat = psutil.sensors_battery()
            if bat is None:
                return None
            plugged = bat.power_plugged is True
            return BatteryInfo(percent=float(bat.percent), plugged_in=plugged)
        except Exception:
            return None

    def temp_and_cache_dirs(self) -> list[CacheLocation]:
        """Temporary and cache locations on macOS."""
        locations: list[CacheLocation] = []
        home = str(Path.home())

        # ~/Library/Caches (cache)
        user_caches = os.path.join(home, "Library", "Caches")
        if os.path.exists(user_caches):
            locations.append(CacheLocation(path=user_caches, kind="cache", safe_to_clean=True))

        # tempfile temp dir (temp)
        temp_dir = tempfile.gettempdir()
        if os.path.exists(temp_dir):
            locations.append(CacheLocation(path=temp_dir, kind="temp", safe_to_clean=True))

        # ~/.Trash (trash, safe_to_clean False)
        user_trash = os.path.join(home, ".Trash")
        if os.path.exists(user_trash):
            locations.append(CacheLocation(path=user_trash, kind="trash", safe_to_clean=False))

        return locations

    def protected_process_names(self) -> frozenset[str]:
        """Protected macOS processes."""
        return _MACOS_PROTECTED_PROCESSES

    def process_count(self) -> int | None:
        """Total active process count via psutil.pids()."""
        try:
            return len(psutil.pids())
        except Exception:
            return None

    def thread_count(self) -> int | None:
        """Return None as no low-overhead thread counting source exists on macOS."""
        return None

    def get_window_owning_pids(self) -> set[int]:
        """Return process IDs that own visible UI on macOS (background only false).

        Note: Best effort via osascript System Events with 5s timeout.
        Cannot be tested locally on Windows/Linux environments.
        """
        script = (
            'tell application "System Events" to get unix id of every process '
            "whose background only is false"
        )
        try:
            res = subprocess.run(
                ["osascript", "-e", script],
                capture_output=True,
                text=True,
                timeout=5.0,
                check=False,
            )
            if res.returncode == 0 and res.stdout.strip():
                # Output format: "123, 456, 789"
                parts = [p.strip() for p in res.stdout.split(",") if p.strip().isdigit()]
                return {int(p) for p in parts}
        except Exception as e:
            logger.debug("Failed to query macOS visible UI processes: %s", e)
        return set()

    def get_foreground_pid(self) -> int | None:
        """Return process ID of frontmost application on macOS via osascript."""
        script = (
            'tell application "System Events" to get unix id of '
            "first process whose frontmost is true"
        )
        try:
            res = subprocess.run(
                ["osascript", "-e", script],
                capture_output=True,
                text=True,
                timeout=5.0,
                check=False,
            )
            if res.returncode == 0 and res.stdout.strip().isdigit():
                return int(res.stdout.strip())
        except Exception as e:
            logger.debug("Failed to query macOS frontmost process: %s", e)
        return None

    def close_process_windows(self, pid: int, timeout_s: float = 8.0) -> bool:
        """Gracefully quit an application on macOS using AppleScript."""
        try:
            proc = psutil.Process(pid)
            app_name = proc.name()
            # Try to tell application to quit
            script = f'tell application "{app_name}" to quit'
            subprocess.run(
                ["osascript", "-e", script],
                capture_output=True,
                timeout=5.0,
                check=False,
            )
            start = time.monotonic()
            while time.monotonic() - start < timeout_s:
                if not psutil.pid_exists(pid):
                    return True
                try:
                    if not proc.is_running() or proc.status() == psutil.STATUS_ZOMBIE:
                        return True
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    return True
                time.sleep(0.2)
            return False
        except Exception as e:
            logger.debug("Failed graceful close for pid %d on macOS: %s", pid, e)
            return False

    def never_offer_names(self) -> frozenset[str]:
        """Return lowercase set of process names that must never be offered for termination."""
        return _MACOS_NEVER_OFFER
