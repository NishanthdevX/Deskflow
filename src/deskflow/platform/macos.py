"""macOS platform adapter implementation."""

import os
import tempfile
from pathlib import Path

import psutil

from deskflow.platform.base import BatteryInfo, CacheLocation, CpuFreq, PlatformAdapter

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
