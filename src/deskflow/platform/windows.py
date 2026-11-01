"""Windows platform adapter implementation."""

import ctypes
import os
import tempfile
from ctypes import wintypes

import psutil

from deskflow.platform.base import BatteryInfo, CacheLocation, CpuFreq, PlatformAdapter

_WINDOWS_PROTECTED_PROCESSES = frozenset(
    {
        "system",
        "system idle process",
        "registry",
        "smss.exe",
        "csrss.exe",
        "wininit.exe",
        "services.exe",
        "lsass.exe",
        "winlogon.exe",
        "svchost.exe",
        "explorer.exe",
        "dwm.exe",
        "fontdrvhost.exe",
        "msmpeng.exe",
        "searchindexer.exe",
    }
)


class THREADENTRY32(ctypes.Structure):
    """Win32 thread entry structure."""

    _fields_ = [
        ("dwSize", wintypes.DWORD),
        ("cntUsage", wintypes.DWORD),
        ("th32ThreadID", wintypes.DWORD),
        ("th32OwnerProcessID", wintypes.DWORD),
        ("tpBasePri", wintypes.LONG),
        ("tpDeltaPri", wintypes.LONG),
        ("dwFlags", wintypes.DWORD),
    ]


class WindowsAdapter(PlatformAdapter):
    """Platform adapter for Windows 10/11."""

    @property
    def name(self) -> str:
        """Return platform name."""
        return "windows"

    def cpu_freq(self) -> CpuFreq | None:
        """Sample CPU frequency via psutil.

        On Windows, psutil often reports nominal/base frequency rather than live clock,
        meaning current frequently equals max. We mark reliable=False when current == max
        or when psutil data is absent/zero.
        """
        try:
            freq = psutil.cpu_freq()
            if freq is None or freq.current <= 0:
                return None
            max_mhz = freq.max if freq.max > 0 else freq.current
            # If current exactly equals max, it is likely reporting static base clock
            reliable = freq.current != max_mhz
            return CpuFreq(current_mhz=freq.current, max_mhz=max_mhz, reliable=reliable)
        except Exception:
            return None

    def temperatures(self) -> dict[str, float]:
        """Hardware temperatures on Windows.

        psutil.sensors_temperatures() is typically unavailable without elevation or
        special kernel drivers on Windows. Returns empty dict if not available.
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
        """Temporary and cache locations on Windows."""
        locations: list[CacheLocation] = []
        seen_paths: set[str] = set()

        # User temporary directory
        user_temp = tempfile.gettempdir()
        if os.path.exists(user_temp):
            norm = os.path.normpath(user_temp).lower()
            seen_paths.add(norm)
            locations.append(CacheLocation(path=user_temp, kind="temp", safe_to_clean=True))

        # %LOCALAPPDATA%\Temp
        local_app_data = os.environ.get("LOCALAPPDATA")
        if local_app_data:
            local_temp = os.path.join(local_app_data, "Temp")
            norm_local = os.path.normpath(local_temp).lower()
            if os.path.exists(local_temp) and norm_local not in seen_paths:
                seen_paths.add(norm_local)
                locations.append(CacheLocation(path=local_temp, kind="temp", safe_to_clean=True))

        # C:\Windows\Temp (System temp - read only, not safe to clean by user)
        win_dir = os.environ.get("WINDIR", r"C:\Windows")
        win_temp = os.path.join(win_dir, "Temp")
        norm_win = os.path.normpath(win_temp).lower()
        if os.path.exists(win_temp) and norm_win not in seen_paths:
            seen_paths.add(norm_win)
            locations.append(CacheLocation(path=win_temp, kind="temp", safe_to_clean=False))

        return locations

    def protected_process_names(self) -> frozenset[str]:
        """Protected Windows processes."""
        return _WINDOWS_PROTECTED_PROCESSES

    def thread_count(self) -> int | None:
        """Fast enumeration of total system threads via ToolHelp snapshot."""
        try:
            # TH32CS_SNAPTHREAD = 0x00000004
            h_snap = ctypes.windll.kernel32.CreateToolhelp32Snapshot(0x00000004, 0)
            if h_snap == -1:
                return None
            te = THREADENTRY32()
            te.dwSize = ctypes.sizeof(THREADENTRY32)
            count = 0
            if ctypes.windll.kernel32.Thread32First(h_snap, ctypes.byref(te)):
                count += 1
                while ctypes.windll.kernel32.Thread32Next(h_snap, ctypes.byref(te)):
                    count += 1
            ctypes.windll.kernel32.CloseHandle(h_snap)
            return count
        except Exception:
            return None
