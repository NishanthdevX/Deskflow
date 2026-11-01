"""Windows platform adapter implementation."""

import ctypes
import os
import tempfile
from ctypes import wintypes

import psutil

from deskflow.platform.base import (
    BatteryInfo,
    CacheLocation,
    CpuFreq,
    PlatformAdapter,
    ProcessMeta,
)

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


class PERFORMANCE_INFORMATION(ctypes.Structure):
    """Win32 performance information structure from psapi.h."""

    _fields_ = [
        ("cb", wintypes.DWORD),
        ("CommitTotal", ctypes.c_size_t),
        ("CommitLimit", ctypes.c_size_t),
        ("CommitPeak", ctypes.c_size_t),
        ("PhysicalTotal", ctypes.c_size_t),
        ("PhysicalAvailable", ctypes.c_size_t),
        ("SystemCache", ctypes.c_size_t),
        ("KernelTotal", ctypes.c_size_t),
        ("KernelPaged", ctypes.c_size_t),
        ("KernelNonpaged", ctypes.c_size_t),
        ("PageSize", ctypes.c_size_t),
        ("HandleCount", wintypes.DWORD),
        ("ProcessCount", wintypes.DWORD),
        ("ThreadCount", wintypes.DWORD),
    ]


class UNICODE_STRING(ctypes.Structure):
    """Win32 Unicode string structure."""

    _fields_ = [
        ("Length", wintypes.USHORT),
        ("MaximumLength", wintypes.USHORT),
        ("Buffer", ctypes.c_wchar_p),
    ]


class SYSTEM_PROCESS_INFORMATION(ctypes.Structure):
    """Win32 system process information structure for NtQuerySystemInformation."""

    _fields_ = [
        ("NextEntryOffset", wintypes.ULONG),
        ("NumberOfThreads", wintypes.ULONG),
        ("WorkingSetPrivateSize", ctypes.c_int64),
        ("HardFaultCount", wintypes.ULONG),
        ("NumberOfThreadsHighWatermark", wintypes.ULONG),
        ("CycleTime", ctypes.c_uint64),
        ("CreateTime", ctypes.c_int64),
        ("UserTime", ctypes.c_int64),
        ("KernelTime", ctypes.c_int64),
        ("ImageName", UNICODE_STRING),
        ("BasePriority", wintypes.LONG),
        ("UniqueProcessId", ctypes.c_void_p),
        ("InheritedFromUniqueProcessId", ctypes.c_void_p),
        ("HandleCount", wintypes.ULONG),
        ("SessionId", wintypes.ULONG),
        ("UniqueProcessKey", ctypes.c_void_p),
        ("PeakVirtualSize", ctypes.c_size_t),
        ("VirtualSize", ctypes.c_size_t),
        ("PageFaultCount", wintypes.ULONG),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
    ]


def _query_system_processes() -> list[ProcessMeta] | None:
    """Fetch all running processes in a single NtQuerySystemInformation syscall (<5ms)."""
    try:
        buf_size = 2 * 1024 * 1024
        buf = ctypes.create_string_buffer(buf_size)
        ret_len = ctypes.c_ulong(0)
        res = ctypes.windll.ntdll.NtQuerySystemInformation(5, buf, buf_size, ctypes.byref(ret_len))
        if res != 0:
            return None
        offset = 0
        entries: list[ProcessMeta] = []
        while True:
            p = SYSTEM_PROCESS_INFORMATION.from_buffer(buf, offset)
            pid = int(p.UniqueProcessId or 0)
            name = str(p.ImageName.Buffer or "")
            ctime_raw = int(p.CreateTime)
            ctime_epoch = (
                max(0.0, (ctime_raw - 116444736000000000) / 10000000.0)
                if ctime_raw > 116444736000000000
                else 0.0
            )
            entries.append(
                ProcessMeta(
                    pid=pid,
                    name=name,
                    rss_bytes=int(p.WorkingSetSize),
                    num_threads=int(p.NumberOfThreads),
                    create_time_s=ctime_epoch,
                )
            )
            if p.NextEntryOffset == 0:
                break
            offset += p.NextEntryOffset
        return entries
    except Exception:
        return None


def _get_performance_info() -> PERFORMANCE_INFORMATION | None:
    """Fetch system-wide performance info using psapi.GetPerformanceInfo."""
    try:
        pi = PERFORMANCE_INFORMATION()
        pi.cb = ctypes.sizeof(PERFORMANCE_INFORMATION)
        if ctypes.windll.psapi.GetPerformanceInfo(ctypes.byref(pi), pi.cb):
            return pi
    except Exception:
        pass
    return None


class MEMORYSTATUSEX(ctypes.Structure):
    """Win32 memory status structure."""

    _fields_ = [
        ("dwLength", wintypes.DWORD),
        ("dwMemoryLoad", wintypes.DWORD),
        ("ullTotalPhys", ctypes.c_uint64),
        ("ullAvailPhys", ctypes.c_uint64),
        ("ullTotalPageFile", ctypes.c_uint64),
        ("ullAvailPageFile", ctypes.c_uint64),
        ("ullTotalVirtual", ctypes.c_uint64),
        ("ullAvailVirtual", ctypes.c_uint64),
        ("ullAvailExtendedVirtual", ctypes.c_uint64),
    ]


def _get_swap_status() -> tuple[int, float] | None:
    """Fetch swap/pagefile used bytes and percentage via GlobalMemoryStatusEx (<0.01ms)."""
    try:
        ms = MEMORYSTATUSEX()
        ms.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(ms)):
            total = ms.ullTotalPageFile
            avail = ms.ullAvailPageFile
            used = total - avail
            pct = (used / total) * 100.0 if total > 0 else 0.0
            return int(used), float(pct)
    except Exception:
        pass
    return None


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

    def process_count(self) -> int | None:
        """Total system process count via GetPerformanceInfo."""
        pi = _get_performance_info()
        return int(pi.ProcessCount) if pi is not None else None

    def thread_count(self) -> int | None:
        """Total system thread count via GetPerformanceInfo."""
        pi = _get_performance_info()
        return int(pi.ThreadCount) if pi is not None else None

    def process_snapshot(self) -> list[ProcessMeta] | None:
        """Low-overhead process snapshot via NtQuerySystemInformation."""
        return _query_system_processes()

    def swap_memory(self) -> tuple[int, float] | None:
        """Low-overhead swap/pagefile status via GlobalMemoryStatusEx."""
        return _get_swap_status()

    def is_trim_memory_supported(self) -> bool:
        """Windows supports EmptyWorkingSet."""
        return True

    def trim_process_memory(self, pid: int) -> bool:
        """Trim process working set using EmptyWorkingSet Win32 API."""
        try:
            # PROCESS_SET_QUOTA (0x0100) | PROCESS_QUERY_INFORMATION (0x0400)
            desired_access = 0x0100 | 0x0400
            handle = ctypes.windll.kernel32.OpenProcess(desired_access, False, wintypes.DWORD(pid))
            if not handle:
                return False
            try:
                res = ctypes.windll.psapi.EmptyWorkingSet(handle)
                return bool(res != 0)
            finally:
                ctypes.windll.kernel32.CloseHandle(handle)
        except Exception:
            return False
