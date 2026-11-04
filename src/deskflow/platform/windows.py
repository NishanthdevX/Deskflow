"""Windows platform adapter implementation."""

from __future__ import annotations

import ctypes
import os
import tempfile
import time
from ctypes import wintypes
from typing import TYPE_CHECKING, Any

import psutil

from deskflow.platform.base import (
    BatteryInfo,
    CacheLocation,
    CpuFreq,
    PlatformAdapter,
    ProcessMeta,
)

if TYPE_CHECKING:
    from deskflow.collector.models import ProcessSample

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

_WINDOWS_NEVER_OFFER = (
    frozenset(
        {
            # Shells and terminals
            "cmd.exe",
            "powershell.exe",
            "pwsh.exe",
            "bash.exe",
            "zsh.exe",
            "sh.exe",
            "wt.exe",
            "windowsterminal.exe",
            "conhost.exe",
            "mintty.exe",
            "alacritty.exe",
            "wezterm-gui.exe",
            "wezterm.exe",
            "kitty.exe",
            "hyper.exe",
            # IDEs and editors
            "code.exe",
            "cursor.exe",
            "antigravity.exe",
            "pycharm64.exe",
            "pycharm.exe",
            "idea64.exe",
            "idea.exe",
            "devenv.exe",
            "sublime_text.exe",
            "notepad++.exe",
            # File manager and shell UI
            "explorer.exe",
            "dwm.exe",
            "shellexperiencehost.exe",
            "startmenuexperiencehost.exe",
            "searchhost.exe",
            "taskbar.exe",
            # Security tools
            "msmpeng.exe",
            "securityhealthservice.exe",
            "nissrv.exe",
            "windowsdefender.exe",
            # Presentation and conferencing apps
            "powerpnt.exe",
            "zoom.exe",
            "teams.exe",
            "ms-teams.exe",
            "obs64.exe",
            "obs32.exe",
            "obs.exe",
            "slack.exe",
            "webex.exe",
            "discord.exe",
        }
    )
    | _WINDOWS_PROTECTED_PROCESSES
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


WNDENUMPROC = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
WM_CLOSE = 0x0010


def _ensure_interactive_desktop() -> None:
    """Ensure thread is attached to interactive desktop if running in background session."""
    try:
        user32 = ctypes.windll.user32
        h_desk = user32.OpenDesktopW(
            "Default", 0, False, 0x0100 | 0x0040 | 0x0001 | 0x0002 | 0x0020
        )
        if h_desk:
            user32.SetThreadDesktop(h_desk)
    except Exception:
        pass


def _get_window_owning_pids_win32() -> set[int]:
    """Return process IDs that own at least one visible top-level window with a non-empty title."""
    try:
        user32 = ctypes.windll.user32
        _ensure_interactive_desktop()
        pids: set[int] = set()

        def _enum_cb(hwnd: wintypes.HWND, lparam: wintypes.LPARAM) -> bool:
            if user32.IsWindowVisible(hwnd) and user32.GetWindowTextLengthW(hwnd) > 0:
                pid = wintypes.DWORD()
                user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
                if pid.value > 0:
                    pids.add(int(pid.value))
            return True

        cb = WNDENUMPROC(_enum_cb)
        user32.EnumWindows(cb, 0)
        return pids
    except Exception:
        return set()


def _get_foreground_pid_win32() -> int | None:
    """Return the process ID owning the currently active foreground window, or None."""
    try:
        user32 = ctypes.windll.user32
        _ensure_interactive_desktop()
        hwnd = user32.GetForegroundWindow()
        if hwnd:
            pid = wintypes.DWORD()
            user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
            if pid.value > 0:
                return int(pid.value)
    except Exception:
        pass
    return None


def _close_process_windows_win32(pid: int, timeout_s: float = 8.0) -> bool:
    """Post WM_CLOSE to visible windows of pid, waiting up to timeout_s for process exit."""
    try:
        user32 = ctypes.windll.user32
        _ensure_interactive_desktop()
        hwnds: list[wintypes.HWND] = []

        def _enum_cb(hwnd: wintypes.HWND, lparam: wintypes.LPARAM) -> bool:
            if user32.IsWindowVisible(hwnd):
                proc_pid = wintypes.DWORD()
                user32.GetWindowThreadProcessId(hwnd, ctypes.byref(proc_pid))
                if proc_pid.value == pid:
                    hwnds.append(hwnd)
            return True

        cb = WNDENUMPROC(_enum_cb)
        user32.EnumWindows(cb, 0)

        for hwnd in hwnds:
            user32.PostMessageW(hwnd, WM_CLOSE, 0, 0)

        start = time.monotonic()
        while time.monotonic() - start < timeout_s:
            if not psutil.pid_exists(pid):
                return True
            try:
                proc = psutil.Process(pid)
                if not proc.is_running() or proc.status() == psutil.STATUS_ZOMBIE:
                    return True
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                return True
            time.sleep(0.2)

        return False
    except Exception:
        return False


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

    def get_window_owning_pids(self) -> set[int]:
        """Return process IDs that own at least one visible top-level titled window."""
        return _get_window_owning_pids_win32()

    def get_foreground_pid(self) -> int | None:
        """Return the process ID owning the currently active foreground window, or None."""
        return _get_foreground_pid_win32()

    def close_process_windows(self, pid: int, timeout_s: float = 8.0) -> bool:
        """Post WM_CLOSE to process windows, waiting up to timeout_s for graceful exit."""
        return _close_process_windows_win32(pid=pid, timeout_s=timeout_s)

    def never_offer_names(self) -> frozenset[str]:
        """Return lowercase set of process names that must never be offered for termination."""
        return _WINDOWS_NEVER_OFFER

    def query_recycle_bin(self) -> tuple[int, int]:
        """Return (num_items, total_size_bytes) in the Windows Recycle Bin."""
        return _query_recycle_bin_win32()

    def empty_recycle_bin(self) -> bool:
        """Empty the Windows Recycle Bin across all drives without prompts."""
        return _empty_recycle_bin_win32()

    def app_cache_dirs(self) -> list[CacheLocation]:
        """Return browser and application cache directories on Windows."""
        return _get_browser_cache_dirs_win32()

    def get_static_hardware(self) -> dict[str, Any]:
        """Return static hardware facts: host model, CPU, GPUs, VRAM, and resolution."""
        return _get_static_hardware_win32()

    def full_process_snapshot(self) -> list[ProcessSample]:
        """Return a complete process pass across the system for heavy-task analysis."""
        return _full_process_snapshot_win32()


class SHQUERYRBINFO(ctypes.Structure):
    """Win32 SHQUERYRBINFO structure for SHQueryRecycleBinW."""

    _fields_ = [
        ("cbSize", wintypes.DWORD),
        ("i64Size", ctypes.c_int64),
        ("i64NumItems", ctypes.c_int64),
    ]


def _query_recycle_bin_win32() -> tuple[int, int]:
    """Query total item count and bytes in the Windows Recycle Bin."""
    try:
        rb = SHQUERYRBINFO()
        rb.cbSize = ctypes.sizeof(SHQUERYRBINFO)
        hr = ctypes.windll.shell32.SHQueryRecycleBinW(None, ctypes.byref(rb))
        if hr == 0:
            return (int(rb.i64NumItems), int(rb.i64Size))
    except Exception:
        pass
    return (0, 0)


def _empty_recycle_bin_win32() -> bool:
    """Empty the Windows Recycle Bin without UI or sound."""
    try:
        # SHERB_NOCONFIRMATION (0x1) | SHERB_NOPROGRESSUI (0x2) | SHERB_NOSOUND (0x4)
        flags = 0x00000001 | 0x00000002 | 0x00000004
        hr = ctypes.windll.shell32.SHEmptyRecycleBinW(None, None, wintypes.DWORD(flags))
        return bool(hr == 0)
    except Exception:
        return False


def _get_static_hardware_win32() -> dict[str, Any]:
    """Read CPU, BIOS, GPU, and resolution from Windows registry and APIs."""
    import winreg

    info: dict[str, Any] = {
        "host_model": "unknown",
        "os_build": "unknown",
        "cpu_model": "unknown",
        "gpus": [],
        "resolution": "unknown",
    }

    # Host model from BIOS
    try:
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\BIOS") as k:
            mfr = str(winreg.QueryValueEx(k, "SystemManufacturer")[0]).strip()
            prod = str(winreg.QueryValueEx(k, "SystemProductName")[0]).strip()
            if mfr and prod:
                info["host_model"] = f"{mfr} {prod}".strip()
            elif prod:
                info["host_model"] = prod
    except Exception:
        pass

    # CPU model from CentralProcessor
    try:
        with winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
        ) as k:
            cpu_name = str(winreg.QueryValueEx(k, "ProcessorNameString")[0]).strip()
            if cpu_name:
                info["cpu_model"] = cpu_name
    except Exception:
        pass

    # Windows build and version
    try:
        with winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Microsoft\Windows NT\CurrentVersion"
        ) as k:
            build = str(winreg.QueryValueEx(k, "CurrentBuildNumber")[0]).strip()
            try:
                disp = str(winreg.QueryValueEx(k, "DisplayVersion")[0]).strip()
            except FileNotFoundError:
                disp = ""
            info["os_build"] = f"{disp} (Build {build})".strip() if disp else f"Build {build}"
    except Exception:
        pass

    # GPUs from display adapter class key
    try:
        class_guid = "{4d36e968-e325-11ce-bfc1-08002be10318}"
        key_path = rf"SYSTEM\CurrentControlSet\Control\Class\{class_guid}"
        gpus: list[dict[str, Any]] = []
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key_path) as k:
            num_subkeys = winreg.QueryInfoKey(k)[0]
            for i in range(num_subkeys):
                sub = winreg.EnumKey(k, i)
                if sub.isdigit():
                    try:
                        with winreg.OpenKey(k, sub) as sk:
                            desc = str(winreg.QueryValueEx(sk, "DriverDesc")[0]).strip()
                            vram_bytes = 0
                            for v_name in [
                                "HardwareInformation.qwMemorySize",
                                "HardwareInformation.MemorySize",
                            ]:
                                try:
                                    vram_bytes = int(winreg.QueryValueEx(sk, v_name)[0])
                                    if vram_bytes > 0:
                                        break
                                except (FileNotFoundError, OSError):
                                    pass
                            if desc:
                                gpus.append({"name": desc, "vram_bytes": vram_bytes})
                    except Exception:
                        pass
        info["gpus"] = gpus
    except Exception:
        pass

    # Resolution: physical display resolution via GetDeviceCaps with GetSystemMetrics fallback
    try:
        user32 = ctypes.windll.user32
        gdi32 = ctypes.windll.gdi32
        hdc = user32.GetDC(0)
        w = gdi32.GetDeviceCaps(hdc, 118)  # DESKTOPHORZRES (physical width)
        h = gdi32.GetDeviceCaps(hdc, 117)  # DESKTOPVERTRES (physical height)
        user32.ReleaseDC(0, hdc)
        if w > 0 and h > 0:
            info["resolution"] = f"{w}x{h}"
        else:
            w = int(user32.GetSystemMetrics(0))
            h = int(user32.GetSystemMetrics(1))
            if w > 0 and h > 0:
                info["resolution"] = f"{w}x{h}"
    except Exception:
        pass

    return info


def _get_browser_cache_dirs_win32() -> list[CacheLocation]:
    """Locate browser and app cache directories on Windows."""
    import os
    from pathlib import Path

    localapp_str = os.environ.get("LOCALAPPDATA")
    if not localapp_str:
        return []

    localapp = Path(localapp_str)
    caches: list[CacheLocation] = []

    chromium_browsers = [
        ("Google Chrome", localapp / "Google" / "Chrome" / "User Data"),
        ("Microsoft Edge", localapp / "Microsoft" / "Edge" / "User Data"),
        ("Brave", localapp / "BraveSoftware" / "Brave-Browser" / "User Data"),
        ("Opera", localapp / "Opera Software" / "Opera Stable"),
        ("Opera GX", localapp / "Opera Software" / "Opera GX Stable"),
    ]

    for browser_name, user_data in chromium_browsers:
        if not user_data.exists():
            continue
        if "Opera" in browser_name:
            profile_dirs = [user_data]
        else:
            profile_dirs = [user_data / "Default"] + list(user_data.glob("Profile *"))

        for p_dir in profile_dirs:
            if not p_dir.is_dir():
                continue
            for cache_folder in ("Cache", "Code Cache", "GPUCache"):
                target = p_dir / cache_folder
                if target.exists() and target.is_dir():
                    caches.append(
                        CacheLocation(
                            path=str(target),
                            kind="cache",
                            safe_to_clean=True,
                            label=f"{browser_name} {cache_folder}",
                        )
                    )

    ff_profiles = localapp / "Mozilla" / "Firefox" / "Profiles"
    if ff_profiles.exists():
        for p in ff_profiles.glob("*/cache2"):
            if p.is_dir():
                caches.append(
                    CacheLocation(
                        path=str(p),
                        kind="cache",
                        safe_to_clean=True,
                        label="Mozilla Firefox cache2",
                    )
                )

    return caches


def _full_process_snapshot_win32() -> list[ProcessSample]:
    """Capture full process pass across system for heavy-task ranking."""
    import time

    import psutil

    from deskflow.collector.models import ProcessSample
    from deskflow.safety import is_protected

    now = time.time()
    results: list[ProcessSample] = []
    for proc in psutil.process_iter(
        ["pid", "name", "username", "cpu_percent", "memory_info", "num_threads", "create_time"]
    ):
        try:
            info = proc.info
            pid = info["pid"]
            if pid is None or pid <= 4:
                continue
            name = info.get("name") or "unknown"
            user = info.get("username")
            mem_info = info.get("memory_info")
            rss = mem_info.rss if mem_info else 0
            cpu = float(info.get("cpu_percent") or 0.0)
            threads = int(info.get("num_threads") or 1)
            create_t = float(info.get("create_time") or now)
            age_s = max(0.0, now - create_t)

            results.append(
                ProcessSample(
                    pid=pid,
                    name=name,
                    rss_bytes=rss,
                    cpu_percent=cpu,
                    num_threads=threads,
                    age_s=age_s,
                    is_protected=is_protected(name, pid, username=user),
                )
            )
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

    return results
