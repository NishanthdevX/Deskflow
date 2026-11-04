"""macOS platform adapter implementation."""

from __future__ import annotations

import logging
import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import psutil

from deskflow.platform.base import BatteryInfo, CacheLocation, CpuFreq, PlatformAdapter

if TYPE_CHECKING:
    from deskflow.collector.models import ProcessSample

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

    def query_recycle_bin(self) -> tuple[int, int]:
        """Return (num_items, total_size_bytes) in ~/.Trash."""
        trash = Path.home() / ".Trash"
        if not trash.exists() or not trash.is_dir():
            return (0, 0)
        count = 0
        total_size = 0
        try:
            for entry in trash.iterdir():
                count += 1
                try:
                    if entry.is_file():
                        total_size += entry.stat().st_size
                    elif entry.is_dir():
                        for f in entry.rglob("*"):
                            if f.is_file() and not f.is_symlink():
                                total_size += f.stat().st_size
                except (PermissionError, OSError):
                    pass
        except Exception:
            pass
        return (count, total_size)

    def empty_recycle_bin(self) -> bool:
        """Empty ~/.Trash on macOS."""
        import shutil

        trash = Path.home() / ".Trash"
        if not trash.exists():
            return True
        success = True
        for entry in trash.iterdir():
            try:
                if entry.is_dir() and not entry.is_symlink():
                    shutil.rmtree(entry, ignore_errors=True)
                else:
                    entry.unlink(missing_ok=True)
            except Exception:
                success = False
        return success

    def app_cache_dirs(self) -> list[CacheLocation]:
        """Return application cache directories in ~/Library/Caches on macOS."""
        caches_dir = Path.home() / "Library" / "Caches"
        if not caches_dir.exists() or not caches_dir.is_dir():
            return []
        locs: list[CacheLocation] = []
        try:
            for p in caches_dir.iterdir():
                if p.is_dir() and not p.is_symlink():
                    locs.append(
                        CacheLocation(
                            path=str(p),
                            kind="cache",
                            safe_to_clean=True,
                            label=p.name,
                        )
                    )
        except Exception:
            pass
        return locs

    def get_static_hardware(self) -> dict[str, Any]:
        """Return macOS hardware facts: host model, CPU, GPUs, and resolution."""
        info: dict[str, Any] = {
            "host_model": "unknown",
            "os_build": "unknown",
            "cpu_model": "unknown",
            "gpus": [],
            "resolution": "unknown",
        }
        try:
            res = subprocess.run(
                ["sysctl", "-n", "hw.model"],
                check=False,
                capture_output=True,
                text=True,
                timeout=2.0,
            )
            if res.returncode == 0 and res.stdout.strip():
                info["host_model"] = res.stdout.strip()
        except Exception:
            pass
        try:
            res = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                check=False,
                capture_output=True,
                text=True,
                timeout=2.0,
            )
            if res.returncode == 0 and res.stdout.strip():
                info["cpu_model"] = res.stdout.strip()
        except Exception:
            pass
        try:
            res = subprocess.run(
                ["sw_vers", "-buildVersion"],
                check=False,
                capture_output=True,
                text=True,
                timeout=2.0,
            )
            if res.returncode == 0 and res.stdout.strip():
                info["os_build"] = f"Build {res.stdout.strip()}"
        except Exception:
            pass
        # Display/GPU via system_profiler in thread with 5s timeout
        try:
            res = subprocess.run(
                ["system_profiler", "SPDisplaysDataType"],
                check=False,
                capture_output=True,
                text=True,
                timeout=5.0,
            )
            if res.returncode == 0:
                output = res.stdout
                for line in output.splitlines():
                    if "Chipset Model:" in line:
                        gpu_name = line.split(":", 1)[1].strip()
                        info["gpus"].append({"name": gpu_name, "vram_bytes": 0})
                    elif "Resolution:" in line:
                        res_str = line.split(":", 1)[1].strip().split()[0]
                        info["resolution"] = res_str
        except Exception:
            pass
        return info

    def full_process_snapshot(self) -> list[ProcessSample]:
        """Capture full process pass across macOS system."""
        from deskflow.collector.models import ProcessSample
        from deskflow.safety import is_protected

        now = time.time()
        results: list[ProcessSample] = []
        for proc in psutil.process_iter(
            ["pid", "name", "username", "cpu_percent", "memory_info", "num_threads", "create_time"]
        ):
            try:
                p_info = proc.info
                pid = p_info["pid"]
                if pid is None or pid <= 1:
                    continue
                name = p_info.get("name") or "unknown"
                user = p_info.get("username")
                mem = p_info.get("memory_info")
                rss = mem.rss if mem else 0
                cpu = float(p_info.get("cpu_percent") or 0.0)
                threads = int(p_info.get("num_threads") or 1)
                create_t = float(p_info.get("create_time") or now)
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
