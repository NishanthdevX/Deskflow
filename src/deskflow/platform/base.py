"""Abstract base class and data types for platform-specific adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from deskflow.collector.models import ProcessSample


@dataclass(frozen=True)
class CpuFreq:
    """CPU frequency report in MHz."""

    current_mhz: float
    max_mhz: float
    reliable: bool


@dataclass(frozen=True)
class BatteryInfo:
    """Battery state."""

    percent: float
    plugged_in: bool


@dataclass(frozen=True)
class CacheLocation:
    """A filesystem directory containing temporary or cached files."""

    path: str
    kind: Literal["temp", "cache", "trash"]
    safe_to_clean: bool
    label: str = ""


@dataclass(frozen=True)
class ProcessMeta:
    """Lightweight process metadata snapshot."""

    pid: int
    name: str
    rss_bytes: int
    num_threads: int
    create_time_s: float = 0.0


class PlatformAdapter(ABC):
    """Abstract interface for platform-specific capabilities."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the platform, e.g. 'windows' or 'macos'."""

    @abstractmethod
    def cpu_freq(self) -> CpuFreq | None:
        """Return current CPU frequency information, or None if unavailable."""

    @abstractmethod
    def temperatures(self) -> dict[str, float]:
        """Return hardware temperature readings in Celsius, or empty dict if unavailable."""

    @abstractmethod
    def battery(self) -> BatteryInfo | None:
        """Return battery status, or None if no battery is present or readable."""

    @abstractmethod
    def temp_and_cache_dirs(self) -> list[CacheLocation]:
        """Return known temporary and cache directories on this platform."""

    @abstractmethod
    def protected_process_names(self) -> frozenset[str]:
        """Return the set of lowercased process names that must never be terminated."""

    def process_count(self) -> int | None:
        """Return total active process count across the system, or None if unavailable."""
        return None

    def thread_count(self) -> int | None:
        """Return total active thread count across the system, or None if unavailable."""
        return None

    def swap_memory(self) -> tuple[int, float] | None:
        """Return (swap_used_bytes, swap_percent) or None if unavailable/unsupported."""
        return None

    def process_snapshot(self) -> list[ProcessMeta] | None:
        """Return low-overhead system process snapshot, or None if unavailable."""
        return None

    def is_trim_memory_supported(self) -> bool:
        """Return True if the platform supports trimming process working sets."""
        return False

    def trim_process_memory(self, pid: int) -> bool:
        """Trim working set memory for the specified process.

        Returns True if successful, False otherwise.
        """
        return False

    def get_window_owning_pids(self) -> set[int]:
        """Return process IDs that own at least one visible top-level titled window."""
        return set()

    def get_foreground_pid(self) -> int | None:
        """Return the process ID owning the currently active foreground window, or None."""
        return None

    def close_process_windows(self, pid: int, timeout_s: float = 8.0) -> bool:
        """Post a graceful close message (e.g. WM_CLOSE or AppleScript quit) to process windows.

        Waits up to timeout_s for the process to terminate.
        Returns True if the process exited within timeout_s, False otherwise.
        """
        return False

    def never_offer_names(self) -> frozenset[str]:
        """Return lowercase set of process names that must never be offered for termination."""
        return frozenset()

    def query_recycle_bin(self) -> tuple[int, int]:
        """Return (num_items, total_size_bytes) in the system recycle bin or trash."""
        return (0, 0)

    def empty_recycle_bin(self) -> bool:
        """Empty the system recycle bin or trash. Returns True on success."""
        return False

    def app_cache_dirs(self) -> list[CacheLocation]:
        """Return browser and application cache directories (safe to delete cache only)."""
        return []

    def get_static_hardware(self) -> dict[str, Any]:
        """Return static hardware facts: host model, CPU model, GPUs, VRAM, and resolution."""
        return {
            "host_model": "unknown",
            "os_build": "unknown",
            "cpu_model": "unknown",
            "gpus": [],
            "resolution": "unknown",
        }

    def full_process_snapshot(self) -> list[ProcessSample]:
        """Return a complete process pass across the system for heavy-task analysis."""
        return []
