"""Abstract base class and data types for platform-specific adapters."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal


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

    def thread_count(self) -> int | None:
        """Return total active thread count across the system, or None if unavailable."""
        return None
