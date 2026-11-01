"""Base scenario definitions and safety constants for the fault-injection lab."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Literal

import psutil

Intensity = Literal["low", "medium", "high"]
Pattern = Literal["constant", "ramp", "burst"]

# Safety floor constants
DISK_MIN_FREE_BYTES = 5 * 1024 * 1024 * 1024  # 5 GiB
MEM_FLOOR_MIN_BYTES = 2 * 1024 * 1024 * 1024  # 2 GiB
MEM_FLOOR_FRACTION = 0.15  # 15% total RAM


def get_memory_floor_bytes() -> int:
    """Return dynamic safety floor for available memory."""
    try:
        total = psutil.virtual_memory().total
        return max(MEM_FLOOR_MIN_BYTES, int(total * MEM_FLOOR_FRACTION))
    except Exception:
        return MEM_FLOOR_MIN_BYTES


@dataclass
class ScenarioStats:
    """Telemetry report of allocated resources during an injection run."""

    worker_count: int = 0
    bytes_allocated: int = 0
    extra: dict[str, Any] | None = None


class Scenario(ABC):
    """Abstract base class for all fault injection scenarios."""

    def __init__(self, intensity: Intensity = "medium", pattern: Pattern = "constant") -> None:
        self.intensity = intensity
        self.pattern = pattern
        self._is_running = False

    @property
    @abstractmethod
    def name(self) -> str:
        """Scenario name identifier."""

    @abstractmethod
    def start(self, hold_duration_s: float = 180.0) -> None:
        """Start injecting stress for the planned hold duration."""

    @abstractmethod
    def stop(self) -> None:
        """Stop all injection and clean up child processes/threads."""

    @abstractmethod
    def get_stats(self) -> ScenarioStats:
        """Return actual worker counts and allocated resources."""

    @property
    def is_running(self) -> bool:
        """Return whether scenario is actively injecting stress."""
        return self._is_running

