"""Fault-injection lab for generating labeled system stress datasets."""

from deskflow.lab.background import BackgroundLoadScenario
from deskflow.lab.base import (
    DISK_MIN_FREE_BYTES,
    Intensity,
    Pattern,
    Scenario,
    ScenarioStats,
    get_memory_floor_bytes,
)
from deskflow.lab.combo import ComboScenario
from deskflow.lab.cpu import CpuPressureScenario
from deskflow.lab.disk import DiskPressureScenario
from deskflow.lab.idle import IdleScenario
from deskflow.lab.lock import cleanup_leftovers, get_scratch_dir
from deskflow.lab.memory import MemoryPressureScenario
from deskflow.lab.watchdog import LabWatchdog


def create_scenario(
    scenario_str: str,
    intensity: Intensity = "medium",
    pattern: Pattern = "constant",
    **kwargs: object,
) -> Scenario:
    """Create a Scenario instance from a string name."""
    s_names = [s.strip() for s in scenario_str.split("+")]
    scenarios: list[Scenario] = []

    for name in s_names:
        if name == "cpu_pressure":
            max_cores = kwargs.get("max_cores")
            scenarios.append(
                CpuPressureScenario(
                    intensity=intensity,
                    pattern=pattern,
                    max_cores=int(str(max_cores)) if max_cores is not None else None,
                )
            )
        elif name == "memory_pressure":
            target_bytes = kwargs.get("target_bytes")
            scenarios.append(
                MemoryPressureScenario(
                    intensity=intensity,
                    pattern=pattern,
                    target_bytes_override=int(str(target_bytes))
                    if target_bytes is not None
                    else None,
                )
            )
        elif name == "disk_pressure":
            scratch_dir = kwargs.get("scratch_dir")
            max_size = kwargs.get("max_file_size")
            scenarios.append(
                DiskPressureScenario(
                    intensity=intensity,
                    pattern=pattern,
                    scratch_dir=scratch_dir if scratch_dir is not None else None,  # type: ignore[arg-type]
                    max_file_size_override=int(str(max_size)) if max_size is not None else None,
                )
            )
        elif name == "background_load":
            scenarios.append(BackgroundLoadScenario(intensity=intensity, pattern=pattern))
        elif name == "idle":
            scenarios.append(IdleScenario(intensity=intensity, pattern=pattern))
        else:
            raise ValueError(f"Unknown lab scenario: {name}")

    if len(scenarios) == 1:
        return scenarios[0]
    return ComboScenario(sub_scenarios=scenarios, intensity=intensity, pattern=pattern)


__all__ = [
    "BackgroundLoadScenario",
    "ComboScenario",
    "CpuPressureScenario",
    "DISK_MIN_FREE_BYTES",
    "DiskPressureScenario",
    "IdleScenario",
    "Intensity",
    "LabWatchdog",
    "MemoryPressureScenario",
    "Pattern",
    "Scenario",
    "ScenarioStats",
    "cleanup_leftovers",
    "create_scenario",
    "get_memory_floor_bytes",
    "get_scratch_dir",
]
