"""Execution plans for automated lab sessions."""

from dataclasses import dataclass
from typing import Literal

from deskflow.lab.base import Intensity, Pattern


@dataclass(frozen=True)
class LabRunSpec:
    """Specification of a single fault injection run in a plan."""

    scenario: str
    intensity: Intensity
    pattern: Pattern
    pre_s: float = 30.0
    hold_s: float = 180.0
    post_s: float = 60.0

    @property
    def total_duration_s(self) -> float:
        """Total duration of this run in seconds."""
        return self.pre_s + self.hold_s + self.post_s

    @property
    def key(self) -> str:
        """Deterministic run key for session resumption tracking."""
        return f"{self.scenario}:{self.intensity}:{self.pattern}"


PlanName = Literal["quick", "standard"]


def get_plan(plan_name: PlanName) -> list[LabRunSpec]:
    """Generate the ordered list of run specifications for a named plan."""
    if plan_name == "quick":
        return [
            LabRunSpec(scenario="cpu_pressure", intensity="medium", pattern="constant"),
            LabRunSpec(scenario="memory_pressure", intensity="medium", pattern="constant"),
            LabRunSpec(scenario="disk_pressure", intensity="medium", pattern="constant"),
            LabRunSpec(scenario="background_load", intensity="medium", pattern="constant"),
            LabRunSpec(scenario="idle", intensity="medium", pattern="constant"),
        ]

    if plan_name == "standard":
        scenarios = ["cpu_pressure", "memory_pressure", "disk_pressure", "background_load"]
        run_matrix: list[tuple[Intensity, Pattern]] = [
            ("low", "constant"),
            ("medium", "constant"),
            ("medium", "ramp"),
            ("high", "constant"),
            ("high", "burst"),
        ]

        runs: list[LabRunSpec] = []
        for s in scenarios:
            for intensity, pattern in run_matrix:
                runs.append(LabRunSpec(scenario=s, intensity=intensity, pattern=pattern))

        # Two idle runs included (total 20 runs)
        runs.append(LabRunSpec(scenario="idle", intensity="medium", pattern="constant"))
        runs.append(LabRunSpec(scenario="idle", intensity="medium", pattern="constant"))
        return runs

    raise ValueError(f"Unknown plan: {plan_name}")
