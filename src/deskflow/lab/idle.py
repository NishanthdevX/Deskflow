"""Idle reference scenario."""

from deskflow.lab.base import Intensity, Pattern, Scenario, ScenarioStats


class IdleScenario(Scenario):
    """Reference idle scenario performing no injection."""

    def __init__(
        self,
        intensity: Intensity = "medium",
        pattern: Pattern = "constant",
    ) -> None:
        super().__init__(intensity=intensity, pattern=pattern)

    @property
    def name(self) -> str:
        return "idle"

    def start(self, hold_duration_s: float = 180.0) -> None:
        self._is_running = True

    def stop(self) -> None:
        self._is_running = False

    def get_stats(self) -> ScenarioStats:
        return ScenarioStats(worker_count=0, bytes_allocated=0)

