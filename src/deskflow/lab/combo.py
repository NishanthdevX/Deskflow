"""Combination scenario holding multiple concurrent injectors."""

from deskflow.lab.base import Intensity, Pattern, Scenario, ScenarioStats


class ComboScenario(Scenario):
    """Composite scenario executing multiple fault injection scenarios concurrently."""

    def __init__(
        self,
        sub_scenarios: list[Scenario],
        intensity: Intensity = "medium",
        pattern: Pattern = "constant",
    ) -> None:
        super().__init__(intensity=intensity, pattern=pattern)
        self.sub_scenarios = sub_scenarios

    @property
    def name(self) -> str:
        return "+".join(s.name for s in self.sub_scenarios)

    def start(self, hold_duration_s: float = 180.0) -> None:
        for s in self.sub_scenarios:
            s.start(hold_duration_s=hold_duration_s)
        self._is_running = True

    def stop(self) -> None:
        for s in self.sub_scenarios:
            s.stop()
        self._is_running = False

    def get_child_pids(self) -> list[int]:
        pids: list[int] = []
        for s in self.sub_scenarios:
            if hasattr(s, "get_child_pids"):
                pids.extend(s.get_child_pids())
        return pids

    def get_stats(self) -> ScenarioStats:
        workers = sum(s.get_stats().worker_count for s in self.sub_scenarios)
        alloc_bytes = sum(s.get_stats().bytes_allocated for s in self.sub_scenarios)
        return ScenarioStats(worker_count=workers, bytes_allocated=alloc_bytes)
