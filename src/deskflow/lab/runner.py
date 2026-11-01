"""Orchestration of lab fault-injection runs with concurrent telemetry collection."""

import atexit
import json
import logging
import signal
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from deskflow import __version__
from deskflow.baseline import get_or_create_machine_id
from deskflow.collector import (
    SCHEMA_VERSION,
    ProbeResult,
    Record,
    RunRecorder,
    Sampler,
    StorageScanner,
    run_probe,
)
from deskflow.lab import (
    LabWatchdog,
    Scenario,
    cleanup_leftovers,
    get_scratch_dir,
)
from deskflow.lab.lock import register_child_pids, unregister_children
from deskflow.platform import get_adapter

logger = logging.getLogger(__name__)


class LabRunner:
    """Coordinates a 3-phase lab run: Pre (baseline), Inject (stress), Post (recovery)."""

    def __init__(
        self,
        scenario: Scenario,
        pre_duration_s: float = 30.0,
        hold_duration_s: float = 180.0,
        post_duration_s: float = 60.0,
        interval_s: float = 2.0,
        probe_interval_s: float = 4.0,
        session_id: str | None = None,
        out_dir: Path | None = None,
        config_dir: Path | None = None,
        status_callback: Callable[[str, float, float], None] | None = None,
    ) -> None:
        self.scenario = scenario
        self.pre_duration_s = pre_duration_s
        self.hold_duration_s = hold_duration_s
        self.post_duration_s = post_duration_s
        self.interval_s = interval_s
        self.probe_interval_s = probe_interval_s
        self.config_dir = config_dir
        self.status_callback = status_callback

        self.machine_id = get_or_create_machine_id(config_dir=config_dir)
        self.session_id = session_id or f"lab-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
        self.scratch_dir = get_scratch_dir()
        self.recorder = RunRecorder(user_data_dir=out_dir)

        self._aborted = False
        self._abort_reason: str | None = None
        self._watchdog: LabWatchdog | None = None

    def _handle_abort(self, reason: str) -> None:
        """Watchdog callback triggered on resource threshold breach."""
        self._aborted = True
        self._abort_reason = reason
        logger.warning("Lab execution aborted: %s", reason)
        self.scenario.stop()
        unregister_children(scratch_dir=self.scratch_dir)

    def run(self) -> Path:
        """Execute the 3-phase lab run and write JSONL and metadata sidecar."""
        # Clean any leftovers from past runs
        cleanup_leftovers(scratch_dir=self.scratch_dir)

        adapter = get_adapter()
        sampler = Sampler(adapter=adapter, interval_s=self.interval_s)
        storage_scanner = StorageScanner(adapter=adapter)
        storage_scanner.start_background_scan()

        label = f"lab:{self.scenario.name}:{self.scenario.intensity}:{self.scenario.pattern}"
        total_duration = self.pre_duration_s + self.hold_duration_s + self.post_duration_s

        # Metadata timeline tracking
        planned_dict = {
            "pre_s": self.pre_duration_s,
            "hold_s": self.hold_duration_s,
            "post_s": self.post_duration_s,
            "total_s": total_duration,
        }
        actual_timestamps: dict[str, str | None] = {
            "pre_start": None,
            "inject_start": None,
            "inject_end": None,
            "post_end": None,
        }

        # Watchdog setup
        self._watchdog = LabWatchdog(
            on_abort=self._handle_abort,
            scratch_dir=self.scratch_dir,
            check_interval_s=1.0,
        )

        def _cleanup() -> None:
            if self._watchdog:
                self._watchdog.stop()
            self.scenario.stop()
            unregister_children(scratch_dir=self.scratch_dir)

        atexit.register(_cleanup)

        # Signal handlers for graceful abort
        original_sigint = signal.getsignal(signal.SIGINT)
        original_sigterm = signal.getsignal(signal.SIGTERM)

        def _sig_handler(signum: int, frame: Any) -> None:
            self._handle_abort(f"Terminated by signal {signum}")
            _cleanup()
            raise KeyboardInterrupt()

        try:
            signal.signal(signal.SIGINT, _sig_handler)
            signal.signal(signal.SIGTERM, _sig_handler)
        except (ValueError, AttributeError):
            pass

        start_mono = time.monotonic()
        inject_start_mono = start_mono + self.pre_duration_s
        inject_end_mono = inject_start_mono + self.hold_duration_s
        end_mono = inject_end_mono + self.post_duration_s

        actual_timestamps["pre_start"] = datetime.now(UTC).isoformat()
        current_phase = "pre"

        next_sample_mono = start_mono
        next_probe_mono = start_mono

        try:
            self._watchdog.start()

            while time.monotonic() < end_mono:
                if self._aborted:
                    break

                now_mono = time.monotonic()

                # Phase transitions
                if current_phase == "pre" and now_mono >= inject_start_mono:
                    current_phase = "inject"
                    actual_timestamps["inject_start"] = datetime.now(UTC).isoformat()
                    self.scenario.start(hold_duration_s=self.hold_duration_s)
                    if hasattr(self.scenario, "get_child_pids"):
                        register_child_pids(
                            self.scenario.get_child_pids(), scratch_dir=self.scratch_dir
                        )

                elif current_phase == "inject" and now_mono >= inject_end_mono:
                    current_phase = "post"
                    actual_timestamps["inject_end"] = datetime.now(UTC).isoformat()
                    self.scenario.stop()
                    unregister_children(scratch_dir=self.scratch_dir)

                # Collect telemetry sample
                sample, processes = sampler.sample()

                # Collect probe if due
                probe: ProbeResult | None = None
                if now_mono >= next_probe_mono:
                    probe = run_probe()
                    while next_probe_mono <= now_mono:
                        next_probe_mono += self.probe_interval_s

                storage = storage_scanner.get_cached_report()
                record = Record(
                    schema_version=SCHEMA_VERSION,
                    run_id=self.recorder.run_id,
                    label=label,
                    sample=sample,
                    probe=probe,
                    top_processes=processes[:10],
                    storage=storage,
                )
                self.recorder.write(record)

                if self.status_callback is not None:
                    elapsed = now_mono - start_mono
                    self.status_callback(current_phase, elapsed, total_duration)

                # Sleep to maintain sampling interval
                next_sample_mono += self.interval_s
                sleep_s = max(0.0, next_sample_mono - time.monotonic())
                if sleep_s > 0:
                    time.sleep(sleep_s)

        finally:
            _cleanup()
            atexit.unregister(_cleanup)
            try:
                signal.signal(signal.SIGINT, original_sigint)
                signal.signal(signal.SIGTERM, original_sigterm)
            except (ValueError, AttributeError):
                pass

        if actual_timestamps["inject_end"] is None and current_phase == "inject":
            actual_timestamps["inject_end"] = datetime.now(UTC).isoformat()
        actual_timestamps["post_end"] = datetime.now(UTC).isoformat()

        stats = self.scenario.get_stats()

        # Write sidecar JSON
        meta_dict = {
            "scenario": self.scenario.name,
            "intensity": self.scenario.intensity,
            "pattern": self.scenario.pattern,
            "machine_id": self.machine_id,
            "session_id": self.session_id,
            "deskflow_version": __version__,
            "planned": planned_dict,
            "actual": actual_timestamps,
            "aborted": self._aborted,
            "abort_reason": self._abort_reason,
            "worker_count": stats.worker_count,
            "bytes_allocated": stats.bytes_allocated,
        }

        meta_path = self.recorder.path.with_suffix(".meta.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta_dict, f, indent=2)

        return self.recorder.path
