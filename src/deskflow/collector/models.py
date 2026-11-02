"""Telemetry and performance data models."""

from dataclasses import asdict, dataclass
from typing import Any

SCHEMA_VERSION: int = 1


@dataclass(frozen=True)
class SystemSample:
    """Snapshot of raw system telemetry and computed delta rates."""

    ts_utc: str
    monotonic_s: float
    cpu_percent_mean: float
    cpu_percent_max_core: float
    cpu_freq_current_mhz: float | None
    cpu_freq_max_mhz: float | None
    cpu_freq_reliable: bool | None
    ctx_switches_per_s: float | None
    interrupts_per_s: float | None
    mem_total_bytes: int
    mem_available_bytes: int
    mem_percent: float
    swap_used_bytes: int
    swap_percent: float
    disk_read_bytes_per_s: float | None
    disk_write_bytes_per_s: float | None
    disk_busy_ms_per_s: float | None
    net_recv_bytes_per_s: float | None
    net_sent_bytes_per_s: float | None
    net_errors_per_s: float | None
    net_drops_per_s: float | None
    process_count: int
    thread_count: int | None
    battery_percent: float | None
    battery_plugged: bool | None
    max_temperature_c: float | None
    uptime_s: float

    def to_dict(self) -> dict[str, Any]:
        """Convert sample to JSON-serializable dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SystemSample":
        """Reconstruct sample from dictionary."""
        return cls(**data)


@dataclass(frozen=True)
class ProbeResult:
    """Measured execution times of fixed micro-workloads in nanoseconds."""

    ts_utc: str
    cpu_ns: int
    mem_ns: int
    disk_ns: int
    total_ns: int

    def to_dict(self) -> dict[str, Any]:
        """Convert probe result to JSON-serializable dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ProbeResult":
        """Reconstruct probe result from dictionary."""
        return cls(**data)


@dataclass(frozen=True)
class ProcessSample:
    """Resource snapshot for a single process."""

    pid: int
    name: str
    rss_bytes: int
    cpu_percent: float
    num_threads: int
    age_s: float
    is_protected: bool

    def to_dict(self) -> dict[str, Any]:
        """Convert process sample to JSON-serializable dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ProcessSample":
        """Reconstruct process sample from dictionary."""
        return cls(**data)


@dataclass(frozen=True)
class StorageEntry:
    """Size report for a temporary or cache folder."""

    path: str
    kind: str
    bytes: int
    cleanable_bytes: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert entry to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StorageEntry":
        """Reconstruct entry from dictionary."""
        return cls(
            path=data["path"],
            kind=data["kind"],
            bytes=int(data["bytes"]),
            cleanable_bytes=int(data.get("cleanable_bytes", 0)),
        )


@dataclass(frozen=True)
class StorageReport:
    """Report on temporary, cache, and trash storage consumption."""

    ts_utc: str
    entries: list[StorageEntry]
    truncated: bool
    duration_s: float

    def to_dict(self) -> dict[str, Any]:
        """Convert report to JSON-serializable dictionary."""
        return {
            "ts_utc": self.ts_utc,
            "entries": [e.to_dict() for e in self.entries],
            "truncated": self.truncated,
            "duration_s": self.duration_s,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StorageReport":
        """Reconstruct report from dictionary."""
        entries = [
            StorageEntry.from_dict(e) if isinstance(e, dict) else e for e in data.get("entries", [])
        ]
        return cls(
            ts_utc=data["ts_utc"],
            entries=entries,
            truncated=bool(data["truncated"]),
            duration_s=float(data["duration_s"]),
        )


@dataclass(frozen=True)
class Record:
    """Complete collection record written to run logs."""

    schema_version: int
    run_id: str
    label: str
    sample: SystemSample
    probe: ProbeResult | None
    top_processes: list[ProcessSample]
    storage: StorageReport | None

    def to_dict(self) -> dict[str, Any]:
        """Convert record to JSON-serializable dictionary."""
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "label": self.label,
            "sample": self.sample.to_dict(),
            "probe": self.probe.to_dict() if self.probe is not None else None,
            "top_processes": [p.to_dict() for p in self.top_processes],
            "storage": self.storage.to_dict() if self.storage is not None else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Record":
        """Reconstruct record from dictionary and validate schema version."""
        schema_v = data.get("schema_version")
        if schema_v != SCHEMA_VERSION:
            raise ValueError(
                f"Schema version mismatch: record has version {schema_v}, "
                f"expected {SCHEMA_VERSION}."
            )

        sample = SystemSample.from_dict(data["sample"])
        probe = ProbeResult.from_dict(data["probe"]) if data.get("probe") is not None else None
        top_processes = [ProcessSample.from_dict(p) for p in data.get("top_processes", [])]
        storage = (
            StorageReport.from_dict(data["storage"]) if data.get("storage") is not None else None
        )

        return cls(
            schema_version=int(schema_v),
            run_id=str(data["run_id"]),
            label=str(data["label"]),
            sample=sample,
            probe=probe,
            top_processes=top_processes,
            storage=storage,
        )
