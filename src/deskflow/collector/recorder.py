"""Run recording and JSONL persistence."""

import json
import secrets
from datetime import UTC, datetime
from pathlib import Path
from typing import TextIO

import platformdirs

from deskflow.collector.models import Record


def generate_run_id() -> str:
    """Generate a unique run ID composed of UTC timestamp and 6 random hex characters."""
    now_str = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    random_hex = secrets.token_hex(3)
    return f"{now_str}_{random_hex}"


class RunRecorder:
    """Writes telemetry and probe records as JSON Lines, flushing after every write."""

    def __init__(
        self,
        user_data_dir: Path | None = None,
        out_path: Path | None = None,
        run_id: str | None = None,
    ) -> None:
        """Initialize the recorder.

        Args:
            user_data_dir: Root directory for user data. Defaults to platformdirs location.
            out_path: Explicit output file path. Overrides default naming if supplied.
            run_id: Explicit run_id. If not supplied, generated automatically.
        """
        self.run_id = run_id or generate_run_id()

        if out_path is not None:
            self.path = Path(out_path)
        else:
            base_dir = (
                user_data_dir
                if user_data_dir is not None
                else Path(platformdirs.user_data_dir("deskflow", ensure_exists=True))
            )
            runs_dir = base_dir / "runs"
            runs_dir.mkdir(parents=True, exist_ok=True)
            self.path = runs_dir / f"{self.run_id}.jsonl"

        self.path.parent.mkdir(parents=True, exist_ok=True)
        # File is held open across multiple write() calls and closed via close() or context manager
        self._file: TextIO | None = open(self.path, "a", encoding="utf-8")  # noqa: SIM115
        self._records_written = 0

    @property
    def records_written(self) -> int:
        """Total number of records written in this session."""
        return self._records_written

    def write(self, record: Record) -> None:
        """Serialize a Record to JSON and append it to the log file, flushing immediately."""
        if self._file is None:
            raise RuntimeError("RunRecorder is closed.")
        payload = json.dumps(record.to_dict())
        self._file.write(payload + "\n")
        self._file.flush()
        self._records_written += 1

    def close(self) -> None:
        """Flush and close the underlying file."""
        if self._file is not None:
            self._file.flush()
            self._file.close()
            self._file = None

    def __enter__(self) -> "RunRecorder":
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        self.close()


def load_run(path: Path | str) -> list[Record]:
    """Read a JSONL run file and return the list of validated Record instances.

    Args:
        path: Filepath to the .jsonl file.

    Returns:
        List of deserialized Record instances.

    Raises:
        ValueError: If a record has an incompatible schema_version or invalid JSON.
        FileNotFoundError: If the file does not exist.
    """
    file_path = Path(path)
    records: list[Record] = []

    with open(file_path, encoding="utf-8") as f:
        for line_no, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError as err:
                raise ValueError(f"Invalid JSON at {file_path}:{line_no}: {err}") from err

            # from_dict enforces schema_version validation
            record = Record.from_dict(data)
            records.append(record)

    return records
