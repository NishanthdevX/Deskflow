Read AGENTS.md fully. Phase 0 is complete. Implement Phase 1 only: the telemetry collector,
the performance probe, storage scanning, run recording, and three CLI commands. No ML.

A. Platform layer (src/deskflow/platform/)
- base.py: abstract PlatformAdapter with methods:
  cpu_freq() -> CpuFreq | None (current_mhz, max_mhz, reliable: bool)
  temperatures() -> dict[str, float] (empty if unavailable)
  battery() -> BatteryInfo | None (percent, plugged_in)
  temp_and_cache_dirs() -> list[CacheLocation] (path, kind: "temp" | "cache" | "trash",
  safe_to_clean: bool)
  protected_process_names() -> frozenset[str]
  name: str
- windows.py and macos.py implement it, selected by get_adapter() in __init__.py.
  Windows: temp dirs = user temp and %LOCALAPPDATA%\Temp; also C:\Windows\Temp as
  read-only (safe_to_clean False). macOS: ~/Library/Caches (cache), tempfile temp dir
  (temp), ~/.Trash (trash, safe_to_clean False).
  cpu_freq: use psutil.cpu_freq(); set reliable=False when current equals max on every
  reading or when psutil returns None (psutil often reports base MHz on Windows and nothing
  on Apple Silicon). Do not attempt PDH or WMI in this phase.
- Protected process names (case-insensitive) must cover at least:
  Windows: System, System Idle Process, Registry, smss.exe, csrss.exe, wininit.exe,
  services.exe, lsass.exe, winlogon.exe, svchost.exe, explorer.exe, dwm.exe,
  fontdrvhost.exe, MsMpEng.exe, SearchIndexer.exe.
  macOS: kernel_task, launchd, WindowServer, loginwindow, Finder, Dock, SystemUIServer,
  coreaudiod, mds, mds_stores, distnoted, cfprefsd.
- src/deskflow/safety.py: is_protected(name, pid, username) returns True for protected
  names, processes not owned by the current user, the current process, its parent chain,
  and PID 0/1/4. Pure function, fully unit tested.

B. Data models (src/deskflow/collector/models.py), frozen dataclasses with to_dict() and
from_dict(), SCHEMA_VERSION = 1:
- SystemSample: ts_utc (ISO 8601), monotonic_s, cpu_percent_mean, cpu_percent_max_core,
  cpu_freq_current_mhz, cpu_freq_max_mhz, cpu_freq_reliable, ctx_switches_per_s,
  interrupts_per_s, mem_total_bytes, mem_available_bytes, mem_percent, swap_used_bytes,
  swap_percent, disk_read_bytes_per_s, disk_write_bytes_per_s, disk_busy_ms_per_s (None if
  unavailable), net_recv_bytes_per_s, net_sent_bytes_per_s, net_errors_per_s,
  net_drops_per_s, process_count, thread_count, battery_percent, battery_plugged,
  max_temperature_c, uptime_s. All optional fields are None when unavailable.
  Rates are computed from deltas between consecutive samples; the first sample has rates
  set to None.
- ProbeResult: ts_utc, cpu_ns, mem_ns, disk_ns, total_ns.
- ProcessSample: pid, name, rss_bytes, cpu_percent, num_threads, age_s, is_protected.
  Never record command lines or usernames (privacy).
- StorageReport: ts_utc, entries (path, kind, bytes), truncated: bool, duration_s.
- Record: schema_version, run_id, label, sample, probe (nullable), top_processes (top 10 by
  rss), storage (nullable).

C. Sampler (src/deskflow/collector/sampler.py)
- Sampler(adapter, interval_s) with sample() -> tuple[SystemSample, list[ProcessSample]].
  Prime psutil cpu_percent counters at construction. Per-process cpu_percent must be
  primed on first pass (first reading is 0.0; document this).
- Catch psutil.AccessDenied, NoSuchProcess and ZombieProcess per process and skip.
- Process enumeration must stay under 100 ms on a typical machine; use
  psutil.process_iter with an attrs list, not repeated per-process calls.

D. Probe (src/deskflow/collector/probe.py)
- Three fixed workloads measured with time.perf_counter_ns:
  cpu: a pure-Python integer arithmetic loop of CPU_PROBE_ITERATIONS using a linear
  congruential update; choose the constant so a modern laptop takes roughly 10 to 20 ms
  idle. mem: allocate a bytearray of 8 MiB and write one byte per 4096-byte stride, then
  release it. disk: write 1 MiB to a temp file in the DESKFLOW data directory, fsync, then
  delete it; always clean up even on error.
- run_probe() -> ProbeResult. Constants are module-level and documented. Total probe cost
  must stay under about 60 ms on an idle laptop; document the measured value in
  docs/architecture.md after measuring it.
- The probe runs on the sampler thread, sequentially, never concurrently with sampling.

E. Storage scanner (src/deskflow/collector/storage.py)
- StorageScanner.scan() walks the adapter's cache locations with os.scandir (no
  recursion limit by stack; use an explicit stack), never follows symlinks, enforces a
  time budget (default 3 s) and an entry cap (default 200000), and sets truncated when a
  limit is hit. Results are cached for 300 s by default. Runs in a background thread so it
  never delays sampling.

F. Recording (src/deskflow/collector/recorder.py)
- RunRecorder writes one JSON object per line to
  <user_data_dir>/runs/<run_id>.jsonl, flushing every record. run_id is a UTC timestamp
  plus 6 random hex characters. Provide load_run(path) -> list[Record] that validates
  schema_version and raises a clear error on mismatch.

G. CLI (register in cli.py; keep each command in its own module under src/deskflow/commands/)
- deskflow snapshot [--json]: take two samples 1 second apart (so rates exist), run one
  probe, print a compact table (CPU, memory, swap, disk, network, probe timings, top 5
  processes by memory). With --json print the Record as JSON.
- deskflow collect --duration SECONDS (default 600) --interval SECONDS (default 2)
  --probe-interval SECONDS (default 5, must be a multiple of interval) --label TEXT
  (default "unlabeled") [--out PATH]. Prints one status line every 10 seconds (samples
  written, last probe total in ms). Ctrl+C stops cleanly, flushes the file and prints the
  output path.
- deskflow doctor: prints platform, Python version, and a capability table: CPU
  frequency (available / reliable), temperatures, battery, disk busy time, cache
  locations found with sizes, probe baseline (median of 5 probes with min and max), and a
  one-line verdict on what this machine can and cannot measure.

H. Tests
- Unit tests with monkeypatched psutil: rate computation from deltas, first-sample None
  handling, per-process exceptions skipped, schema round trip (to_dict/from_dict), recorder
  write/load, schema_version mismatch error, safety.is_protected cases, storage scanner
  limits (use tmp_path with generated files and a symlink loop).
- Integration tests (@pytest.mark.integration, real psutil, run on both OSes in CI):
  Sampler returns plausible values; run_probe returns positive nanoseconds under 500 ms;
  `deskflow collect --duration 6 --interval 1 --probe-interval 2 --label test` writes at
  least 5 valid records.
- Pure-logic tests must not depend on the host OS.

I. Documentation
- docs/data-schema.md: every field, unit, nullability, and the platform caveats.
- docs/architecture.md: update with collector, probe and platform layer, the probe
  workloads, and the measured probe cost.
- README: replace the "not yet available" usage section for snapshot, collect and doctor
  with real usage examples.
- CHANGELOG: Phase 1 entry.

J. Verification (run and paste real output)
uv run ruff check . && uv run ruff format --check . && uv run mypy src && uv run pytest
uv run deskflow doctor
uv run deskflow snapshot
uv run deskflow collect --duration 20 --interval 1 --probe-interval 5 --label smoke
Then confirm the JSONL file has at least 19 lines, each loadable with load_run.

Acceptance criteria
- All checks pass; CI green on Windows and macOS (state if you could not run macOS
  locally and rely on CI).
- Commands run as a normal non-admin user with no exceptions on a missing capability.
- Collector overhead: measure CPU use of `deskflow collect` over 60 seconds on an idle
  machine and report the number; target under 3 percent of one core.
- No emojis, no print() outside CLI output modules, no sys.platform checks outside
  src/deskflow/platform/.