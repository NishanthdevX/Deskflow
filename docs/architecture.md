# Architecture

## Module map

```
src/deskflow/
    __init__.py          Package version (read from installed metadata).
    cli.py               Typer application; root entry point for the deskflow command.
    safety.py            Process protection rules (system PIDs, shells, user ownership).
    py.typed             PEP 561 marker.

    platform/            OS-specific code behind a common abstract interface.
        __init__.py      Public platform API exposing PlatformAdapter and types.
        base.py          Abstract base class defining PlatformAdapter, CpuFreq, BatteryInfo.
        windows.py       Windows implementation using psutil, tempfile, env vars.
        macos.py         macOS implementation using standard Apple cache paths and psutil.
        factory.py       get_adapter() runtime dispatcher.

    collector/           Telemetry sampling, micro-benchmark probe, run persistence.
        __init__.py      Public collector exports.
        models.py        SystemSample, ProbeResult, ProcessSample, StorageReport, Record.
        sampler.py       psutil telemetry sampler with delta rates and process enumeration.
        probe.py         Micro-workload performance probe (CPU, memory, disk).
        storage.py       Storage scanner with explicit stack, symlink guards, and TTL cache.
        recorder.py      RunRecorder (JSONL writer) and load_run reader.

    commands/            CLI subcommands.
        snapshot.py      `deskflow snapshot` command (instant table or JSON).
        collect.py       `deskflow collect` continuous sampling loop.
        doctor.py        `deskflow doctor` platform capability and baseline inspector.

    features/            Feature engineering (Phase 2).
    ml/                  Machine-learning models (Phase 3).
    health/              Health score derivation (Phase 4).
    remediation/         Safe system fixes (Phase 6).
    evaluation/          Live accuracy tracking (Phase 7).
```

## Performance Probe Specifications

The probe serves as ground truth for system responsiveness by timing three fixed synthetic workloads with `time.perf_counter_ns()` sequentially on the sampling thread:

1. **CPU Micro-workload**:
   - Pure-Python integer arithmetic loop executing 120,000 iterations of a linear congruential update:
     `state = (state * 1103515245 + 12345) & 0x7FFFFFFF`
   - Target execution time on idle modern CPU: ~15-25 ms.
2. **Memory Micro-workload**:
   - Allocates an 8 MiB `bytearray` and writes one byte every 4096-byte page stride (2048 writes total), then deallocates.
   - Target execution time on idle modern CPU: ~1-3 ms.
3. **Disk Micro-workload**:
   - Writes 1 MiB in 64 KiB chunks to a temporary file in `<user_data_dir>`, executes an explicit `os.fsync()`, and unlinks the file in a `try...finally` block.
   - Target execution time on idle modern SSD: ~5-15 ms.

### Measured Probe Latency

On a reference modern multi-core development laptop under normal idle state:
- CPU probe: ~25.49 ms
- Memory probe: ~1.95 ms
- Disk probe: ~5.58 ms
- **Total Probe Cost**: ~33.02 ms (well within the 60 ms ceiling).

## Measured Collector Overhead

Collector overhead is measured using `scripts/measure_overhead.py`, executing `deskflow collect --duration 60 --interval 2 --probe-interval 5` as a subprocess and tracking child process user + system CPU time with `psutil`.

### Baseline (Before Phase 2.1 Fixes)

- **Machine**: AMD64 Family 25 Model 68 Stepping 1, AuthenticAMD (Windows 11, 8 physical cores, 16 logical cores)
- **Subprocess wall time**: 62.62 s
- **User CPU time**: 2.67 s
- **System CPU time**: 19.34 s
- **Total CPU time**: 22.02 s
- **Measured CPU Overhead**: 35.15% of one logical core (2.20% of overall 16-core system capacity).

#### Stage-by-Stage Profile (30 ticks)

| Stage | Mean (ms) | Max (ms) | Notes |
|---|---|---|---|
| `system_counters` | 679.47 | 788.61 | Dominated by per-tick process table iteration |
| `process_enumeration` | 519.55 | 736.62 | `psutil.process_iter` per-tick CPU time and memory stats |
| `thread_counting` | 76.37 | 89.01 | ToolHelp32 system-wide thread snapshot traversal |
| `storage_report` | 0.01 | 0.01 | Retrieval of cached storage report |
| `probe` | 47.81 | 62.87 | Sequential synthetic CPU, memory, and disk probe |
| `json_write` | 0.50 | 0.79 | JSON formatting and line flush |
| `total_tick` | 1323.71 | 1537.25 | Over 1.3 seconds elapsed per sampling tick |

#### cProfile Root Cause Analysis

In a 30s collection run (11 ticks):
- `psutil_windows.proc_info`: 5.85s cumulative time inside Windows kernel process queries.
- `psutil.cpu_percent` per process: 5.31s cumulative time querying per-process kernel/user time across 200+ running processes every 2 seconds.
- `CreateToolhelp32Snapshot` / `Thread32Next`: 76 ms per tick traversing thousands of threads.
- Background `StorageScanner`: background file scanning (`nt.scandir`) running concurrently during probe execution.

### Optimized (After Phase 2.1 Fixes)

- **Machine**: AMD64 Family 25 Model 68 Stepping 1, AuthenticAMD (Windows 11, 8 physical cores, 16 logical cores)
- **Subprocess wall time**: 61.09 s
- **User CPU time**: 0.89 s (down from 2.67 s)
- **System CPU time**: 0.30 s (down from 19.34 s)
- **Total CPU time**: 1.19 s (down from 22.02 s)
- **Measured CPU Overhead**: **1.94% of one logical core** (0.12% of overall 16-core system capacity, well under the 3.0% acceptance ceiling).

#### Optimized Stage-by-Stage Profile (30 ticks)

| Stage | Mean (ms) | Max (ms) | Notes |
|---|---|---|---|
| `system_counters` | 15.88 | 21.49 | `GetPerformanceInfo` + `GlobalMemoryStatusEx` + cached boot time |
| `thread_counting` | 0.13 | 0.29 | Direct `GetPerformanceInfo` (0.13 ms vs 76 ms ToolHelp snapshot) |
| `process_enumeration` | 0.02 | 0.02 | Amortized: `NtQuerySystemInformation` (4.6 ms every 10s) |
| `storage_report` | 0.00 | 0.01 | Paused during probe; disabled by default in `collect` / `lab run` |
| `probe` | 40.27 | 57.59 | Fixed synthetic CPU, memory, and disk micro-workloads |
| `json_write` | 0.40 | 0.73 | Fast line serialization |
| `total_tick` | 56.70 | 79.80 | Total active work under 60 ms per sampling tick |

#### Fixes Implemented

1. **Kernel Snapshot via `GetPerformanceInfo`**: Replaced thread enumeration and ToolHelp snapshots with `GetPerformanceInfo` from `psapi.dll`, dropping thread counting cost from 76 ms to 0.13 ms.
2. **Fast Memory Status via `GlobalMemoryStatusEx`**: Replaced `psutil.swap_memory()` with `GlobalMemoryStatusEx`, avoiding repeated pagefile disk enumeration (0.01 ms vs 17.4 ms per tick).
3. **Low-Overhead Process Snapshot**: Replaced full-system `psutil.process_iter` with `NtQuerySystemInformation(SystemProcessInformation)` in `src/deskflow/platform/windows.py`. Processes are fetched in 4.6 ms, and per-process enumeration runs on its own 10s cadence (`--process-interval`).
4. **Ownership Caching**: Process user ownership is cached by `(pid, create_time)` and never re-evaluated via slow LSASS RPC calls. Usernames are never logged.
5. **Storage Scanner Isolation**: Background storage scan is disabled by default in `collect` and `lab run` (enabled only with `--storage`). When enabled, a thread synchronization event guarantees it pauses whenever a probe executes.
6. **Lazy Module Loading**: Heavy data processing packages (`pandas`, `pyarrow`, `numpy`) are loaded lazily inside dataset execution commands, reducing CLI startup CPU time by 0.86 seconds.

## Platform Layer Isolation

All operating system distinctions live exclusively within `src/deskflow/platform/`. Modules outside this package never inspect `sys.platform` or call OS-specific bindings. If an operating system capability is absent or requires elevated privileges, the adapter safely returns `None` or an empty container.
