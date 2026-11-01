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

## Platform Layer Isolation

All operating system distinctions live exclusively within `src/deskflow/platform/`. Modules outside this package never inspect `sys.platform` or call OS-specific bindings. If an operating system capability is absent or requires elevated privileges, the adapter safely returns `None` or an empty container.
