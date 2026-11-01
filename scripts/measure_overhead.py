"""Script to measure CPU overhead of deskflow collect subprocess."""

import argparse
import platform
import subprocess
import sys
import time

import psutil


def measure_collector_overhead(
    duration: int = 60, interval: int = 2, probe_interval: int = 5
) -> None:
    cmd = [
        sys.executable,
        "-m",
        "deskflow.cli",
        "collect",
        "--duration",
        str(duration),
        "--interval",
        str(interval),
        "--probe-interval",
        str(probe_interval),
    ]

    print(f"Executing: {' '.join(cmd)}")
    start_wall = time.monotonic()
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    pid = proc.pid
    ps_proc = psutil.Process(pid)

    last_user = 0.0
    last_system = 0.0

    while proc.poll() is None:
        try:
            times = ps_proc.cpu_times()
            last_user = times.user
            last_system = times.system
            # Include child processes if any
            for child in ps_proc.children(recursive=True):
                try:
                    c_times = child.cpu_times()
                    last_user += c_times.user
                    last_system += c_times.system
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            break
        time.sleep(0.2)

    # Final check
    try:
        times = ps_proc.cpu_times()
        last_user = times.user
        last_system = times.system
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass

    end_wall = time.monotonic()
    wall_elapsed = end_wall - start_wall
    total_cpu_time = last_user + last_system
    cpu_percent_of_one_core = (total_cpu_time / wall_elapsed) * 100.0 if wall_elapsed > 0 else 0.0

    print("-" * 50)
    print("DESKFLOW COLLECT OVERHEAD MEASUREMENT")
    print("-" * 50)
    print(f"OS: {platform.system()} {platform.release()} ({platform.machine()})")
    print(f"Processor: {platform.processor()}")
    print(f"Logical cores: {psutil.cpu_count(logical=True)}")
    print(f"Physical cores: {psutil.cpu_count(logical=False)}")
    print(f"Subprocess duration (wall): {wall_elapsed:.2f} s")
    print(f"User CPU time: {last_user:.3f} s")
    print(f"System CPU time: {last_system:.3f} s")
    print(f"Total CPU time: {total_cpu_time:.3f} s")
    print(f"CPU overhead: {cpu_percent_of_one_core:.2f}% of one core")
    print("-" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Measure deskflow collect CPU overhead.")
    parser.add_argument(
        "--duration", type=int, default=60, help="Run duration in seconds (default 60)"
    )
    parser.add_argument(
        "--interval", type=int, default=2, help="Sample interval in seconds (default 2)"
    )
    parser.add_argument(
        "--probe-interval", type=int, default=5, help="Probe interval in seconds (default 5)"
    )
    args = parser.parse_args()
    measure_collector_overhead(
        duration=args.duration, interval=args.interval, probe_interval=args.probe_interval
    )
