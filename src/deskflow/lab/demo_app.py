"""A deliberately visible, safely closable workload for Deskflow demonstrations."""

import argparse
import threading
import time

import psutil


def capped_memory_mb(requested_mb: int) -> int:
    """Cap requested demo allocation while retaining the lab's 2 GiB/15 percent memory floor."""
    available_mb = int(psutil.virtual_memory().available / (1024 * 1024))
    floor_mb = max(2048, int(psutil.virtual_memory().total * 0.15 / (1024 * 1024)))
    cap = max(0, available_mb - floor_mb)
    return max(0, min(requested_mb, cap))


def run_demo(memory_mb: int, cpu_percent: float) -> int:
    """Run the visible tkinter demo process until its window is closed."""
    try:
        import tkinter as tk
    except ImportError:
        return 2
    actual_mb = capped_memory_mb(memory_mb)
    allocation = bytearray(actual_mb * 1024 * 1024)
    for idx in range(0, len(allocation), 4096):
        allocation[idx] = 1
    stop = threading.Event()

    def burn() -> None:
        period = 0.1
        duty = max(0.0, min(100.0, cpu_percent)) / 100.0
        while not stop.is_set():
            end = time.perf_counter() + period * duty
            while time.perf_counter() < end:
                pass
            stop.wait(max(0.0, period * (1.0 - duty)))

    if cpu_percent > 0:
        threading.Thread(target=burn, daemon=True).start()
    root = tk.Tk()
    root.title("DESKFLOW Demo App (safe to close)")
    text = f"Memory: {actual_mb} MiB\nCPU: {cpu_percent:.0f}%\nSafe to close"
    tk.Label(root, text=text).pack(padx=28, pady=24)

    def close() -> None:
        """Release allocation by ending the Tk event loop."""
        stop.set()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", close)
    root.mainloop()
    return 0


def main() -> None:
    """Parse detached demo process arguments and run the window."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--memory-mb", type=int, default=2048)
    parser.add_argument("--cpu-percent", type=float, default=0.0)
    args = parser.parse_args()
    raise SystemExit(run_demo(args.memory_mb, args.cpu_percent))


if __name__ == "__main__":
    main()
