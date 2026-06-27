"""Memory watchdog and phase tracking.

Ported from train_lfm25_roundpipe_lora.py:
  - start_memory_watchdog()  — daemon thread aborting on RSS limit
  - mark_phase()             — elapsed-time phase marker
  - mark_memory_phase()      — system RAM snapshot
  - mark_gpu_memory_phase()  — GPU allocator snapshot

Call set_log_file(f) from train.py after opening the log file so that
phase/memory events are routed there instead of being dropped.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from typing import Any, Optional

import torch

from stratum.telemetry import gpu_memory_snapshot


# ---------------------------------------------------------------------------
# Log file routing
# ---------------------------------------------------------------------------

_log_file = None


def set_log_file(f) -> None:
    """Register the open log file handle for diagnostic event routing."""
    global _log_file
    _log_file = f


def _log(d: dict) -> None:
    """Write a structured event to the log file; silently drop if not set."""
    if _log_file is not None:
        _log_file.write(json.dumps(d) + "\n")
        _log_file.flush()


# ---------------------------------------------------------------------------
# System memory helpers
# ---------------------------------------------------------------------------

def _proc_status_kib(key: str) -> int:
    """Read a value from /proc/self/status in KiB."""
    try:
        with open("/proc/self/status", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith(f"{key}:"):
                    return int(line.split()[1])
    except OSError:
        return 0
    return 0


def _meminfo_kib(key: str) -> int:
    """Read a value from /proc/meminfo in KiB."""
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith(f"{key}:"):
                    return int(line.split()[1])
    except OSError:
        return 0
    return 0


def memory_snapshot() -> dict[str, float]:
    """Return system RAM snapshot: RSS, VMS, MemAvailable in GiB."""
    return {
        "rss_gib": round(_proc_status_kib("VmRSS") / 1024**2, 3),
        "vms_gib": round(_proc_status_kib("VmSize") / 1024**2, 3),
        "mem_available_gib": round(_meminfo_kib("MemAvailable") / 1024**2, 3),
    }


# ---------------------------------------------------------------------------
# Phase markers
# ---------------------------------------------------------------------------

_PHASE_T0 = time.perf_counter()


def mark_phase(name: str) -> None:
    """Record a timing phase marker to the log file."""
    now = time.perf_counter()
    _log({"event": "phase", "phase": name, "elapsed_sec": round(now - _PHASE_T0, 2)})


def mark_memory_phase(name: str, host_ram_limit_gib: float = 0.0) -> None:
    """Record a memory phase marker to the log file.

    If *host_ram_limit_gib* > 0 and RSS exceeds it, raises MemoryError.
    """
    snap = memory_snapshot()
    _log({"event": "phase_memory", "phase": name, **snap})
    if host_ram_limit_gib and snap["rss_gib"] > host_ram_limit_gib:
        raise MemoryError(
            f"host RSS {snap['rss_gib']} GiB exceeded --host-ram-limit-gib={host_ram_limit_gib}"
        )


def mark_gpu_memory_phase(name: str) -> None:
    """Record a GPU memory phase marker to the log file."""
    if not torch.cuda.is_available():
        return
    _log({"event": "phase_gpu_memory", "phase": name, **gpu_memory_snapshot()})


# ---------------------------------------------------------------------------
# Watchdog — abort when RSS exceeds limit
# ---------------------------------------------------------------------------

def start_memory_watchdog(host_ram_limit_gib: float, interval_sec: float = 1.0) -> None:
    """Start a daemon thread that aborts the process if RSS exceeds the limit.

    Reads /proc/self/status VmRSS every *interval_sec* seconds.
    Calls os._exit(137) on excess (SIGKILL-style, can't be caught).
    """
    if host_ram_limit_gib <= 0:
        return

    def watch() -> None:
        while True:
            rss_gib = _proc_status_kib("VmRSS") / 1024**2
            if rss_gib > host_ram_limit_gib:
                # Always emit to stderr — this is a fatal error event.
                print(json.dumps({
                    "event": "error",
                    "msg": "memory_watchdog_rss_limit_exceeded",
                    "rss_gib": round(rss_gib, 3),
                    "host_ram_limit_gib": host_ram_limit_gib,
                }), file=sys.stderr, flush=True)
                os._exit(137)
            time.sleep(interval_sec)

    thread = threading.Thread(target=watch, name="host-memory-watchdog", daemon=True)
    thread.start()
