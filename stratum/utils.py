"""Device detection, memory probes, and logging utilities."""

import ctypes
import gc
import json
import logging
import os
import sys

import torch

# Stratum logger — library-level, not application-level.
# Users can control verbosity via stratum.setLogLevel() or
# by configuring the "stratum" logger directly.
_log = logging.getLogger("stratum")
_log.setLevel(logging.INFO)
_log.addHandler(logging.StreamHandler(sys.stderr))


def set_log_level(level: int | str) -> None:
    """Set stratum's log level. Accepts logging.DEBUG, logging.INFO, etc."""
    _log.setLevel(level)


def log_event(event: str, **fields) -> None:
    """Emit a structured log line at INFO level.

    Produces JSON-compatible output for machine parsing:
      [stratum] event field1=val1 field2=val2
    """
    parts = [f"[stratum] {event}"]
    for k, v in fields.items():
        parts.append(f"{k}={v}")
    _log.info(" ".join(parts))


def get_device_info() -> list[dict]:
    """Return info for each visible CUDA device."""
    if not torch.cuda.is_available():
        return []
    info = []
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        info.append({
            "id": i,
            "name": props.name,
            "total_gib": round(props.total_memory / 1024**3, 1),
            "sm_version": f"{props.major}.{props.minor}",
        })
    return info


def has_peer_access(dev_a: int, dev_b: int) -> bool:
    """Check if device A can directly access device B memory."""
    if not torch.cuda.is_available():
        return False
    try:
        return torch.cuda.can_device_access_peer(dev_a, dev_b)
    except RuntimeError:
        return False


def gpu_memory_snapshot(device_id: int | None = None) -> dict:
    """Snapshot of GPU memory allocator state for one device.

    Ported from RoundPipe's gpu_memory_snapshot().
    """
    if not torch.cuda.is_available():
        return {}
    if device_id is None:
        device_id = torch.cuda.current_device()
    snap = {
        "dev": device_id,
        "alloc": round(torch.cuda.memory_allocated(device_id) / 1024**3, 3),
        "reserved": round(torch.cuda.memory_reserved(device_id) / 1024**3, 3),
        "peak_alloc": round(torch.cuda.max_memory_allocated(device_id) / 1024**3, 3),
        "peak_reserved": round(torch.cuda.max_memory_reserved(device_id) / 1024**3, 3),
    }
    try:
        free, total = torch.cuda.mem_get_info(device_id)
        snap["free"] = round(free / 1024**3, 3)
        snap["total"] = round(total / 1024**3, 3)
    except RuntimeError:
        pass
    return snap


def get_optimal_tensor_split(
    device_ids: list[int] | None = None,
) -> list[float]:
    """Suggest tensor_split ratios based on free VRAM per device."""
    if not torch.cuda.is_available():
        return [1.0]

    if device_ids is None:
        device_ids = list(range(torch.cuda.device_count()))

    free = []
    for d in device_ids:
        torch.cuda.set_device(d)
        free_bytes, total_bytes = torch.cuda.mem_get_info(d)
        free.append(free_bytes)

    total_free = sum(free)
    if total_free == 0:
        return [1.0 / len(device_ids)] * len(device_ids)

    return [f / total_free for f in free]


def host_rss_gib() -> float:
    """Return the current process RSS in GiB."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024**2
    except (OSError, IndexError, ValueError):
        return 0.0
    return 0.0


def release_cached_memory(*, log_file=None) -> None:
    """Force release of cached heap memory back to the OS.

    Calls ``gc.collect()`` to reap Python-level garbage and
    ``malloc_trim(0)`` to release glibc free pages.  This is safe
    to call after ``prepare_nf4()`` frees the FP16 model weights;
    without it, freed pages stay in RSS because the allocator caches
    them for reuse.

    Does nothing on non-glibc platforms (Windows, macOS, musl).
    """
    before = host_rss_gib()
    gc.collect()
    try:
        libc = ctypes.CDLL("libc.so.6")
        libc.malloc_trim(0)
    except (OSError, AttributeError):
        pass  # not glibc
    after = host_rss_gib()
    if log_file is not None:
        log_file.write(json.dumps({
            "event": "release_cached_memory",
            "rss_before_gib": round(before, 2),
            "rss_after_gib": round(after, 2),
            "freed_gib": round(before - after, 2),
        }) + "\n")
        log_file.flush()
