"""Device detection, memory probes, and logging utilities."""

import logging
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
