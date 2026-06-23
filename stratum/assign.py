"""Layer-to-device assignment using llama.cpp's upper_bound algorithm."""

from bisect import bisect_left
from typing import Optional

from stratum.utils import log_event


def assign_layers_to_devices(
    n_layers: int,
    tensor_split: Optional[list[float]] = None,
    device_ids: Optional[list[int]] = None,
    n_devices: Optional[int] = None,
) -> dict[int, int]:
    """Assign each decoder layer to a device.

    Uses the same cumulative-split / upper_bound algorithm as llama.cpp's
    LLAMA_SPLIT_MODE_LAYER (llama-model.cpp:1265-1277).

    Args:
        n_layers: Total number of decoder layers.
        tensor_split: VRAM weights per device, e.g. [10, 32] for 10GB+32GB.
            If None, splits evenly across devices.
        device_ids: CUDA device IDs to assign to, e.g. [0, 1].
            If None, uses [0, 1, ..., n_devices-1].
        n_devices: Number of devices (used when tensor_split is None).
            Defaults to 1.

    Returns:
        dict mapping layer_idx -> device_id.
    """
    if tensor_split is not None:
        nd = len(tensor_split)
        if nd == 0:
            raise ValueError("tensor_split list is empty")
    elif n_devices is not None:
        nd = n_devices
        if nd <= 0:
            raise ValueError(f"n_devices must be >= 1, got {nd}")
    else:
        nd = 1

    if device_ids is None:
        device_ids = list(range(nd))
    elif len(device_ids) != nd:
        raise ValueError(
            f"device_ids length {len(device_ids)} != "
            f"n_devices {nd}"
        )

    if nd == 1:
        assignment = {i: device_ids[0] for i in range(n_layers)}
        log_event("assign_layers", n_layers=n_layers, n_devices=1)
        return assignment

    # Normalise splits to cumulative 0..1 (same as llama.cpp)
    if tensor_split is not None:
        total = sum(tensor_split)
        if total <= 0:
            raise ValueError("tensor_split must sum to > 0")
        splits = []
        cum = 0.0
        for s in tensor_split:
            cum += s
            splits.append(cum / total)
    else:
        splits = [(i + 1) / nd for i in range(nd)]

    # Each layer maps to a device via upper_bound on (layer_idx / n_layers)
    assignment = {}
    for i in range(n_layers):
        frac = i / n_layers
        dev_idx = bisect_left(splits, frac)
        dev_idx = min(dev_idx, nd - 1)
        assignment[i] = device_ids[dev_idx]

    # Summarise distribution
    per_dev = {d: 0 for d in device_ids}
    for dev in assignment.values():
        per_dev[dev] = per_dev.get(dev, 0) + 1
    log_event("assign_layers", n_layers=n_layers, n_devices=nd,
              distribution=per_dev, tensor_split=tensor_split)
    return assignment
