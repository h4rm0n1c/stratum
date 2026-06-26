"""Stage planning helpers.

RoundPipe can schedule layer groups under a model-memory budget. Stratum keeps
the existing layer-to-device assignment, then optionally splits each device's
contiguous layers into smaller stages to cap per-stage streamed weight size.
"""

from __future__ import annotations

import torch.nn as nn

from stratum.upload import NF4_ATTR


def estimate_module_bytes(module: nn.Module, *, floor_gib: float = 0.0) -> int:
    """Estimate uploaded parameter+buffer bytes for one module.

    ``floor_gib`` mirrors qz-roundpipe's NF4 scheduler-size floor: after the
    layer's upload footprint is estimated, planning can force a minimum size so
    memory-budgeted stage grouping does not over-pack layers whose frozen
    weights were compressed or otherwise under-estimated.
    """
    total = 0
    seen: set[int] = set()
    for param in module.parameters():
        if id(param) in seen:
            continue
        seen.add(id(param))
        payload = getattr(param, NF4_ATTR, None)
        if payload is not None:
            total += payload.source_bytes
        else:
            total += param.numel() * param.element_size()
        if param.grad is not None:
            total += param.grad.numel() * param.grad.element_size()
    for buf in module.buffers():
        total += buf.numel() * buf.element_size()
    if floor_gib > 0:
        total = max(total, int(floor_gib * 1024**3))
    return total


def split_layers_by_memory_limit(
    layers: list[nn.Module],
    limit_gib: float,
    *,
    layer_size_floor_gib: float = 0.0,
) -> list[list[nn.Module]]:
    """Split ordered layers into stage groups below *limit_gib* when possible.

    Layers are never reordered or split internally. A single layer larger than
    the limit is emitted as its own group rather than rejected.
    """
    if not layers:
        return []
    if limit_gib <= 0:
        return [layers]

    limit_bytes = int(limit_gib * 1024**3)
    groups: list[list[nn.Module]] = []
    current: list[nn.Module] = []
    current_bytes = 0

    for layer in layers:
        layer_bytes = estimate_module_bytes(layer, floor_gib=layer_size_floor_gib)
        if current and current_bytes + layer_bytes > limit_bytes:
            groups.append(current)
            current = []
            current_bytes = 0
        current.append(layer)
        current_bytes += layer_bytes

    if current:
        groups.append(current)
    return groups
