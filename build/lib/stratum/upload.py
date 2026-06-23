"""Unified weight upload: NF4 for frozen 2D, FP16 for everything else.

One function, one call site per module. Covers every parameter type and buffer.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Optional

import time

import torch

from stratum.utils import log_event

# bitsandbytes is imported lazily inside upload_to_device() to avoid
# breaking users who pass use_nf4=False without having bnb installed.


def _cache_path(cache_dir: Path, name: str) -> Path:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", name)[-160:]
    digest = hashlib.sha1(name.encode("utf-8")).hexdigest()[:12]
    return cache_dir / f"{digest}-{safe}.pt"


@torch.no_grad()
def upload_to_device(
    module: torch.nn.Module,
    device_id: int,
    *,
    use_nf4: bool = True,
    min_numel: int = 4096,
    cache_dir: Optional[Path | str] = None,
    verbose: bool = True,
) -> int:
    """Move all parameters and buffers of *module* to *device_id*.

    Frozen 2D weights with ``numel >= min_numel`` use NF4-compressed H2D
    transfer (4x less PCIe traffic, dequantised in place on the GPU).
    Everything else — trainable parameters, 1D norms, biases, conv weights,
    buffers — is uploaded as FP16 directly.

    Args:
        module: The module whose params/buffers to upload.
        device_id: Target CUDA device.
        use_nf4: Enable NF4 compression for eligible frozen 2D weights.
        min_numel: Minimum number of elements for NF4 eligibility.
        cache_dir: Optional directory to cache quantised payloads.
        verbose: Print per-tensor progress.

    Returns:
        Number of tensors uploaded.
    """
    device = torch.device(f"cuda:{device_id}")
    tensors = 0
    t0 = time.time()

    for name, param in module.named_parameters():
        if param.data.device == device:
            continue

        # NF4 eligible: frozen, 2D, large enough
        if use_nf4 and not param.requires_grad \
           and param.ndim == 2 and param.numel() >= min_numel \
           and param.dtype in (torch.float16, torch.bfloat16, torch.float32):

            try:
                from bitsandbytes.functional import QuantState, quantize_4bit, dequantize_4bit
            except ImportError as exc:
                raise ImportError(
                    "bitsandbytes is required for NF4 weight upload. "
                    "Install it with: pip install bitsandbytes, "
                    "or pass use_nf4=False / --no-nf4"
                ) from exc

            weight = param.data.cpu().contiguous()

            # Try cache (same _cache_path for load and save)
            cached = None
            cache_path = None
            if cache_dir is not None:
                cache_path = _cache_path(Path(cache_dir), name)
                cached = _load_cache(cache_path, weight.shape, weight.dtype)

            if cached is not None:
                quantized_cpu, absmax_cpu, code_cpu = cached
            else:
                quantized_cpu, q_state = quantize_4bit(
                    weight, blocksize=64, compress_statistics=False,
                    quant_type="nf4",
                )
                absmax_cpu = q_state.absmax
                code_cpu = q_state.code
                if cache_path is not None:
                    _save_cache(cache_path, quantized_cpu, absmax_cpu,
                                code_cpu, weight.shape, weight.dtype)

            # Upload compressed, dequantise on GPU
            quantized_gpu = quantized_cpu.to(device, non_blocking=True)
            absmax_gpu = absmax_cpu.to(device, non_blocking=True)
            code_gpu = code_cpu.to(device, non_blocking=True)

            q_state = QuantState(
                absmax=absmax_gpu, shape=tuple(weight.shape), code=code_gpu,
                blocksize=64, quant_type="nf4", dtype=weight.dtype,
            )
            param.data = dequantize_4bit(quantized_gpu, q_state).contiguous()
            tensors += 1

            if verbose:
                source_mib = weight.numel() * weight.element_size() / 1024**2
                print(f"  nf4 {name}: {list(weight.shape)} {source_mib:.1f} MiB -> "
                      f"{source_mib / 4:.1f} MiB PCIe", flush=True)
            continue

        # Default: FP16 direct upload (trainable, non-2D, or too small)
        param.data = param.data.to(device, non_blocking=True)
        tensors += 1

    # Move buffers (running stats, conv state, etc.)
    for buf in module.buffers():
        if buf.data.device == device:
            continue
        buf.data = buf.data.to(device, non_blocking=True)

    dt = time.time() - t0
    source_mib = sum(
        p.numel() * p.element_size() for p in module.parameters()
        if p.data.device == device
    ) / 1024**2
    log_event("upload_complete", device=device_id, tensors=tensors,
              total_mib=round(source_mib, 1), seconds=round(dt, 1))

    if verbose:
        print(f"  device {device_id}: {tensors} tensors, ~{source_mib:.0f} MiB"
              f" in {dt:.1f}s", flush=True)

    return tensors


def _load_cache(path: Path, shape, dtype) -> Optional[tuple]:
    if not path.exists():
        return None
    try:
        obj = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        return None
    if obj.get("shape") != tuple(shape) or obj.get("dtype") != str(dtype):
        return None
    return obj["quantized"], obj["absmax"], obj["code"]


def _save_cache(path, quantized, absmax, code, shape, dtype):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp")
    torch.save({
        "shape": tuple(shape),
        "dtype": str(dtype),
        "quantized": quantized.contiguous().cpu(),
        "absmax": absmax.contiguous().cpu(),
        "code": code.contiguous().cpu(),
    }, tmp)
    tmp.replace(path)
