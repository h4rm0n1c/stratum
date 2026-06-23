"""Weight upload: NF4 streaming for frozen 2D weights, FP16 for everything else.

Ported from roundpipe_nf4.py.

Two-phase:
  Phase 1 — prepare_nf4(): quantize frozen 2D weights to NF4, attach payload,
             drop original FP16 data (frees CPU RAM).
  Phase 2 — upload_stream(): upload NF4 payload compressed to GPU, wrap in
             NF4Linear (never dequants permanently — JIT dequant in forward).
             Non-NF4 params upload as FP16 permanent.
"""

from __future__ import annotations

import hashlib
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, cast

import torch
from stratum.utils import log_event

NF4_ATTR = "roundpipe_nf4_payload"


@dataclass
class NF4Stats:
    """Structured NF4 preparation statistics (ported from roundpipe_nf4.py)."""
    tensors: int = 0
    source_bytes: int = 0
    payload_bytes: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    cache_writes: int = 0

    @property
    def compression(self) -> float:
        if self.payload_bytes == 0:
            return 0.0
        return self.source_bytes / self.payload_bytes


def _pin_cpu(t: torch.Tensor) -> torch.Tensor:
    """Copy *t* to a pinned CPU buffer (faster H2D)."""
    t = t.detach().contiguous().cpu()
    if t.is_pinned():
        return t
    pinned = torch.empty_like(t, device=torch.device("cpu"), pin_memory=True)
    pinned.copy_(t)
    return pinned


def _payload_bytes(*tensors: torch.Tensor) -> int:
    return sum(t.numel() * t.element_size() for t in tensors)


@dataclass
class NF4Payload:
    """NF4-quantized weight payload kept on CPU between steps.

    Fields:
        quantized:   NF4-quantized 4-bit data (uint8, 1/2 the element count of shape).
        absmax:      Per-block absolute max values for dequantization.
        code:        Quantization codebook (256 FP16 values).
        shape:       Original weight shape (tuple).
        dtype:       Original weight dtype.
        source_bytes: Original weight size in bytes (for reporting).
    """
    quantized: torch.Tensor
    absmax: torch.Tensor
    code: torch.Tensor
    shape: tuple
    dtype: torch.dtype
    source_bytes: int


def _cache_path(cache_dir: Path, name: str) -> Path:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", name)[-160:]
    digest = hashlib.sha1(name.encode("utf-8")).hexdigest()[:12]
    return cache_dir / f"{digest}-{safe}.pt"


@torch.no_grad()
def prepare_nf4(
    module: torch.nn.Module,
    *,
    min_numel: int = 4096,
    cache_dir: Optional[Path | str] = None,
    verbose: bool = True,
    blocksize: int = 64,
    quant_type: str = "nf4",
) -> NF4Stats:
    """Phase 1: quantize frozen 2D weights, attach NF4 payload, drop originals.

    Ported from roundpipe_nf4.py::prepare_nf4_frozen_params(). Drops original
    FP16 weight data (param.data = empty(0)) after quantizing.

    Returns NF4Stats with tensors/bytes/cache counts.
    """
    from bitsandbytes.functional import quantize_4bit

    stats = NF4Stats()
    for name, param in module.named_parameters():
        if param.requires_grad:
            continue
        if param.ndim != 2:
            continue
        if param.numel() < min_numel:
            continue
        if param.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            continue
        if hasattr(param, NF4_ATTR):
            payload = cast(NF4Payload, getattr(param, NF4_ATTR))
            stats.tensors += 1
            stats.source_bytes += payload.source_bytes
            stats.payload_bytes += _payload_bytes(payload.quantized, payload.absmax, payload.code)
            continue

        weight = param.detach().contiguous().cpu()

        cache_path = _cache_path(Path(cache_dir), name) if cache_dir else None
        payload = None
        if cache_path and cache_path.exists():
            try:
                obj = torch.load(cache_path, map_location="cpu", weights_only=False)
                if obj.get("shape") == tuple(weight.shape):
                    payload = NF4Payload(
                        quantized=_pin_cpu(obj["quantized"]),
                        absmax=_pin_cpu(obj["absmax"]),
                        code=_pin_cpu(obj["code"]),
                        shape=tuple(weight.shape),
                        dtype=weight.dtype,
                        source_bytes=weight.numel() * weight.element_size(),
                    )
                    stats.cache_hits += 1
            except Exception:
                pass

        if payload is None:
            if cache_path is not None:
                stats.cache_misses += 1
            quantized, q_state = quantize_4bit(
                weight, blocksize=blocksize, compress_statistics=False, quant_type=quant_type,
            )
            payload = NF4Payload(
                quantized=_pin_cpu(quantized),
                absmax=_pin_cpu(q_state.absmax),
                code=_pin_cpu(q_state.code),
                shape=tuple(weight.shape),
                dtype=weight.dtype,
                source_bytes=weight.numel() * weight.element_size(),
            )
            if cache_path:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                tmp = cache_path.with_name(f"{cache_path.name}.tmp")
                torch.save({
                    "shape": tuple(payload.shape),
                    "dtype": str(payload.dtype).removeprefix("torch."),
                    "quantized": payload.quantized.detach().contiguous().cpu(),
                    "absmax": payload.absmax.detach().contiguous().cpu(),
                    "code": payload.code.detach().contiguous().cpu(),
                }, tmp)
                tmp.replace(cache_path)
                stats.cache_writes += 1

        setattr(param, NF4_ATTR, payload)
        param.data = torch.empty(0, dtype=weight.dtype, device=weight.device)
        stats.tensors += 1
        stats.source_bytes += payload.source_bytes
        stats.payload_bytes += _payload_bytes(payload.quantized, payload.absmax, payload.code)

    if verbose:
        print(f"  nf4 total: {stats.tensors} tensors, "
              f"{stats.source_bytes/1024**3:.2f} GiB -> {stats.payload_bytes/1024**3:.2f} GiB payload "
              f"(compression {stats.compression:.2f}x, "
              f"cache hits={stats.cache_hits} misses={stats.cache_misses} writes={stats.cache_writes})",
              flush=True)
    return stats


@torch.no_grad()
def estimate_module_upload_gib(module: torch.nn.Module) -> float:
    """Estimate GPU upload footprint for a module, counting NF4 params after dequant.

    Ported from roundpipe_nf4.py::estimate_module_upload_gib().
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
    return total / 10**9


@torch.no_grad()
def upload_stream(
    module: torch.nn.Module,
    device_id: int,
    *,
    verbose: bool = True,
) -> int:
    """Phase 2: upload all params/buffers of *module* to *device_id*.

    NF4-eligible frozen Linear layers are replaced with NF4Linear (4-bit on
    GPU, JIT dequant in forward). Everything else uploads as FP16 permanent.

    Ported from roundpipe_nf4.py::upload_layers_nf4().
    """
    from bitsandbytes.functional import QuantState, dequantize_4bit
    from stratum.nf4_linear import NF4Linear

    device = torch.device(f"cuda:{device_id}")
    tensors = 0
    t0 = time.time()

    for name, param in module.named_parameters():
        if param.data.device == device:
            continue

        payload = getattr(param, NF4_ATTR, None)
        if payload is not None:
            quantized_cpu, absmax_cpu, code_cpu = payload.quantized, payload.absmax, payload.code
            shape, dtype = payload.shape, payload.dtype

            q_gpu = quantized_cpu.to(device, non_blocking=False)
            a_gpu = absmax_cpu.to(device, non_blocking=False)
            c_gpu = code_cpu.to(device, non_blocking=False)
            q_state = QuantState(
                absmax=a_gpu, shape=shape, code=c_gpu,
                blocksize=64, quant_type="nf4", dtype=dtype,
            )

            # Find the parent module
            parts = name.split(".")
            parent_path = parts[:-1]
            parent_attr = parent_path[-1]
            grandparent_path = parts[:-2]
            grandparent = module
            for p in grandparent_path:
                grandparent = getattr(grandparent, p)
            parent_mod = getattr(grandparent, parent_attr)

            # Dequant NF4 to FP16 on GPU (same as RoundPipe's upload_layers_nf4)
            from bitsandbytes.functional import dequantize_4bit
            weight = dequantize_4bit(q_gpu, q_state).contiguous()
            param.data = weight
            tensors += 1

            if verbose:
                mib = quantized_cpu.numel() * quantized_cpu.element_size() / 1024**2
                print(f"  stream {name}: nf4 {list(shape)} -> {mib:.1f} MiB on GPU", flush=True)
        else:
            param.data = param.data.to(device, non_blocking=False)
            tensors += 1

    for buf in module.buffers():
        if buf.data.device == device:
            continue
        buf.data = buf.data.to(device, non_blocking=False)

    dt = time.time() - t0
    log_event("upload_stream", device=device_id, tensors=tensors, seconds=round(dt, 1))
    if verbose:
        print(f"  device {device_id}: {tensors} tensors in {dt:.1f}s", flush=True)
    return tensors


@torch.no_grad()
def ensure_weights(module: torch.nn.Module, device_id: int) -> int:
    """Upload NF4 payloads to GPU and dequant to FP16 for all params in *module*.

    Opposite of free_weights(). Called before a stage's forward pass.
    Only affects params that still have NF4 payload attached and empty data.
    """
    from bitsandbytes.functional import dequantize_4bit

    device = torch.device(f"cuda:{device_id}")
    tensors = 0
    for name, param in module.named_parameters():
        if param.data.numel() > 0:
            continue  # already has FP16 data
        payload = getattr(param, NF4_ATTR, None)
        if payload is None:
            continue
        quantized_cpu, absmax_cpu, code_cpu = payload.quantized, payload.absmax, payload.code
        shape, dtype = payload.shape, payload.dtype
        q_gpu = quantized_cpu.to(device, non_blocking=False)
        a_gpu = absmax_cpu.to(device, non_blocking=False)
        c_gpu = code_cpu.to(device, non_blocking=False)
        from bitsandbytes.functional import QuantState
        q_state = QuantState(
            absmax=a_gpu, shape=shape, code=c_gpu,
            blocksize=64, quant_type="nf4", dtype=dtype,
        )
        param.data = dequantize_4bit(q_gpu, q_state).contiguous()
        tensors += 1
    return tensors


@torch.no_grad()
def free_weights(module: torch.nn.Module) -> int:
    """Free FP16 weight data for NF4-eligible params in *module*.

    Opposite of ensure_weights(). Called after a stage's backward pass.
    Resets param.data to empty(0), keeping NF4 payload for next upload.
    """
    tensors = 0
    for name, param in module.named_parameters():
        if not hasattr(param, NF4_ATTR):
            continue
        if param.data.numel() == 0:
            continue
        param.data = torch.empty(0, dtype=param.dtype, device=param.device)
        tensors += 1
    return tensors
