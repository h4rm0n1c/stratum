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
from typing import Optional

import torch
from stratum.utils import log_event

NF4_ATTR = "roundpipe_nf4_payload"


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
) -> int:
    """Phase 1: quantize frozen 2D weights, attach NF4 payload, drop originals.

    Ported from roundpipe_nf4.py::prepare_nf4_frozen_params(). Drops original
    FP16 weight data (param.data = empty(0)) after quantizing.
    """
    from bitsandbytes.functional import quantize_4bit

    tensors = 0
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
            tensors += 1
            continue

        weight = param.detach().contiguous().cpu()

        cache_path = _cache_path(Path(cache_dir), name) if cache_dir else None
        cached = None
        if cache_path and cache_path.exists():
            try:
                obj = torch.load(cache_path, map_location="cpu", weights_only=False)
                if obj.get("shape") == tuple(weight.shape):
                    cached = (obj["quantized"], obj["absmax"], obj["code"])
            except Exception:
                pass

        if cached:
            quantized, absmax, code = cached
        else:
            quantized, q_state = quantize_4bit(
                weight, blocksize=64, compress_statistics=False, quant_type="nf4",
            )
            absmax = q_state.absmax
            code = q_state.code
            if cache_path:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                tmp = cache_path.with_name(f"{cache_path.name}.tmp")
                torch.save({
                    "shape": tuple(weight.shape),
                    "dtype": str(weight.dtype).removeprefix("torch."),
                    "quantized": quantized.contiguous().cpu(),
                    "absmax": absmax.contiguous().cpu(),
                    "code": code.contiguous().cpu(),
                }, tmp)
                tmp.replace(cache_path)

        setattr(param, NF4_ATTR, (quantized, absmax, code, tuple(weight.shape),
                                  weight.dtype, weight.numel() * weight.element_size()))
        param.data = torch.empty(0, dtype=weight.dtype, device=weight.device)
        tensors += 1

    if verbose:
        src = sum(
            getattr(p, NF4_ATTR)[5] for p in module.parameters() if hasattr(p, NF4_ATTR)
        ) / 1024**3
        pay = sum(
            getattr(p, NF4_ATTR)[0].numel() * getattr(p, NF4_ATTR)[0].element_size()
            + getattr(p, NF4_ATTR)[1].numel() * getattr(p, NF4_ATTR)[1].element_size()
            + getattr(p, NF4_ATTR)[2].numel() * getattr(p, NF4_ATTR)[2].element_size()
            for p in module.parameters() if hasattr(p, NF4_ATTR)
        ) / 1024**3
        print(f"  nf4 total: {tensors} tensors, {src:.2f} GiB -> {pay:.2f} GiB payload", flush=True)
    return tensors


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
            quantized_cpu, absmax_cpu, code_cpu, shape, dtype, _ = payload

            q_gpu = quantized_cpu.to(device, non_blocking=False)
            a_gpu = absmax_cpu.to(device, non_blocking=False)
            c_gpu = code_cpu.to(device, non_blocking=False)
            q_state = QuantState(
                absmax=a_gpu, shape=shape, code=c_gpu,
                blocksize=64, quant_type="nf4", dtype=dtype,
            )

            # Find the parent module (the nn.Linear holding this weight)
            parts = name.split(".")
            # parts[-1] is "weight" — parent is the Linear
            # For PEFT: ["...", "base_layer", "weight"] → replace base_layer
            # For non-PEFT: ["...", "q_proj", "weight"] → replace q_proj
            parent_path = parts[:-1]  # path to the Linear module
            parent_attr = parent_path[-1]  # attribute name of Linear
            grandparent_path = parts[:-2]
            grandparent = module
            for p in grandparent_path:
                grandparent = getattr(grandparent, p)

            # Get the bias if it exists
            bias = None
            linear = getattr(grandparent, parent_attr)
            if hasattr(linear, "bias") and linear.bias is not None:
                bias = linear.bias.to(device, non_blocking=False)

            # Replace with NF4Linear (4-bit on GPU, JIT dequant)
            nf4_linear = NF4Linear(q_gpu, q_state, bias)
            setattr(grandparent, parent_attr, nf4_linear)
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
