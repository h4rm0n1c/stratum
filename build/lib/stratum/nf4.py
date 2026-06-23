"""Per-device NF4 weight upload.

Adapted from roundpipe_nf4.py — same bitsandbytes quantize/dequant logic,
but dispatched per-device instead of through RoundPipe's streaming scheduler.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any, Optional

import torch
import bitsandbytes as bnb
from bitsandbytes.functional import QuantState, quantize_4bit, dequantize_4bit


@torch.no_grad()
def upload_weights_nf4(
    module: torch.nn.Module,
    device_id: int,
    *,
    min_numel: int = 4096,
    blocksize: int = 64,
    quant_type: str = "nf4",
    cache_dir: Optional[Path | str] = None,
    verbose: bool = True,
) -> int:
    """Upload frozen 2D weights of *module* to *device_id* via NF4 dequant.

    Frozen 2D params are quantized on CPU, transferred compressed (4-bit),
    then dequantized on the GPU. Returns the number of tensors uploaded.

    The module's parameters are replaced in-place with the dequantized GPU
    versions. Trainable (LoRA) parameters are uploaded directly as FP16.
    """
    device = torch.device(f"cuda:{device_id}")
    tensors = 0

    for name, param in module.named_parameters():
        if param.ndim != 2:
            continue
        if param.numel() < min_numel:
            continue
        if param.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            continue

        if param.requires_grad:
            # Trainable — upload as FP16 directly
            param.data = param.data.to(device, non_blocking=True)
            tensors += 1
            continue

        # Frozen — quantize, upload compressed, dequantize on GPU
        weight = param.detach().contiguous().cpu()

        if cache_dir is not None:
            cache_path = _cache_path(Path(cache_dir), name)
            payload = _load_cache(cache_path, weight.shape, param.dtype, blocksize, quant_type)
            if payload is not None:
                quantized_cpu, absmax_cpu, code_cpu = payload
            else:
                quantized_cpu, quant_state = quantize_4bit(
                    weight, blocksize=blocksize,
                    compress_statistics=False, quant_type=quant_type,
                )
                absmax_cpu = quant_state.absmax
                code_cpu = quant_state.code
                _save_cache(cache_path, quantized_cpu, absmax_cpu, code_cpu,
                            weight.shape, param.dtype, blocksize, quant_type)
        else:
            quantized_cpu, quant_state = quantize_4bit(
                weight, blocksize=blocksize,
                compress_statistics=False, quant_type=quant_type,
            )
            absmax_cpu = quant_state.absmax
            code_cpu = quant_state.code

        # Upload compressed payload
        quantized_gpu = quantized_cpu.to(device, non_blocking=True)
        absmax_gpu = absmax_cpu.to(device, non_blocking=True)
        code_gpu = code_cpu.to(device, non_blocking=True)

        quant_state = QuantState(
            absmax=absmax_gpu,
            shape=tuple(weight.shape),
            code=code_gpu,
            blocksize=blocksize,
            quant_type=quant_type,
            dtype=weight.dtype,
        )
        dequantized = dequantize_4bit(quantized_gpu, quant_state)
        param.data = dequantized.contiguous()
        tensors += 1

    # Upload remaining params not handled above (non-2D, small 2D):
    # layer norms, biases, conv1d weights, etc.
    for name, param in module.named_parameters():
        if param.data.device == device:
            continue
        param.data = param.data.to(device, non_blocking=True)
        tensors += 1

    # Also move buffers (e.g. running stats, conv state)
    for name, buf in module.named_buffers():
        if buf.data.device == device:
            continue
        buf.data = buf.data.to(device, non_blocking=True)

    if verbose:
        print(
            {"nf4_upload": f"device={device_id}", "tensors": tensors},
            flush=True,
        )

    return tensors


def _cache_path(cache_dir: Path, name: str) -> Path:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", name)[-160:]
    digest = hashlib.sha1(name.encode("utf-8")).hexdigest()[:12]
    return cache_dir / f"{digest}-{safe}.pt"


def _load_cache(path: Path, shape, dtype, blocksize, quant_type) -> Optional[tuple]:
    if not path.exists():
        return None
    try:
        obj = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        return None
    if (
        obj.get("shape") != tuple(shape)
        or obj.get("dtype") != str(dtype).removeprefix("torch.")
        or obj.get("blocksize") != blocksize
        or obj.get("quant_type") != quant_type
    ):
        return None
    return (obj["quantized"], obj["absmax"], obj["code"])


def _save_cache(path, quantized, absmax, code, shape, dtype, blocksize, quant_type):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp")
    torch.save({
        "shape": tuple(shape),
        "dtype": str(dtype).removeprefix("torch."),
        "blocksize": blocksize,
        "quant_type": quant_type,
        "quantized": quantized.contiguous().cpu(),
        "absmax": absmax.contiguous().cpu(),
        "code": code.contiguous().cpu(),
    }, tmp)
    tmp.replace(path)
