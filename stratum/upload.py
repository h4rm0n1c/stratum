"""Weight upload: NF4 streaming for frozen 2D weights, FP16 for everything else.

Ported from roundpipe_nf4.py.

Two-phase:
  Phase 1 — prepare_nf4(): quantize frozen 2D weights to NF4, attach payload,
             drop original FP16 data (frees CPU RAM).
  Phase 2 — upload_stream()/ensure_weights(): upload NF4 payload compressed to
             GPU, dequantize into the staged parameter data for use, then let
             free_weights() drop the materialized GPU copy after the stage.
             Non-NF4 params upload as regular tensors.
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
from stratum.layer_transfer import copy_tensor_chunked, DEFAULT_CHUNK_UPLOAD_BYTES

NF4_ATTR = "roundpipe_nf4_payload"
FP16_ATTR = "stratum_fp16_staged"


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


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).removeprefix("torch.")


def _dtype_from_name(name: str) -> torch.dtype:
    return cast(torch.dtype, getattr(torch, name.removeprefix("torch.")))


@dataclass
class NF4Payload:
    """NF4-quantized weight payload kept on CPU between steps.

    Fields:
        quantized:   NF4-quantized 4-bit data (uint8, 1/2 the element count of shape).
        absmax:      Per-block absolute max values for dequantization.
        code:        Quantization codebook (256 FP16 values).
        shape:       Original weight shape (tuple).
        dtype:       Original weight dtype.
        blocksize:   NF4 quantization block size.
        quant_type:  bitsandbytes quantization type.
        source_numel: Original weight element count.
        source_bytes: Original weight size in bytes (for reporting).
        payload_bytes: Compressed payload size in bytes.
    """
    quantized: torch.Tensor
    absmax: torch.Tensor
    code: torch.Tensor
    shape: tuple
    dtype: torch.dtype
    blocksize: int
    quant_type: str
    source_numel: int
    source_bytes: int
    payload_bytes: int


@dataclass
class FP16StagedPayload:
    """FP16 frozen weight kept on CPU for per-step upload (analogous to NF4Payload).

    Used by the --no-nf4 runtime path: frozen params are pinned on CPU and
    uploaded to GPU each step via copy_tensor_chunked, then freed after backward.
    """
    data: torch.Tensor
    shape: tuple
    dtype: torch.dtype
    source_bytes: int


@torch.no_grad()
def prepare_fp16_staged(
    module: torch.nn.Module,
    *,
    min_numel: int = 4096,
    verbose: bool = True,
) -> int:
    """Pin frozen params for per-step FP16 upload (non-NF4 analogue of prepare_nf4).

    Marks each qualifying frozen parameter with FP16_ATTR and sets param.data to
    empty(0), matching the NF4 lifecycle. ensure_weights() will upload the pinned
    CPU tensor to GPU each step; free_weights() will release the GPU copy afterward.

    Only 2D-or-larger params above min_numel are staged; small params (biases,
    small norms) and trainable params are left as-is and uploaded permanently.
    """
    count = 0
    for name, param in module.named_parameters():
        if param.requires_grad:
            continue
        if param.numel() < min_numel:
            continue
        if param.ndim < 2:
            continue
        if hasattr(param, NF4_ATTR) or hasattr(param, FP16_ATTR):
            continue
        if param.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            continue
        cpu_data = _pin_cpu(param.data)
        payload = FP16StagedPayload(
            data=cpu_data,
            shape=tuple(param.shape),
            dtype=param.dtype,
            source_bytes=param.numel() * param.element_size(),
        )
        setattr(param, FP16_ATTR, payload)
        param.data = torch.empty(0, dtype=param.dtype, device=param.device)
        count += 1
    if verbose:
        total_bytes = sum(
            getattr(p, FP16_ATTR).source_bytes
            for p in module.parameters()
            if hasattr(p, FP16_ATTR)
        )
        print(
            f"  fp16 staged: {count} tensors, "
            f"{total_bytes / 1024**3:.2f} GiB to upload per step",
            flush=True,
        )
    return count


@dataclass
class _PrefetchEntry:
    param: torch.nn.Parameter
    payload: NF4Payload
    quantized: torch.Tensor
    absmax: torch.Tensor
    code: torch.Tensor


@dataclass
class NF4Prefetch:
    """Pending async NF4 payload upload for one module.

    `finalize()` fences the copy stream and dequantizes the prefetched payloads
    into parameter data. This mirrors RoundPipe's upload-before-use event model
    while preserving Stratum's existing in-place module layout.
    """
    entries: list[_PrefetchEntry]
    device: torch.device
    stream: Optional[torch.cuda.Stream] = None
    event: Optional[torch.cuda.Event] = None

    def finalize(self) -> int:
        if not self.entries:
            return 0
        if self.device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError(f"CUDA is not available for NF4 prefetch finalize on {self.device}")

        from bitsandbytes.functional import QuantState, dequantize_4bit

        current = torch.cuda.current_stream(device=self.device)
        if self.event is not None:
            current.wait_event(self.event)

        tensors = 0
        with torch.cuda.device(self.device), torch.cuda.stream(current), torch.no_grad():
            for entry in self.entries:
                if entry.param.data.numel() > 0:
                    if entry.param.data.device == self.device:
                        continue
                    entry.param.data = torch.empty(0, dtype=entry.param.dtype, device=entry.param.device)
                payload = entry.payload
                q_state = QuantState(
                    absmax=entry.absmax,
                    shape=payload.shape,
                    code=entry.code,
                    blocksize=payload.blocksize,
                    quant_type=payload.quant_type,
                    dtype=payload.dtype,
                )
                entry.param.data = dequantize_4bit(entry.quantized, q_state).contiguous()
                tensors += 1
        return tensors

    def wait(self) -> int:
        return self.finalize()


def _cache_path(cache_dir: Path, name: str) -> Path:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", name)[-160:]
    digest = hashlib.sha1(name.encode("utf-8")).hexdigest()[:12]
    return cache_dir / f"{digest}-{safe}.pt"


def _load_payload_from_cache(
    path: Path,
    param: torch.nn.Parameter,
    *,
    blocksize: int,
    quant_type: str,
) -> Optional[NF4Payload]:
    if not path.exists():
        return None
    try:
        obj = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        return None

    expected = {
        "version": 1,
        "shape": tuple(param.shape),
        "dtype": _dtype_name(param.dtype),
        "blocksize": int(blocksize),
        "quant_type": str(quant_type),
        "source_numel": int(param.numel()),
        "source_bytes": int(param.numel() * param.element_size()),
    }
    for key, value in expected.items():
        if obj.get(key) != value:
            return None

    quantized = cast(torch.Tensor, obj["quantized"])
    absmax = cast(torch.Tensor, obj["absmax"])
    code = cast(torch.Tensor, obj["code"])
    return NF4Payload(
        quantized=_pin_cpu(quantized),
        absmax=_pin_cpu(absmax),
        code=_pin_cpu(code),
        shape=tuple(obj["shape"]),
        dtype=_dtype_from_name(str(obj["dtype"])),
        blocksize=int(obj["blocksize"]),
        quant_type=str(obj["quant_type"]),
        source_numel=int(obj["source_numel"]),
        source_bytes=int(obj["source_bytes"]),
        payload_bytes=_payload_bytes(quantized, absmax, code),
    )


def _save_payload_to_cache(path: Path, payload: NF4Payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp")
    torch.save({
        "version": 1,
        "shape": tuple(payload.shape),
        "dtype": _dtype_name(payload.dtype),
        "blocksize": int(payload.blocksize),
        "quant_type": str(payload.quant_type),
        "source_numel": int(payload.source_numel),
        "source_bytes": int(payload.source_bytes),
        "quantized": payload.quantized.detach().contiguous().cpu(),
        "absmax": payload.absmax.detach().contiguous().cpu(),
        "code": payload.code.detach().contiguous().cpu(),
    }, tmp)
    tmp.replace(path)


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
    """Phase 1: quantize frozen weights (2D and higher), attach NF4 payload, drop originals.

    Ported from roundpipe_nf4.py::prepare_nf4_frozen_params(). Drops original
    FP16 weight data (param.data = empty(0)) after quantizing.

    Handles stacked MoE expert weight tensors (ndim > 2, e.g. [N, A, B]) by
    reshaping to 2D for bitsandbytes quantization; payload.shape retains the
    original tensor shape so all dequant paths reconstruct the correct rank.

    Returns NF4Stats with tensors/bytes/cache counts.
    """
    stats = NF4Stats()
    for name, param in module.named_parameters():
        if param.requires_grad:
            continue
        if param.ndim < 2:
            continue
        if param.numel() < min_numel:
            continue
        if param.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            continue
        if hasattr(param, NF4_ATTR):
            payload = cast(NF4Payload, getattr(param, NF4_ATTR))
            stats.tensors += 1
            stats.source_bytes += payload.source_bytes
            stats.payload_bytes += payload.payload_bytes
            continue

        cache_path = _cache_path(Path(cache_dir), name) if cache_dir else None
        payload = (
            _load_payload_from_cache(
                cache_path, param, blocksize=blocksize, quant_type=quant_type
            )
            if cache_path is not None
            else None
        )
        if payload is not None:
            stats.cache_hits += 1

        if payload is None:
            from bitsandbytes.functional import quantize_4bit

            if cache_path is not None:
                stats.cache_misses += 1
            orig_shape = tuple(param.shape)
            weight = param.detach().contiguous().cpu()
            # bitsandbytes quantize_4bit requires a 2D contiguous input; higher-rank
            # stacked weight tensors (e.g. MoE gate_up_proj [N, A, B]) are reshaped
            # to [-1, last_dim] for quantization and restored via payload.shape at
            # dequant time.
            if weight.ndim > 2:
                weight = weight.reshape(-1, weight.shape[-1])
            quantized, q_state = quantize_4bit(
                weight, blocksize=blocksize, compress_statistics=False, quant_type=quant_type,
            )
            payload = NF4Payload(
                quantized=_pin_cpu(quantized),
                absmax=_pin_cpu(q_state.absmax),
                code=_pin_cpu(q_state.code),
                shape=orig_shape,
                dtype=q_state.dtype,
                blocksize=int(q_state.blocksize),
                quant_type=str(q_state.quant_type),
                source_numel=int(param.numel()),
                source_bytes=int(param.numel() * param.element_size()),
                payload_bytes=_payload_bytes(quantized, q_state.absmax, q_state.code),
            )
            if cache_path:
                _save_payload_to_cache(cache_path, payload)
                stats.cache_writes += 1

        setattr(param, NF4_ATTR, payload)
        param.data = torch.empty(0, dtype=payload.dtype, device=param.device)
        stats.tensors += 1
        stats.source_bytes += payload.source_bytes
        stats.payload_bytes += payload.payload_bytes

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

    NF4-eligible frozen parameters are uploaded compressed and dequantized into
    the existing Parameter data on the target GPU. Everything else uploads as a
    regular tensor.

    Ported from roundpipe_nf4.py::upload_layers_nf4().
    """
    from bitsandbytes.functional import QuantState, dequantize_4bit

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
                blocksize=payload.blocksize, quant_type=payload.quant_type, dtype=dtype,
            )

            # Dequant NF4 to FP16 on GPU (same as RoundPipe's upload_layers_nf4)
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
    device = torch.device(f"cuda:{device_id}")
    tensors = 0
    dequantize_4bit = None
    QuantState = None
    for name, param in module.named_parameters():
        fp16_payload = getattr(param, FP16_ATTR, None)
        if fp16_payload is not None:
            if param.data.numel() > 0:
                if param.data.device == device:
                    continue
                # Shared FP16-staged param reached from a different device context:
                # drop the current materialization and re-upload for this device.
                param.data = torch.empty(0, dtype=param.dtype, device=param.device)
            dst = torch.empty(fp16_payload.shape, dtype=fp16_payload.dtype, device=device)
            copy_tensor_chunked(fp16_payload.data, dst, chunk_bytes=DEFAULT_CHUNK_UPLOAD_BYTES, non_blocking=True)
            param.data = dst
            tensors += 1
            continue
        payload = getattr(param, NF4_ATTR, None)
        if payload is None:
            if param.data.numel() > 0 and param.data.device != device:
                if param.requires_grad:
                    raise RuntimeError(
                        f"trainable shared parameter {name!r} is on {param.data.device}, "
                        f"but module is running on {device}; Stratum cannot safely move "
                        "trainable parameters between stages"
                    )
                param.data = param.data.to(device, non_blocking=True)
                tensors += 1
            continue
        if param.data.numel() > 0:
            if param.data.device == device:
                continue  # already has FP16 data on the target device
            # Shared frozen weights can be used by prefix and postfix on
            # different GPUs. Re-materialize from the CPU NF4 payload instead
            # of trying to keep one Parameter's data on two devices.
            param.data = torch.empty(0, dtype=param.dtype, device=param.device)
        if dequantize_4bit is None or QuantState is None:
            from bitsandbytes.functional import QuantState as _QuantState
            from bitsandbytes.functional import dequantize_4bit as _dequantize_4bit
            QuantState = _QuantState
            dequantize_4bit = _dequantize_4bit
        quantized_cpu, absmax_cpu, code_cpu = payload.quantized, payload.absmax, payload.code
        shape, dtype = payload.shape, payload.dtype
        q_gpu = quantized_cpu.to(device, non_blocking=True)
        a_gpu = absmax_cpu.to(device, non_blocking=True)
        c_gpu = code_cpu.to(device, non_blocking=True)
        q_state = QuantState(
            absmax=a_gpu, shape=shape, code=c_gpu,
            blocksize=payload.blocksize, quant_type=payload.quant_type, dtype=dtype,
        )
        param.data = dequantize_4bit(q_gpu, q_state).contiguous()
        tensors += 1
    for buf in module.buffers():
        if buf.data.device != device:
            buf.data = buf.data.to(device, non_blocking=True)
            tensors += 1
    return tensors


@torch.no_grad()
def prefetch_weights(module: torch.nn.Module, device_id: int) -> NF4Prefetch:
    """Start async NF4 payload H2D copies for empty streamed params.

    This does not mutate parameter data. Call `NF4Prefetch.finalize()` before
    running the module to fence the copy stream and dequantize into FP16 data.
    If there is nothing to upload, the returned object is a cheap no-op.
    """
    device = torch.device(f"cuda:{device_id}")
    entries: list[_PrefetchEntry] = []
    for name, param in module.named_parameters():
        payload = getattr(param, NF4_ATTR, None)
        if payload is None:
            continue
        if param.data.numel() > 0 and param.data.device == device:
            continue
        entries.append(
            _PrefetchEntry(
                param=param,
                payload=cast(NF4Payload, payload),
                quantized=cast(NF4Payload, payload).quantized,
                absmax=cast(NF4Payload, payload).absmax,
                code=cast(NF4Payload, payload).code,
            )
        )

    if not entries:
        return NF4Prefetch([], device)

    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError(f"CUDA is not available for NF4 prefetch to {device}")

    stream = torch.cuda.Stream(device=device)
    prefetched: list[_PrefetchEntry] = []
    with torch.cuda.device(device), torch.cuda.stream(stream):
        for entry in entries:
            prefetched.append(
                _PrefetchEntry(
                    param=entry.param,
                    payload=entry.payload,
                    quantized=entry.quantized.to(device, non_blocking=True),
                    absmax=entry.absmax.to(device, non_blocking=True),
                    code=entry.code.to(device, non_blocking=True),
                )
            )
        event = torch.cuda.Event()
        stream.record_event(event)
    return NF4Prefetch(prefetched, device=device, stream=stream, event=event)


@torch.no_grad()
def ensure_prefetched_weights(
    module: torch.nn.Module,
    device_id: int,
    prefetch: Optional[NF4Prefetch],
) -> int:
    """Finalize a matching prefetch or fall back to synchronous ensure."""
    if prefetch is None:
        return ensure_weights(module, device_id)
    if prefetch.device != torch.device(f"cuda:{device_id}"):
        raise ValueError(f"prefetch device {prefetch.device} does not match cuda:{device_id}")
    # Prefetch only stages NF4 payload tensors. The regular ensure path still
    # has to move non-NF4 frozen params and buffers, matching upload_stream().
    return prefetch.finalize() + ensure_weights(module, device_id)


@torch.no_grad()
def free_weights(module: torch.nn.Module) -> int:
    """Free FP16 weight data for NF4-eligible params in *module*.

    Opposite of ensure_weights(). Called after a stage's backward pass.
    Resets param.data to empty(0), keeping NF4 payload for next upload.
    """
    tensors = 0
    for name, param in module.named_parameters():
        if not hasattr(param, NF4_ATTR) and not hasattr(param, FP16_ATTR):
            continue
        if param.data.numel() == 0:
            continue
        param.data = torch.empty(0, dtype=param.dtype, device=param.device)
        tensors += 1
    return tensors
