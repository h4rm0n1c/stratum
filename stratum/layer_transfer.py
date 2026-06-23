"""Layer upload/download helpers adapted from RoundPipe.

RoundPipe's runtime copies CPU-resident layers to a device, runs them, then
downloads gradients or buffers back. Stratum does not use that execution model
today, but these helpers keep the copy semantics available for future offload
and prefetch paths without changing the active training loop.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Optional, Sequence

import torch


DEFAULT_CHUNK_UPLOAD_BYTES = 256 * 1024 * 1024


@dataclass
class LayerTransferResult:
    layers: list[torch.nn.Module]
    event: Optional[torch.cuda.Event] = None
    stream: Optional[torch.cuda.Stream] = None

    def wait(self) -> list[torch.nn.Module]:
        if self.event is not None:
            self.event.synchronize()
        return self.layers


@dataclass
class DownloadResult:
    gradients: int
    buffers: int
    event: Optional[torch.cuda.Event] = None
    stream: Optional[torch.cuda.Stream] = None

    def wait(self) -> "DownloadResult":
        if self.event is not None:
            self.event.synchronize()
        return self


def _as_device(device: torch.device | str | int) -> torch.device:
    if isinstance(device, int):
        return torch.device(f"cuda:{device}")
    return torch.device(device)


def _chunk_numel(tensor: torch.Tensor, chunk_bytes: int) -> int:
    if chunk_bytes <= 0:
        raise ValueError(f"chunk_bytes must be positive, got {chunk_bytes}")
    return max(1, chunk_bytes // max(1, tensor.element_size()))


@torch.no_grad()
def copy_tensor_chunked(
    src: torch.Tensor,
    dst: torch.Tensor,
    *,
    chunk_bytes: int = DEFAULT_CHUNK_UPLOAD_BYTES,
    non_blocking: bool = False,
) -> int:
    """Copy *src* into *dst* in RoundPipe-style flat chunks.

    Returns the number of chunks copied. The tensors must have matching shape
    and dtype; callers allocate *dst* on the desired device.
    """
    if src.shape != dst.shape:
        raise ValueError(f"shape mismatch: {tuple(src.shape)} != {tuple(dst.shape)}")
    if src.dtype != dst.dtype:
        raise ValueError(f"dtype mismatch: {src.dtype} != {dst.dtype}")
    if src.numel() == 0:
        return 0
    if not dst.is_contiguous():
        dst.copy_(src, non_blocking=non_blocking)
        return 1

    flat_src = src.detach().contiguous().view(-1)
    flat_dst = dst.view(-1)
    chunk_numel = _chunk_numel(src, chunk_bytes)
    chunks = 0
    for start in range(0, src.numel(), chunk_numel):
        end = min(start + chunk_numel, src.numel())
        flat_dst[start:end].copy_(flat_src[start:end], non_blocking=non_blocking)
        chunks += 1
    return chunks


def _new_event_if_cuda(device: torch.device, stream: Optional[torch.cuda.Stream]) -> Optional[torch.cuda.Event]:
    if device.type != "cuda" or not torch.cuda.is_available() or stream is None:
        return None
    event = torch.cuda.Event()
    stream.record_event(event)
    return event


def _module_device(module: torch.nn.Module) -> torch.device:
    for tensor in module.parameters():
        return tensor.device
    for tensor in module.buffers():
        return tensor.device
    return torch.device("cpu")


def _copy_modules(layers: Sequence[torch.nn.Module]) -> dict[Optional[torch.nn.Module], Optional[torch.nn.Module]]:
    module_copies: dict[Optional[torch.nn.Module], Optional[torch.nn.Module]] = {None: None}
    for layer in layers:
        for module in layer.modules():
            if module not in module_copies:
                module_copies[module] = copy.copy(module)
    return module_copies


def _rewire_module_graph(
    module_copies: dict[Optional[torch.nn.Module], Optional[torch.nn.Module]],
    param_copies: dict[Optional[torch.nn.Parameter], Optional[torch.nn.Parameter]],
    buffer_copies: dict[Optional[torch.Tensor], Optional[torch.Tensor]],
) -> None:
    for old_module, new_module in module_copies.items():
        if old_module is None or new_module is None:
            continue
        new_module._modules = {
            name: module_copies[child] for name, child in old_module._modules.items()
        }
        new_module._parameters = {
            name: param_copies[param] for name, param in old_module._parameters.items()
        }
        new_module._buffers = {
            name: buffer_copies[buf] for name, buf in old_module._buffers.items()
        }


@torch.no_grad()
def upload_layer_copies(
    layers: Sequence[torch.nn.Module],
    device: torch.device | str | int,
    *,
    with_grad: bool = False,
    chunk_bytes: int = DEFAULT_CHUNK_UPLOAD_BYTES,
    stream: Optional[torch.cuda.Stream] = None,
) -> LayerTransferResult:
    """Create shallow layer copies with params/buffers copied to *device*.

    Shared parameters and buffers remain shared inside the copied graph. This is
    a utility equivalent of RoundPipe's `upload_layers()`, not a runtime hook.
    """
    device = _as_device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA is not available for layer upload to {device}")

    module_copies = _copy_modules(layers)
    param_copies: dict[Optional[torch.nn.Parameter], Optional[torch.nn.Parameter]] = {None: None}
    buffer_copies: dict[Optional[torch.Tensor], Optional[torch.Tensor]] = {None: None}
    non_blocking = device.type == "cuda"

    def issue_copies() -> None:
        for layer in layers:
            for param in layer.parameters():
                if param in param_copies:
                    continue
                data = torch.empty_like(param.detach(), device=device)
                copy_tensor_chunked(param.detach(), data, chunk_bytes=chunk_bytes, non_blocking=non_blocking)
                param_copy = torch.nn.Parameter(data, requires_grad=param.requires_grad)
                if with_grad and param.grad is not None:
                    grad = torch.empty_like(param.grad.detach(), device=device)
                    copy_tensor_chunked(param.grad.detach(), grad, chunk_bytes=chunk_bytes, non_blocking=non_blocking)
                    param_copy.grad = grad
                param_copies[param] = param_copy

            for buffer in layer.buffers():
                if buffer in buffer_copies:
                    continue
                buffer_copy = torch.empty_like(buffer.detach(), device=device)
                copy_tensor_chunked(buffer.detach(), buffer_copy, chunk_bytes=chunk_bytes, non_blocking=non_blocking)
                buffer_copies[buffer] = buffer_copy

    if device.type == "cuda":
        if stream is None:
            stream = torch.cuda.Stream(device=device)
        with torch.cuda.device(device), torch.cuda.stream(stream):
            issue_copies()
            event = _new_event_if_cuda(device, stream)
    else:
        issue_copies()
        event = None

    _rewire_module_graph(module_copies, param_copies, buffer_copies)
    copied_layers = [module_copies[layer] for layer in layers]
    return LayerTransferResult([layer for layer in copied_layers if layer is not None], event=event, stream=stream)


@torch.no_grad()
def download_layer_state(
    cpu_layer: torch.nn.Module,
    device_layer: torch.nn.Module,
    *,
    with_buffer: bool = False,
    with_grad: bool = True,
    chunk_bytes: int = DEFAULT_CHUNK_UPLOAD_BYTES,
    stream: Optional[torch.cuda.Stream] = None,
) -> DownloadResult:
    """Copy gradients and optional buffers from a device-layer copy to CPU."""
    device = _module_device(device_layer)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA layer download requested but CUDA is not available")
    non_blocking = device.type == "cuda"
    gradients = 0
    buffers = 0

    def issue_copies() -> None:
        nonlocal gradients, buffers
        if with_grad:
            cpu_params = list(cpu_layer.named_parameters())
            dev_params = list(device_layer.named_parameters())
            if [name for name, _ in cpu_params] != [name for name, _ in dev_params]:
                raise ValueError("layer parameter structure changed during transfer")
            for (cpu_name, cpu_param), (dev_name, dev_param) in zip(cpu_params, dev_params):
                if cpu_name != dev_name:
                    raise ValueError("layer parameter structure changed during transfer")
                if dev_param.grad is None:
                    continue
                if cpu_param.grad is None or cpu_param.grad.shape != dev_param.grad.shape:
                    cpu_param.grad = torch.empty_like(dev_param.grad, device=torch.device("cpu"))
                copy_tensor_chunked(
                    dev_param.grad.detach(),
                    cpu_param.grad,
                    chunk_bytes=chunk_bytes,
                    non_blocking=non_blocking,
                )
                gradients += 1

        if with_buffer:
            cpu_buffers = list(cpu_layer.named_buffers())
            dev_buffers = list(device_layer.named_buffers())
            if [name for name, _ in cpu_buffers] != [name for name, _ in dev_buffers]:
                raise ValueError("layer buffer structure changed during transfer")
            for (cpu_name, cpu_buffer), (dev_name, dev_buffer) in zip(cpu_buffers, dev_buffers):
                if cpu_name != dev_name:
                    raise ValueError("layer buffer structure changed during transfer")
                copy_tensor_chunked(
                    dev_buffer.detach(),
                    cpu_buffer.data,
                    chunk_bytes=chunk_bytes,
                    non_blocking=non_blocking,
                )
                buffers += 1

    if device.type == "cuda":
        if stream is None:
            stream = torch.cuda.Stream(device=device)
        with torch.cuda.device(device), torch.cuda.stream(stream):
            issue_copies()
            event = _new_event_if_cuda(device, stream)
    else:
        issue_copies()
        event = None

    return DownloadResult(gradients=gradients, buffers=buffers, event=event, stream=stream)
