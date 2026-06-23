"""Async transfer primitives adapted from RoundPipe.

These helpers are intentionally standalone for now. They provide the stream /
event semantics needed for future NF4 prefetch and activation offload without
changing the current synchronous training path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol, Sequence

import torch


class _WaitableEvent(Protocol):
    def synchronize(self) -> None: ...


@dataclass
class TransferResult:
    tensor: torch.Tensor
    event: Optional[_WaitableEvent] = None
    stream: Optional[torch.cuda.Stream] = None
    pinned_buffer: Optional[torch.Tensor] = None

    def wait(self) -> torch.Tensor:
        """Wait for the transfer event, then return the tensor."""
        if self.event is not None:
            self.event.synchronize()
        return self.tensor


def _as_device(device: torch.device | str | int) -> torch.device:
    if isinstance(device, int):
        return torch.device(f"cuda:{device}")
    return torch.device(device)


def _pin_cpu_tensor(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.device.type != "cpu":
        raise ValueError(f"expected CPU tensor, got {tensor.device}")
    if not torch.cuda.is_available() or tensor.is_pinned():
        return tensor
    pinned = torch.empty_like(tensor, device=torch.device("cpu"), pin_memory=True)
    pinned.copy_(tensor.detach())
    return pinned


def _copy_sync(tensor: torch.Tensor, device: torch.device, keep_requires_grad: bool) -> torch.Tensor:
    requires_grad = tensor.requires_grad
    with torch.no_grad():
        out = tensor.detach().to(device).clone()
    out.requires_grad_(keep_requires_grad and requires_grad)
    return out


def _wait_events(stream: torch.cuda.Stream, events: Optional[Sequence[torch.cuda.Event]]) -> None:
    if not events:
        return
    for event in events:
        stream.wait_event(event)


def async_h2d(
    tensor: torch.Tensor,
    device: torch.device | str | int,
    *,
    stream: Optional[torch.cuda.Stream] = None,
    wait_events: Optional[Sequence[torch.cuda.Event]] = None,
    keep_requires_grad: bool = False,
) -> TransferResult:
    """Copy a host tensor to *device* with pinned fallback and event output."""
    device = _as_device(device)
    requires_grad = tensor.requires_grad
    if device.type != "cuda":
        return TransferResult(_copy_sync(tensor, device, keep_requires_grad))

    if not torch.cuda.is_available():
        raise RuntimeError(f"CUDA is not available for async_h2d to {device}")

    if tensor.device.type != "cpu":
        raise ValueError(f"async_h2d expected a CPU tensor, got {tensor.device}")

    host = _pin_cpu_tensor(tensor)
    if stream is None:
        stream = torch.cuda.Stream(device=device)
    with torch.cuda.device(device), torch.cuda.stream(stream):
        _wait_events(stream, wait_events)
        with torch.no_grad():
            out = host.detach().to(device, non_blocking=True)
        out.requires_grad_(keep_requires_grad and requires_grad)
        event = torch.cuda.Event()
        stream.record_event(event)
    return TransferResult(out, event=event, stream=stream, pinned_buffer=host)


def async_d2h(
    tensor: torch.Tensor,
    *,
    stream: Optional[torch.cuda.Stream] = None,
    keep_requires_grad: bool = False,
) -> TransferResult:
    """Copy a CUDA tensor to pinned host memory with event output."""
    requires_grad = tensor.requires_grad
    if tensor.device.type != "cuda":
        return TransferResult(_copy_sync(tensor, torch.device("cpu"), keep_requires_grad))

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA tensor transfer requested but CUDA is not available")

    if stream is None:
        stream = torch.cuda.Stream(device=tensor.device)
    host = torch.empty_like(tensor, device=torch.device("cpu"), pin_memory=True)
    with torch.cuda.device(tensor.device), torch.cuda.stream(stream):
        host.copy_(tensor.detach(), non_blocking=True)
        host.requires_grad_(keep_requires_grad and requires_grad)
        event = torch.cuda.Event()
        stream.record_event(event)
    return TransferResult(host, event=event, stream=stream, pinned_buffer=host)


class PinnedUpload(torch.autograd.Function):
    """Autograd H2D copy that mirrors pageable CPU tensors through pinned RAM."""

    @staticmethod
    def forward(ctx, tensor: torch.Tensor, device: torch.device | str | int) -> torch.Tensor:
        result = async_h2d(
            tensor,
            _as_device(device),
            keep_requires_grad=False,
        )
        return result.wait()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        if grad_output.device.type == "cuda" and torch.cuda.is_available():
            grad = async_d2h(grad_output).wait()
        else:
            grad = grad_output.to(torch.device("cpu"))
        return grad, None


class RegisterBackwardEvent(torch.autograd.Function):
    """Pass-through tensor whose backward waits on a recorded CUDA event."""

    @staticmethod
    def forward(ctx, tensor: torch.Tensor, event: Optional[torch.cuda.Event]) -> torch.Tensor:
        ctx.event = event
        return tensor

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        event = ctx.event
        if event is not None:
            event.synchronize()
        return grad_output, None
