"""Per-device optimiser with synchronised LR scheduling and optional CPU offload.

Ported from RoundPipe's PerDeviceOptimizer + RoundPipeBase optimizer methods.

Two modes:
  - **Synchronous** (default): AdamW on GPU, same as before. Optimizer state
    lives on the same device as the trainable parameters.
  - **CPU-offloaded** (``--cpu-offload-optim``): AdamW operates on fp32 CPU
    copies of trainable parameters. The GPU only holds fp16 forward params;
    gradients are moved to the CPU optimizer copies before each step, and
    updated params are copied back. Frees ~2× trainable-param GPU memory.
"""

from __future__ import annotations

from typing import Any, Iterator, Optional, Sequence

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from stratum.attribute import ParamAttribute
from stratum._threads import AnnotatedEvent
from stratum.optim_stream import (
    launch_optim_kernel,
    on_optim_stream,
    synchronize_optim,
)
from stratum.utils import log_event


class PerDeviceOptimizer:
    """Manages one AdamW optimiser per device for local LoRA parameters.

    LR schedulers are synchronised across devices so all optimisers see the
    same LR at the same step.

    Args:
        modules_per_device: Mapping of device_id -> list of modules whose
            trainable parameters should be optimised.
        lr: Learning rate.
        weight_decay: Weight decay.
        scheduler: LR scheduler type.
        warmup_steps: Warmup steps for cosine_with_warmup scheduler.
        total_steps: Total training steps.
        cpu_offload: If True, maintain fp32 CPU copies of trainable params
            and move gradients/updates between GPU and CPU. Saves GPU memory.
        optim_dtype: Data type for CPU optimizer copies (default fp32).
            Only used when cpu_offload=True.
    """

    def __init__(
        self,
        modules_per_device: dict[int, list[torch.nn.Module]],
        lr: float = 1e-4,
        weight_decay: float = 0.0,
        scheduler: str = "cosine_with_warmup",
        warmup_steps: int = 500,
        total_steps: int = 25000,
        *,
        cpu_offload: bool = False,
        optim_dtype: torch.dtype = torch.float32,
    ):
        self.cpu_offload = cpu_offload
        self.optim_dtype = optim_dtype
        self.modules_per_device = modules_per_device

        self.optimizers: dict[int, torch.optim.Optimizer | None] = {}
        self.schedulers: dict[int, object] = {}
        self._optim_updated: AnnotatedEvent = AnnotatedEvent(f"opt_upd")
        self._optim_updated.set()
        self._last_step_was_skipped = False

        for device_id, modules in modules_per_device.items():
            if cpu_offload:
                # Create AdamW on CPU fp32 copies of trainable params.
                # The GPU still holds fp16 forward params; we move grads
                # to the CPU optim copies after backward.
                optim_params = list(self._cpu_optim_params(modules))
            else:
                # Original behaviour: AdamW on GPU params directly.
                optim_params = [
                    p for m in modules for p in m.parameters() if p.requires_grad
                ]

            if not optim_params:
                self.optimizers[device_id] = None
                self.schedulers[device_id] = None
                continue

            opt = AdamW(optim_params, lr=lr, weight_decay=weight_decay, betas=(0.9, 0.95))
            self.optimizers[device_id] = opt

            if scheduler == "constant" or total_steps <= 0:
                self.schedulers[device_id] = None
            elif scheduler == "cosine":
                self.schedulers[device_id] = CosineAnnealingLR(
                    opt, T_max=total_steps, eta_min=lr * 0.1
                )
            elif scheduler == "cosine_with_warmup":
                warmup = LinearLR(
                    opt, start_factor=0.01, end_factor=1.0,
                    total_iters=warmup_steps,
                )
                cosine = CosineAnnealingLR(
                    opt, T_max=max(1, total_steps - warmup_steps),
                    eta_min=lr * 0.1,
                )
                self.schedulers[device_id] = SequentialLR(
                    opt, schedulers=[warmup, cosine],
                    milestones=[warmup_steps],
                )
            else:
                self.schedulers[device_id] = None

    # ---- CPU-offloaded param management ----

    def _cpu_optim_params(self, modules: list[torch.nn.Module]) -> Iterator[torch.nn.Parameter]:
        """Yield lazy-created fp32 CPU copies of trainable params.

        Ported from RoundPipeBase.optim_named_parameters(). Each trainable
        parameter gets a ParamAttribute with an fp32 CPU copy on first access.
        """
        visited: set[int] = set()
        for m in modules:
            for p in m.parameters():
                if not p.requires_grad:
                    continue
                if id(p) in visited:
                    continue
                visited.add(id(p))
                attr = ParamAttribute.ensure(p)
                if attr.optim is None:
                    attr.optim = torch.nn.Parameter(
                        p.detach().to(dtype=self.optim_dtype, device="cpu", copy=True),
                        requires_grad=True,
                    )
                yield attr.optim

    def ensure_optim_params(self) -> None:
        """Ensure all CPU optim copies exist (call after LoRA setup)."""
        if not self.cpu_offload:
            return
        for modules in self.modules_per_device.values():
            for _ in self._cpu_optim_params(modules):
                pass

    # ---- Gradient movement (runs on optimizer stream) ----

    def _record_grad_ready_events(self) -> list[torch.cuda.Event]:
        """Record per-device CUDA events after backward before CPU grad reads.

        The CPU optimizer stream is a Python thread, not a CUDA stream. When it
        copies CUDA gradients to CPU optimizer params, it must first wait for
        the compute streams that produced those gradients. Stratum currently
        pins backward/recompute work to the device default streams, so recording
        one event per CUDA gradient device is the adapted RoundPipe handoff.
        """
        if not torch.cuda.is_available():
            return []

        devices: list[torch.device] = []
        seen_devices: set[torch.device] = set()
        seen_params: set[int] = set()
        for modules in self.modules_per_device.values():
            for module in modules:
                for param in module.parameters():
                    if not param.requires_grad or id(param) in seen_params:
                        continue
                    seen_params.add(id(param))
                    grad = param.grad
                    if grad is None or not grad.is_cuda:
                        continue
                    device = grad.device
                    if device in seen_devices:
                        continue
                    seen_devices.add(device)
                    devices.append(device)

        events: list[torch.cuda.Event] = []
        for device in devices:
            event = torch.cuda.Event()
            event.record(torch.cuda.default_stream(device))
            events.append(event)
        return events

    @staticmethod
    def _wait_grad_ready_events(events: Sequence[Any] | None) -> None:
        if not events:
            return
        for event in events:
            event.synchronize()

    def _move_grad_to_optim(
        self,
        grad_ready_events: Sequence[Any] | None = None,
    ) -> None:
        """Move GPU parameter gradients to fp32 CPU optimizer copies.

        Ported from RoundPipe's _move_grad_to_optim(). Designed to run on
        the optimizer stream thread. Collects gradients from GPU params
        and copies them into the fp32 CPU optim copies' .grad.
        """
        if not on_optim_stream():
            raise RuntimeError("_move_grad_to_optim must run on the optim stream")
        self._wait_grad_ready_events(grad_ready_events)
        visited: set[int] = set()
        for modules in self.modules_per_device.values():
            for m in modules:
                for name, param in m.named_parameters():
                    if not param.requires_grad:
                        continue
                    if id(param) in visited:
                        continue
                    visited.add(id(param))
                    attr = ParamAttribute.get(param)
                    if attr is None or attr.optim is None:
                        continue

                    grad = param.grad
                    if grad is None:
                        attr.optim_grad_buffer = None
                        attr.optim.grad = None
                        continue

                    cpu_grad = grad.detach().to(
                        dtype=attr.optim.dtype,
                        device=attr.optim.device,
                    )

                    # Copy GPU grad to CPU optim copy's grad
                    if attr.optim.grad is None:
                        attr.optim.grad = attr.optim_grad_buffer
                        if attr.optim.grad is None:
                            attr.optim.grad = torch.empty_like(attr.optim)
                        attr.optim.grad.copy_(cpu_grad)
                    else:
                        attr.optim.grad.add_(cpu_grad)
                    attr.optim_grad_buffer = attr.optim.grad

                    # Zero the GPU grad to free memory
                    param.grad = None

    # ---- Parameter sync (runs on optimizer stream) ----

    def sync_optim_param(self) -> None:
        """Copy updated fp32 CPU optim params back to GPU model params.

        Ported from RoundPipe's sync_optim_param(). Runs on the optimizer
        stream after optimizer.step().
        """
        visited: set[int] = set()
        for modules in self.modules_per_device.values():
            for m in modules:
                for param in m.parameters():
                    if not param.requires_grad:
                        continue
                    if id(param) in visited:
                        continue
                    visited.add(id(param))
                    attr = ParamAttribute.get(param)
                    if attr is not None and attr.optim is not None:
                        param.data.copy_(attr.optim.data.to(dtype=param.dtype, device=param.device))

    def synchronize(self) -> None:
        """Wait for any queued CPU-offloaded optimizer step to finish.

        For CPU offload, ``_optim_updated`` is set only after updated CPU
        optimizer parameters have been copied back to the live model params.
        """
        if not self.cpu_offload:
            return
        self._optim_updated.wait()
        synchronize_optim()

    def last_step_was_skipped(self) -> bool:
        """Return whether the most recent optimizer step was skipped by AMP."""
        return self._last_step_was_skipped

    # ---- Step ----

    def step(self, *, async_step: bool = False,
             scaler: Any = None) -> None:
        """Run optimizer step on all devices.

        Ported from RoundPipeBase.step(), extended with GradScaler support
        from roundpipe/grad_scaler.py.

        Args:
            async_step: If True, schedule the optimizer step on the background
                optimizer stream and return immediately. The next iteration
                will use 1-step-old parameters. If False, run synchronously
                (default, matching current behaviour).
            scaler: Optional ``GradScaler`` instance. When provided, uses
                ``scaler.step(opt)`` instead of ``opt.step()`` to unscale
                gradients and handle inf/NaN detection.
        """
        self._last_step_was_skipped = False
        if self.cpu_offload:
            self._step_cpu_offload(async_step=async_step, scaler=scaler)
        else:
            # Synchronous original path
            for opt in self.optimizers.values():
                if opt is not None:
                    if scaler is not None:
                        scaler.step(opt)
                        self._last_step_was_skipped |= _scaler_step_was_skipped(
                            scaler, opt
                        )
                    else:
                        opt.step()

    def _step_cpu_offload(self, *, async_step: bool = False,
                          scaler: Any = None) -> None:
        """Async optimizer step with CPU-offloaded parameters."""
        self._optim_updated.wait()  # ensure previous step is done
        grad_ready_events = self._record_grad_ready_events()

        launch_optim_kernel(self._move_grad_to_optim, grad_ready_events)
        self._optim_updated.clear()

        # Actual optimizer.step() on the optim stream
        for device_id, opt in self.optimizers.items():
            if opt is not None:

                def _step_one(opt=opt) -> None:
                    if scaler is not None:
                        scaler.step(opt)
                        self._last_step_was_skipped |= _scaler_step_was_skipped(
                            scaler, opt
                        )
                    else:
                        opt.step()

                launch_optim_kernel(_step_one)

        launch_optim_kernel(self.sync_optim_param)
        launch_optim_kernel(self._optim_updated.set)

        if not async_step:
            self.synchronize()

    def zero_grad(self) -> None:
        """Zero all gradients."""
        if self.cpu_offload:
            for opt in self.optimizers.values():
                if opt is not None:
                    opt.zero_grad(set_to_none=True)
            # Zero GPU param grads (CPU optim grads reused via _move_grad_to_optim)
            for modules in self.modules_per_device.values():
                for m in modules:
                    for p in m.parameters():
                        if p.requires_grad and p.grad is not None:
                            p.grad = None
        else:
            for opt in self.optimizers.values():
                if opt is not None:
                    opt.zero_grad()

    def scheduler_step(self) -> None:
        """Step all LR schedulers."""
        for sched in self.schedulers.values():
            if sched is not None:
                sched.step()

    def log_lr(self, step: int) -> None:
        """Log current LR for each device (call periodically)."""
        lrs = self.get_lr()
        for dev, lr in lrs.items():
            if lr > 0:
                log_event("lr", step=step, device=dev, lr=f"{lr:.2e}")

    def get_lr(self) -> dict[int, float]:
        """Get current LR per device."""
        return {
            dev: opt.param_groups[0]["lr"] if opt is not None else 0.0
            for dev, opt in self.optimizers.items()
        }


def _scaler_step_was_skipped(scaler: Any, optimizer: torch.optim.Optimizer) -> bool:
    """Detect whether GradScaler skipped ``optimizer.step()`` for inf/NaN grads.

    PyTorch optimizers usually return ``None`` even when they run, so the return
    value of ``GradScaler.step()`` cannot reliably drive LR scheduling. The
    public Stratum scaler exposes ``step_was_skipped``; for torch.amp we read
    the same per-optimizer inf state GradScaler uses internally before
    ``update()`` clears it.
    """
    is_enabled = getattr(scaler, "is_enabled", None)
    if callable(is_enabled):
        if not is_enabled():
            return False
    elif not getattr(scaler, "enabled", True):
        return False

    step_was_skipped = getattr(scaler, "step_was_skipped", None)
    if callable(step_was_skipped):
        return bool(step_was_skipped(optimizer))

    inner = getattr(scaler, "main_scaler", scaler)
    states = getattr(inner, "_per_optimizer_states", None)
    if states is None:
        return False
    state = states.get(id(optimizer))
    if not state:
        return False
    found_inf = state.get("found_inf_per_device", {})
    return any(bool(t.item()) for t in found_inf.values())
