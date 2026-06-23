"""Per-device optimiser with synchronised LR scheduling."""

from typing import Optional

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from stratum.utils import log_event


class PerDeviceOptimizer:
    """Manages one AdamW optimiser per device for local LoRA parameters.

    LR schedulers are synchronised across devices so all optimisers see the
    same LR at the same step.
    """

    def __init__(
        self,
        modules_per_device: dict[int, list[torch.nn.Module]],
        lr: float = 1e-4,
        weight_decay: float = 0.0,
        scheduler: str = "cosine_with_warmup",
        warmup_steps: int = 500,
        total_steps: int = 25000,
    ):
        self.optimizers: dict[int, torch.optim.Optimizer] = {}
        self.schedulers: dict[int, object] = {}

        for device_id, modules in modules_per_device.items():
            params = [p for m in modules for p in m.parameters() if p.requires_grad]
            if not params:
                self.optimizers[device_id] = None
                self.schedulers[device_id] = None
                continue

            opt = AdamW(params, lr=lr, weight_decay=weight_decay, betas=(0.9, 0.95))
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

    def step(self) -> None:
        """Step all optimisers."""
        for opt in self.optimizers.values():
            if opt is not None:
                opt.step()

    def zero_grad(self) -> None:
        """Zero all gradients."""
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
