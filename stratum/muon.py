"""Hybrid Muon + AdamW optimizer.

Muon is applied to matrix-shaped parameters. Higher-rank tensors are treated as
batches of matrices over their final two dimensions. Vector/scalar trainable
parameters fall back to AdamW-style updates.
"""

from __future__ import annotations

import math
from typing import Iterable

import torch
from torch.optim.optimizer import Optimizer


def _orthogonalize_newton_schulz(grad: torch.Tensor, steps: int) -> torch.Tensor:
    """Return Newton-Schulz orthogonalized matrix updates in fp32.

    For tensors with ndim > 2, each final-two-dimension slice is handled as an
    independent matrix. This matches stacked expert-weight layouts without
    coupling unrelated experts through one flattened orthogonalization.
    """
    if grad.ndim < 2:
        raise ValueError("Muon orthogonalization expects at least a 2D tensor")

    x = grad.float()
    if x.numel() == 0:
        return x

    original_shape = x.shape
    x = x.reshape(-1, original_shape[-2], original_shape[-1])
    transposed = False
    if x.shape[-2] > x.shape[-1]:
        x = x.transpose(-2, -1)
        transposed = True

    norm = x.norm(dim=(-2, -1), keepdim=True)
    finite = torch.isfinite(norm) & (norm > 0)
    x = torch.where(finite, x / (norm + 1e-7), torch.zeros_like(x))
    # Coefficients used by common Muon implementations for quintic
    # Newton-Schulz orthogonalization.
    a, b, c = 3.4445, -4.7750, 2.0315
    for _ in range(max(0, steps)):
        xx_t = x @ x.transpose(-2, -1)
        x = a * x + (b * xx_t + c * (xx_t @ xx_t)) @ x
    out = x

    if transposed:
        out = out.transpose(-2, -1)
    return out.reshape(original_shape)


class MuonAdamW(Optimizer):
    """Hybrid optimizer: Muon for ndim >= 2, AdamW fallback otherwise.

    ``adamw_param_ids`` forces selected matrix parameters through the AdamW
    path. Stratum uses this for Q/K attention projections by default because
    Muon on Q/K without QK-Clip can collapse attention patterns.
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        *,
        lr: float = 1e-4,
        weight_decay: float = 0.0,
        betas: tuple[float, float] = (0.9, 0.95),
        eps: float = 1e-8,
        muon_momentum: float = 0.95,
        muon_ns_steps: int = 5,
        muon_update_scale: float = 0.2,
        adamw_param_ids: Iterable[int] | None = None,
    ) -> None:
        if lr < 0:
            raise ValueError(f"invalid lr: {lr}")
        if weight_decay < 0:
            raise ValueError(f"invalid weight_decay: {weight_decay}")
        if not 0 <= muon_momentum < 1:
            raise ValueError(f"invalid muon_momentum: {muon_momentum}")
        if muon_ns_steps < 0:
            raise ValueError(f"invalid muon_ns_steps: {muon_ns_steps}")

        defaults = {
            "lr": lr,
            "weight_decay": weight_decay,
            "betas": betas,
            "eps": eps,
            "muon_momentum": muon_momentum,
            "muon_ns_steps": muon_ns_steps,
            "muon_update_scale": muon_update_scale,
        }
        super().__init__(list(params), defaults)
        self._adamw_param_ids = set(adamw_param_ids or [])

    @torch.no_grad()
    def step(self, closure=None):  # type: ignore[override]
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            muon_momentum = group["muon_momentum"]
            muon_ns_steps = group["muon_ns_steps"]
            muon_update_scale = group["muon_update_scale"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("MuonAdamW does not support sparse gradients")

                if p.ndim >= 2 and id(p) not in self._adamw_param_ids:
                    self._step_muon(
                        p,
                        grad,
                        lr=lr,
                        weight_decay=weight_decay,
                        momentum=muon_momentum,
                        ns_steps=muon_ns_steps,
                        update_scale=muon_update_scale,
                    )
                else:
                    self._step_adamw(
                        p,
                        grad,
                        lr=lr,
                        weight_decay=weight_decay,
                        beta1=beta1,
                        beta2=beta2,
                        eps=eps,
                    )
        return loss

    def _step_muon(
        self,
        p: torch.Tensor,
        grad: torch.Tensor,
        *,
        lr: float,
        weight_decay: float,
        momentum: float,
        ns_steps: int,
        update_scale: float,
    ) -> None:
        state = self.state[p]
        if len(state) == 0:
            state["step"] = torch.tensor(0.0, device=p.device)
            state["momentum_buffer"] = torch.zeros_like(p, dtype=torch.float32)

        state["step"] += 1
        if weight_decay != 0:
            p.mul_(1 - lr * weight_decay)

        buf = state["momentum_buffer"]
        buf.mul_(momentum).add_(grad.detach().float(), alpha=1 - momentum)
        update = _orthogonalize_newton_schulz(buf, ns_steps)
        scale = update_scale * math.sqrt(max(p.shape[-2:]))
        p.add_(update.to(dtype=p.dtype, device=p.device), alpha=-lr * scale)

    def _step_adamw(
        self,
        p: torch.Tensor,
        grad: torch.Tensor,
        *,
        lr: float,
        weight_decay: float,
        beta1: float,
        beta2: float,
        eps: float,
    ) -> None:
        state = self.state[p]
        if len(state) == 0:
            state["step"] = torch.tensor(0.0, device=p.device)
            state["exp_avg"] = torch.zeros_like(p, dtype=torch.float32)
            state["exp_avg_sq"] = torch.zeros_like(p, dtype=torch.float32)

        state["step"] += 1
        step = int(state["step"].item())
        if weight_decay != 0:
            p.mul_(1 - lr * weight_decay)

        exp_avg = state["exp_avg"]
        exp_avg_sq = state["exp_avg_sq"]
        grad_f = grad.detach().float()
        exp_avg.mul_(beta1).add_(grad_f, alpha=1 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad_f, grad_f, value=1 - beta2)

        bias_correction1 = 1 - beta1**step
        bias_correction2 = 1 - beta2**step
        step_size = lr / bias_correction1
        denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)
        p.addcdiv_(exp_avg.to(dtype=p.dtype), denom.to(dtype=p.dtype), value=-step_size)
