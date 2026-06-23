"""Chunked linear cross-entropy with RoundPipe-style inner backward.

This ports the useful behavior of RoundPipe's
`ChunkedCompileLinearCrossEntropy`: split token rows, run
`linear + cross_entropy(reduction="sum")` per chunk, call backward inside the
custom forward, and return the accumulated gradients from the outer backward.

The result avoids keeping a full `[tokens, vocab]` logits graph alive while
preserving the same loss and gradients as regular linear cross-entropy.
"""

from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn


_COMPILED_LINEAR_CE = None
_COMPILE_WARNING_PRINTED = False


def _linear_cross_entropy(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    labels: torch.Tensor,
    ignore_index: int,
) -> torch.Tensor:
    logits = nn.functional.linear(hidden_states, weight, bias)
    return nn.functional.cross_entropy(
        logits.float(),
        labels,
        ignore_index=ignore_index,
        reduction="sum",
    )


def _compiled_linear_cross_entropy(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    labels: torch.Tensor,
    ignore_index: int,
) -> torch.Tensor:
    global _COMPILED_LINEAR_CE, _COMPILE_WARNING_PRINTED
    if _COMPILED_LINEAR_CE is None:
        try:
            _COMPILED_LINEAR_CE = torch.compile(_linear_cross_entropy)
        except Exception as exc:
            if not _COMPILE_WARNING_PRINTED:
                print(f"torch_compile_loss not available: {exc}", flush=True)
                _COMPILE_WARNING_PRINTED = True
            return _linear_cross_entropy(hidden_states, weight, bias, labels, ignore_index)
    return _COMPILED_LINEAR_CE(hidden_states, weight, bias, labels, ignore_index)


class ChunkedLinearCrossEntropyFunction(torch.autograd.Function):
    """Autograd function that saves per-input grads instead of logits graphs."""

    @staticmethod
    def forward(
        ctx: Any,
        hidden_states: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
        labels: torch.Tensor,
        ignore_index: int,
        token_chunk_size: int,
        global_requires_grad: bool,
        use_torch_compile: bool,
    ) -> torch.Tensor:
        if token_chunk_size <= 0:
            raise ValueError(f"token_chunk_size must be > 0, got {token_chunk_size}")

        hidden_shape = tuple(hidden_states.shape)
        hidden_flat = hidden_states.reshape(-1, hidden_states.shape[-1])
        labels_flat = labels.reshape(-1).to(hidden_flat.device)

        detached_hidden = hidden_flat.detach().requires_grad_(ctx.needs_input_grad[0])
        detached_weight = weight.detach().requires_grad_(ctx.needs_input_grad[1])
        if bias is None:
            detached_bias = None
        else:
            detached_bias = bias.detach().requires_grad_(ctx.needs_input_grad[2])

        loss_sum = hidden_flat.new_zeros(())
        loss_fn = _compiled_linear_cross_entropy if use_torch_compile else _linear_cross_entropy
        grad_context = torch.enable_grad() if global_requires_grad else torch.no_grad()

        with grad_context:
            for chunk_h, chunk_l in zip(
                detached_hidden.split(token_chunk_size, dim=0),
                labels_flat.split(token_chunk_size, dim=0),
            ):
                chunk_loss = loss_fn(
                    chunk_h,
                    detached_weight,
                    detached_bias,
                    chunk_l,
                    ignore_index,
                )
                if chunk_loss.requires_grad:
                    chunk_loss.backward()
                loss_sum = loss_sum + chunk_loss.detach()

        grads: list[torch.Tensor] = []
        grad_hidden = detached_hidden.grad
        grad_weight = detached_weight.grad
        grad_bias = detached_bias.grad if detached_bias is not None else None

        if grad_hidden is not None:
            grad_hidden = grad_hidden.reshape(hidden_shape)
            grads.append(grad_hidden)
        if grad_weight is not None:
            grads.append(grad_weight)
        if grad_bias is not None:
            grads.append(grad_bias)

        ctx.hidden_shape = hidden_shape
        ctx.save_for_backward(*grads)
        return loss_sum

    @staticmethod
    def backward(ctx: Any, grad_loss: torch.Tensor) -> tuple[Optional[torch.Tensor], ...]:
        grads = ctx.saved_tensors
        grads_idx = 0
        grad_outputs: list[Optional[torch.Tensor]] = []

        for input_idx in range(3):
            if ctx.needs_input_grad[input_idx]:
                if grads_idx >= len(grads):
                    grad_outputs.append(None)
                else:
                    grad_outputs.append(grads[grads_idx].mul(grad_loss))
                    grads_idx += 1
            else:
                grad_outputs.append(None)

        return *grad_outputs, None, None, None, None, None


def chunked_linear_cross_entropy(
    hidden_states: torch.Tensor,
    lm_head: nn.Linear,
    labels: torch.Tensor,
    *,
    num_items: Optional[torch.Tensor] = None,
    ignore_index: int = -100,
    token_chunk_size: int = 4096,
    use_torch_compile: bool = False,
) -> torch.Tensor:
    """Return token-normalized CE for already aligned hidden states and labels.

    Args:
        hidden_states: Tensor shaped `[..., hidden_size]`.
        lm_head: Linear projection to vocabulary logits.
        labels: Tensor shaped like `hidden_states.shape[:-1]`.
        num_items: Optional non-ignored-token count. Computed from labels if
            omitted.
        ignore_index: Label value ignored by cross entropy.
        token_chunk_size: Number of flattened token rows per inner backward.
        use_torch_compile: Compile the `linear + CE` chunk function on demand.
    """
    flat_labels = labels.reshape(-1)
    if num_items is None:
        num_items = (flat_labels != ignore_index).sum().to(hidden_states.device)
    else:
        num_items = num_items.to(hidden_states.device)

    if int(num_items.detach().cpu().item()) == 0:
        return hidden_states.sum() * 0.0

    loss_sum = ChunkedLinearCrossEntropyFunction.apply(
        hidden_states,
        lm_head.weight,
        lm_head.bias,
        labels,
        ignore_index,
        token_chunk_size,
        torch.is_grad_enabled(),
        use_torch_compile,
    )
    return loss_sum / num_items
