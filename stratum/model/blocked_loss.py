"""BlockedPostfixCausalLMLoss — norm + lm_head with per-block backward.

Ported from train_lfm25_roundpipe_lora.py::BlockedPostfixCausalLMLoss.

Saves VRAM at long context by:
  1. Splitting the sequence into token blocks for the norm + lm_head forward
  2. Backward-propagating through each block's lm_head output via
     per-block .backward() calls (accumulates lm_head gradients)
  3. Saving the hidden-states gradient to CPU and restoring it in the
     outer autograd backward

This is an alternative to the simpler chunked loss in the postfix.
Enable with --postfix-loss-token-chunk-size N.
"""

from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn
from stratum.telemetry import assert_finite_tensor, mark_model_gpu_phase as _log_phase


def _make_block_loss(
    block_hidden: torch.Tensor,
    lm_head: nn.Linear,
    block_labels: torch.Tensor,
    vocab_size: int,
    num_items: torch.Tensor,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Compute cross-entropy loss for one block.

    Hidden: [batch, chunk, hidden] after norm
    Labels: [batch, chunk] pre-shift (the outer code handles shifting)
    """
    logits = lm_head(block_hidden)
    loss = nn.functional.cross_entropy(
        logits.reshape(-1, vocab_size),
        block_labels.reshape(-1),
        ignore_index=ignore_index,
        reduction="sum",
    )
    return loss / num_items


class BlockedPostfixCausalLMLoss(torch.autograd.Function):
    """Custom autograd: split norm + lm_head over token blocks.

    Forward:
      - Shifts labels for causal LM
      - Counts non-ignored tokens
      - Iterates token blocks: norm → lm_head → CE(reduction=sum) / total_items
      - Calls .backward() on each block's loss to accumulate lm_head gradients
      - Saves hidden_states gradient to CPU

    Backward:
      - Restores hidden_states gradient from CPU
      - Multiplies by grad_loss (typically 1.0)
    """

    @staticmethod
    def forward(
        ctx: Any,
        hidden_states: torch.Tensor,
        labels: torch.Tensor,
        norm: nn.Module,
        lm_head: nn.Linear,
        vocab_size: int,
        token_chunk_size: int,
        ignore_index: int,
        memory_telemetry: bool = False,
        debug_finite: bool = False,
    ) -> torch.Tensor:
        if any(param.requires_grad for param in norm.parameters()):
            raise ValueError("--postfix-loss-token-chunk-size does not yet support trainable final norm")
        if any(param.requires_grad for param in lm_head.parameters()):
            raise ValueError("--postfix-loss-token-chunk-size does not yet support trainable lm_head")

        # Detach hidden_states and re-attach grad so backward stops here
        detached_hidden = hidden_states.detach().requires_grad_(ctx.needs_input_grad[0])

        # Shift labels for causal LM: predict token t+1 from hidden at position t
        shifted_labels = nn.functional.pad(labels, (0, 1), value=ignore_index)[..., 1:].contiguous()
        flat_shifted_labels = shifted_labels.view(-1)
        num_items = (flat_shifted_labels != ignore_index).sum().to(hidden_states.device)

        if int(num_items.detach().cpu().item()) == 0:
            grad_saved = torch.zeros_like(hidden_states) if ctx.needs_input_grad[0] else None
            ctx.save_for_backward(grad_saved) if grad_saved is not None else ctx.save_for_backward()
            return hidden_states.new_zeros(())

        batch, seq_len, _hidden_size = hidden_states.shape
        if batch != 1:
            raise ValueError("--postfix-loss-token-chunk-size currently expects batch-size 1")

        loss_sum = hidden_states.new_zeros(())
        with torch.enable_grad():
            for start in range(0, seq_len, token_chunk_size):
                end = min(start + token_chunk_size, seq_len)
                block_norm = norm(detached_hidden[:, start:end, :])
                if debug_finite:
                    assert_finite_tensor(f"post_norm_hidden_{start}_{end}", block_norm)

                block_labels = flat_shifted_labels[start:end]
                block_labels_2d = shifted_labels[:, start:end]
                block_loss = _make_block_loss(
                    block_norm, lm_head, block_labels_2d,
                    vocab_size, num_items, ignore_index,
                )
                if block_loss.requires_grad:
                    block_loss.backward()
                loss_sum = loss_sum + block_loss.detach()

                if memory_telemetry and (start == 0 or end == seq_len):
                    _log_phase("postfix_loss_block_after_loss",
                               token_start=start, token_end=end, seq_len=seq_len)

        grad_hidden_states = detached_hidden.grad
        if grad_hidden_states is None and ctx.needs_input_grad[0]:
            raise RuntimeError("blocked postfix loss did not produce hidden-state gradients")

        if grad_hidden_states is not None:
            ctx.hidden_states_device = hidden_states.device
            ctx.hidden_states_dtype = hidden_states.dtype
            ctx.memory_telemetry = memory_telemetry
            if memory_telemetry:
                print({
                    "postfix_saved_grad": {
                        "shape": list(grad_hidden_states.shape),
                        "dtype": str(grad_hidden_states.dtype),
                        "element_size": grad_hidden_states.element_size(),
                    }
                }, flush=True)
            ctx.save_for_backward(grad_hidden_states.detach().to("cpu", copy=True))
            detached_hidden.grad = None
            del grad_hidden_states
            if hidden_states.device.type == "cuda":
                torch.cuda.empty_cache()
        else:
            ctx.memory_telemetry = memory_telemetry
            ctx.save_for_backward()

        return loss_sum

    @staticmethod
    def backward(ctx: Any, grad_loss: torch.Tensor) -> tuple[Optional[torch.Tensor], ...]:
        saved = ctx.saved_tensors
        if saved:
            if getattr(ctx, "memory_telemetry", False):
                _log_phase("postfix_backward_before_grad_restore")
            grad_hidden_states = saved[0].to(
                device=ctx.hidden_states_device,
                dtype=ctx.hidden_states_dtype,
                non_blocking=False,
            ).mul_(grad_loss)
            if getattr(ctx, "memory_telemetry", False):
                _log_phase("postfix_backward_after_grad_restore")
        else:
            grad_hidden_states = None
        return grad_hidden_states, None, None, None, None, None, None, None, None


# (shared helpers used from stratum.telemetry)
