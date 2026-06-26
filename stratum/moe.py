"""MoE utilities: router logit capture and auxiliary load-balancing loss.

Ported from roundpipe/models/qwen3_moe.py and HF transformers.

Two utilities:
  1. ``patch_moe_block_for_router_logits(module)`` — monkey-patches an
     ``Lfm2MoeSparseMoeBlock`` (or similar) so its forward records router
     logits without changing the block's normal tensor return value.
  2. ``load_balancing_loss_func(router_logits, num_experts, top_k)`` —
     computes the Switch Transformer auxiliary load-balancing loss.
"""

from __future__ import annotations

from typing import Callable, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


def patch_moe_block_for_router_logits(module: nn.Module) -> int:
    """Monkey-patch MoE blocks to record router logits as side-channel state.

    The original HF ``Lfm2MoeSparseMoeBlock.forward()`` computes
    ``router_logits = self.gate(...)`` internally but discards them.  This
    patch re-wraps forward to capture the logits without changing the return
    value that the HF decoder layer expects, at the cost of one extra gate
    linear call.

    Supports multiple MoE block types via a registry check.  Currently:
    - ``Lfm2MoeSparseMoeBlock`` (LFM2.5)

    Args:
        module: A module that may contain MoE submodules.

    Returns:
        Number of MoE blocks patched.
    """
    # Collect available MoE block types (guarded imports)
    _moe_block_types = []

    try:
        from transformers.models.lfm2_moe.modeling_lfm2_moe import (
            Lfm2MoeSparseMoeBlock,
        )
        _moe_block_types.append(Lfm2MoeSparseMoeBlock)
    except (ImportError, ModuleNotFoundError):
        pass
    try:
        from transformers.models.qwen3_moe.modeling_qwen3_moe import (
            Qwen3MoeSparseMoeBlock,
        )
        _moe_block_types.append(Qwen3MoeSparseMoeBlock)
    except (ImportError, ModuleNotFoundError):
        pass

    if not _moe_block_types:
        return 0

    patched = 0
    for sub in module.modules():
        if not any(isinstance(sub, t) for t in _moe_block_types):
            continue
        if getattr(sub, "_router_patched", False):
            continue
        orig_forward = sub.forward

        def _make_patched_forward(
            block: nn.Module,
            original_forward: Callable[[torch.Tensor], torch.Tensor | tuple],
        ) -> Callable[[torch.Tensor], torch.Tensor | tuple]:
            def patched_forward(hidden_states: torch.Tensor) -> torch.Tensor | tuple:
                # Original forward computes gate internally but returns only hiddens.
                result = original_forward(hidden_states)
                # Recompute gate to capture router_logits (single linear layer).
                hidden_dim = hidden_states.shape[-1]
                flat_hidden = hidden_states.reshape(-1, hidden_dim)
                gate_output = block.gate(flat_hidden)
                router_logits = gate_output[0] if isinstance(gate_output, tuple) else gate_output
                block._last_router_logits = router_logits
                return result

            return patched_forward

        sub.forward = _make_patched_forward(sub, orig_forward)
        sub._router_patched = True
        sub._last_router_logits = None
        patched += 1

    return patched


def pop_router_logits(module: nn.Module) -> list[torch.Tensor]:
    """Collect and clear router logits recorded by patched MoE blocks."""
    router_logits: list[torch.Tensor] = []
    for sub in module.modules():
        captured = getattr(sub, "_last_router_logits", None)
        if captured is not None:
            router_logits.append(captured)
            sub._last_router_logits = None
    return router_logits


def load_balancing_loss_func(
    router_logits: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
    num_experts: int,
    top_k: int,
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute the Switch Transformer load-balancing aux loss.

    Ported from ``transformers.models.qwen3_moe.modeling_qwen3_moe``.

    Args:
        router_logits: Tuple of per-layer router logit tensors, each of shape
            ``(batch * seq, num_experts)``.
        num_experts: Total number of experts.
        top_k: Number of experts selected per token.
        attention_mask: Optional ``(batch, seq)`` mask.  Padding positions
            are excluded from the balancing computation.

    Returns:
        Scalar loss tensor encouraging uniform expert utilisation.
    """
    if router_logits is None or not isinstance(router_logits, tuple) or not router_logits:
        return torch.tensor(0.0)

    compute_device = router_logits[0].device
    concatenated_logits = torch.cat(
        [layer_gate.to(compute_device) for layer_gate in router_logits],
        dim=0,
    )
    routing_weights = F.softmax(concatenated_logits, dim=-1)
    _, selected_experts = torch.topk(routing_weights, top_k, dim=-1)
    expert_mask = F.one_hot(selected_experts, num_classes=num_experts)

    if attention_mask is None:
        tokens_per_expert = torch.mean(expert_mask.float(), dim=0)
        router_prob_per_expert = torch.mean(routing_weights, dim=0)
    else:
        batch_size, sequence_length = attention_mask.shape
        num_hidden_layers = concatenated_logits.shape[0] // (batch_size * sequence_length)
        expert_attention_mask = (
            attention_mask[None, :, :, None, None]
            .expand((num_hidden_layers, batch_size, sequence_length, top_k, num_experts))
            .reshape(-1, top_k, num_experts)
            .to(compute_device)
        )
        tokens_per_expert = torch.sum(
            expert_mask.float() * expert_attention_mask,
            dim=0,
        ) / torch.sum(expert_attention_mask, dim=0)
        router_per_expert_attention_mask = (
            attention_mask[None, :, :, None]
            .expand((num_hidden_layers, batch_size, sequence_length, num_experts))
            .reshape(-1, num_experts)
            .to(compute_device)
        )
        router_prob_per_expert = torch.sum(
            routing_weights * router_per_expert_attention_mask,
            dim=0,
        ) / torch.sum(router_per_expert_attention_mask, dim=0)

    overall_loss = torch.sum(tokens_per_expert * router_prob_per_expert.unsqueeze(0))
    return overall_loss * num_experts
