"""Qwen3.5 model architecture adapter for Stratum.

Includes Qwen35VoltaAttention for V100 flash-attention support.
"""

from __future__ import annotations

from typing import Any, Optional, Union, cast

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5Attention,
    apply_rotary_pos_emb,
)
from transformers.modeling_outputs import CausalLMOutputWithPast

from stratum.model.registry import ModelArch, register


class Qwen35VoltaAttention(Qwen3_5Attention):
    """Qwen3.5 attention using flash-attention.

    Dispatches to the right backend based on GPU architecture:
    - sm_70 (V100): flash_attn_v100
    - sm_75+ (Ampere): standard flash_attn
    - Fallback: eager (any GPU)
    """

    def _select_flash_fn(self, device: torch.device) -> callable:
        """Return flash_attn_v100 for Volta GPUs, None (eager) otherwise."""
        try:
            sm = torch.cuda.get_device_capability(device)
            if sm[0] == 7 and sm[1] == 0:
                from flash_attn_v100 import flash_attn_func as fn
                return fn
        except (RuntimeError, ImportError):
            pass
        return None

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
        past_key_values: Any = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, None]:
        if kwargs.get("output_attentions", False):
            raise ValueError("Qwen35VoltaAttention does not return attention weights")
        if self.training and self.attention_dropout:
            raise ValueError("flash-attn requires dropout=0.0")

        flash_fn = self._select_flash_fn(hidden_states.device)

        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states, gate = torch.chunk(
            self.q_proj(hidden_states).view(*input_shape, -1, self.head_dim * 2),
            2, dim=-1,
        )
        gate = gate.reshape(*input_shape, -1)

        query_states = self.q_norm(query_states.view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if flash_fn is not None:
            q = query_states.transpose(1, 2).contiguous()
            k = key_states.transpose(1, 2).contiguous()
            v = value_states.transpose(1, 2).contiguous()
            try:
                attn_output = flash_fn(
                    q, k, v, dropout_p=0.0, softmax_scale=self.scaling, causal=True,
                )
                attn_output = attn_output.transpose(1, 2).contiguous()
            except RuntimeError:
                flash_fn = None

        if flash_fn is None:
            from transformers.models.llama.modeling_llama import eager_attention_forward
            attn_output, _ = eager_attention_forward(
                self, query_states, key_states, value_states,
                attention_mask, dropout=0.0, scaling=self.scaling,
            )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = attn_output * torch.sigmoid(gate)
        attn_output = self.o_proj(attn_output)
        return attn_output, None


class Qwen35ForCausalLMPrefix(nn.Module):
    """Prefix: embedding + rotary embedding."""

    def __init__(self, model):
        super().__init__()
        core = model.get_base_model() if hasattr(model, "get_base_model") else model
        import copy
        self.embed_tokens = copy.deepcopy(core.model.embed_tokens)
        self.rotary_emb = copy.deepcopy(core.model.rotary_emb)
        self.config = core.config

    def forward(self, input_ids=None, attention_mask=None, position_ids=None,
                past_key_values=None, labels=None, **kwargs):
        inputs_embeds = self.embed_tokens(input_ids)
        batch, seq_len = input_ids.shape
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)

        causal_mask = {
            "full_attention": None,
            "linear_attention": attention_mask,
        }

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        return (
            hidden_states, causal_mask, position_ids, position_embeddings,
            kwargs, labels, 0,
        )


class Qwen35ForCausalLMWrappedLayer(nn.Module):
    """One decoder layer wrapper."""

    def __init__(self, layer, *, idx: int, use_flash_attention: bool = False):
        super().__init__()
        self.layer = layer
        self.idx = idx
        self.use_flash_attention = use_flash_attention

    def forward(self, input_data):
        hidden, causal_mask, pos_ids, pos_embeds, kwargs, labels, _lk = input_data
        attn_mask = causal_mask.get("full_attention", causal_mask.get("linear_attention"))
        hidden = self.layer(
            hidden, attention_mask=attn_mask, position_ids=pos_ids,
            past_key_value=None, use_cache=False, cache_position=None,
            position_embeddings=pos_embeds,
        )
        if isinstance(hidden, tuple):
            hidden = hidden[0]
        return (hidden, causal_mask, pos_ids, pos_embeds, kwargs, labels, 0)


class Qwen35ForCausalLMPostfix(nn.Module):
    """Postfix: final norm + lm_head."""

    def __init__(self, model):
        super().__init__()
        core = model.get_base_model() if hasattr(model, "get_base_model") else model
        import copy
        self.norm = copy.deepcopy(core.model.norm)
        self.lm_head = copy.deepcopy(core.lm_head)
        self.vocab_size = core.config.vocab_size

    def forward(self, input_data):
        hidden, causal_mask, pos_ids, pos_embeds, kwargs, labels, _lk = input_data
        hidden = self.norm(hidden)

        loss = None
        if labels is not None:
            shift_hidden = hidden[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            logits = self.lm_head(shift_hidden)
            loss = nn.CrossEntropyLoss()(
                logits.view(-1, self.vocab_size), shift_labels.view(-1),
            )

        return CausalLMOutputWithPast(loss=loss)


@register("qwen3.5")
class Qwen35Arch(ModelArch):
    def get_num_layers(self, config):
        return config.num_hidden_layers

    def build_prefix(self, model):
        return Qwen35ForCausalLMPrefix(model)

    def build_wrapped_layer(self, layer, idx):
        return Qwen35ForCausalLMWrappedLayer(layer, idx=idx, use_flash_attention=True)

    def build_postfix(self, model):
        return Qwen35ForCausalLMPostfix(model)

    def build(self, hf_model, tensor_split=None, device_ids=None, **kwargs):
        core = hf_model.get_base_model() if hasattr(hf_model, "get_base_model") else hf_model
        _patch_qwen35_attention(core)
        return super().build(hf_model, tensor_split, device_ids, **kwargs)


def _patch_qwen35_attention(model):
    """Replace Qwen3_5Attention with Qwen35VoltaAttention on all layers."""
    core = model.get_base_model() if hasattr(model, "get_base_model") else model
    patched = 0
    for idx, layer in enumerate(core.model.layers):
        if not hasattr(layer, "self_attn"):
            continue
        old_attn = layer.self_attn
        if not isinstance(old_attn, Qwen3_5Attention):
            continue
        if isinstance(old_attn, Qwen35VoltaAttention):
            patched += 1
            continue

        new_attn = Qwen35VoltaAttention(old_attn.config, layer_idx=idx)
        for attr in ["q_proj", "k_proj", "v_proj", "o_proj", "q_norm", "k_norm"]:
            if hasattr(old_attn, attr):
                setattr(new_attn, attr, getattr(old_attn, attr))
        new_attn.training = old_attn.training
        layer.self_attn = new_attn
        patched += 1

    print(f"Patched {patched} Qwen3.5 attention layers with Volta flash attention",
          flush=True)
