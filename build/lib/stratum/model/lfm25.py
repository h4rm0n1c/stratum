"""LFM2.5 model architecture adapter for Stratum.

Includes Lfm25VoltaAttention for V100 flash-attention support.
"""

from __future__ import annotations

from typing import Any, Callable, Optional, Union, cast

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from transformers.models.lfm2_moe.modeling_lfm2_moe import (
    Lfm2MoeAttention,
    apply_rotary_pos_emb,
)
from transformers.models.llama.modeling_llama import repeat_kv
from transformers.modeling_outputs import CausalLMOutputWithPast

from stratum.model.registry import ModelArch, register


# ---------------------------------------------------------------------------
# Flash-attention-v100 for LFM2.5 (head_dim=64, natively supported)
# ---------------------------------------------------------------------------

class Lfm25VoltaAttention(Lfm2MoeAttention):
    """LFM2.5 MoE attention using flash-attn-v100 for dense causal attention."""

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
        past_key_values: Any = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, None]:
        from flash_attn_v100 import flash_attn_func as flash_attn_v100_func

        if kwargs.get("output_attentions", False):
            raise ValueError("Lfm25VoltaAttention does not return attention weights")

        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_layernorm(
            self.q_proj(hidden_states).view(*hidden_shape)
        ).transpose(1, 2)
        key_states = self.k_layernorm(
            self.k_proj(hidden_states).view(*hidden_shape)
        ).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(*hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # GQA
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # (B, H, M, D) -> (B, M, H, D) for flash_attn_v100
        q = query_states.transpose(1, 2).contiguous()
        k = key_states.transpose(1, 2).contiguous()
        v = value_states.transpose(1, 2).contiguous()

        attn_output = flash_attn_v100_func(
            q, k, v, dropout_p=0.0, softmax_scale=self.scaling, causal=True,
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        output = self.out_proj(attn_output)
        return output, None


# ---------------------------------------------------------------------------
# RoundPipe-compatible wrappers (same interface as before, no RoundPipe dep)
# ---------------------------------------------------------------------------

class LFM25ForCausalLMPrefix(nn.Module):
    """Prefix: embedding + token embedding norm + position embedding."""

    def __init__(self, model, *, memory_telemetry: bool = False):
        super().__init__()
        core = model.get_base_model() if hasattr(model, "get_base_model") else model
        self.embed_tokens = core.model.embed_tokens
        self.token_embd_norm = getattr(core.model, "token_embd_norm", None)
        self.pos_emb = getattr(core.model, "pos_emb", None) or getattr(
            core.model, "rotary_emb", None
        )
        self.config = core.model.config
        self.has_sliding_layers = getattr(core.model, "has_sliding_layers", False)
        self.memory_telemetry = memory_telemetry

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[Any] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.Tensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Any,
    ):
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = cast(torch.Tensor, self.embed_tokens(input_ids))
            if self.token_embd_norm is not None:
                inputs_embeds = self.token_embd_norm(inputs_embeds)

        use_cache = False
        past_key_values = None
        if cache_position is None:
            cache_position = torch.arange(
                0, inputs_embeds.shape[1], device=inputs_embeds.device
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        # Build causal mask
        from transformers.masking_utils import create_causal_mask

        mask_kwargs = {
            "config": self.config,
            "inputs_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "cache_position": cache_position,
            "past_key_values": past_key_values,
            "position_ids": position_ids,
        }
        causal_mask = {
            "full_attention": create_causal_mask(**mask_kwargs),
            "linear_attention": attention_mask,
        }
        if self.has_sliding_layers:
            from transformers.masking_utils import create_sliding_window_causal_mask

            causal_mask["sliding_attention"] = create_sliding_window_causal_mask(
                **mask_kwargs
            )

        hidden_states = inputs_embeds
        position_embeddings = self.pos_emb(hidden_states, position_ids=position_ids)

        return (
            hidden_states,
            causal_mask,
            position_ids,
            position_embeddings,
            kwargs,
            labels,
            logits_to_keep,
        )


class LFM25ForCausalLMWrappedLayer(nn.Module):
    """One decoder layer wrapper with optional checkpointing."""

    def __init__(
        self,
        layer: nn.Module,
        *,
        idx: int,
        checkpoint_decoder_layer: bool = False,
        use_flash_attention: bool = False,
    ):
        super().__init__()
        self.layer = layer
        self.idx = idx
        self.checkpoint_decoder_layer = checkpoint_decoder_layer
        self.use_flash_attention = use_flash_attention

    def forward(self, input_data):
        (
            hidden_states,
            causal_mask_mapping,
            position_ids,
            position_embeddings,
            kwargs,
            labels,
            logits_to_keep,
        ) = input_data

        layer_kwargs = dict(kwargs)
        layer_kwargs.pop("return_logits", None)

        # Select attention mask based on layer type
        attention_type = getattr(self.layer, "attention_type", "full_attention")
        if attention_type in causal_mask_mapping:
            attn_mask = causal_mask_mapping[attention_type]
        elif attention_type == "full_attention":
            attn_mask = causal_mask_mapping.get("full_attention")
        else:
            attn_mask = None

        def run_layer(hs):
            return self.layer(
                hs,
                attention_mask=attn_mask,
                position_ids=position_ids,
                past_key_value=None,
                use_cache=False,
                cache_position=None,
                position_embeddings=position_embeddings,
                **layer_kwargs,
            )

        if self.checkpoint_decoder_layer and self.training:
            hidden_states = checkpoint(run_layer, hidden_states, use_reentrant=False)
        else:
            hidden_states = run_layer(hidden_states)

        if isinstance(hidden_states, tuple):
            hidden_states = hidden_states[0]

        return (
            hidden_states,
            causal_mask_mapping,
            position_ids,
            position_embeddings,
            kwargs,
            labels,
            logits_to_keep,
        )


class LFM25ForCausalLMPostfix(nn.Module):
    """Postfix: final norm + lm_head."""

    def __init__(self, model, *, debug_finite: bool = False):
        super().__init__()
        self.debug_finite = debug_finite
        core = model.get_base_model() if hasattr(model, "get_base_model") else model
        self.norm = getattr(core.model, "output_norm", None) or getattr(
            core.model, "norm", None
        )
        self.vocab_size = getattr(core.config, "vocab_size", None)
        if self.vocab_size is None and hasattr(core.config, "text_config"):
            self.vocab_size = core.config.text_config.vocab_size
        self.lm_head = getattr(core, "output", None) or getattr(core, "lm_head", None)

    def forward(self, input_data):
        (
            hidden_states,
            causal_mask_mapping,
            position_ids,
            position_embeddings,
            kwargs,
            labels,
            logits_to_keep,
        ) = input_data

        if self.norm is not None:
            hidden_states = self.norm(hidden_states)

        # Compute loss if labels provided
        loss = None
        if labels is not None and not kwargs.get("return_logits", False):
            shift_hidden = hidden_states[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            batch, seq_len, _ = shift_hidden.shape

            logits = self.lm_head(shift_hidden)
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                logits.view(-1, self.vocab_size),
                shift_labels.view(-1),
            )

        return CausalLMOutputWithPast(loss=loss, logits=None)


# ---------------------------------------------------------------------------
# ModelArch registration
# ---------------------------------------------------------------------------

@register("lfm25-8b-a1b")
class LFM25Arch(ModelArch):
    """LFM2.5-8B-A1B architecture adapter."""

    def get_config(self, model):
        return model.config

    def get_num_layers(self, config):
        return config.num_hidden_layers

    def build_prefix(self, model):
        return LFM25ForCausalLMPrefix(model)

    def build_wrapped_layer(self, layer, idx):
        return LFM25ForCausalLMWrappedLayer(
            layer, idx=idx,
            use_flash_attention=True,
        )

    def build_postfix(self, model):
        return LFM25ForCausalLMPostfix(model)

    def build(self, hf_model, tensor_split=None, device_ids=None, **kwargs):
        # Patch full-attention layers with Volta flash attention before building pipeline
        core = hf_model.get_base_model() if hasattr(hf_model, "get_base_model") else hf_model
        _patch_lfm25_attention(core)
        return super().build(hf_model, tensor_split, device_ids, **kwargs)


def _patch_lfm25_attention(model):
    """Replace Lfm2MoeAttention with Lfm25VoltaAttention on full-attention layers."""
    core = model.get_base_model() if hasattr(model, "get_base_model") else model
    config = core.config
    patched = 0
    for idx, layer in enumerate(core.model.layers):
        layer_types = getattr(config, "layer_types", None)
        is_attn = (
            layer_types[idx] == "full_attention"
            if layer_types and idx < len(layer_types)
            else hasattr(layer, "self_attn")
        )
        if not is_attn:
            continue
        if not hasattr(layer, "self_attn"):
            continue

        old_attn = layer.self_attn
        if isinstance(old_attn, Lfm25VoltaAttention):
            patched += 1
            continue

        new_attn = Lfm25VoltaAttention(old_attn.config, layer_idx=idx)
        # Preserve existing projection modules (keeps LoRA adapters intact)
        for attr in ["q_proj", "k_proj", "v_proj", "out_proj",
                     "q_layernorm", "k_layernorm"]:
            if hasattr(old_attn, attr):
                setattr(new_attn, attr, getattr(old_attn, attr))
        layer.self_attn = new_attn
        patched += 1

    if patched:
        print(f"Patched {patched} LFM2.5 attention layers with Volta flash attention",
              flush=True)
