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


# causal_conv1d is compiled for sm_70 + sm_86 — fast path works on both GPUs.
# is_fast_path_available is checked at runtime by Lfm2MoeShortConv.
import transformers.models.lfm2_moe.modeling_lfm2_moe as _lfm2_mod
_lfm2_mod.is_fast_path_available = True


# ---------------------------------------------------------------------------
# Flash-attention-v100 for LFM2.5 (head_dim=64, natively supported)
# ---------------------------------------------------------------------------

class Lfm25VoltaAttention(Lfm2MoeAttention):
    """LFM2.5 MoE attention using flash-attention.

    Dispatches to the right backend based on GPU architecture:
    - sm_70 (V100): flash_attn_v100
    - sm_75+ (Ampere): standard flash_attn
    - Fallback: eager (any GPU)
    """

    def _select_flash_fn(self, device: torch.device) -> callable:
        """Pick the best flash-attention variant for the GPU."""
        try:
            sm = torch.cuda.get_device_capability(device)
        except RuntimeError:
            return None
        if sm[0] == 7 and sm[1] == 0:
            try:
                from flash_attn_v100 import flash_attn_func as fn
                return fn
            except ImportError:
                pass
        # Standard flash-attn for Ampere+; eager fallback if unavailable
        try:
            from flash_attn import flash_attn_func as fn
            return fn
        except ImportError:
            return None

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
        past_key_values: Any = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, None]:
        flash_fn = self._select_flash_fn(hidden_states.device)

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

        if flash_fn is not None:
            # Flash path with GQA
            k_fa = repeat_kv(key_states, self.num_key_value_groups)
            v_fa = repeat_kv(value_states, self.num_key_value_groups)
            q = query_states.transpose(1, 2).contiguous()
            k = k_fa.transpose(1, 2).contiguous()
            v = v_fa.transpose(1, 2).contiguous()
            try:
                attn_output = flash_fn(
                    q, k, v, dropout_p=0.0, softmax_scale=self.scaling, causal=True,
                )
                attn_output = attn_output.transpose(1, 2).contiguous()
            except RuntimeError:
                flash_fn = None

        if flash_fn is None:
            # Eager fallback — pre-GQA tensors, GQA handled internally
            from transformers.models.llama.modeling_llama import eager_attention_forward
            attn_output, _ = eager_attention_forward(
                self, query_states, key_states, value_states,
                attention_mask, dropout=0.0, scaling=self.scaling,
            )

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
        import copy
        self.embed_tokens = copy.deepcopy(core.model.embed_tokens)
        self.token_embd_norm = copy.deepcopy(getattr(core.model, "token_embd_norm", None))
        self.pos_emb = copy.deepcopy(
            getattr(core.model, "pos_emb", None) or getattr(core.model, "rotary_emb", None)
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

        # Causal mask not needed — transformers 5.x handles causality
        # internally via is_causal. Volta flash attention also uses
        # causal=True. Pass None to avoid format mismatches.
        causal_mask = {
            "full_attention": None,
            "linear_attention": attention_mask,
        }

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
        import copy
        self.norm = copy.deepcopy(
            getattr(core.model, "output_norm", None) or getattr(core.model, "norm", None)
        )
        self.vocab_size = getattr(core.config, "vocab_size", None)
        if self.vocab_size is None and hasattr(core.config, "text_config"):
            self.vocab_size = core.config.text_config.vocab_size
        self.lm_head = copy.deepcopy(
            getattr(core, "output", None) or getattr(core, "lm_head", None)
        )

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

        loss = None
        if labels is not None and not kwargs.get("return_logits", False):
            shift_hidden = hidden_states[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            # Count non-ignored tokens for normalization (same as
            # RoundPipe's ChunkedCompileLinearForCausalLMLoss).
            flat_labels = shift_labels.reshape(-1)
            num_items = (flat_labels != -100).sum()
            if num_items == 0:
                return CausalLMOutputWithPast(
                    loss=shift_hidden.new_zeros(()), logits=None
                )

            # Chunked loss: split into 4096-token chunks for lm_head
            # to avoid OOM from full [seq_len, 124893] logits matrix.
            # Uses reduction="sum" per chunk, normalizes by num_items
            # (same pattern as RoundPipe's BlockedPostfixCausalLMLoss).
            chunk_size = 4096
            seq_len = shift_hidden.shape[1]
            loss_sum = shift_hidden.new_zeros(())
            for start in range(0, seq_len, chunk_size):
                end = min(start + chunk_size, seq_len)
                chunk_h = shift_hidden[:, start:end, :].contiguous()
                chunk_l = shift_labels[:, start:end].contiguous()
                logits = self.lm_head(chunk_h)
                cl = nn.functional.cross_entropy(
                    logits.reshape(-1, self.vocab_size),
                    chunk_l.reshape(-1),
                    ignore_index=-100,
                    reduction="sum",
                )
                loss_sum = loss_sum + cl
            loss = loss_sum / num_items

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

    def build_wrapped_layer(self, layer, idx, **kwargs):
        return LFM25ForCausalLMWrappedLayer(
            layer, idx=idx,
            use_flash_attention=True,
            checkpoint_decoder_layer=kwargs.get("checkpoint_decoder_layer", False),
        )

    def build_postfix(self, model):
        return LFM25ForCausalLMPostfix(model)

    def build(self, hf_model, tensor_split=None, device_ids=None, **kwargs):
        # Replace ALL Lfm2MoeAttention modules with Lfm25VoltaAttention.
        # The VoltaAttention falls back to eager on non-Volta GPUs at
        # forward time, so no SM-capability check needed here.
        core = hf_model.get_base_model() if hasattr(hf_model, "get_base_model") else hf_model
        _patch_lfm25_attention(core)
        return super().build(hf_model, tensor_split, device_ids, **kwargs)


def _patch_lfm25_attention(model):
    """Replace all Lfm2MoeAttention with Lfm25VoltaAttention."""
    core = model.get_base_model() if hasattr(model, "get_base_model") else model
    config = core.config
    patched = 0
    for idx, layer in enumerate(core.model.layers):
        if not hasattr(layer, "self_attn"):
            continue
        old_attn = layer.self_attn
        if isinstance(old_attn, Lfm25VoltaAttention):
            patched += 1
            continue

        new_attn = Lfm25VoltaAttention(old_attn.config, layer_idx=idx)
        for attr in ["q_proj", "k_proj", "v_proj", "out_proj",
                     "q_layernorm", "k_layernorm"]:
            if hasattr(old_attn, attr):
                setattr(new_attn, attr, getattr(old_attn, attr))
        layer.self_attn = new_attn
        patched += 1

    if patched:
        print(f"Patched {patched} LFM2.5 attention layers with Volta flash attention",
              flush=True)
