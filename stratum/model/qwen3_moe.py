"""Qwen3-MoE model architecture adapter for Stratum.

Includes capability-dispatched flash attention for mixed RTX 3080/V100 runs,
and MoE router-logit side-channel capture for auxiliary load-balancing loss.

Qwen3MoeAttention has per-head q_norm/k_norm like Qwen3Attention.
Router logits are captured via patch_moe_block_for_router_logits (side-channel,
no change to the MoE block return value) and accumulated in kwargs["_router_logits"].
"""

from __future__ import annotations

import inspect
from typing import Any, Callable, NamedTuple, Optional

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from transformers.models.qwen3_moe.modeling_qwen3_moe import (
    Qwen3MoeAttention,
    apply_rotary_pos_emb,
)
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
from transformers.modeling_outputs import CausalLMOutputWithPast

from stratum.model.registry import ModelArch, register
from stratum.model.mlp_opt import apply_mlp_optimizations
from stratum.model.blocked_loss import BlockedPostfixCausalLMLoss
from stratum.model.chunked_loss import chunked_linear_cross_entropy
from stratum.telemetry import assert_finite_tensor, mark_model_gpu_phase
from stratum.context import (
    checkpoint_context_fn,
    doing_recompute,
    get_recompute_data,
    save_for_recompute,
)
from stratum.moe import (
    load_balancing_loss_func,
    patch_moe_block_for_router_logits,
    pop_router_logits,
)


def _compat_mask_call(fn: Callable[..., Any], kwargs: dict[str, Any]) -> Any:
    sig = inspect.signature(fn)
    return fn(**{k: v for k, v in kwargs.items() if k in sig.parameters})


class _FlashBackend(NamedTuple):
    name: str
    fn: Callable[..., torch.Tensor]


class Qwen3MoeFlashAttention(Qwen3MoeAttention):
    """Qwen3-MoE attention using capability-dispatched flash-attention.

    Dispatches to the right backend based on GPU architecture:
    - sm_70 (V100): flash_attn_v100
    - sm_80+ (Ampere+): standard flash_attn

    Qwen3MoeAttention has q_norm/k_norm (same pattern as Qwen3Attention) and
    no gate projection. CUDA execution requires one of the flash backends;
    eager is CPU/test-only.
    """

    def _select_flash_backend(self, device: torch.device) -> _FlashBackend | None:
        """Pick the non-quadratic attention backend for the current GPU."""
        if device.type != "cuda":
            return None
        try:
            sm = torch.cuda.get_device_capability(device)
        except RuntimeError:
            return None

        if sm[0] == 7 and sm[1] == 0:
            try:
                from flash_attn_v100 import flash_attn_func as fn
                return _FlashBackend("flash_attn_v100", fn)
            except ImportError:
                return None

        if sm[0] >= 8:
            try:
                from flash_attn import flash_attn_func as fn
                return _FlashBackend("flash_attn", fn)
            except ImportError:
                return None

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
            raise ValueError("Qwen3MoeFlashAttention does not return attention weights")
        if self.training and getattr(self, "attention_dropout", 0.0):
            raise ValueError("flash-attn requires dropout=0.0")

        flash_backend = self._select_flash_backend(hidden_states.device)

        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        # Qwen3-MoE: q_norm/k_norm applied after projection, no gate.
        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if flash_backend is not None:
            if not getattr(self, "_stratum_flash_backend_logged", False):
                print({
                    "event": "flash_attention_backend",
                    "model": "qwen3-moe",
                    "layer": int(self.layer_idx),
                    "backend": flash_backend.name,
                    "device": str(hidden_states.device),
                }, flush=True)
                self._stratum_flash_backend_logged = True
            q = query_states.transpose(1, 2).contiguous()
            k = key_states.transpose(1, 2).contiguous()
            v = value_states.transpose(1, 2).contiguous()
            try:
                attn_output = flash_backend.fn(
                    q, k, v, dropout_p=0.0, softmax_scale=self.scaling, causal=True,
                )
            except RuntimeError as exc:
                raise RuntimeError(
                    f"{flash_backend.name} failed for Qwen3-MoE layer {self.layer_idx}; "
                    "not falling back to quadratic eager attention"
                ) from exc

        if flash_backend is None:
            if hidden_states.device.type == "cuda":
                raise RuntimeError(
                    f"no flash-attention backend available for Qwen3-MoE layer {self.layer_idx} "
                    f"on {hidden_states.device}; not falling back to quadratic eager attention"
                )
            from transformers.models.llama.modeling_llama import eager_attention_forward
            attn_output, _ = eager_attention_forward(
                self, query_states, key_states, value_states,
                attention_mask, dropout=0.0, scaling=self.scaling,
            )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, None


class Qwen3MoeForCausalLMPrefix(nn.Module):
    """Prefix: embedding + rotary embedding.

    Qwen3-MoE uses a single causal mask (not a dict), matching the reference
    qwen3_moe.py. The mask function is selected based on config.sliding_window:
    None → create_causal_mask; non-None → create_sliding_window_causal_mask.
    In flash mode (default) the mask is None regardless.
    """

    def __init__(
        self,
        model,
        *,
        dense_attention_masks: bool = False,
        memory_telemetry: bool = False,
        output_router_logits: bool = False,
    ):
        super().__init__()
        core = model.get_base_model() if hasattr(model, "get_base_model") else model
        self.embed_tokens = core.model.embed_tokens
        self.rotary_emb = core.model.rotary_emb
        self.config = core.model.config
        self.sliding_window = getattr(self.config, "sliding_window", None)
        self.dense_attention_masks = dense_attention_masks
        self.memory_telemetry = memory_telemetry
        self.output_router_logits = output_router_logits

    def forward(self, input_ids=None, attention_mask=None, position_ids=None,
                past_key_values=None, labels=None, **kwargs):
        if self.memory_telemetry:
            mark_model_gpu_phase("prefix_enter")

        inputs_embeds = kwargs.get("inputs_embeds")
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Specify exactly one of input_ids or inputs_embeds")

        # On checkpoint backward recompute, restore saved non-grad tensors.
        if doing_recompute():
            causal_mask, position_ids, position_embeddings = get_recompute_data()
            if self.memory_telemetry:
                mark_model_gpu_phase("prefix_recompute_loaded")
            return (
                inputs_embeds if inputs_embeds is not None else self.embed_tokens(input_ids),
                causal_mask,
                position_ids,
                position_embeddings,
                kwargs,
                labels,
                0,
            )

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        seq_len = inputs_embeds.shape[1]
        ref_device = inputs_embeds.device
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=ref_device).unsqueeze(0)

        if self.dense_attention_masks:
            cache_position = torch.arange(seq_len, device=ref_device)
            mask_kwargs = {
                "config": self.config,
                "inputs_embeds": inputs_embeds,
                "input_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }
            mask_fn = (
                create_sliding_window_causal_mask
                if self.sliding_window is not None
                else create_causal_mask
            )
            causal_mask = _compat_mask_call(mask_fn, mask_kwargs)
        else:
            # Long-context flash mode: kernel handles causality natively.
            causal_mask = None

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        if self.memory_telemetry:
            mark_model_gpu_phase("prefix_after_rope", seq_len=int(hidden_states.shape[1]))

        save_for_recompute(causal_mask, position_ids, position_embeddings)

        # Initialise the router_logits accumulator (shared mutable dict flowing
        # through the pipeline). The wrapped layer appends to this list after
        # pop_router_logits() collects from patched MoE blocks.
        if self.output_router_logits and "_router_logits" not in kwargs:
            kwargs["_router_logits"] = []

        return (
            hidden_states, causal_mask, position_ids, position_embeddings,
            kwargs, labels, 0,
        )


class Qwen3MoeForCausalLMWrappedLayer(nn.Module):
    """One Qwen3-MoE decoder layer wrapper with optional activation checkpointing.

    Collects router logits from patched Qwen3MoeSparseMoeBlock instances via
    the pop_router_logits() side channel and appends them to kwargs["_router_logits"].
    """

    def __init__(
        self,
        layer,
        *,
        idx: int,
        checkpoint_decoder_layer: bool = False,
        use_flash_attention: bool = False,
        memory_telemetry: bool = False,
        debug_finite: bool = False,
    ):
        super().__init__()
        self.layer = layer
        self.idx = idx
        self.checkpoint_decoder_layer = checkpoint_decoder_layer
        self.use_flash_attention = use_flash_attention
        self.memory_telemetry = memory_telemetry
        self.debug_finite = debug_finite

    def forward(self, input_data):
        hidden, causal_mask, pos_ids, pos_embeds, kwargs, labels, _lk = input_data

        layer_kwargs = dict(kwargs)
        layer_kwargs.pop("return_logits", None)
        layer_kwargs.pop("_router_logits", None)

        def run_layer(hs):
            if doing_recompute():
                (
                    recompute_attn_mask,
                    recompute_pos_ids,
                    recompute_pos_embeds,
                    recompute_layer_kwargs,
                ) = get_recompute_data()
            else:
                recompute_attn_mask = causal_mask
                recompute_pos_ids = pos_ids
                recompute_pos_embeds = pos_embeds
                recompute_layer_kwargs = layer_kwargs
                save_for_recompute(causal_mask, pos_ids, pos_embeds, layer_kwargs)
            return self.layer(
                hs,
                attention_mask=recompute_attn_mask,
                position_ids=recompute_pos_ids,
                past_key_values=None,
                use_cache=False,
                cache_position=None,
                position_embeddings=recompute_pos_embeds,
                **recompute_layer_kwargs,
            )

        if self.checkpoint_decoder_layer and self.training:
            checkpoint_fields = {
                "layer_idx": int(self.idx),
                "stage_device": (
                    int(hidden.device.index)
                    if hidden.is_cuda and hidden.device.index is not None
                    else None
                ),
                "attention_type": "full_attention",
                "recompute_grain": "layer",
            }
            hidden = checkpoint(
                run_layer,
                hidden,
                use_reentrant=False,
                context_fn=lambda: checkpoint_context_fn(**checkpoint_fields),
            )
        else:
            hidden = run_layer(hidden)

        # HF layer returns tensor or tuple. MoE blocks are patched to record
        # router logits as side-channel state, so the return value is still
        # just the hidden-states tensor (not a tuple with router_logits).
        if isinstance(hidden, tuple):
            if (
                len(hidden) >= 2
                and isinstance(hidden[1], torch.Tensor)
                and hidden[1].dim() == 2
                and kwargs.get("_router_logits") is not None
            ):
                kwargs["_router_logits"].append(hidden[1])
                hidden = hidden[0]
            else:
                hidden = hidden[0]

        # Collect any router logits captured by patched MoE blocks.
        if kwargs.get("_router_logits") is not None:
            kwargs["_router_logits"].extend(pop_router_logits(self.layer))

        if self.debug_finite:
            assert_finite_tensor(f"layer_{self.idx}_output", hidden)

        return (hidden, causal_mask, pos_ids, pos_embeds, kwargs, labels, _lk)


class Qwen3MoeForCausalLMPostfix(nn.Module):
    """Postfix: final norm + lm_head.

    Two loss modes (matching RoundPipe):
      1. postfix_loss_token_chunk_size == 0 (default):
         norm runs full-seq, lm_head chunked by loss_token_chunk_size.
      2. postfix_loss_token_chunk_size > 0:
         BlockedPostfixCausalLMLoss — splits norm + lm_head into blocks.

    When ``router_aux_loss_coef > 0``, adds MoE load-balancing loss from
    accumulated router_logits to the main LM loss, mirroring LFM2.5.
    """

    def __init__(self, model, *, loss_token_chunk_size: int = 4096,
                 postfix_loss_token_chunk_size: int = 0,
                 memory_telemetry: bool = False,
                 debug_finite: bool = False,
                 torch_compile_loss: bool = False,
                 router_aux_loss_coef: float = 0.0):
        super().__init__()
        self.loss_token_chunk_size = loss_token_chunk_size
        self.postfix_loss_token_chunk_size = postfix_loss_token_chunk_size
        self.memory_telemetry = memory_telemetry
        self.debug_finite = debug_finite
        self.torch_compile_loss = torch_compile_loss
        self.router_aux_loss_coef = router_aux_loss_coef
        core = model.get_base_model() if hasattr(model, "get_base_model") else model
        self.norm = core.model.norm
        self.lm_head = core.lm_head
        self.vocab_size = core.config.vocab_size
        self.num_experts = getattr(core.config, "num_experts", 8)
        self.num_experts_per_tok = getattr(core.config, "num_experts_per_tok", 2)

    def forward(self, input_data):
        hidden, causal_mask, pos_ids, pos_embeds, kwargs, labels, _lk = input_data

        loss = None
        if labels is not None:
            if self.postfix_loss_token_chunk_size > 0:
                loss = BlockedPostfixCausalLMLoss.apply(
                    hidden, labels,
                    self.norm, self.lm_head, self.vocab_size,
                    self.postfix_loss_token_chunk_size,
                    -100, self.memory_telemetry, self.debug_finite,
                )
                if self.debug_finite:
                    assert_finite_tensor("blocked_postfix_loss", loss)
            else:
                hidden = self.norm(hidden)
                if self.debug_finite:
                    assert_finite_tensor("post_norm_hidden_states", hidden)
                shift_hidden = hidden[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                flat_labels = shift_labels.reshape(-1)
                num_items = (flat_labels != -100).sum()
                if num_items == 0:
                    loss = shift_hidden.sum() * 0.0
                else:
                    loss = chunked_linear_cross_entropy(
                        shift_hidden, self.lm_head, shift_labels,
                        num_items=num_items, ignore_index=-100,
                        token_chunk_size=self.loss_token_chunk_size,
                        use_torch_compile=self.torch_compile_loss,
                    )
                    if self.debug_finite:
                        assert_finite_tensor("chunked_loss", loss)

        # MoE auxiliary load-balancing loss from accumulated router_logits.
        if self.router_aux_loss_coef > 0:
            router_logits = kwargs.get("_router_logits")
            if router_logits:
                aux_loss = load_balancing_loss_func(
                    tuple(router_logits),
                    num_experts=self.num_experts,
                    top_k=self.num_experts_per_tok,
                )
                if loss is not None:
                    loss = loss + self.router_aux_loss_coef * aux_loss.to(loss.device)
                elif aux_loss.requires_grad:
                    loss = self.router_aux_loss_coef * aux_loss

        return CausalLMOutputWithPast(loss=loss)


@register("qwen3-moe")
class Qwen3MoeArch(ModelArch):
    def get_num_layers(self, config):
        return config.num_hidden_layers

    def build_prefix(self, model, **kwargs):
        return Qwen3MoeForCausalLMPrefix(
            model,
            dense_attention_masks=kwargs.get("dense_attention_masks", False),
            memory_telemetry=kwargs.get("memory_telemetry", False),
            output_router_logits=kwargs.get("output_router_logits", False),
        )

    def build_wrapped_layer(self, layer, idx, **kwargs):
        return Qwen3MoeForCausalLMWrappedLayer(
            layer,
            idx=idx,
            checkpoint_decoder_layer=kwargs.get("checkpoint_decoder_layer", False),
            use_flash_attention=True,
            memory_telemetry=kwargs.get("memory_telemetry", False),
            debug_finite=kwargs.get("debug_finite", False),
        )

    def build_postfix(self, model, **kwargs):
        return Qwen3MoeForCausalLMPostfix(
            model,
            loss_token_chunk_size=kwargs.get("loss_token_chunk_size", 4096),
            postfix_loss_token_chunk_size=kwargs.get("postfix_loss_token_chunk_size", 0),
            memory_telemetry=kwargs.get("memory_telemetry", False),
            debug_finite=kwargs.get("debug_finite", False),
            torch_compile_loss=kwargs.get("torch_compile_loss", False),
            router_aux_loss_coef=kwargs.get("router_aux_loss_coef", 0.0),
        )

    def build(self, hf_model, tensor_split=None, device_ids=None, **kwargs):
        core = hf_model.get_base_model() if hasattr(hf_model, "get_base_model") else hf_model
        from stratum.telemetry import parse_int_set
        flash_layers_str = kwargs.get("flash_layers", "") or kwargs.get("volta_layers", "")
        if flash_layers_str.strip().lower() in {"none", "off", "false"}:
            raise ValueError("--flash-layers cannot disable flash attention")
        flash_layer_indices = parse_int_set(flash_layers_str) if flash_layers_str else None
        _patch_qwen3moe_attention(core, layer_indices=flash_layer_indices)
        apply_mlp_optimizations(
            core,
            checkpoint_mlp=kwargs.get("checkpoint_mlp", False),
            memory_flat_frozen_mlp=kwargs.get("memory_flat_frozen_mlp", False),
            mlp_token_chunk_size=kwargs.get("mlp_token_chunk_size", 0),
        )
        # MoE router logit capture: patch blocks to expose router logits
        # as side-channel state without changing block return values.
        output_router_logits = kwargs.get("output_router_logits", False)
        if output_router_logits:
            n_patched = patch_moe_block_for_router_logits(core)
            if n_patched > 0:
                print(f"MoE router logit capture: patched {n_patched} MoE blocks", flush=True)
            else:
                print("MoE router logit capture: no MoE blocks found to patch", flush=True)
        return super().build(hf_model, tensor_split, device_ids, **kwargs)


def _patch_qwen3moe_attention(
    model,
    layer_indices: Optional[set[int]] = None,
):
    """Replace Qwen3MoeAttention with capability-dispatched flash attention.

    Args:
        model: HF model to patch.
        layer_indices: Set of layer indices to patch. None = patch all.
    """
    core = model.get_base_model() if hasattr(model, "get_base_model") else model
    patched = 0
    for idx, layer in enumerate(core.model.layers):
        if layer_indices is not None and idx not in layer_indices:
            continue
        if not hasattr(layer, "self_attn"):
            continue
        old_attn = layer.self_attn
        if not isinstance(old_attn, Qwen3MoeAttention):
            continue
        if isinstance(old_attn, Qwen3MoeFlashAttention):
            if layer_indices is None or idx in layer_indices:
                patched += 1
            continue

        new_attn = Qwen3MoeFlashAttention(old_attn.config, layer_idx=idx)
        for attr in ["q_proj", "k_proj", "v_proj", "o_proj", "q_norm", "k_norm"]:
            if hasattr(old_attn, attr):
                setattr(new_attn, attr, getattr(old_attn, attr))
        new_attn.training = old_attn.training
        layer.self_attn = new_attn
        patched += 1

    print(f"Patched {patched} Qwen3-MoE attention layers with capability-dispatched flash attention",
          flush=True)
