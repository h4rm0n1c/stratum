"""LFM2.5 model architecture adapter for Stratum.

Includes capability-dispatched flash attention for mixed RTX 3080/V100 runs.
"""

from __future__ import annotations

import inspect
from typing import Any, Callable, NamedTuple, Optional, Union, cast

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from transformers.models.lfm2_moe.modeling_lfm2_moe import (
    Lfm2MoeAttention,
    apply_rotary_pos_emb,
)
from transformers.models.llama.modeling_llama import repeat_kv
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
from stratum.moe import load_balancing_loss_func, patch_moe_block_for_router_logits, pop_router_logits


# causal_conv1d is compiled for sm_70 + sm_86 — fast path works on both GPUs.
# is_fast_path_available is checked at runtime by Lfm2MoeShortConv.
import transformers.models.lfm2_moe.modeling_lfm2_moe as _lfm2_mod
_lfm2_mod.is_fast_path_available = True


def _compat_mask_call(fn: Callable[..., Any], kwargs: dict[str, Any]) -> Any:
    sig = inspect.signature(fn)
    return fn(**{k: v for k, v in kwargs.items() if k in sig.parameters})


# ---------------------------------------------------------------------------
# Capability-dispatched flash attention for LFM2.5 (head_dim=64).
# ---------------------------------------------------------------------------

class _FlashBackend(NamedTuple):
    name: str
    fn: Callable[..., torch.Tensor]


class Lfm25FlashAttention(Lfm2MoeAttention):
    """LFM2.5 MoE attention using flash-attention.

    Dispatches to the right backend based on GPU architecture:
    - sm_70 (V100): flash_attn_v100
    - sm_80+ / sm_86 (Ampere+): flash_attn

    CPU/non-CUDA execution may use eager for local tests. CUDA execution must
    use a flash backend; silent quadratic fallback is not acceptable here.
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
        flash_backend = self._select_flash_backend(hidden_states.device)

        if kwargs.get("output_attentions", False):
            raise ValueError("Lfm25FlashAttention does not return attention weights")

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

        if flash_backend is not None:
            if not getattr(self, "_stratum_flash_backend_logged", False):
                print({
                    "event": "flash_attention_backend",
                    "model": "lfm25",
                    "layer": int(self.layer_idx),
                    "backend": flash_backend.name,
                    "device": str(hidden_states.device),
                }, flush=True)
                self._stratum_flash_backend_logged = True
            # Flash path with GQA
            k_fa = repeat_kv(key_states, self.num_key_value_groups)
            v_fa = repeat_kv(value_states, self.num_key_value_groups)
            q = query_states.transpose(1, 2).contiguous()
            k = k_fa.transpose(1, 2).contiguous()
            v = v_fa.transpose(1, 2).contiguous()
            try:
                attn_output = flash_backend.fn(
                    q, k, v, dropout_p=0.0, softmax_scale=self.scaling, causal=True,
                )
                attn_output = attn_output.transpose(1, 2).contiguous()
            except RuntimeError as exc:
                raise RuntimeError(
                    f"{flash_backend.name} failed for LFM2.5 layer {self.layer_idx}; "
                    "not falling back to quadratic eager attention"
                ) from exc

        if flash_backend is None:
            if hidden_states.device.type == "cuda":
                raise RuntimeError(
                    f"no flash-attention backend available for LFM2.5 layer {self.layer_idx} "
                    f"on {hidden_states.device}; not falling back to quadratic eager attention"
                )
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
        self.token_embd_norm = getattr(core.model, "token_embd_norm", None)
        self.pos_emb = getattr(core.model, "pos_emb", None) or getattr(core.model, "rotary_emb", None)
        self.config = core.model.config
        self.has_sliding_layers = getattr(core.model, "has_sliding_layers", False)
        self.dense_attention_masks = dense_attention_masks
        self.memory_telemetry = memory_telemetry
        self.output_router_logits = output_router_logits

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
        if self.memory_telemetry:
            mark_model_gpu_phase("prefix_enter")

        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Specify exactly one of input_ids or inputs_embeds")

        # On checkpoint backward recompute, restore saved non-grad tensors
        # and skip the expensive embedding lookup + mask construction.
        if doing_recompute():
            causal_mask_mapping, position_ids, position_embeddings = get_recompute_data()
            if self.memory_telemetry:
                mark_model_gpu_phase("prefix_recompute_loaded")
            return (
                inputs_embeds if inputs_embeds is not None else self.embed_tokens(input_ids),
                causal_mask_mapping,
                position_ids,
                position_embeddings,
                kwargs,
                labels,
                logits_to_keep,
            )

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

        if not isinstance(attention_mask, dict):
            mask_kwargs = {
                "config": self.config,
                "inputs_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }
            if self.dense_attention_masks:
                causal_mask_mapping = {
                    "full_attention": _compat_mask_call(create_causal_mask, mask_kwargs),
                    "linear_attention": attention_mask,
                }
                if self.has_sliding_layers:
                    causal_mask_mapping["sliding_attention"] = _compat_mask_call(
                        create_sliding_window_causal_mask,
                        mask_kwargs,
                    )
            else:
                # Long-context flash mode relies on kernel-native causal handling.
                causal_mask_mapping = {
                    "full_attention": None,
                    "linear_attention": attention_mask,
                }
        else:
            causal_mask_mapping = attention_mask

        hidden_states = inputs_embeds
        if self.pos_emb is None:
            raise RuntimeError("LFM2.5 model has no pos_emb or rotary_emb")
        position_embeddings = self.pos_emb(hidden_states, position_ids=position_ids)

        if self.memory_telemetry:
            mark_model_gpu_phase("prefix_after_rope", seq_len=int(hidden_states.shape[1]))

        # Save non-grad data for recompute (qz-roundpipe parity: saves
        # causal_mask_mapping, position_ids, position_embeddings).
        save_for_recompute(causal_mask_mapping, position_ids, position_embeddings)

        # Initialize the router_logits accumulator in kwargs (shared mutable
        # dict flows through the pipeline). The wrapped layer appends to this
        # list when its MoE block returns router_logits.
        if self.output_router_logits and "_router_logits" not in kwargs:
            kwargs["_router_logits"] = []

        return (
            hidden_states,
            causal_mask_mapping,
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
        (
            hidden_states,
            causal_mask_mapping,
            position_ids,
            position_embeddings,
            kwargs,
            labels,
            logits_to_keep,
        ) = input_data

        if self.memory_telemetry:
            mark_model_gpu_phase(
                "layer_enter",
                layer_idx=self.idx,
                seq_len=int(hidden_states.shape[1]),
                hidden_size=int(hidden_states.shape[2]),
                attention_type=getattr(self.layer, "attention_type", "full_attention"),
                checkpoint_decoder_layer=bool(self.checkpoint_decoder_layer),
            )

        layer_kwargs = dict(kwargs)
        layer_kwargs.pop("return_logits", None)
        layer_kwargs.pop("_router_logits", None)

        attention_type = getattr(self.layer, "attention_type", "full_attention")
        if attention_type in causal_mask_mapping:
            attn_mask = causal_mask_mapping[attention_type]
        elif attention_type == "full_attention":
            attn_mask = causal_mask_mapping.get("full_attention")
        else:
            attn_mask = None

        def run_layer(hs):
            if doing_recompute():
                (
                    recompute_attn_mask,
                    recompute_position_ids,
                    recompute_position_embeddings,
                    recompute_layer_kwargs,
                ) = get_recompute_data()
            else:
                recompute_attn_mask = attn_mask
                recompute_position_ids = position_ids
                recompute_position_embeddings = position_embeddings
                recompute_layer_kwargs = layer_kwargs
                save_for_recompute(
                    attn_mask, position_ids, position_embeddings, layer_kwargs,
                )
            return self.layer(
                hs,
                attention_mask=recompute_attn_mask,
                position_ids=recompute_position_ids,
                past_key_values=None,
                use_cache=False,
                cache_position=None,
                position_embeddings=recompute_position_embeddings,
                **recompute_layer_kwargs,
            )

        if self.checkpoint_decoder_layer and self.training:
            checkpoint_fields = {
                "layer_idx": int(self.idx),
                "stage_device": (
                    int(hidden_states.device.index)
                    if hidden_states.is_cuda and hidden_states.device.index is not None
                    else None
                ),
                "attention_type": str(attention_type),
                "recompute_grain": "layer",
            }
            hidden_states = checkpoint(
                run_layer,
                hidden_states,
                use_reentrant=False,
                context_fn=lambda: checkpoint_context_fn(**checkpoint_fields),
            )
        else:
            hidden_states = run_layer(hidden_states)

        # HF layer returns tensor or tuple depending on version.
        if isinstance(hidden_states, tuple):
            # Some adapters may propagate (hidden_states, router_logits).
            # The standard HF LFM2.5 path records router logits side-channel
            # on patched MoE blocks and keeps the decoder return as a tensor.
            if (len(hidden_states) >= 2
                    and isinstance(hidden_states[1], torch.Tensor)
                    and hidden_states[1].dim() == 2
                    and kwargs.get("_router_logits") is not None):
                router_logit = hidden_states[1]
                hidden_states = hidden_states[0]
                kwargs["_router_logits"].append(router_logit)
            else:
                hidden_states = hidden_states[0]
        if kwargs.get("_router_logits") is not None:
            kwargs["_router_logits"].extend(pop_router_logits(self.layer))

        if self.debug_finite:
            assert_finite_tensor(f"layer_{self.idx}_output", hidden_states)

        if self.memory_telemetry:
            mark_model_gpu_phase(
                "layer_after_run",
                layer_idx=self.idx,
                attention_type=attention_type,
            )

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
    """Postfix: final norm + lm_head.

    Two loss modes (matching RoundPipe):
      1. postfix_loss_token_chunk_size == 0 (default):
         norm runs full-seq, lm_head chunked by loss_token_chunk_size.
      2. postfix_loss_token_chunk_size > 0:
         BlockedPostfixCausalLMLoss — splits norm + lm_head into blocks,
         backprops per-block within forward, saves grads to CPU.

    When ``router_aux_loss_coef > 0``, adds MoE load-balancing loss from
    accumulated router_logits to the main LM loss.
    """

    def __init__(self, model, *, debug_finite: bool = False,
                 loss_token_chunk_size: int = 4096,
                 postfix_loss_token_chunk_size: int = 0,
                 memory_telemetry: bool = False,
                 torch_compile_loss: bool = False,
                 router_aux_loss_coef: float = 0.0):
        super().__init__()
        self.debug_finite = debug_finite
        self.loss_token_chunk_size = loss_token_chunk_size
        self.postfix_loss_token_chunk_size = postfix_loss_token_chunk_size
        self.memory_telemetry = memory_telemetry
        self.torch_compile_loss = torch_compile_loss
        self.router_aux_loss_coef = router_aux_loss_coef
        core = model.get_base_model() if hasattr(model, "get_base_model") else model
        self.norm = getattr(core.model, "output_norm", None) or getattr(core.model, "norm", None)
        self.vocab_size = getattr(core.config, "vocab_size", None)
        if self.vocab_size is None and hasattr(core.config, "text_config"):
            self.vocab_size = core.config.text_config.vocab_size
        self.lm_head = getattr(core, "output", None) or getattr(core, "lm_head", None)
        # Read MoE config (conservative defaults if keys missing)
        self.num_experts = getattr(core.config, "num_experts", 8)
        self.num_experts_per_tok = getattr(core.config, "num_experts_per_tok", 2)

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

        if self.memory_telemetry:
            mark_model_gpu_phase("postfix_enter",
                                 seq_len=int(hidden_states.shape[1]))

        loss = None
        if labels is not None and not kwargs.get("return_logits", False):
            # Mode 2: BlockedPostfixCausalLMLoss (norm + lm_head in blocks)
            if self.postfix_loss_token_chunk_size > 0:
                loss = BlockedPostfixCausalLMLoss.apply(
                    hidden_states, labels,
                    self.norm, self.lm_head, self.vocab_size,
                    self.postfix_loss_token_chunk_size,
                    -100, self.memory_telemetry, self.debug_finite,
                )
                if self.debug_finite:
                    assert_finite_tensor("blocked_postfix_loss", loss)
            else:
                # Mode 1: norm full-seq, then chunked lm_head
                if self.norm is not None:
                    hidden_states = self.norm(hidden_states)
                    if self.debug_finite:
                        assert_finite_tensor("post_norm_hidden_states", hidden_states)

                shift_hidden = hidden_states[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()

                # Count non-ignored tokens for normalization (same as
                # RoundPipe's ChunkedCompileLinearForCausalLMLoss).
                flat_labels = shift_labels.reshape(-1)
                num_items = (flat_labels != -100).sum()
                if num_items == 0:
                    loss = shift_hidden.sum() * 0.0
                else:
                    loss = chunked_linear_cross_entropy(
                        shift_hidden,
                        self.lm_head,
                        shift_labels,
                        num_items=num_items,
                        ignore_index=-100,
                        token_chunk_size=self.loss_token_chunk_size,
                        use_torch_compile=self.torch_compile_loss,
                    )
                    if self.debug_finite:
                        assert_finite_tensor("chunked_loss", loss)

        # MoE auxiliary load-balancing loss from accumulated router_logits
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

    def build_prefix(self, model, **kwargs):
        return LFM25ForCausalLMPrefix(
            model,
            dense_attention_masks=kwargs.get("dense_attention_masks", False),
            memory_telemetry=kwargs.get("memory_telemetry", False),
            output_router_logits=kwargs.get("output_router_logits", False),
        )

    def build_wrapped_layer(self, layer, idx, **kwargs):
        return LFM25ForCausalLMWrappedLayer(
            layer, idx=idx,
            use_flash_attention=True,
            checkpoint_decoder_layer=kwargs.get("checkpoint_decoder_layer", False),
            memory_telemetry=kwargs.get("memory_telemetry", False),
            debug_finite=kwargs.get("debug_finite", False),
        )

    def build_postfix(self, model, **kwargs):
        return LFM25ForCausalLMPostfix(
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
        # Selective flash-attention patching.
        from stratum.telemetry import parse_int_set
        flash_layers_str = kwargs.get("flash_layers", "")
        if flash_layers_str.strip().lower() in {"none", "off", "false"}:
            raise ValueError("--flash-layers cannot disable flash attention")
        flash_layer_indices = parse_int_set(flash_layers_str) if flash_layers_str else None
        _patch_lfm25_attention(core, layer_indices=flash_layer_indices)
        # MLP optimizations (checkpoint_mlp, memory_flat_frozen_mlp, mlp_token_chunk_size)
        apply_mlp_optimizations(
            core,
            checkpoint_mlp=kwargs.get("checkpoint_mlp", False),
            memory_flat_frozen_mlp=kwargs.get("memory_flat_frozen_mlp", False),
            mlp_token_chunk_size=kwargs.get("mlp_token_chunk_size", 0),
        )
        # MoE router logit capture (patch MoE blocks to return router_logits)
        output_router_logits = kwargs.get("output_router_logits", False)
        if output_router_logits:
            n_patched = patch_moe_block_for_router_logits(core)
            if n_patched > 0:
                print(f"MoE router logit capture: patched {n_patched} MoE blocks", flush=True)
            else:
                print("MoE router logit capture: no MoE blocks found to patch", flush=True)
        return super().build(hf_model, tensor_split, device_ids, **kwargs)


def _is_full_attention_layer(config, layer, idx: int) -> bool:
    """Check if a decoder layer is a full-attention layer (vs conv/ShortConv).

    Ported from qz-roundpipe's LFM2.5 attention patch.
    """
    layer_types = getattr(config, "layer_types", None)
    if layer_types is not None and idx < len(layer_types):
        return layer_types[idx] == "full_attention"
    return hasattr(layer, "self_attn")


def _preserve_attention_modules(
    new_attn: Lfm25FlashAttention, old_attn: Lfm2MoeAttention,
) -> None:
    """Preserve module objects to keep PEFT/LoRA adapters intact.

    Ported from qz-roundpipe's LFM2.5 attention patch.
    """
    new_attn.q_proj = old_attn.q_proj
    new_attn.k_proj = old_attn.k_proj
    new_attn.v_proj = old_attn.v_proj
    new_attn.out_proj = old_attn.out_proj
    new_attn.q_layernorm = old_attn.q_layernorm
    new_attn.k_layernorm = old_attn.k_layernorm


def _patch_lfm25_attention(model, layer_indices: Optional[set[int]] = None):
    """Replace Lfm2MoeAttention with capability-dispatched flash attention.

    Ported from qz-roundpipe's LFM2.5 attention patch.

    Only patches full-attention layers (ShortConv/linear_attention layers
    are skipped — they have different internal structure and must not be
    replaced with Lfm25FlashAttention).

    Args:
        model: HF model to patch.
        layer_indices: Set of layer indices to patch. None = patch all.
    """
    core = model.get_base_model() if hasattr(model, "get_base_model") else model
    config = core.config
    patched = 0
    for idx, layer in enumerate(core.model.layers):
        if not _is_full_attention_layer(config, layer, idx):
            continue
        if layer_indices is not None and idx not in layer_indices:
            continue
        if not hasattr(layer, "self_attn"):
            raise TypeError(f"layer {idx} is full_attention but has no self_attn")

        old_attn = layer.self_attn
        if isinstance(old_attn, Lfm25FlashAttention):
            if layer_indices is None or idx in layer_indices:
                patched += 1
            continue
        if not isinstance(old_attn, Lfm2MoeAttention):
            raise TypeError(
                f"layer {idx} self_attn is {type(old_attn).__name__}, "
                "expected Lfm2MoeAttention"
            )

        new_attn = Lfm25FlashAttention(old_attn.config, layer_idx=idx)
        _preserve_attention_modules(new_attn, old_attn)
        new_attn.training = old_attn.training
        layer.self_attn = new_attn
        patched += 1

    if patched:
        print(f"Patched {patched} LFM2.5 attention layers with capability-dispatched flash attention",
              flush=True)
