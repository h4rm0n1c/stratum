"""MLP memory optimizations ported from RoundPipe.

Three mutually-exclusive MLP patching strategies for decoder layers:

1. CheckpointedModule   — wrap MLP in torch.utils.checkpoint.checkpoint()
2. TokenChunkedModule   — split MLP forward into sequence-token chunks
3. MemoryFlatFrozenMLP  — custom autograd with token-chunked backward recompute

All three are safe on multi-GPU — they modify layer.mlp in-place before
the DecoderStage extracts them, so the pipeline topology is unaffected.

Source: train_lfm25_roundpipe_lora.py and train_qwen35_roundpipe_lora.py
"""

from __future__ import annotations

from typing import Any, NamedTuple, Optional

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint


# ---------------------------------------------------------------------------
# 1. CheckpointedModule — wrap MLP in activation checkpointing
# ---------------------------------------------------------------------------

class CheckpointedModule(nn.Module):
    """Wrap a submodule in activation checkpointing when gradients are active."""

    def __init__(self, module: nn.Module) -> None:
        super().__init__()
        self.module = module

    def forward(self, *args, **kwargs):
        if self.training and any(torch.is_tensor(arg) and arg.requires_grad for arg in args):
            if kwargs:
                return checkpoint(lambda *flat_args: self.module(*flat_args, **kwargs), *args, use_reentrant=False)
            return checkpoint(self.module, *args, use_reentrant=False)
        return self.module(*args, **kwargs)


# ---------------------------------------------------------------------------
# 2. TokenChunkedModule — split MLP forward over sequence-token chunks
# ---------------------------------------------------------------------------

class TokenChunkedModule(nn.Module):
    """Run a positionwise module over sequence-token chunks.

    Since MLPs process each token independently, the output of applying
    the MLP to token chunks and concatenating is identical to applying
    it to the full sequence. This lowers the peak 3x-hidden intermediate.
    """

    def __init__(self, module: nn.Module, token_chunk_size: int) -> None:
        super().__init__()
        if token_chunk_size <= 0:
            raise ValueError("token_chunk_size must be positive")
        self.module = module
        self.token_chunk_size = token_chunk_size

    def forward(self, hidden_states: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        if args:
            raise ValueError("TokenChunkedModule only supports positional hidden_states plus keyword args")
        if hidden_states.dim() < 3 or hidden_states.shape[1] <= self.token_chunk_size:
            return self.module(hidden_states, **kwargs)

        outputs = []
        seq_len = hidden_states.shape[1]
        for start in range(0, seq_len, self.token_chunk_size):
            end = min(start + self.token_chunk_size, seq_len)
            outputs.append(self.module(hidden_states[:, start:end, :], **kwargs))
        return torch.cat(outputs, dim=1)


# ---------------------------------------------------------------------------
# 3. MemoryFlatFrozenMLP — custom autograd with chunked backward
# ---------------------------------------------------------------------------

class _MlpProjectionNames(NamedTuple):
    gate: str
    up: str
    down: str


def _mlp_projection_names(module: nn.Module) -> _MlpProjectionNames:
    """Return the dense MLP projection names used by *module*."""
    if hasattr(module, "gate_exps"):
        raise TypeError("memory-flat frozen MLP does not support MoE expert modules")
    if hasattr(module, "ffn_gate_proj"):
        return _MlpProjectionNames("ffn_gate_proj", "ffn_up_proj", "ffn_down_proj")
    return _MlpProjectionNames("gate_proj", "up_proj", "down_proj")


def _assert_frozen_mlp(module: nn.Module) -> _MlpProjectionNames:
    """Validate that *module* is a frozen dense MLP.

    Supports both LFM2.5 naming (ffn_*_proj) and standard Qwen naming.
    """
    proj_names = _mlp_projection_names(module)

    missing = [name for name in proj_names if not hasattr(module, name)]
    if missing:
        raise TypeError(
            f"memory-flat frozen MLP requires gate/up/down projections, "
            f"missing: {missing}"
        )
    if not hasattr(module, "act_fn"):
        raise TypeError("memory-flat frozen MLP requires act_fn (SiLU)")

    trainable = [name for name, param in module.named_parameters() if param.requires_grad]
    if trainable:
        raise ValueError(
            "--memory-flat-frozen-mlp only supports frozen MLP parameters; "
            f"trainable parameters found: {trainable[:8]}"
        )

    for name in proj_names:
        proj = getattr(module, name)
        if not isinstance(proj, nn.Linear):
            raise TypeError(
                f"memory-flat frozen MLP expected {name} to be nn.Linear, "
                f"got {type(proj)!r}"
            )
        if proj.bias is not None:
            raise TypeError(f"memory-flat frozen MLP currently expects bias=False for {name}")
    return proj_names


class MemoryFlatFrozenMLPFunction(torch.autograd.Function):
    """Custom autograd: forward runs MLP in chunks, backward recomputes one chunk at a time."""

    @staticmethod
    def forward(ctx: Any, hidden_states: torch.Tensor, module: nn.Module, token_chunk_size: int) -> torch.Tensor:
        proj_names = _assert_frozen_mlp(module)
        if token_chunk_size <= 0:
            raise ValueError("token_chunk_size must be positive")
        if hidden_states.dim() < 3:
            raise ValueError("memory-flat frozen MLP expects [batch, seq, hidden] input")

        ctx.module = module
        ctx.proj_names = proj_names
        ctx.token_chunk_size = token_chunk_size
        ctx.save_for_backward(hidden_states)

        seq_len = hidden_states.shape[1]
        gate_proj = getattr(module, proj_names.gate)
        up_proj = getattr(module, proj_names.up)
        down_proj = getattr(module, proj_names.down)
        out_features = down_proj.out_features
        output = hidden_states.new_empty(*hidden_states.shape[:-1], out_features)

        with torch.no_grad():
            for start in range(0, seq_len, token_chunk_size):
                end = min(start + token_chunk_size, seq_len)
                chunk = hidden_states[:, start:end, :]
                output[:, start:end, :] = down_proj(
                    module.act_fn(gate_proj(chunk)) * up_proj(chunk)
                )
        return output

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[Optional[torch.Tensor], None, None]:
        (hidden_states,) = ctx.saved_tensors
        module = ctx.module
        proj_names = ctx.proj_names
        token_chunk_size = ctx.token_chunk_size
        seq_len = hidden_states.shape[1]
        grad_hidden_states = torch.empty_like(hidden_states) if ctx.needs_input_grad[0] else None

        if grad_hidden_states is not None:
            gate_proj = getattr(module, proj_names.gate)
            up_proj = getattr(module, proj_names.up)
            down_proj = getattr(module, proj_names.down)
            with torch.enable_grad():
                for start in range(0, seq_len, token_chunk_size):
                    end = min(start + token_chunk_size, seq_len)
                    chunk = hidden_states[:, start:end, :].detach().requires_grad_(True)
                    chunk_output = down_proj(
                        module.act_fn(gate_proj(chunk)) * up_proj(chunk)
                    )
                    (grad_chunk,) = torch.autograd.grad(
                        chunk_output,
                        chunk,
                        grad_outputs=grad_output[:, start:end, :],
                        retain_graph=False,
                        create_graph=False,
                    )
                    grad_hidden_states[:, start:end, :] = grad_chunk

        return grad_hidden_states, None, None


class MemoryFlatFrozenMLP(nn.Module):
    """Frozen MLP with token-sliced backward recompute.

    Wraps a frozen (non-trainable) dense MLP. The forward runs the MLP
    in token chunks. The backward recomputes the MLP one token chunk at
    a time and computes per-chunk gradients via torch.autograd.grad().
    """

    def __init__(self, module: nn.Module, token_chunk_size: int) -> None:
        super().__init__()
        if token_chunk_size <= 0:
            raise ValueError("token_chunk_size must be positive")
        _assert_frozen_mlp(module)
        self.module = module
        self.token_chunk_size = token_chunk_size

    def forward(self, hidden_states: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        if args or kwargs:
            raise ValueError("MemoryFlatFrozenMLP only supports a hidden_states positional argument")
        if hidden_states.shape[1] <= self.token_chunk_size and not hidden_states.requires_grad:
            return self.module(hidden_states)
        return MemoryFlatFrozenMLPFunction.apply(hidden_states, self.module, self.token_chunk_size)


# ---------------------------------------------------------------------------
# Patcher functions — apply MLP strategies to a model in-place
# ---------------------------------------------------------------------------

def _unwrap_peft(model: nn.Module) -> nn.Module:
    return model.get_base_model() if hasattr(model, "get_base_model") else model


def enable_decoder_mlp_checkpointing(model: nn.Module) -> int:
    """Wrap each decoder layer's MLP in activation checkpointing."""
    core = _unwrap_peft(model)
    layers = getattr(core.model, "layers", None)
    if layers is None:
        raise TypeError("expected model with core.model.layers")

    patched = 0
    for layer in layers:
        if hasattr(layer, "mlp") and not isinstance(layer.mlp, CheckpointedModule):
            layer.mlp = CheckpointedModule(layer.mlp)
            patched += 1
    return patched


def enable_memory_flat_frozen_mlp(model: nn.Module, token_chunk_size: int) -> int:
    """Replace each decoder layer's MLP with MemoryFlatFrozenMLP."""
    core = _unwrap_peft(model)
    layers = getattr(core.model, "layers", None)
    if layers is None:
        raise TypeError("expected model with core.model.layers")

    patched = 0
    for layer in layers:
        if hasattr(layer, "mlp") and not isinstance(layer.mlp, MemoryFlatFrozenMLP):
            layer.mlp = MemoryFlatFrozenMLP(layer.mlp, token_chunk_size)
            patched += 1
    return patched


def enable_decoder_mlp_token_chunking(model: nn.Module, token_chunk_size: int) -> int:
    """Wrap each decoder layer's MLP in TokenChunkedModule."""
    core = _unwrap_peft(model)
    layers = getattr(core.model, "layers", None)
    if layers is None:
        raise TypeError("expected model with core.model.layers")

    patched = 0
    for layer in layers:
        if hasattr(layer, "mlp") and not isinstance(layer.mlp, TokenChunkedModule):
            layer.mlp = TokenChunkedModule(layer.mlp, token_chunk_size)
            patched += 1
    return patched


def apply_mlp_optimizations(
    model: nn.Module,
    *,
    checkpoint_mlp: bool = False,
    memory_flat_frozen_mlp: bool = False,
    mlp_token_chunk_size: int = 0,
) -> None:
    """Apply the correct MLP optimization to *model* in-place.

    Mutex rules (same as RoundPipe):
      - memory_flat_frozen_mlp requires mlp_token_chunk_size > 0
      - memory_flat_frozen_mlp conflicts with checkpoint_mlp
      - mlp_token_chunk_size without memory_flat_frozen_mlp → TokenChunkedModule
    """
    if memory_flat_frozen_mlp:
        if mlp_token_chunk_size <= 0:
            raise ValueError("--memory-flat-frozen-mlp requires --mlp-token-chunk-size > 0")
        if checkpoint_mlp:
            raise ValueError("--memory-flat-frozen-mlp is not compatible with --checkpoint-mlp")
        patched = enable_memory_flat_frozen_mlp(model, mlp_token_chunk_size)
        print(f"  mlp memory-flat frozen: {patched} layers, chunk_size={mlp_token_chunk_size}", flush=True)
    elif checkpoint_mlp:
        patched = enable_decoder_mlp_checkpointing(model)
        print(f"  mlp checkpointed: {patched} layers", flush=True)
    elif mlp_token_chunk_size > 0:
        patched = enable_decoder_mlp_token_chunking(model, mlp_token_chunk_size)
        print(f"  mlp token-chunked: {patched} layers, chunk_size={mlp_token_chunk_size}", flush=True)
