"""Telemetry, debugging, and memory-phase utilities.

Ported from train_lfm25_roundpipe_lora.py / train_qwen35_roundpipe_lora.py.

Components:
  - mark_model_gpu_phase()       — structured GPU allocator snapshot
  - enable_operator_telemetry()  — per-operator forward/backward hooks
  - assert_finite_tensor()       — NaN/Inf detection
  - parse_int_set() / parse_name_list() — CLI parsing helpers
"""

from __future__ import annotations

from typing import Any, Optional

import json
import torch
import torch.nn as nn
from stratum.output import vprint, vwrite


# ---------------------------------------------------------------------------
# GPU memory snapshot (enriched)
# ---------------------------------------------------------------------------

def gpu_memory_snapshot(device_id: Optional[int] = None) -> dict[str, float | int | str]:
    """GPU allocator snapshot with all RoundPipe fields."""
    if not torch.cuda.is_available():
        return {}
    if device_id is None:
        device_id = torch.cuda.current_device()
    snap: dict[str, float | int | str] = {
        "cuda_device": device_id,
        "allocated_gib": round(torch.cuda.memory_allocated(device_id) / 1024**3, 3),
        "reserved_gib": round(torch.cuda.memory_reserved(device_id) / 1024**3, 3),
        "max_allocated_gib": round(torch.cuda.max_memory_allocated(device_id) / 1024**3, 3),
        "max_reserved_gib": round(torch.cuda.max_memory_reserved(device_id) / 1024**3, 3),
    }
    try:
        free_bytes, total_bytes = torch.cuda.mem_get_info(device_id)
        snap["cuda_free_gib"] = round(free_bytes / 1024**3, 3)
        snap["cuda_total_gib"] = round(total_bytes / 1024**3, 3)
    except RuntimeError:
        pass
    stats = torch.cuda.memory_stats(device_id)
    for src, dst in (
        ("active_bytes.all.current", "active_gib"),
        ("inactive_split_bytes.all.current", "inactive_split_gib"),
        ("allocated_bytes.all.peak", "allocator_peak_gib"),
        ("reserved_bytes.all.peak", "reserved_peak_gib"),
        ("num_alloc_retries", "alloc_retries"),
        ("num_ooms", "cuda_ooms"),
    ):
        if src in stats:
            value = int(stats[src])
            snap[dst] = (
                round(value / 1024**3, 3)
                if src.endswith("bytes.all.current") or src.endswith("bytes.all.peak")
                else value
            )
    return snap


def mark_model_gpu_phase(name: str, **fields: Any) -> None:
    """Print structured JSON with GPU allocator state at a model boundary."""
    payload: dict[str, Any] = {
        "model_gpu_phase": name,
        **gpu_memory_snapshot(),
        **fields,
    }
    print(payload, flush=True)


# ---------------------------------------------------------------------------
# Finite-check helper
# ---------------------------------------------------------------------------

def assert_finite_tensor(name: str, tensor: torch.Tensor) -> None:
    """Raise FloatingPointError if *tensor* contains non-finite values."""
    if torch.isfinite(tensor).all():
        return
    bad = ~torch.isfinite(tensor)
    raise FloatingPointError(
        f"{name} contains non-finite values: "
        f"shape={tuple(tensor.shape)} dtype={tensor.dtype} "
        f"bad={int(bad.sum().item())}"
    )


# ---------------------------------------------------------------------------
# Operator telemetry — per-submodule forward/backward hooks
# ---------------------------------------------------------------------------

def _tensor_tree_nbytes(values: Any) -> int:
    """Total element bytes in a (possibly nested) tensor tree."""
    if torch.is_tensor(values):
        return int(values.numel() * values.element_size())
    if isinstance(values, (list, tuple)):
        return sum(_tensor_tree_nbytes(item) for item in values)
    if isinstance(values, dict):
        return sum(_tensor_tree_nbytes(item) for item in values.values())
    return 0


def _first_tensor_shape(values: Any) -> Optional[list[int]]:
    """Return the shape of the first tensor found in a (possibly nested) structure."""
    if torch.is_tensor(values):
        return list(values.shape)
    if isinstance(values, (list, tuple)):
        for item in values:
            shape = _first_tensor_shape(item)
            if shape is not None:
                return shape
    if isinstance(values, dict):
        for item in values.values():
            shape = _first_tensor_shape(item)
            if shape is not None:
                return shape
    return None


def _module_param_summary(module: nn.Module) -> dict[str, float | int]:
    """Count frozen/trainable params and their total bytes."""
    frozen_bytes = 0
    trainable_bytes = 0
    param_count = 0
    trainable_param_count = 0
    for param in module.parameters(recurse=True):
        param_count += 1
        nbytes = int(param.numel() * param.element_size())
        if param.requires_grad:
            trainable_bytes += nbytes
            trainable_param_count += 1
        else:
            frozen_bytes += nbytes
    return {
        "param_count": param_count,
        "trainable_param_count": trainable_param_count,
        "param_gib": round((frozen_bytes + trainable_bytes) / 1024**3, 3),
        "trainable_param_gib": round(trainable_bytes / 1024**3, 6),
    }


def enable_operator_telemetry(
    model: nn.Module,
    *,
    layer_indices: set[int],
    module_names: list[str],
    verbose: bool = True,
) -> int:
    """Register forward/backward hooks on selected submodules.

    For each decoder layer index in *layer_indices*, find submodules
    whose names are in *module_names* (e.g. "self_attn", "mlp") and
    register pre/post hooks that emit structured JSON with GPU allocator
    snapshots.

    Returns the number of hooks registered.
    """
    core = model.get_base_model() if hasattr(model, "get_base_model") else model
    layers = getattr(core.model, "layers", None)
    if layers is None:
        raise TypeError("expected a model with core.model.layers")
    if not layer_indices:
        if verbose:
            vprint({"operator_telemetry": "no layers selected"})
        return 0

    registered = 0
    for idx, layer in enumerate(layers):
        if idx not in layer_indices:
            continue
        for module_name in module_names:
            module = getattr(layer, module_name, None)
            if module is None:
                if verbose:
                    vprint({
                        "operator_telemetry_missing_module": module_name,
                        "layer_idx": idx,
                    })
                continue

            def make_pre_hook(_idx, _name):
                def pre_hook(_mod, inputs):
                    mark_model_gpu_phase(
                        "operator_forward_pre",
                        layer_idx=_idx,
                        module=_name,
                        module_type=type(_mod).__name__,
                        input_shape=_first_tensor_shape(inputs),
                        input_gib=round(_tensor_tree_nbytes(inputs) / 1024**3, 3),
                        **_module_param_summary(_mod),
                    )
                return pre_hook

            def make_post_hook(_idx, _name):
                def post_hook(_mod, inputs, outputs):
                    mark_model_gpu_phase(
                        "operator_forward_post",
                        layer_idx=_idx,
                        module=_name,
                        module_type=type(_mod).__name__,
                        input_shape=_first_tensor_shape(inputs),
                        output_shape=_first_tensor_shape(outputs),
                        input_gib=round(_tensor_tree_nbytes(inputs) / 1024**3, 3),
                        output_gib=round(_tensor_tree_nbytes(outputs) / 1024**3, 3),
                        **_module_param_summary(_mod),
                    )
                return post_hook

            def make_backward_pre_hook(_idx, _name):
                def backward_pre_hook(_mod, grad_outputs):
                    mark_model_gpu_phase(
                        "operator_backward_pre",
                        layer_idx=_idx,
                        module=_name,
                        module_type=type(_mod).__name__,
                        grad_output_shape=_first_tensor_shape(grad_outputs),
                        grad_output_gib=round(_tensor_tree_nbytes(grad_outputs) / 1024**3, 3),
                        **_module_param_summary(_mod),
                    )
                return backward_pre_hook

            def make_backward_post_hook(_idx, _name):
                def backward_post_hook(_mod, grad_inputs, grad_outputs):
                    mark_model_gpu_phase(
                        "operator_backward_post",
                        layer_idx=_idx,
                        module=_name,
                        module_type=type(_mod).__name__,
                        grad_input_shape=_first_tensor_shape(grad_inputs),
                        grad_output_shape=_first_tensor_shape(grad_outputs),
                        grad_input_gib=round(_tensor_tree_nbytes(grad_inputs) / 1024**3, 3),
                        grad_output_gib=round(_tensor_tree_nbytes(grad_outputs) / 1024**3, 3),
                        **_module_param_summary(_mod),
                    )
                return backward_post_hook

            module.register_forward_pre_hook(make_pre_hook(idx, module_name))
            module.register_forward_hook(make_post_hook(idx, module_name))
            if hasattr(module, "register_full_backward_pre_hook"):
                module.register_full_backward_pre_hook(make_backward_pre_hook(idx, module_name))
            module.register_full_backward_hook(make_backward_post_hook(idx, module_name))
            registered += 1

    if verbose and registered:
        vwrite(f"  operator telemetry: {registered} hooks on layers {sorted(layer_indices)}")
    return registered


# ---------------------------------------------------------------------------
# CLI parsing helpers
# ---------------------------------------------------------------------------

def parse_int_set(value: str) -> set[int]:
    """Parse comma-separated integer string into a set, e.g. '0,2,4' -> {0,2,4}."""
    if not value.strip():
        return set()
    if value.strip().lower() in {"none", "off", "false"}:
        return set()
    return {int(part.strip()) for part in value.split(",") if part.strip()}


def parse_name_list(value: str) -> list[str]:
    """Parse comma-separated name string into a list, e.g. 'a,b' -> ['a','b']."""
    return [part.strip() for part in value.split(",") if part.strip()]
