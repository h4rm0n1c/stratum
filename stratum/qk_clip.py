"""QK-Clip support for Muon training.

Attention wrappers record per-head attention-logit statistics for the current
step. Exact flash-kernel max logits are preferred when the installed backend
exposes them; otherwise the portable fallback is a conservative norm-product
upper bound. After the optimizer update, trainable Q/K projection rows are
rescaled when the statistic exceeds the configured threshold.
"""

from __future__ import annotations

from typing import Any, Callable, Iterable

import torch
import torch.nn as nn

from stratum.attribute import ParamAttribute


QK_CLIP_STATS_ATTR = "_stratum_qk_clip_smax"
QK_CLIP_ENABLED_ATTR = "_stratum_qk_clip_enabled"
QK_CLIP_STAT_MODE_ATTR = "_stratum_qk_clip_stat_mode"
QK_CLIP_STAT_SOURCE_ATTR = "_stratum_qk_clip_stat_source"
QK_CLIP_STAT_MODES = {"auto", "bound", "exact_flash"}


def record_qk_clip_stats(
    module: nn.Module,
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    *,
    scaling: float,
) -> None:
    """Record per-query-head attention-logit upper bounds on *module*.

    ``query_states``/``key_states`` may be ``(batch, heads, seq, dim)`` or
    packed ``(tokens, heads, dim)``. When key/value heads are grouped, key head
    bounds are repeated to query-head count before combining.
    """
    if not bool(getattr(module, QK_CLIP_ENABLED_ATTR, False)):
        return
    if _qk_clip_stat_mode(module) == "exact_flash":
        return
    with torch.no_grad():
        q = _heads_first(query_states.detach())
        k = _heads_first(key_states.detach())
        if q.numel() == 0 or k.numel() == 0:
            return
        q_max = q.float().norm(dim=-1).amax(dim=1)
        k_max = k.float().norm(dim=-1).amax(dim=1)
        if q_max.numel() != k_max.numel():
            if q_max.numel() % k_max.numel() != 0:
                return
            k_max = k_max.repeat_interleave(q_max.numel() // k_max.numel())
        smax = q_max * k_max * float(scaling)
        _record_qk_clip_smax(module, smax, source="bound")


def record_qk_clip_exact_stats(module: nn.Module, max_logits: torch.Tensor) -> None:
    """Record exact per-head max attention logits returned by a flash backend."""
    if not bool(getattr(module, QK_CLIP_ENABLED_ATTR, False)):
        return
    if not torch.is_tensor(max_logits) or max_logits.numel() == 0:
        return
    with torch.no_grad():
        _record_qk_clip_smax(module, max_logits.detach().flatten().float(), source="exact_flash")


def flash_attention_with_qk_clip_stats(
    module: nn.Module,
    backend_name: str,
    fn: Callable[..., Any],
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    scaling: float,
    **flash_kwargs: Any,
) -> torch.Tensor:
    """Call flash attention and collect QK-Clip stats according to module mode.

    Patched flash backends may accept ``return_max_logits=True`` and return
    either ``(out, meta)`` with ``meta.max_logits`` or ``(out, max_logits)``.
    In ``auto`` mode unsupported backends fall back to the norm-product bound;
    in ``exact_flash`` mode unsupported or malformed returns fail loudly.
    """
    if not bool(getattr(module, QK_CLIP_ENABLED_ATTR, False)):
        return fn(q, k, v, **flash_kwargs)

    mode = _qk_clip_stat_mode(module)
    if mode == "bound":
        record_qk_clip_stats(module, query_states, key_states, scaling=scaling)
        return fn(q, k, v, **flash_kwargs)

    exact_kwargs = dict(flash_kwargs)
    exact_kwargs["return_max_logits"] = True
    try:
        result = fn(q, k, v, **exact_kwargs)
    except TypeError as exc:
        if mode == "exact_flash":
            raise RuntimeError(
                f"{backend_name} does not expose return_max_logits=True required "
                "by --muon-qk-stat-mode exact_flash"
            ) from exc
        record_qk_clip_stats(module, query_states, key_states, scaling=scaling)
        return fn(q, k, v, **flash_kwargs)

    attn_output, max_logits = _split_flash_attention_result(result)
    if max_logits is None:
        if mode == "exact_flash":
            raise RuntimeError(
                f"{backend_name} accepted return_max_logits=True but did not return "
                "per-head max logits for QK-Clip"
            )
        record_qk_clip_stats(module, query_states, key_states, scaling=scaling)
    else:
        record_qk_clip_exact_stats(module, max_logits)
    return attn_output


def clear_qk_clip_stats(module: nn.Module) -> None:
    if hasattr(module, QK_CLIP_STATS_ATTR):
        delattr(module, QK_CLIP_STATS_ATTR)
    if hasattr(module, QK_CLIP_STAT_SOURCE_ATTR):
        delattr(module, QK_CLIP_STAT_SOURCE_ATTR)


def apply_qk_clip_to_modules(
    modules: Iterable[nn.Module],
    *,
    threshold: float,
    cpu_offload: bool,
) -> dict[str, int | float]:
    """Apply QK-Clip post optimizer update.

    Returns compact stats for training logs/tests.
    """
    if threshold <= 0:
        return {
            "layers": 0,
            "heads": 0,
            "max_s": 0.0,
            "min_gamma": 1.0,
            "exact_layers": 0,
            "bound_layers": 0,
        }

    layers = 0
    clipped_heads = 0
    max_s = 0.0
    min_gamma = 1.0
    exact_layers = 0
    bound_layers = 0
    visited_modules: set[int] = set()
    for root in modules:
        for module in root.modules():
            if id(module) in visited_modules:
                continue
            visited_modules.add(id(module))
            smax = getattr(module, QK_CLIP_STATS_ATTR, None)
            if smax is None:
                continue
            source = str(getattr(module, QK_CLIP_STAT_SOURCE_ATTR, "bound"))
            clear_qk_clip_stats(module)
            if not torch.is_tensor(smax) or smax.numel() == 0:
                continue
            if source == "exact_flash":
                exact_layers += 1
            else:
                bound_layers += 1
            smax_f = smax.float()
            gamma = torch.clamp(threshold / torch.clamp_min(smax_f, 1e-12), max=1.0)
            if bool(torch.all(gamma >= 1.0)):
                max_s = max(max_s, float(smax_f.max().item()))
                continue

            if _scale_separate_qk_projections(
                module,
                gamma,
                num_q_heads=int(gamma.numel()),
                cpu_offload=cpu_offload,
            ):
                layers += 1
                clipped_heads += int((gamma < 1.0).sum().item())
                max_s = max(max_s, float(smax_f.max().item()))
                min_gamma = min(min_gamma, float(gamma.min().item()))
                continue

            if _scale_fused_qkv_projection(
                module,
                gamma,
                num_q_heads=int(gamma.numel()),
                cpu_offload=cpu_offload,
            ):
                layers += 1
                clipped_heads += int((gamma < 1.0).sum().item())
                max_s = max(max_s, float(smax_f.max().item()))
                min_gamma = min(min_gamma, float(gamma.min().item()))
                continue

            max_s = max(max_s, float(smax_f.max().item()))

    return {
        "layers": layers,
        "heads": clipped_heads,
        "max_s": max_s,
        "min_gamma": min_gamma,
        "exact_layers": exact_layers,
        "bound_layers": bound_layers,
    }


def _qk_clip_stat_mode(module: nn.Module) -> str:
    mode = str(getattr(module, QK_CLIP_STAT_MODE_ATTR, "bound"))
    return mode if mode in QK_CLIP_STAT_MODES else "bound"


def _record_qk_clip_smax(module: nn.Module, smax: torch.Tensor, *, source: str) -> None:
    if not torch.is_tensor(smax) or smax.numel() == 0:
        return
    smax = smax.detach().flatten().float()
    prev = getattr(module, QK_CLIP_STATS_ATTR, None)
    if prev is not None and prev.numel() == smax.numel():
        smax = torch.maximum(prev.to(device=smax.device, dtype=smax.dtype), smax)
        prev_source = str(getattr(module, QK_CLIP_STAT_SOURCE_ATTR, source))
        if prev_source == "exact_flash":
            source = "exact_flash"
    setattr(module, QK_CLIP_STATS_ATTR, smax.cpu())
    setattr(module, QK_CLIP_STAT_SOURCE_ATTR, source)


def _split_flash_attention_result(result: Any) -> tuple[torch.Tensor, torch.Tensor | None]:
    if not isinstance(result, tuple):
        return result, None
    if not result:
        raise RuntimeError("flash attention returned an empty tuple")
    attn_output = result[0]
    for item in result[1:]:
        max_logits = getattr(item, "max_logits", None)
        if torch.is_tensor(max_logits):
            return attn_output, max_logits
        if torch.is_tensor(item) and item.ndim <= 1:
            return attn_output, item
    return attn_output, None


def _heads_first(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim == 4:
        # (batch, heads, seq, dim) -> (heads, batch * seq, dim)
        return tensor.permute(1, 0, 2, 3).reshape(tensor.shape[1], -1, tensor.shape[-1])
    if tensor.ndim == 3:
        # (tokens, heads, dim) -> (heads, tokens, dim)
        return tensor.permute(1, 0, 2)
    raise ValueError(f"unsupported QK tensor rank for QK-Clip: {tensor.ndim}")


def _projection_out_features(proj: nn.Module) -> int:
    if hasattr(proj, "out_features"):
        return int(getattr(proj, "out_features"))
    weight = getattr(proj, "weight", None)
    if torch.is_tensor(weight) and weight.ndim >= 2:
        return int(weight.shape[0])
    for param in proj.parameters(recurse=True):
        if param.ndim >= 2:
            return int(param.shape[0])
    return 0


def _scale_separate_qk_projections(
    module: nn.Module,
    gamma: torch.Tensor,
    *,
    num_q_heads: int,
    cpu_offload: bool,
) -> bool:
    q_proj = getattr(module, "q_proj", None)
    k_proj = getattr(module, "k_proj", None)
    if q_proj is None or k_proj is None:
        return False
    q_head_dim = _infer_q_head_dim(module, q_proj, num_q_heads)
    k_head_dim = _infer_k_head_dim(module, k_proj, num_q_heads)
    if q_head_dim <= 0 or k_head_dim <= 0:
        return False

    _scale_projection_rows(
        q_proj,
        _row_scale(gamma.sqrt(), q_head_dim),
        offset=0,
        cpu_offload=cpu_offload,
    )

    num_k_heads = _projection_out_features(k_proj) // k_head_dim
    k_gamma = _key_head_gamma(gamma, num_q_heads=num_q_heads, num_k_heads=num_k_heads)
    _scale_projection_rows(
        k_proj,
        _row_scale(k_gamma.sqrt(), k_head_dim),
        offset=0,
        cpu_offload=cpu_offload,
    )
    return True


def _scale_fused_qkv_projection(
    module: nn.Module,
    gamma: torch.Tensor,
    *,
    num_q_heads: int,
    cpu_offload: bool,
) -> bool:
    fused = _fused_qkv_projection(module)
    if fused is None:
        return False
    head_dim = _infer_fused_head_dim(module, fused, num_q_heads)
    if head_dim <= 0:
        return False
    out_features = _projection_out_features(fused)
    q_rows = num_q_heads * head_dim
    if out_features <= q_rows:
        return False

    num_k_heads = _infer_num_key_value_heads(module, out_features, q_rows, head_dim)
    if num_k_heads <= 0:
        return False
    k_rows = num_k_heads * head_dim
    if out_features < q_rows + k_rows:
        return False

    _scale_projection_rows(
        fused,
        _row_scale(gamma.sqrt(), head_dim),
        offset=0,
        cpu_offload=cpu_offload,
    )
    k_gamma = _key_head_gamma(gamma, num_q_heads=num_q_heads, num_k_heads=num_k_heads)
    _scale_projection_rows(
        fused,
        _row_scale(k_gamma.sqrt(), head_dim),
        offset=q_rows,
        cpu_offload=cpu_offload,
    )
    return True


def _fused_qkv_projection(module: nn.Module) -> nn.Module | None:
    for attr in ("query_key_value", "qkv", "in_proj_qkv", "c_attn"):
        candidate = getattr(module, attr, None)
        if isinstance(candidate, nn.Module):
            return candidate
    return None


def _infer_q_head_dim(module: nn.Module, q_proj: nn.Module, num_q_heads: int) -> int:
    head_dim = int(getattr(module, "head_dim", 0) or 0)
    if head_dim > 0:
        return head_dim
    out_features = _projection_out_features(q_proj)
    return out_features // max(1, num_q_heads)


def _infer_k_head_dim(module: nn.Module, k_proj: nn.Module, num_q_heads: int) -> int:
    head_dim = int(getattr(module, "head_dim", 0) or 0)
    if head_dim > 0:
        return head_dim
    out_features = _projection_out_features(k_proj)
    return out_features // max(1, num_q_heads)


def _infer_fused_head_dim(module: nn.Module, proj: nn.Module, num_q_heads: int) -> int:
    head_dim = int(getattr(module, "head_dim", 0) or 0)
    if head_dim > 0:
        return head_dim
    out_features = _projection_out_features(proj)
    # Common packed QKV is q + k + v with equal head counts.
    return out_features // max(1, 3 * num_q_heads)


def _infer_num_key_value_heads(
    module: nn.Module,
    out_features: int,
    q_rows: int,
    head_dim: int,
) -> int:
    for attr in ("num_key_value_heads", "num_kv_heads", "n_kv_heads"):
        value = int(getattr(module, attr, 0) or 0)
        if value > 0:
            return value
    remaining = out_features - q_rows
    # Common packed layout after Q is K then V, with K and V equal size.
    return remaining // max(1, 2 * head_dim)


def _key_head_gamma(
    gamma: torch.Tensor,
    *,
    num_q_heads: int,
    num_k_heads: int,
) -> torch.Tensor:
    if num_k_heads > 0 and num_k_heads != num_q_heads:
        if num_q_heads % num_k_heads != 0:
            return gamma
        group = max(1, num_q_heads // num_k_heads)
        return gamma.reshape(num_k_heads, group).amin(dim=1)
    return gamma


def _row_scale(head_scale: torch.Tensor, head_dim: int) -> torch.Tensor:
    return head_scale.repeat_interleave(head_dim).to(dtype=torch.float32)


def _scale_projection_rows(
    proj: nn.Module,
    row_scale: torch.Tensor,
    *,
    offset: int,
    cpu_offload: bool,
) -> None:
    for name, param in proj.named_parameters():
        if not param.requires_grad:
            continue
        if not _is_output_row_parameter(name, param, row_scale.numel(), offset):
            continue
        target = _optim_or_live_param(param, cpu_offload)
        rows = min(row_scale.numel(), max(0, target.shape[0] - offset))
        if rows <= 0:
            continue
        scale = row_scale[:rows].to(device=target.device, dtype=target.dtype)
        view_shape = (rows,) + (1,) * (target.ndim - 1)
        target.data[offset : offset + rows].mul_(scale.reshape(view_shape))


def _optim_or_live_param(param: torch.nn.Parameter, cpu_offload: bool) -> torch.nn.Parameter:
    if cpu_offload:
        attr = ParamAttribute.get(param)
        if attr is not None and attr.optim is not None:
            return attr.optim
    return param


def _is_output_row_parameter(
    name: str,
    param: torch.nn.Parameter,
    rows: int,
    offset: int,
) -> bool:
    if param.ndim < 1 or param.shape[0] < offset + 1:
        return False
    lower = name.lower()
    if lower == "weight" or lower == "bias":
        return param.shape[0] >= offset + 1
    if "lora_b" in lower and lower.endswith("weight"):
        return param.ndim >= 2 and param.shape[0] >= offset + 1
    # Do not scale LoRA A: it is rank-by-input, not output-head rows.
    if "lora_a" in lower:
        return False
    if param.shape[0] >= rows + offset and lower.endswith("weight"):
        return True
    return False
