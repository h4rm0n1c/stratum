"""Model architecture registry — add new architectures by decorating a build class."""

from typing import Any, Optional

import torch
from stratum.output import vwrite
import torch.nn as nn

from stratum.pipeline import StratumPipeline
from stratum.assign import assign_layers_to_devices
from stratum.planner import split_layers_by_memory_limit
from stratum.stage import DeviceStage
from stratum.upload import (
    prepare_nf4, prepare_fp16_staged, upload_stream,
    ensure_weights, free_weights, NF4_ATTR, FP16_ATTR,
)
from stratum.nf4_linear import NF4Linear
from stratum.utils import log_event


class ModelArch:
    """Base class for model architecture adapters.

    Subclasses must implement build_prefix(), build_wrapped_layer(),
    and build_postfix(). The build() method assembles them into a
    StratumPipeline, then uploads all weights to their assigned devices.
    """

    def build_prefix(self, model: nn.Module, **kwargs) -> nn.Module:
        raise NotImplementedError

    def build_wrapped_layer(self, layer: nn.Module, idx: int, **kwargs) -> nn.Module:
        raise NotImplementedError

    def build_postfix(self, model: nn.Module, **kwargs) -> nn.Module:
        raise NotImplementedError

    def get_config(self, model: nn.Module) -> Any:
        return model.config

    def get_num_layers(self, config: Any) -> int:
        raise NotImplementedError

    def build(
        self,
        hf_model: nn.Module,
        tensor_split: Optional[list[float]] = None,
        device_ids: Optional[list[int]] = None,
        *,
        use_nf4: bool = True,
        nf4_cache_dir: Optional[str] = None,
        nf4_scope: str = "all",
        nf4_min_numel: int = 4096,
        checkpoint_decoder_layer: bool = False,
        stage_memory_limit_gib: float = 0.0,
        nf4_layer_size_floor_gib: float = 0.0,
        prefetch_nf4: bool = False,
        offload_stage_inputs: bool = False,
        hf_model_name_or_path: Optional[str] = None,
        verbose: bool = True,
        **kwargs,
    ) -> StratumPipeline:
        """Build a StratumPipeline from a HuggingFace model.

        1. Get base model (unwrap PEFT)
        2. Determine devices and layer assignment
        3. Build prefix, wrapped layers, postfix (all on CPU)
        4. Group layers into DeviceStages per device
        5. Upload all weights to their assigned devices via upload_to_device()
        6. Return StratumPipeline
        """
        core = hf_model.get_base_model() if hasattr(hf_model, "get_base_model") else hf_model
        config = self.get_config(core)
        n_layers = self.get_num_layers(config)

        n_devices = len(device_ids) if device_ids else (len(tensor_split) if tensor_split else 1)
        if device_ids is None:
            device_ids = list(range(n_devices))

        assignment = assign_layers_to_devices(
            n_layers, tensor_split=tensor_split, device_ids=device_ids,
        )

        # Build prefix (stays on CPU for now).
        # Pop MoE params from kwargs to avoid duplicate keyword errors
        # when passed explicitly (kwargs still carries them from subclass).
        _moe_rl = kwargs.pop("output_router_logits", False)
        _moe_coef = kwargs.pop("router_aux_loss_coef", 0.0)
        prefix = self.build_prefix(
            core, output_router_logits=_moe_rl, **kwargs,
        )

        # Build wrapped layers
        raw_layers = list(core.model.layers)
        wrapped = [
            self.build_wrapped_layer(
                layer,
                idx,
                checkpoint_decoder_layer=checkpoint_decoder_layer,
                **kwargs,
            )
            for idx, layer in enumerate(raw_layers)
        ]

        # Group by device
        device_groups: dict[int, list] = {d: [] for d in device_ids}
        for idx, wl in enumerate(wrapped):
            dev = assignment[idx]
            device_groups[dev].append(wl)

        # Build DeviceStages (params stay on CPU)
        stages = []
        for dev in device_ids:
            if device_groups[dev]:
                stage_groups = split_layers_by_memory_limit(
                    device_groups[dev],
                    stage_memory_limit_gib,
                    layer_size_floor_gib=nf4_layer_size_floor_gib,
                )
                if stage_memory_limit_gib > 0 and len(stage_groups) > 1:
                    log_event(
                        "stage_memory_split",
                        device=dev,
                        layers=len(device_groups[dev]),
                        substages=len(stage_groups),
                        limit_gib=stage_memory_limit_gib,
                        layer_size_floor_gib=nf4_layer_size_floor_gib,
                    )
                for group in stage_groups:
                    stages.append(DeviceStage(group, device_id=dev))

        # Build postfix (stays on CPU).
        postfix = self.build_postfix(
            core, router_aux_loss_coef=_moe_coef, **kwargs,
        )

        pipeline = StratumPipeline(
            prefix,
            stages,
            postfix,
            prefetch_nf4=prefetch_nf4,
            offload_stage_inputs=offload_stage_inputs,
        )

        # Phase 1: Prepare frozen weight streaming.
        # NF4 mode: quantize frozen 2D weights, drop originals (upload per-step as FP16).
        # Non-NF4 mode: pin frozen 2D weights on CPU, drop GPU copy (upload per-step as FP16).
        # Both paths leave param.data = empty(0) for staged weights; Phase 2 uploads only
        # the non-staged params (trainable LoRA, small norms/biases) permanently to GPU.
        if nf4_scope not in {"all", "layers"}:
            raise ValueError(f"nf4_scope must be 'all' or 'layers', got {nf4_scope!r}")

        if use_nf4:
            nf4_kwargs = dict(cache_dir=nf4_cache_dir, verbose=verbose, min_numel=nf4_min_numel)
            nf4_modules = []
            if nf4_scope == "all":
                nf4_modules.append(pipeline.prefix)
            nf4_modules.extend(stages)
            if nf4_scope == "all":
                nf4_modules.append(pipeline.postfix)

            _is_meta_model = any(
                param.device.type == "meta"
                for _mod in nf4_modules
                for param in _mod.parameters()
            )

            def _resolve_hf_source() -> str:
                _hf_name = (
                    hf_model_name_or_path
                    or getattr(hf_model, "name_or_path", None)
                    or getattr(core, "name_or_path", None)
                )
                if not _hf_name:
                    raise RuntimeError(
                        "staged FP16 load requires hf_model_name_or_path"
                    )
                return _hf_name

            if _is_meta_model:
                # Single streaming pass per module: check NF4 cache first for
                # each param, quantize cache misses inline, load small non-NF4
                # params as FP16.  Peak RSS = accumulated NF4 payloads + one
                # FP16 tensor rather than the entire module in FP16.
                from stratum.upload import stream_load_and_quantize_module
                from stratum.utils import release_cached_memory as _rcm
                _hf_name = _resolve_hf_source()
                for idx, _mod in enumerate(nf4_modules):
                    _stats = stream_load_and_quantize_module(
                        _mod,
                        _hf_name,
                        min_numel=nf4_min_numel,
                        cache_dir=nf4_cache_dir,
                        blocksize=nf4_kwargs.get("blocksize", 64),
                        quant_type=nf4_kwargs.get("quant_type", "nf4"),
                        verbose=verbose,
                    )
                    if _stats is None:
                        raise RuntimeError(f"stream NF4 build failed for module {idx}")
                    _rcm()
                if verbose:
                    vwrite("  stream NF4 build: all modules processed")
            else:
                # Model already on CPU: params populated, just quantize.
                for _mod in nf4_modules:
                    prepare_nf4(_mod, **nf4_kwargs)
        else:
            fp16_kwargs = dict(verbose=verbose, min_numel=nf4_min_numel)
            prepare_fp16_staged(pipeline.prefix, **fp16_kwargs)
            for stage in stages:
                prepare_fp16_staged(stage, **fp16_kwargs)
            prepare_fp16_staged(pipeline.postfix, **fp16_kwargs)

        # After NF4/FP16 prep the original FP16 model weights are freed
        # (param.data = empty(0)).  Force the allocator to release the
        # cached pages back to the OS so RSS reflects our actual memory
        # footprint, not stale heap pages.
        from stratum.utils import release_cached_memory
        release_cached_memory()

        # Phase 2: Upload non-staged params (trainable LoRA, small norms/biases) permanently.
        # NF4-eligible frozen weights and FP16-staged frozen weights stay on CPU and are
        # uploaded per-step by ensure_weights()/free_weights() in the pipeline.
        def _is_staged(param: torch.nn.Parameter) -> bool:
            return hasattr(param, NF4_ATTR) or hasattr(param, FP16_ATTR)

        for name, param in pipeline.prefix.named_parameters():
            if not _is_staged(param) and param.device.type != "meta":
                param.data = param.data.to(f"cuda:{device_ids[0]}", non_blocking=False)
        for dev in set(s.device_id for s in stages):
            for s in stages:
                if s.device_id != dev:
                    continue
                for name, param in s.named_parameters():
                    if not _is_staged(param) and param.device.type != "meta":
                        param.data = param.data.to(f"cuda:{dev}", non_blocking=False)
                for buf in s.buffers():
                    if buf.device.type != "meta":
                        buf.data = buf.data.to(f"cuda:{dev}", non_blocking=False)
        last_device = stages[-1].device_id if stages else device_ids[0]
        for name, param in pipeline.postfix.named_parameters():
            if not _is_staged(param) and param.device.type != "meta":
                param.data = param.data.to(f"cuda:{last_device}", non_blocking=False)

        return pipeline


_registry: dict[str, type[ModelArch]] = {}


def register(name: str):
    """Decorator: register a ModelArch class under *name*."""
    def _inner(cls):
        if not issubclass(cls, ModelArch):
            raise TypeError(f"{cls.__name__} must inherit from ModelArch")
        _registry[name] = cls
        return cls
    return _inner


def build_pipeline(
    model_name: str,
    hf_model: nn.Module,
    tensor_split: Optional[list[float]] = None,
    device_ids: Optional[list[int]] = None,
    *,
    use_nf4: bool = True,
    nf4_cache_dir: Optional[str] = None,
    nf4_scope: str = "all",
    nf4_min_numel: int = 4096,
    checkpoint_decoder_layer: bool = False,
    loss_token_chunk_size: int = 4096,
    postfix_loss_token_chunk_size: int = 0,
    memory_telemetry: bool = False,
    debug_finite: bool = False,
    checkpoint_mlp: bool = False,
    memory_flat_frozen_mlp: bool = False,
    mlp_token_chunk_size: int = 0,
    torch_compile_loss: bool = False,
    stage_memory_limit_gib: float = 0.0,
    nf4_layer_size_floor_gib: float = 0.0,
    prefetch_nf4: bool = False,
    offload_stage_inputs: bool = False,
    hf_model_name_or_path: Optional[str] = None,
    flash_layers: str = "",
    flash_window_left: int = -1,
    flash_window_right: int = 0,
    dense_attention_masks: bool = False,
    output_router_logits: bool = False,
    router_aux_loss_coef: float = 0.0,
) -> StratumPipeline:
    """Build a StratumPipeline for a registered model architecture."""
    if model_name not in _registry:
        available = ", ".join(sorted(_registry.keys()))
        raise ValueError(
            f"Unknown model '{model_name}'. Available: {available}"
        )
    arch = _registry[model_name]()
    return arch.build(
        hf_model, tensor_split=tensor_split, device_ids=device_ids,
        use_nf4=use_nf4, nf4_cache_dir=nf4_cache_dir,
        nf4_scope=nf4_scope, nf4_min_numel=nf4_min_numel,
        output_router_logits=output_router_logits,
        router_aux_loss_coef=router_aux_loss_coef,
        checkpoint_decoder_layer=checkpoint_decoder_layer,
        loss_token_chunk_size=loss_token_chunk_size,
        postfix_loss_token_chunk_size=postfix_loss_token_chunk_size,
        memory_telemetry=memory_telemetry,
        debug_finite=debug_finite,
        checkpoint_mlp=checkpoint_mlp,
        memory_flat_frozen_mlp=memory_flat_frozen_mlp,
        mlp_token_chunk_size=mlp_token_chunk_size,
        torch_compile_loss=torch_compile_loss,
        stage_memory_limit_gib=stage_memory_limit_gib,
        nf4_layer_size_floor_gib=nf4_layer_size_floor_gib,
        prefetch_nf4=prefetch_nf4,
        offload_stage_inputs=offload_stage_inputs,
        hf_model_name_or_path=hf_model_name_or_path,
        flash_layers=flash_layers,
        flash_window_left=flash_window_left,
        flash_window_right=flash_window_right,
        dense_attention_masks=dense_attention_masks,
    )
